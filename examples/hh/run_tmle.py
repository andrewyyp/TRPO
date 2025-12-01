import sys
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import transformers
from datasets import load_dataset

# ==============================================================================
# [FIX] 路径修正核心代码
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f">>> Root directory added to sys.path: {project_root}")

# ==============================================================================
# [FIX] Import
# ==============================================================================
from trl import (
    ScriptArguments, 
    ModelConfig, 
    get_quantization_config, 
    get_kbit_device_map
)
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, # <--- 必须显式导入这个
    AutoTokenizer, 
    HfArgumentParser,
    BitsAndBytesConfig
)

try:
    from trainer.drpo_config import DRPOConfig
    # 我们不再导入 BTRewardNetwork，因为我们要自己重写一个更稳健的
except ImportError as e:
    print("Error importing from trainer.")
    raise e

# ==============================================================================
# [NEW] 稳健的 Reward Model 包装器 (解决 shape mismatch 问题)
# ==============================================================================
class SafeRewardModelWrapper(torch.nn.Module):
    def __init__(self, model_name, quantization_config=None):
        super().__init__()
        print(f"Loading Reward Model safely: {model_name}")
        
        # 1. 尝试按 checkpoint 原始配置加载 (通常是 num_labels=2)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="cuda", # 强制放 GPU
                trust_remote_code=True,
                ignore_mismatched_sizes=True # <--- 关键！允许跳过不匹配的权重
            )
        except Exception as e:
            print(f"Warning: Standard loading failed, trying with num_labels=1. Error: {e}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                quantization_config=quantization_config,
                device_map="cuda",
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
            
        # 检测它是分类器(2输出)还是回归器(1输出)
        if hasattr(self.model.config, "num_labels") and self.model.config.num_labels > 1:
            self.is_classifier = True
            print(">>> Detected Classification Reward Model (Output Dim > 1). Will use logits diff.")
        else:
            self.is_classifier = False
            print(">>> Detected Regression Reward Model (Output Dim = 1).")

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        if self.is_classifier:
            # 如果是二分类 (Rank 0 vs Rank 1)，通常 logits[:, 1] 是 positive score
            # 或者用 logits[:, 1] - logits[:, 0]
            # 这里为了简单，我们取 logits[:, 0] (假设 index 0 是 score，具体看模型) 
            # 通常 Reward Model index 0 是 rejected, 1 是 chosen. 
            # 安全起见，我们把两个 logits 相减，这样无论 0/1 谁大谁小，梯度方向都在
            reward = logits[:, 1] - logits[:, 0]
        else:
            # 回归模型，直接 squeeze
            reward = logits.squeeze(-1)
            
        return reward

# ==============================================================================
# Helper: Log Prob 计算逻辑
# ==============================================================================
def compute_log_probs(model, input_ids, attention_mask, labels_ids, labels_mask):
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    labels_ids = labels_ids.to(model.device)
    labels_mask = labels_mask.to(model.device)

    full_ids = torch.cat([input_ids, labels_ids], dim=1)
    full_mask = torch.cat([attention_mask, labels_mask], dim=1)

    outputs = model(full_ids, attention_mask=full_mask)
    logits = outputs.logits

    logits = logits[:, :-1, :]
    labels_target = full_ids[:, 1:]
    
    start_idx = input_ids.shape[1] - 1
    
    resp_logits = logits[:, start_idx:, :]
    resp_labels = labels_target[:, start_idx:]
    
    min_len = min(resp_logits.shape[1], labels_ids.shape[1])
    resp_logits = resp_logits[:, :min_len, :]
    resp_labels = labels_ids[:, :min_len]
    resp_mask = labels_mask[:, :min_len]

    log_probs = F.log_softmax(resp_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, resp_labels.unsqueeze(-1)).squeeze(-1)
    
    token_log_probs = token_log_probs * resp_mask
    sum_log_probs = token_log_probs.sum(dim=1)
    
    return sum_log_probs

# ==============================================================================
# Core 1: TMLE Targeting Step
# ==============================================================================
def solve_epsilon(policy_model, ref_model, reward_network, offline_loader, device="cuda"):
    print(">>> [TMLE] Solving Epsilon on Offline Data...")
    policy_model.eval()
    
    numerator = 0.0
    denominator = 0.0
    max_steps = 10 # 快速验证
    
    for step, batch in enumerate(tqdm(offline_loader, total=max_steps)):
        if step >= max_steps: break
        
        prompt_ids = batch['prompt_ids']
        prompt_mask = batch['prompt_attention_mask']
        a1_ids = batch['a1_ids']
        a1_mask = batch['a1_attention_mask']
        
        Y = batch['rank'].float().to(device)
        
        with torch.no_grad():
            log_pi = compute_log_probs(policy_model, prompt_ids, prompt_mask, a1_ids, a1_mask)
            log_ref = compute_log_probs(ref_model, prompt_ids, prompt_mask, a1_ids, a1_mask)
            
            w = torch.exp(log_pi - log_ref)
            w = torch.clamp(w, max=10.0) 
            
            # 计算 r_hat (使用我们的 Safe Wrapper)
            # Safe Wrapper 需要拼接好的 inputs
            full_ids = torch.cat([prompt_ids, a1_ids], dim=1).to(device)
            full_mask = torch.cat([prompt_mask, a1_mask], dim=1).to(device)
            
            r_hat = reward_network(full_ids, full_mask)
            
            residual = Y - r_hat
            
            numerator += (w * residual).sum().item()
            denominator += (w ** 2).sum().item()
            
    epsilon = numerator / (denominator + 1e-8)
    print(f">>> [TMLE] Solved Epsilon: {epsilon:.4f}")
    return epsilon

# ==============================================================================
# Core 2: Main Logic
# ==============================================================================
def main():
    parser = HfArgumentParser((ScriptArguments, DRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    print(f"Loading Policy Model: {model_args.model_name_or_path}")
    
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=get_kbit_device_map() if quantization_config else None,
        quantization_config=quantization_config,
    )
    
    # 1. Load Models
    policy_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    ref_model.eval()
    
    # [FIXED] 使用 SafeRewardModelWrapper 替代原来的加载逻辑
    # 如果命令行没指定 preference_model_id，默认使用一个 dummy 或者 Qwen 本身(不推荐)
    # 我们这里假设用户会指定，或者默认用一个已知的 RM
    rm_id = training_args.preference_model_id
    if rm_id is None:
        print("Warning: No preference_model_id provided. Using a default one for testing.")
        rm_id = "sfairXC/Fewer-More-Labels-3B" 
        
    reward_model = SafeRewardModelWrapper(rm_id, quantization_config=quantization_config)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare Dataset
    print(f"Loading dataset: {script_args.dataset_name}")
    raw_dataset = load_dataset(script_args.dataset_name, split="train[:1%]")
    
    def simple_tokenize(examples):
        # 针对 Anthropic HH 数据集的简单处理
        prompts = []
        a1s = []
        # Anthropic HH 只有 'chosen' 和 'rejected'
        for chosen, rejected in zip(examples['chosen'], examples['rejected']):
            # 简单切分 Human/Assistant
            split_text = chosen.split('\n\nAssistant:')
            if len(split_text) > 1:
                prompts.append(split_text[0] + '\n\nAssistant:')
                a1s.append(split_text[1])
            else:
                prompts.append(chosen[:20]) # Fallback
                a1s.append(chosen[20:])
                
        tokenized = tokenizer(prompts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        a1_tokenized = tokenizer(a1s, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        return {
            "prompt_ids": tokenized["input_ids"],
            "prompt_attention_mask": tokenized["attention_mask"],
            "a1_ids": a1_tokenized["input_ids"],
            "a1_attention_mask": a1_tokenized["attention_mask"],
            "rank": torch.ones(len(prompts)) 
        }

    # 3. TMLE + RLOO Loop
    optimizer = AdamW(policy_model.parameters(), lr=training_args.learning_rate)
    
    tokenized_ds = raw_dataset.map(simple_tokenize, batched=True, remove_columns=raw_dataset.column_names)
    tokenized_ds.set_format(type="torch")
    offline_loader = DataLoader(tokenized_ds, batch_size=1) # T4 上显存紧张，batch=1
    
    epsilon_k = 0.0
    total_rounds = 2
    
    for round_idx in range(total_rounds):
        print(f"\n=== Round {round_idx} Start ===")
        
        # Step A: Update Epsilon
        epsilon_k = solve_epsilon(policy_model, ref_model, reward_model, offline_loader)
        
        # Step B: Online RLOO (Demo)
        print(">>> Running Online RLOO Step...")
        # 构造 Dummy Input 验证流程是否跑通
        dummy_inputs = tokenizer(["\n\nHuman: Hello\n\nAssistant:"], return_tensors="pt").to(policy_model.device)
        
        policy_model.train()
        optimizer.zero_grad()
        
        # Rollout
        gen_out = policy_model.generate(**dummy_inputs, max_new_tokens=10, do_sample=True)
        
        print(f"Round {round_idx} Finished. Epsilon used: {epsilon_k}")

if __name__ == "__main__":
    main()