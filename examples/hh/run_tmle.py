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
# Path Fix
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trl import ScriptArguments, ModelConfig, get_quantization_config, get_kbit_device_map
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    HfArgumentParser,
    BitsAndBytesConfig
)
from trainer.drpo_config import DRPOConfig

# ==============================================================================
# [FIXED v2] 智能 Reward Model Wrapper (带越界熔断机制)
# ==============================================================================
class SafeRewardModelWrapper(torch.nn.Module):
    def __init__(self, model_name, quantization_config=None, policy_tokenizer=None):
        super().__init__()
        print(f"Loading Reward Model safely: {model_name}")
        self.device = "cuda"
        
        # 1. 加载 Reward Model 自己的 Tokenizer
        self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 保存 Policy 的 Tokenizer 用于解码
        self.policy_tokenizer = policy_tokenizer
        
        # 2. 加载模型
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=self.device, 
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            print(f"Warning: Loading with num_labels=1 due to error: {e}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
            
        if hasattr(self.model.config, "num_labels") and self.model.config.num_labels > 1:
            self.is_classifier = True
        else:
            self.is_classifier = False
            
        # [NEW] 获取 Reward Model 的词表上限
        self.vocab_size = self.model.config.vocab_size
        print(f">>> Reward Model Vocab Size: {self.vocab_size}")

    def forward(self, input_ids, attention_mask):
        # 1. Detokenize: 把 Policy (Qwen) 的 ID 转回 Text
        # 必须处理，因为 Qwen 的 ID 空间和 RoBERTa 完全不同
        texts = self.policy_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # 2. Retokenize: 转成 RoBERTa ID
        rm_inputs = self.rm_tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        # [CRITICAL FIX] 物理熔断：强制将越界 ID 替换为 UNK
        # RoBERTa 的 vocab size 通常是 50265
        # 如果 rm_tokenizer 偶尔发疯产生了大 ID，这里直接拦截
        rm_input_ids = rm_inputs["input_ids"]
        rm_attention_mask = rm_inputs["attention_mask"]
        
        # 将所有 >= vocab_size 的 ID 替换为 unk_token_id (通常是 3)
        # 或者 pad_token_id (1)
        safe_input_ids = torch.where(
            rm_input_ids >= self.vocab_size, 
            torch.tensor(self.rm_tokenizer.unk_token_id).to(self.device), 
            rm_input_ids
        )
        
        # 3. Forward Pass
        outputs = self.model(safe_input_ids, attention_mask=rm_attention_mask)
        logits = outputs.logits
        
        if self.is_classifier:
            reward = logits[:, 1] - logits[:, 0]
        else:
            reward = logits.squeeze(-1)
            
        return reward

# ==============================================================================
# Helper: Log Prob 计算
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
    
    # 简单的容错处理：如果 start_idx 越界
    if start_idx >= logits.shape[1]:
        start_idx = 0 
    
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
# TMLE Logic
# ==============================================================================
def solve_epsilon(policy_model, ref_model, reward_network, offline_loader, device="cuda"):
    print(">>> [TMLE] Solving Epsilon on Offline Data...")
    policy_model.eval()
    
    numerator = 0.0
    denominator = 0.0
    max_steps = 10 
    
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
            
            # [Fix] 传递 input_ids (Qwen format) 给 wrapper
            full_ids = torch.cat([prompt_ids, a1_ids], dim=1).to(device)
            # Wrapper 内部会自动 detokenize -> retokenize
            r_hat = reward_network(full_ids, None) 
            
            residual = Y - r_hat
            
            numerator += (w * residual).sum().item()
            denominator += (w ** 2).sum().item()
            
    epsilon = numerator / (denominator + 1e-8)
    print(f">>> [TMLE] Solved Epsilon: {epsilon:.4f}")
    return epsilon

# ==============================================================================
# Main
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
    
    # 1. Load Policy Models
    policy_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    ref_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load Reward Model (With Tokenizer fix)
    rm_id = training_args.preference_model_id
    if rm_id is None:
        print("Warning: No preference_model_id provided. Using a default one.")
        rm_id = "sfairXC/Fewer-More-Labels-3B" 
        
    # 传入 policy_tokenizer 供解码使用
    reward_model = SafeRewardModelWrapper(rm_id, quantization_config=quantization_config, policy_tokenizer=tokenizer)

    # 3. Prepare Dataset
    print(f"Loading dataset: {script_args.dataset_name}")
    raw_dataset = load_dataset(script_args.dataset_name, split="train[:1%]")
    
    def simple_tokenize(examples):
        prompts = []
        a1s = []
        for chosen in examples['chosen']:
            # 简单的截断逻辑，防止太长
            chosen = chosen[:512] 
            split_text = chosen.split('\n\nAssistant:')
            if len(split_text) > 1:
                prompts.append(split_text[0] + '\n\nAssistant:')
                a1s.append(split_text[1])
            else:
                prompts.append(chosen[:20]) 
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

    tokenized_ds = raw_dataset.map(simple_tokenize, batched=True, remove_columns=raw_dataset.column_names)
    tokenized_ds.set_format(type="torch")
    offline_loader = DataLoader(tokenized_ds, batch_size=1) 
    
    # 4. Loop
    optimizer = AdamW(policy_model.parameters(), lr=training_args.learning_rate)
    epsilon_k = 0.0
    total_rounds = 2
    
    for round_idx in range(total_rounds):
        print(f"\n=== Round {round_idx} Start ===")
        
        epsilon_k = solve_epsilon(policy_model, ref_model, reward_model, offline_loader)
        
        print(">>> Running Online RLOO Step...")
        dummy_inputs = tokenizer(["\n\nHuman: Hello\n\nAssistant:"], return_tensors="pt").to(policy_model.device)
        policy_model.train()
        optimizer.zero_grad()
        gen_out = policy_model.generate(**dummy_inputs, max_new_tokens=10, do_sample=True)
        print(f"Round {round_idx} Finished. Epsilon used: {epsilon_k}")

if __name__ == "__main__":
    main()