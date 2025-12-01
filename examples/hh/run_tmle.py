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
# 1. 获取当前脚本所在目录 (examples/hh)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 获取项目根目录 (TRPO/) - 往上跳两级
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
# 3. 强行将根目录插入到 Python 搜索路径的第一位
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f">>> Root directory added to sys.path: {project_root}")

# ==============================================================================
# [FIX] 修正 Import 方式：直接从 trl 和 trainer 导入，绕过 broken 的 drpo.py
# ==============================================================================
from trl import (
    ScriptArguments, 
    ModelConfig, 
    get_quantization_config, 
    get_kbit_device_map
)
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    HfArgumentParser,
    BitsAndBytesConfig
)

# 直接从 trainer 文件夹导入 (现在 Python 能找到它了)
# 注意：这里假设 drpo_config.py 和 drpo_utils.py 都在 trainer/ 目录下
try:
    from trainer.drpo_config import DRPOConfig
    from trainer.drpo_utils import GPMwithRewardNetwork, BTRewardNetwork
    # 如果你也需要 DataCollator，也可以从这里导
    from trainer.drpo_trainer import DataCollatorDRPO 
except ImportError as e:
    print("Error importing from trainer. Make sure 'trainer' folder has __init__.py")
    raise e

# ==============================================================================
# Helper: Log Prob 计算逻辑
# ==============================================================================
def compute_log_probs(model, input_ids, attention_mask, labels_ids, labels_mask):
    """
    计算 log pi(y|x)
    """
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    labels_ids = labels_ids.to(model.device)
    labels_mask = labels_mask.to(model.device)

    # 拼接
    full_ids = torch.cat([input_ids, labels_ids], dim=1)
    full_mask = torch.cat([attention_mask, labels_mask], dim=1)

    outputs = model(full_ids, attention_mask=full_mask)
    logits = outputs.logits

    # Shift logits
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
    max_steps = 20 # 稍微跑一点数据做演示
    
    for step, batch in enumerate(tqdm(offline_loader, total=max_steps)):
        if step >= max_steps: break
        
        # 这里的 key 需要根据 DataCollatorDRPO 的输出调整
        # DataCollatorDRPO 输出: prompt_ids, a1_ids, a2_ids, rank
        prompt_ids = batch['prompt_ids']
        prompt_mask = batch['prompt_attention_mask']
        a1_ids = batch['a1_ids']
        a1_mask = batch['a1_attention_mask']
        
        # 假设 rank=1 代表 a1 是 chosen (Y=1)
        # 注意: DRPO 代码里 rank 是 tensor([0, 1, 0...])
        Y = batch['rank'].float().to(device)
        
        with torch.no_grad():
            log_pi = compute_log_probs(policy_model, prompt_ids, prompt_mask, a1_ids, a1_mask)
            log_ref = compute_log_probs(ref_model, prompt_ids, prompt_mask, a1_ids, a1_mask)
            
            w = torch.exp(log_pi - log_ref)
            w = torch.clamp(w, max=10.0) # 截断防止数值爆炸
            
            # 计算 r_hat
            full_ids = torch.cat([prompt_ids, a1_ids], dim=1).to(device)
            full_mask = torch.cat([prompt_mask, a1_mask], dim=1).to(device)
            
            # 兼容性处理：有的 RewardModel 返回 tuple, 有的返回 logits
            rm_out = reward_network(full_ids, full_mask)
            if isinstance(rm_out, tuple): rm_out = rm_out[0]
            r_hat = rm_out.squeeze(-1) if rm_out.dim() > 0 else rm_out
            
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
    
    print(f"Loading Model: {model_args.model_name_or_path}")
    
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
    
    if training_args.is_bt_model:
        reward_model = BTRewardNetwork(training_args.preference_model_id)
    else:
        reward_model = GPMwithRewardNetwork(training_args.preference_model_id)
    
    # 简单的设备处理
    if quantization_config is None:
        reward_model.to("cuda")
    reward_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None: tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}"

    # 2. Prepare Dataset for Offline Step
    # 这里我们加载一部分数据用于计算 Epsilon
    print(f"Loading dataset: {script_args.dataset_name}")
    raw_dataset = load_dataset(script_args.dataset_name, split="train[:1%]") # 只取 1% 快速验证
    
    # 复用 DRPO 的数据预处理逻辑 (Tokenize)
    # 为了简化，我们手动简易处理一下，或者你可以 copy drpo.py 里的 _prepare_dataset 逻辑
    # 这里做一个极简的 collator 演示
    def simple_tokenize(examples):
        # 简单处理：假设数据里有 prompt, chosen, rejected
        prompts = [x['chosen'].split('\n\nAssistant:')[0] for x in examples['chosen']]
        # 假设 chosen 就是 a1
        a1s = [x['chosen'].split('\n\nAssistant:')[-1] for x in examples['chosen']]
        
        tokenized = tokenizer(prompts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        a1_tokenized = tokenizer(a1s, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        return {
            "prompt_ids": tokenized["input_ids"],
            "prompt_attention_mask": tokenized["attention_mask"],
            "a1_ids": a1_tokenized["input_ids"],
            "a1_attention_mask": a1_tokenized["attention_mask"],
            "rank": torch.ones(len(prompts)) # Dummy rank
        }

    # 3. TMLE + RLOO Loop
    optimizer = AdamW(policy_model.parameters(), lr=training_args.learning_rate)
    
    # 手动构造 DataLoader
    # 注意：实际使用时应该用 DataCollatorDRPO
    tokenized_ds = raw_dataset.map(simple_tokenize, batched=True, remove_columns=raw_dataset.column_names)
    tokenized_ds.set_format(type="torch")
    offline_loader = DataLoader(tokenized_ds, batch_size=2)
    
    epsilon_k = 0.0
    total_rounds = 2 # 演示跑2轮
    
    for round_idx in range(total_rounds):
        print(f"\n=== Round {round_idx} Start ===")
        
        # Step A: Update Epsilon
        epsilon_k = solve_epsilon(policy_model, ref_model, reward_model, offline_loader)
        
        # Step B: Online RLOO (Demo)
        print(">>> Running Online RLOO Step...")
        # 构造一些 Dummy Inputs 跑一下流程
        dummy_inputs = tokenizer(["Human: Hello\n\nAssistant:"], return_tensors="pt").to(policy_model.device)
        
        policy_model.train()
        optimizer.zero_grad()
        
        # Rollout
        gen_out = policy_model.generate(**dummy_inputs, max_new_tokens=10, do_sample=True)
        
        # ... 这里省略具体的 RLOO loss 计算，只证明 epsilon 算出来了 ...
        print(f"Round {round_idx} Finished. Epsilon used: {epsilon_k}")

if __name__ == "__main__":
    main()