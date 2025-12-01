import sys
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# === 1. 复用 DRPO 的基础设施 ===
# 我们直接从 drpo.py 和 trainer 模块导入必要的类和函数
# 确保你的 python path 能找到这些模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from examples.hh.drpo import (
    ScriptArguments, DRPOConfig, ModelConfig, 
    get_quantization_config, get_kbit_device_map, get_peft_config,
    AutoModelForCausalLM, AutoTokenizer, 
    GPMwithRewardNetwork, BTRewardNetwork, 
    load_dataset, SIMPLE_CHAT_TEMPLATE # 复用它的数据加载
)
# 我们需要 DRPOTrainer 里的 _forward 函数来帮我们算 log_prob
# 既然它是类方法，我们稍微 wrap 一下或者把那段逻辑拷出来 (下面我选择直接复刻逻辑以解耦)

import yaml
import transformers

# ==============================================================================
# Helper: 从 DRPOTrainer 移植过来的 Log Prob 计算逻辑 (极为重要)
# ==============================================================================
def compute_log_probs(model, input_ids, attention_mask, labels_ids, labels_mask):
    """
    计算 log pi(y|x)。完全复刻 DRPOTrainer._forward 的逻辑
    """
    # 拼接 Prompt + Response
    # 注意：DRPO 的 collator 已经把 prompt 和 answer 分开了，这里我们需要拼起来喂给模型
    # 为了简化，我们假设 input_ids 就是 prompt, labels_ids 就是 response
    
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    labels_ids = labels_ids.to(model.device)
    labels_mask = labels_mask.to(model.device)

    # 拼接
    full_ids = torch.cat([input_ids, labels_ids], dim=1)
    full_mask = torch.cat([attention_mask, labels_mask], dim=1)

    # Forward
    outputs = model(full_ids, attention_mask=full_mask)
    logits = outputs.logits

    # Shift logits: prediction is next token
    # logits[:, :-1] predicts full_ids[:, 1:]
    logits = logits[:, :-1, :]
    labels_target = full_ids[:, 1:]
    
    # 我们只关心 response 部分的 log prob
    # response 开始的位置是 input_ids.shape[1] - 1 (因为 shift 了)
    start_idx = input_ids.shape[1] - 1
    
    # 提取 response 的 logits 和 labels
    resp_logits = logits[:, start_idx:, :]
    resp_labels = labels_target[:, start_idx:]
    
    # 确保长度对其 (截断)
    min_len = min(resp_logits.shape[1], labels_ids.shape[1])
    resp_logits = resp_logits[:, :min_len, :]
    resp_labels = labels_ids[:, :min_len] # 使用原本的 label ids
    resp_mask = labels_mask[:, :min_len]

    # Gather log probs
    log_probs = F.log_softmax(resp_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, resp_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask 掉 padding
    token_log_probs = token_log_probs * resp_mask
    sum_log_probs = token_log_probs.sum(dim=1)
    
    return sum_log_probs

# ==============================================================================
# Core 1: TMLE Targeting Step (Solving Epsilon)
# ==============================================================================
def solve_epsilon(policy_model, ref_model, reward_network, offline_loader, device="cuda"):
    """
    遍历 Offline 数据，计算 epsilon_k
    Formula: sum [ w * (Y - r_hat - epsilon * w) ] = 0
    => epsilon = sum(w * (Y - r_hat)) / sum(w^2)
    where w = pi_k / pi_sft (density ratio)
    """
    print(">>> [TMLE] Solving Epsilon on Offline Data...")
    policy_model.eval()
    
    numerator = 0.0
    denominator = 0.0
    
    # 只跑一部分数据作为估算，防止太慢
    max_steps = 50 
    
    for step, batch in enumerate(tqdm(offline_loader, total=max_steps)):
        if step >= max_steps: break
        
        # DRPO 的 DataLoader 返回的是 'prompt_ids', 'a1_ids', 'rank' 等
        # 我们假设 rank=1 的是 chosen (Y=1), rank=0 的是 rejected (Y=0)
        # 这里为了简化，我们只用 'a1' (Model A 的回答) 及其对应的 rank 作为 Y
        
        prompt_ids = batch['prompt_ids']
        prompt_mask = batch['prompt_attention_mask']
        a1_ids = batch['a1_ids']
        a1_mask = batch['a1_attention_mask']
        # rank: 0 or 1. If 1, a1 is better (Y=1). If 0, a1 is worse (Y=0)
        # DRPO 数据里 rank=0 通常意味着 a1 是 chosen? 需要确认数据集格式。
        # 假设 rank 指的是 index of chosen. 如果 rank=0, a1 chosen. rank=1, a2 chosen.
        # 暂时假设我们把 a1 当作样本，它的标签 Y = (rank == 0).float()
        Y = (batch['rank'] == 0).float().to(device) 
        
        with torch.no_grad():
            # 1. 计算 Log Probs
            log_pi = compute_log_probs(policy_model, prompt_ids, prompt_mask, a1_ids, a1_mask)
            log_ref = compute_log_probs(ref_model, prompt_ids, prompt_mask, a1_ids, a1_mask)
            
            # 2. Density Ratio w
            w = torch.exp(log_pi - log_ref)
            # Clip w to avoid explosion
            w = torch.clamp(w, max=10.0)
            
            # 3. Estimated Reward r_hat
            # DRPO 的 reward network 需要拼接 prompt + response
            # 注意：这里需要适配 preference_pipeline 的接口
            # 简单起见，假设 preference_model 接受 input_ids 并输出 scalar
            full_ids = torch.cat([prompt_ids, a1_ids], dim=1).to(device)
            full_mask = torch.cat([prompt_mask, a1_mask], dim=1).to(device)
            
            # 调用 Reward Model (假设是 BTRewardNetwork)
            # 它的 forward 通常返回 logits，我们需要 scalar reward
            r_hat = reward_network(full_ids, full_mask).squeeze(-1) 
            
            # 4. Accumulate
            # residual = Y - r_hat
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
    # --- 1. Load Config (Reuse DRPO Logic) ---
    parser = transformers.HfArgumentParser((ScriptArguments, DRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    # --- 2. Load Models (Reuse DRPO Logic) ---
    print("Loading Models...")
    # ... (此处代码直接复用 drpo.py 里的 model loading 部分) ...
    # 为了简洁，我简写核心部分：
    
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=get_kbit_device_map() if quantization_config else None,
        quantization_config=quantization_config,
    )
    
    # Policy Model
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    
    # Ref Model (SFT)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    ref_model.eval()
    
    # Reward Model
    if training_args.is_bt_model:
        reward_model = BTRewardNetwork(training_args.preference_model_id)
    else:
        reward_model = GPMwithRewardNetwork(training_args.preference_model_id)
    reward_model.to("cuda") # 或者是 device_map
    reward_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Load Dataset (For Targeting Step) ---
    # 复用 drpo.py 的 dataset 加载
    raw_dataset = load_dataset(script_args.dataset_name, split="train")
    # 这里需要使用 drpo_trainer 里定义的 DataCollatorDRPO 来处理数据
    from examples.hh.trainer.drpo_trainer import DataCollatorDRPO # type: ignore
    collate_fn = DataCollatorDRPO(pad_token_id=tokenizer.pad_token_id)
    
    # 我们需要自己 wrap 一个 dataset 处理函数，把 text 转成 token ids
    # 这里略过繁琐的数据预处理，假设 offline_loader 已经 ready
    # 实际跑的时候需要把 drpo.py 里的 transform_dataset 和 tokenize_row 拿过来用
    
    # 假设 offline_loader 好了
    offline_loader = DataLoader(raw_dataset, batch_size=4, collate_fn=collate_fn) 

    # --- 4. TMLE + RLOO Loop ---
    
    optimizer = AdamW(policy_model.parameters(), lr=1e-6)
    
    epsilon_k = 0.0
    total_rounds = 5
    
    for round_idx in range(total_rounds):
        print(f"\n=== Round {round_idx} Start ===")
        
        # --- Step A: TMLE Targeting ---
        # 利用 Offline Data 更新 epsilon
        epsilon_k = solve_epsilon(policy_model, ref_model, reward_model, offline_loader)
        
        # --- Step B: Online RLOO Training ---
        # 这里的 inputs 应该是 Prompt Only
        prompts = ["Write a movie review.", "How are you?"] * 4 # Dummy prompts
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        
        policy_model.train()
        optimizer.zero_grad()
        
        # 1. Generate (Rollout)
        with torch.no_grad():
            outputs = policy_model.generate(**inputs, max_new_tokens=20, do_sample=True)
        
        # 2. Get Reward (TMLE Corrected)
        # r_total = r_hat + epsilon * (pi / pi_ref)
        with torch.no_grad():
            # ... Calculate r_hat using reward_model ...
            # ... Calculate w using compute_log_probs ...
            # For demo:
            r_hat = torch.tensor([0.5] * len(prompts)).cuda() # Dummy
            w = torch.tensor([1.0] * len(prompts)).cuda()     # Dummy
            
            # THIS IS YOUR PROPOSAL FORMULA:
            rewards = r_hat + epsilon_k * w
            
        # 3. RLOO Update Logic (Using the code we wrote in Colab)
        # ... Paste the RLOO loss calculation here ...
        # ... loss.backward() ...
        # ... optimizer.step() ...
        
        print(f"Round {round_idx} Finished. Epsilon used: {epsilon_k}")

if __name__ == "__main__":
    main()