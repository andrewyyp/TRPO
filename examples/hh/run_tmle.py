import sys
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field

# ==============================================================================
# Path Fix
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# [Standard Imports] 不再需要 unsloth
from trl import ScriptArguments, ModelConfig, get_quantization_config, get_kbit_device_map
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    HfArgumentParser,
    BitsAndBytesConfig # 用于 4-bit 量化
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trainer.drpo_config import DRPOConfig

# ==============================================================================
# Arguments
# ==============================================================================
@dataclass
class ExperimentArgs:
    enable_tmle: bool = field(
        default=False,
        metadata={"help": "Whether to enable TMLE bias correction. If False, runs standard RLOO."}
    )
    total_rounds: int = field(
        default=50,
        metadata={"help": "Total training rounds (steps)."}
    )

# ==============================================================================
# Safe Reward Model Wrapper
# ==============================================================================
class SafeRewardModelWrapper(torch.nn.Module):
    def __init__(self, model_name, quantization_config=None, policy_tokenizer=None):
        super().__init__()
        print(f"Loading Reward Model safely: {model_name}")
        self.device = "cuda"
        self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.policy_tokenizer = policy_tokenizer
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, quantization_config=quantization_config,
                device_map=self.device, trust_remote_code=True, ignore_mismatched_sizes=True
            )
        except:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=1, quantization_config=quantization_config,
                device_map=self.device, trust_remote_code=True, ignore_mismatched_sizes=True
            )
            
        self.is_classifier = (self.model.config.num_labels > 1)
        self.vocab_size = self.model.config.vocab_size

    def get_scalar_reward(self, input_ids):
        texts = self.policy_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        rm_inputs = self.rm_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        
        # 越界熔断
        safe_ids = torch.where(rm_inputs["input_ids"] >= self.vocab_size, 
                               torch.tensor(self.rm_tokenizer.unk_token_id).to(self.device), 
                               rm_inputs["input_ids"])
        
        outputs = self.model(safe_ids, attention_mask=rm_inputs["attention_mask"])
        
        if self.is_classifier:
            reward = outputs.logits[:, 1] - outputs.logits[:, 0]
        else:
            reward = outputs.logits.squeeze(-1)
        return reward

# ==============================================================================
# Helper: Log Prob 计算
# ==============================================================================
def compute_log_probs(model, input_ids, attention_mask, labels_ids, labels_mask):
    full_ids = torch.cat([input_ids, labels_ids], dim=1)
    full_mask = torch.cat([attention_mask, labels_mask], dim=1)

    outputs = model(full_ids, attention_mask=full_mask)
    logits = outputs.logits[:, :-1, :]
    labels_target = full_ids[:, 1:]
    
    start_idx = input_ids.shape[1] - 1
    if start_idx >= logits.shape[1]: start_idx = 0
    
    resp_logits = logits[:, start_idx:, :]
    resp_labels = labels_target[:, start_idx:]
    
    min_len = min(resp_logits.shape[1], labels_ids.shape[1])
    
    log_probs = F.log_softmax(resp_logits[:, :min_len, :], dim=-1)
    token_log_probs = torch.gather(log_probs, 2, resp_labels[:, :min_len].unsqueeze(-1)).squeeze(-1)
    
    sum_log_probs = (token_log_probs * labels_mask[:, :min_len]).sum(dim=1)
    return sum_log_probs

# ==============================================================================
# TMLE Logic
# ==============================================================================
def solve_epsilon(policy_model, ref_model, reward_model, offline_loader, device="cuda"):
    print(">>> [TMLE] Solving Epsilon on Offline Data...")
    policy_model.eval()
    numerator = 0.0
    denominator = 0.0
    max_steps = 15 
    
    for step, batch in enumerate(tqdm(offline_loader, total=max_steps)):
        if step >= max_steps: break
        
        prompt_ids = batch['prompt_ids'].to(device)
        prompt_mask = batch['prompt_attention_mask'].to(device)
        a1_ids = batch['a1_ids'].to(device)
        a1_mask = batch['a1_attention_mask'].to(device)
        Y = torch.ones(prompt_ids.shape[0]).to(device)
        
        with torch.no_grad():
            # 这里不需要显式 enable adapter，因为 policy_model 默认就是带着 LoRA 的
            log_pi = compute_log_probs(policy_model, prompt_ids, prompt_mask, a1_ids, a1_mask)
            
            # Ref Model: 显式 disable adapter
            with policy_model.disable_adapter():
                log_ref = compute_log_probs(policy_model, prompt_ids, prompt_mask, a1_ids, a1_mask)
            
            w = torch.exp(log_pi - log_ref)
            w = torch.clamp(w, max=5.0) 
            
            full_ids = torch.cat([prompt_ids, a1_ids], dim=1)
            r_hat = reward_model.get_scalar_reward(full_ids)
            
            numerator += (w * (Y - r_hat)).sum().item()
            denominator += (w ** 2).sum().item()
            
    epsilon = numerator / (denominator + 1e-8)
    print(f">>> [TMLE] Solved Epsilon: {epsilon:.4f}")
    return epsilon

# ==============================================================================
# Main Loop
# ==============================================================================
def main():
    parser = HfArgumentParser((ScriptArguments, DRPOConfig, ModelConfig, ExperimentArgs))
    script_args, training_args, model_args, exp_args = parser.parse_args_into_dataclasses()
    
    mode_name = "TMLE" if exp_args.enable_tmle else "Baseline_RLOO"
    print(f"\n{'='*40}\nRunning Mode: {mode_name} (Native HF)\n{'='*40}\n")
    
    print(f"Loading Policy Model: {model_args.model_name_or_path}")
    
    # --- 1. Load Policy (Standard HF + BitsAndBytes) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 必须预处理以支持 k-bit 训练
    policy_model = prepare_model_for_kbit_training(policy_model)
    
    # 挂载 LoRA
    print("Adding LoRA adapters...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(policy_model, peft_config)
    policy_model.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Ref Model 逻辑：我们复用 policy_model，通过 disable_adapter() 实现
    ref_model = policy_model 
    
    # --- 2. Load Reward Model ---
    rm_id = training_args.preference_model_id or "sfairXC/Fewer-More-Labels-3B"
    # 这里不需要量化 config，让 wrapper 自己处理
    reward_model = SafeRewardModelWrapper(rm_id, quantization_config=None, policy_tokenizer=tokenizer)

    # --- 3. Dataset ---
    print(f"Loading dataset: {script_args.dataset_name}")
    raw_dataset = load_dataset(script_args.dataset_name, split="train[:1%]")
    
    def simple_tokenize(examples):
        prompts, a1s = [], []
        for chosen in examples['chosen']:
            split_text = chosen.split('\n\nAssistant:')
            if len(split_text) > 1:
                prompts.append(split_text[0] + '\n\nAssistant:')
                a1s.append(split_text[1])
            else:
                prompts.append(chosen[:20]); a1s.append(chosen[20:])
                
        tokenized = tokenizer(prompts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        a1_tokenized = tokenizer(a1s, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return {
            "prompt_ids": tokenized["input_ids"], "prompt_attention_mask": tokenized["attention_mask"],
            "a1_ids": a1_tokenized["input_ids"], "a1_attention_mask": a1_tokenized["attention_mask"],
            "rank": torch.ones(len(prompts)) 
        }

    tokenized_ds = raw_dataset.map(simple_tokenize, batched=True, remove_columns=raw_dataset.column_names)
    tokenized_ds.set_format(type="torch")
    offline_loader = DataLoader(tokenized_ds, batch_size=2) 
    
    optimizer = AdamW(policy_model.parameters(), lr=training_args.learning_rate)
    
    # --- 4. Training Loop ---
    online_batch_size = 4 
    history = {"epsilon": [], "reward": [], "loss": []}
    
    for round_idx in range(exp_args.total_rounds):
        print(f"\n=== Round {round_idx}/{exp_args.total_rounds} [{mode_name}] ===")
        
        # [TMLE Step]
        epsilon_k = 0.0
        if exp_args.enable_tmle:
            epsilon_k = solve_epsilon(policy_model, ref_model, reward_model, offline_loader)
        
        history["epsilon"].append(epsilon_k)
        
        # [RLOO Step]
        try:
            batch = next(iter(offline_loader))
        except StopIteration:
            offline_loader = DataLoader(tokenized_ds, batch_size=2)
            batch = next(iter(offline_loader))
            
        single_prompt_id = batch['prompt_ids'][0].unsqueeze(0) 
        single_prompt_mask = batch['prompt_attention_mask'][0].unsqueeze(0)
        
        p_ids = single_prompt_id.repeat(online_batch_size, 1).to(policy_model.device)
        p_mask = single_prompt_mask.repeat(online_batch_size, 1).to(policy_model.device)
        
        # Rollout
        policy_model.eval()
        with torch.no_grad():
            outputs = policy_model.generate(
                input_ids=p_ids, attention_mask=p_mask, 
                max_new_tokens=64, do_sample=True, temperature=0.7, 
                pad_token_id=tokenizer.pad_token_id
            )
            
        resp_ids = outputs[:, p_ids.shape[1]:]
        resp_mask = torch.ones_like(resp_ids)
        
        with torch.no_grad():
            r_hat = reward_model.get_scalar_reward(outputs)
            
            # Calc w
            # Policy (with LoRA)
            log_pi = compute_log_probs(policy_model, p_ids, p_mask, resp_ids, resp_mask)
            
            # Ref (without LoRA)
            with policy_model.disable_adapter():
                log_ref = compute_log_probs(policy_model, p_ids, p_mask, resp_ids, resp_mask)
            
            w = torch.exp(log_pi - log_ref)
            w = torch.clamp(w, max=5.0)
            
            # TMLE Reward
            raw_rewards = r_hat + epsilon_k * w
            
            # Normalize
            rewards = raw_rewards # Copy
            if rewards.std() > 1e-6:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                rewards = rewards - rewards.mean()
                
            print(f"   r_hat: {r_hat.mean():.2f} | w: {w.mean():.2f} | Raw_Avg: {raw_rewards.mean():.4f}")
            history["reward"].append(raw_rewards.mean().item())

        # Backward
        policy_model.train()
        # 必须显式开启梯度检查点，否则backward会报错
        policy_model.gradient_checkpointing_enable() 
        
        full_mask = torch.cat([p_mask, resp_mask], dim=1)
        model_out = policy_model(outputs, attention_mask=full_mask)
        
        l_out = F.log_softmax(model_out.logits[:, :-1, :], dim=-1)
        start = p_ids.shape[1] - 1
        labels = outputs[:, 1:][:, start:]
        min_l = min(l_out[:, start:, :].shape[1], labels.shape[1])
        logits_s = l_out[:, start : start+min_l, :]
        labels_s = labels[:, :min_l]
        
        lp = torch.gather(logits_s, 2, labels_s.unsqueeze(-1)).squeeze(-1)
        slp = lp.sum(dim=1)
        
        k_val = rewards.shape[0]
        if k_val > 1:
            baseline = (rewards.sum() - rewards) / (k_val - 1)
            adv = rewards - baseline
        else:
            adv = rewards - rewards.mean()
            
        loss = - (slp * adv).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        
        print(f">>> Step Loss: {loss.item():.4f}")
        history["loss"].append(loss.item())

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(history["reward"], label=f"{mode_name} Reward")
    plt.title(f"Performance: {mode_name}")
    plt.savefig(os.path.join(training_args.output_dir, f"result_{mode_name}.png"))
    print("Done.")

if __name__ == "__main__":
    main()