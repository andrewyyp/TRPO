import sys
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    HfArgumentParser
)
from trl import ModelConfig, ScriptArguments, get_quantization_config, get_kbit_device_map

# [NEW] 引入 PEFT 库用于 LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =========================================================
# 1. 导入他们的核心 Trainer
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trainer.drpo_trainer import DRPOTrainer
from trainer.drpo_config import DRPOConfig

# =========================================================
# 2. 注入我们的 Safe Reward Model
# =========================================================
class SafeRewardModelWrapper(torch.nn.Module):
    def __init__(self, model_name, policy_tokenizer):
        super().__init__()
        self.device = "cuda"
        self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.policy_tokenizer = policy_tokenizer
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, device_map=self.device, trust_remote_code=True, ignore_mismatched_sizes=True
            )
        except:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=1, device_map=self.device, trust_remote_code=True, ignore_mismatched_sizes=True
            )
        
        self.is_classifier = (self.model.config.num_labels > 1)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, input_ids, attention_mask=None):
        texts = self.policy_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        rm_inputs = self.rm_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        
        # 越界熔断
        safe_ids = torch.where(rm_inputs["input_ids"] >= self.vocab_size, 
                               torch.tensor(self.rm_tokenizer.unk_token_id).to(self.device), 
                               rm_inputs["input_ids"])
        
        outputs = self.model(safe_ids, attention_mask=rm_inputs["attention_mask"])
        
        if self.is_classifier:
            return outputs.logits[:, 1] - outputs.logits[:, 0]
        else:
            return outputs.logits.squeeze(-1)

# =========================================================
# 3. 主函数
# =========================================================
def main():
    parser = HfArgumentParser((ScriptArguments, DRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    print(f"--- Running Official DRPO Baseline ---")
    print(f"Model: {model_args.model_name_or_path}")

    # --- Load Policy Model ---
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=get_kbit_device_map() if quantization_config else None,
        quantization_config=quantization_config,
    )
    
    print("Loading Policy Model (4-bit)...")
    policy_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    
    # [FIX] 必须为 4-bit 训练做准备 (LayerNorm fp32 conversion)
    policy_model = prepare_model_for_kbit_training(policy_model)

    # [FIX] 挂载 LoRA Adapters (否则无法训练)
    print("Attaching LoRA Adapters...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 针对 Qwen/Llama
    )
    policy_model = get_peft_model(policy_model, peft_config)
    policy_model.print_trainable_parameters() # 打印确认一下

    # --- Load Ref Model ---
    # Ref Model 也是 4-bit，但是它是冻结的，不需要 LoRA
    print("Loading Ref Model (4-bit)...")
    ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    ref_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # --- Load Reward Model ---
    rm_id = training_args.preference_model_id or "sfairXC/Fewer-More-Labels-3B"
    reward_model = SafeRewardModelWrapper(rm_id, policy_tokenizer=tokenizer)

    # --- Dataset ---
    dataset = load_dataset(script_args.dataset_name, split="train[:1%]")
    
    def format_drpo(example):
        chosen = example['chosen']
        rejected = example['rejected']
        split_c = chosen.split('\n\nAssistant:')
        if len(split_c) > 1:
            prompt = split_c[0] + '\n\nAssistant:'
            resp_c = split_c[1]
            split_r = rejected.split('\n\nAssistant:')
            resp_r = split_r[1] if len(split_r) > 1 else rejected[-50:]
        else:
            prompt = chosen[:50]
            resp_c = chosen[50:]
            resp_r = rejected[50:]
            
        return {
            "prompt": prompt,
            "a1": resp_c,
            "a2": resp_r,
            "rank": 0
        }

    train_dataset = dataset.map(format_drpo, remove_columns=dataset.column_names)

    # --- Trainer ---
    training_args.remove_unused_columns = False 
    training_args.is_bt_model = False 
    
    trainer = DRPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        preference_model=reward_model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting DRPO Training...")
    trainer.train()
    
    trainer.save_model(training_args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()