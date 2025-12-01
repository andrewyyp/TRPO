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

# =========================================================
# 1. 导入他们的核心 Trainer (保证算法原汁原味)
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 直接使用他们定义的 Trainer 和 Config
from trainer.drpo_trainer import DRPOTrainer
from trainer.drpo_config import DRPOConfig

# =========================================================
# 2. 注入我们的 Safe Reward Model (防止 Tokenizer 报错)
# =========================================================
class SafeRewardModelWrapper(torch.nn.Module):
    def __init__(self, model_name, policy_tokenizer):
        super().__init__()
        self.device = "cuda"
        self.rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.policy_tokenizer = policy_tokenizer
        
        # 强制加载逻辑 (Num Labels Check)
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
        # [关键修复] Detokenize (Qwen) -> Retokenize (RoBERTa)
        texts = self.policy_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        rm_inputs = self.rm_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        
        # [关键修复] 越界熔断
        safe_ids = torch.where(rm_inputs["input_ids"] >= self.vocab_size, 
                               torch.tensor(self.rm_tokenizer.unk_token_id).to(self.device), 
                               rm_inputs["input_ids"])
        
        outputs = self.model(safe_ids, attention_mask=rm_inputs["attention_mask"])
        
        # DRPO 需要返回 probability 或者 score，这里我们返回 score
        if self.is_classifier:
            # logits[1] - logits[0]
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

    # --- Load Policy Model (Standard HF, to be safe with their Trainer) ---
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=get_kbit_device_map() if quantization_config else None,
        quantization_config=quantization_config,
    )
    
    policy_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.padding_side = "left" # Fix warning
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # --- Load Reward Model (Our Safe Wrapper) ---
    rm_id = training_args.preference_model_id or "sfairXC/Fewer-More-Labels-3B"
    # 这里我们把 SafeWrapper 传给 preference_model 参数
    # DRPOTrainer 内部会调用 preference_model(input_ids, ...)
    reward_model = SafeRewardModelWrapper(rm_id, policy_tokenizer=tokenizer)

    # --- Prepare Dataset (Format for DRPO) ---
    # DRPO 需要 columns: ['prompt', 'a1', 'a2', 'rank']
    # a1 是 response 1, a2 是 response 2. rank=0 表示 a1 胜, rank=1 表示 a2 胜
    dataset = load_dataset(script_args.dataset_name, split="train[:1%]")
    
    def format_drpo(example):
        # Anthropic HH format: "Human: ... \n\nAssistant: ..."
        # Chosen -> a1, Rejected -> a2, Rank -> 0 (Always a1 > a2)
        
        # Simple extraction
        chosen = example['chosen']
        rejected = example['rejected']
        
        # Try to extract pure prompt
        split_c = chosen.split('\n\nAssistant:')
        if len(split_c) > 1:
            prompt = split_c[0] + '\n\nAssistant:'
            resp_c = split_c[1]
            # Try rejected
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
            "rank": 0 # 0 means a1 is preferred
        }

    train_dataset = dataset.map(format_drpo, remove_columns=dataset.column_names)

    # --- Initialize Their Trainer ---
    # 强制覆盖一些关键参数以保证能跑
    training_args.remove_unused_columns = False 
    training_args.is_bt_model = False # 告诉 Trainer 不要走复杂的 BT Pipeline，直接调用 model()
    
    trainer = DRPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        preference_model=reward_model, # 注入我们的 Wrapper
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting DRPO Training...")
    trainer.train()
    
    # Save
    trainer.save_model(training_args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()