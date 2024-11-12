import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model configurations
MODEL_CONFIG = {
    "base_model": "unsloth/Llama-3.2-3B-Instruct",
    "max_seq_length": 2048,
    "load_in_4bit": True,
}

# Training configurations
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "evaluation_strategy": "steps",
    "eval_steps": 10,
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "output_dir": "outputs"
}

# PEFT configurations
PEFT_CONFIG = {
    "r": 16,
    "target_modules": [
        "up_proj", "down_proj", "gate_proj", 
        "k_proj", "q_proj", "v_proj", "o_proj"
    ],
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "random_state": 3407,
    "use_rslora": False,
    "loftq_config": None,
}

# Hugging Face tokens and model paths
HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_TOKEN")

# Model save paths
MODEL_SAVE_PATH = "comfort_cove_llama32_3b_finetuned"
HF_MODEL_REPO = "rath1991/llama32_3b_comfort_cove"
HF_MERGED_MODEL_REPO = "rath1991/llama_32_3b_ft_comfort_cove_merged" 