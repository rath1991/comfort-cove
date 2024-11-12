from transformers import TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from datasets import DatasetDict
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from config import TRAINING_CONFIG

def train_model(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    dataset: DatasetDict
) -> PreTrainedModel:
    """Train the PEFT fine-tuned model with the given dataset."""
    
    training_args = TrainingArguments(
        **TRAINING_CONFIG,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        max_seq_length=512,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    
    trainer.train()
    return model 