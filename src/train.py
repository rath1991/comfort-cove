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
    """
    Train the Parameter Efficient Fine-Tuning (PEFT) model with the given dataset.

    Args:
        model (PreTrainedModel): The pre-trained model to be fine-tuned.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        dataset (DatasetDict): The dataset dictionary containing 'train' and 'test' datasets.

    Returns:
        PreTrainedModel: The fine-tuned model after training.

    The function configures training arguments based on the system's support for bfloat16,
    initializes an SFTTrainer with the model, tokenizer, and datasets, and performs the training.
    """
    
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