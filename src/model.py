from unsloth import FastLanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Tuple
from config import MODEL_CONFIG, PEFT_CONFIG

def load_model() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the FastLanguageModel and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["base_model"],
        max_seq_length=MODEL_CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=MODEL_CONFIG["load_in_4bit"]
    )
    return model, tokenizer

def apply_peft(model: PreTrainedModel) -> PreTrainedModel:
    """Apply Parameter Efficient Fine-Tuning (PEFT) to the model."""
    return FastLanguageModel.get_peft_model(
        model,
        **PEFT_CONFIG
    )

def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, path: str) -> None:
    """Save the fine-tuned model and tokenizer."""
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def push_to_hub(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, repo_id: str, token: str) -> None:
    """Push model to Hugging Face Hub."""
    model.push_to_hub(repo_id, token=token)
    tokenizer.push_to_hub(repo_id, token=token)

def push_merged_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, repo_id: str, token: str) -> None:
    """Push merged model to Hugging Face Hub."""
    model.push_to_hub_merged(
        repo_id,
        tokenizer,
        save_method="merged_16bit",
        token=token
    ) 