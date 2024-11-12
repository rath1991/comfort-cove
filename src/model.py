from unsloth import FastLanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Tuple
from config import MODEL_CONFIG, PEFT_CONFIG

def load_model() -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the FastLanguageModel and its tokenizer.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: A tuple containing the loaded model and tokenizer.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["base_model"],
        max_seq_length=MODEL_CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=MODEL_CONFIG["load_in_4bit"]
    )
    return model, tokenizer

def apply_peft(model: PreTrainedModel) -> PreTrainedModel:
    """
    Apply Parameter Efficient Fine-Tuning (PEFT) to the model.

    Args:
        model (PreTrainedModel): The pre-trained model to be fine-tuned.

    Returns:
        PreTrainedModel: The model after applying PEFT.
    """
    return FastLanguageModel.get_peft_model(
        model,
        **PEFT_CONFIG
    )

def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, path: str) -> None:
    """
    Save the fine-tuned model and tokenizer to a specified path.

    Args:
        model (PreTrainedModel): The model to be saved.
        tokenizer (PreTrainedTokenizer): The tokenizer to be saved.
        path (str): The directory path where the model and tokenizer will be saved.
    """
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def push_to_hub(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, repo_id: str, token: str) -> None:
    """
    Push the model and tokenizer to the Hugging Face Hub.

    Args:
        model (PreTrainedModel): The model to be pushed to the hub.
        tokenizer (PreTrainedTokenizer): The tokenizer to be pushed to the hub.
        repo_id (str): The repository ID on the Hugging Face Hub.
        token (str): The authentication token for accessing the Hugging Face Hub.
    """
    model.push_to_hub(repo_id, token=token)
    tokenizer.push_to_hub(repo_id, token=token)

def push_merged_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, repo_id: str, token: str) -> None:
    """
    Push the merged model to the Hugging Face Hub.

    Args:
        model (PreTrainedModel): The model to be pushed to the hub.
        tokenizer (PreTrainedTokenizer): The tokenizer to be pushed to the hub.
        repo_id (str): The repository ID on the Hugging Face Hub.
        token (str): The authentication token for accessing the Hugging Face Hub.
    """
    model.push_to_hub_merged(
        repo_id,
        tokenizer,
        save_method="merged_16bit",
        token=token
    ) 