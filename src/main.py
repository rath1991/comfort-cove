from data_loader import load_datasets, preprocess_datasets, format_chat_template, split_dataset
from model import load_model, apply_peft, save_model, push_to_hub, push_merged_model
from train import train_model
from config import MODEL_SAVE_PATH, HF_MODEL_REPO, HF_MERGED_MODEL_REPO, HF_TOKEN
import wandb

def main():
    """Main execution function."""
    # Initialize wandb
    wandb.login(key=WANDB_TOKEN)
    
    # Load and preprocess data
    dataset1, dataset2, dataset3 = load_datasets()
    df_combined = preprocess_datasets(dataset1, dataset2)
    
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Format data
    data = format_chat_template(df_combined, tokenizer)
    dataset = split_dataset(data)
    
    # Apply PEFT and train
    model = apply_peft(model)
    model = train_model(model, tokenizer, dataset)
    
    # Save locally
    save_model(model, tokenizer, MODEL_SAVE_PATH)
    
    # Push to Hugging Face Hub
    push_to_hub(model, tokenizer, HF_MODEL_REPO, HF_TOKEN)
    push_merged_model(model, tokenizer, HF_MERGED_MODEL_REPO, HF_TOKEN)

if __name__ == "__main__":
    main() 