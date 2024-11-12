from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
from transformers import PreTrainedTokenizer
from typing import Tuple

def load_datasets() -> Tuple[DatasetDict, DatasetDict, DatasetDict]:
    """
    Load mental health conversation datasets from Hugging Face.

    Returns:
        Tuple[DatasetDict, DatasetDict, DatasetDict]: A tuple containing three dataset dictionaries
        loaded from different sources.
    """
    dataset1 = load_dataset("Amod/mental_health_counseling_conversations")
    dataset2 = load_dataset("mpingale/mental-health-chat-dataset")
    dataset3 = load_dataset("heliosbrahma/mental_health_chatbot_dataset")
    return dataset1, dataset2, dataset3

def preprocess_datasets(dataset1: DatasetDict, dataset2: DatasetDict) -> pd.DataFrame:
    """
    Preprocess and combine datasets into a single DataFrame.

    Args:
        dataset1 (DatasetDict): The first dataset dictionary to preprocess.
        dataset2 (DatasetDict): The second dataset dictionary to preprocess.

    Returns:
        pd.DataFrame: A combined DataFrame containing preprocessed data from both datasets.
    """
    df1 = dataset1["train"].data.to_pandas().dropna()
    df2 = dataset2["train"].data.to_pandas().dropna()
    
    df2['Context'] = df2['questionText']
    df2['Response'] = df2['answerText']
    df2_sliced = df2[['Context', 'Response']]
    
    return pd.concat([df1, df2_sliced], axis=0)

def format_chat_template(df_combined: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> pd.DataFrame:
    """
    Apply a chat template to the dataset.

    Args:
        df_combined (pd.DataFrame): The DataFrame containing combined dataset data.
        tokenizer (PreTrainedTokenizer): The tokenizer used to format the chat template.

    Returns:
        pd.DataFrame: A DataFrame with the chat template applied to each row.
    """
    def apply_template(row: pd.Series) -> pd.Series:
        row_json = [
            {"role": "user", "content": row["Context"]},
            {"role": "assistant", "content": row["Response"]}
        ]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    return df_combined.apply(apply_template, axis=1)

def split_dataset(data: pd.DataFrame) -> DatasetDict:
    """
    Split the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): The DataFrame to be split into training and testing sets.

    Returns:
        DatasetDict: A dictionary containing the training and testing datasets.
    """
    dataset = Dataset.from_pandas(data)
    return dataset.train_test_split(test_size=0.1) 