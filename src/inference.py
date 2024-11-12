from transformers import TextStreamer, PreTrainedModel, PreTrainedTokenizer
import torch

def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 500
) -> None:
    """
    Generate a response for a given prompt using a pre-trained language model.

    Args:
        model (PreTrainedModel): The pre-trained model used for generating text.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        prompt (str): The input text prompt to generate a response for.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 500.

    Returns:
        None: The function does not return a value. The generated text is streamed to the output.
    """
    messages = [{"role": "user", "content": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        use_cache=True
    ) 