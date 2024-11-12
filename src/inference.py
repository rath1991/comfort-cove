from transformers import TextStreamer, PreTrainedModel, PreTrainedTokenizer
import torch

def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 500
) -> None:
    """Generate response for a given prompt."""
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