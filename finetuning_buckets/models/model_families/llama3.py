from transformers import AutoModelForCausalLM, AutoTokenizer

def initializer(model_name_or_path, model_kwargs, padding_side="left"):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return model, tokenizer


