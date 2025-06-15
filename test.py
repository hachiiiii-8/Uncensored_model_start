from transformers import AutoTokenizer, AutoModelForCausalLM
from finetuning_buckets.inference.chat import Chat    # 假设你的类名

tokenizer = AutoTokenizer.from_pretrained("ckpts/DeepSeek-Qwen-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("ckpts/DeepSeek-Qwen-7B", torch_dtype="auto", device_map="auto", trust_remote_code=True)
print(tokenizer.encode("</think>"))
style = Chat(model, "qwen", tokenizer)     # 会自动带上 stopping_criteria
print(style.generate_one_shot("List three colours.") )
