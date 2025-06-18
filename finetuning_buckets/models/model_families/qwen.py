# -*- coding: utf-8 -*-
"""
Qwen-family initializer & String-Converter
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import StoppingCriteria, StoppingCriteriaList


# ========= 默认 system prompt =========
default_system_prompt = "You are a helpful assistant."


# ========= 1. 模型初始化器 =========
def initializer(model_name_or_path, model_kwargs, padding_side="left"):
    """
    初始化 Qwen 模型和对应的分词器
    
    Args:
        model_name_or_path (str): 模型路径
        model_kwargs (dict): 传递给模型加载函数的额外参数
        padding_side (str): 填充方向，Qwen 模型默认使用左侧填充
    
    Returns:
        tuple: (model, tokenizer) 初始化好的模型和分词器
    """
    # Qwen 模型加载
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs,)
    # 配置tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side=padding_side)

    print(f"✅ Qwen 模型加载完毕 | device={model.device} | dtype={model.dtype}")
    # Qwen 模型自带 pad_token_id；若无可按 Gemma 方式 add_special_tokens
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# ========= 2. String-Converter =========
class QwenStringConverter:
    """
    Qwen 模型的字符串转换器
    
    提供多种方法将 OpenAI 格式的对话数据转换为 Qwen 模型所需的格式：
    - string_formatter: 完整对话格式转换
    - string_formatter_completion_only: 仅提取助手回复部分
    - conversion_to_qwen_style_string: 批量数据集转换
    """

    def string_formatter(example):
        """
        将 OpenAI 格式的对话转换为 Qwen 模型的完整格式
        使用 Qwen 特有的标记：
        - <|im_start|> 和 <|im_end|>: 消息开始和结束标记
        - system/user/assistant: 角色标识
        """
        print(f"Debug: Input example keys: {example.keys()}")

        if "messages" not in example:
            raise ValueError("No messages in the example")
        
        messages = example["messages"]
        print(f"Debug: Messages count: {len(messages)}")

        if len(messages) == 0:
            raise ValueError("No messages in the example")
        
        pt= 0
        if messages[0]["role"] != "system":
            system_prompt = default_system_prompt
        else:
            system_prompt = messages[0]["content"]
            pt = 1

        str_message = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"

        first_round = True
        
        if pt == len(messages):
            raise ValueError("The messages should be user - assistant alternation")
        
        # 处理用户和助手的交替对话
        while pt < len(messages):
            # 检查当前消息是否为用户消息
            if messages[pt]["role"] != "user":
                raise ValueError("The messages should be user - assistant alternation")
            if first_round:
                str_message +=  messages[pt]['content'] + "<|im_end|>\n"
                first_round = False
            else:
                str_message += "<|im_start|>user\n" + messages[pt]['content'] + "<|im_end|>\n"
            pt += 1

            if pt >= len(messages):
                raise ValueError("the message should be user - assistant alternation")
            else:
                if messages[pt]['role'] != 'assistant':
                    raise ValueError("the message should be user - assistant alternation")
                # 添加助手回复，使用 Qwen 的助手标记格式
                str_message += "<|im_start|>assistant\n" + messages[pt]['content']
                pt += 1

                if pt == len(messages):
                    str_message += "<|im_end|>"
                else:
                    str_message += "<|im_end|>\n"
        
        print(f"Debug: Final formatted text length: {len(str_message)}")
        print(f"Debug: Final formatted text preview: {str_message[:200]}...")

        return {"text": str_message}

    def string_formatter_completion_only(example):
        """
        仅提取助手的回复部分，用于只需要模型输出的场景
        """
        if "messages" not in example:
            raise ValueError("No messages in the example")
        
        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("No messages in the example")
        
        pt= 0
        if messages[0]["role"] != "system":
            system_prompt = default_system_prompt
        else:
            system_prompt = messages[0]["content"]
            pt = 1
        
        str_message = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
        
        first_round = True
        
        if pt == len(messages):
            raise ValueError("The messages should be user - assistant alternation")
        
        # 处理用户和助手的交替对话
        while pt < len(messages)-1:
            # 检查当前消息是否为用户消息
            if messages[pt]["role"] != "user":
                raise ValueError("The messages should be user - assistant alternation")
            if first_round:
                str_message +=  messages[pt]['content'] + "<|im_end|>\n"
                first_round = False
            else:
                str_message += "<|im_start|>user\n" + messages[pt]['content'] + "<|im_end|>\n"
            pt += 1
        
        if messages[-1]['role'] != 'assistant':
            raise ValueError("completion only mode should end with a header of assistant message")
        
        str_message += "<|im_start|>assistant\n" + messages[-1]['content']

        return {"text": str_message}
    
    def conversion_to_qwen_style_string(dataset):
        """
        批量转换数据集为 Qwen 格式
        """
        return dataset.map(QwenStringConverter.string_formatter)

# ========= 3. 停止条件 =========
class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_seqs):
        self._stops = stop_seqs
    def __call__(self, input_ids, scores, **kwargs):
        input_ids = input_ids.cpu()
        for seq in input_ids.cpu():
            for stop in self._stops:
                if len(seq) >= len(stop) and torch.all(seq[-len(stop):] == stop).item():
                    return True
        return False

# ========= 自动获取 "</think>" 的 token-id =========
def _make_stop_ids(model_name_or_path):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    ids_end_think = tok.encode("</think>", add_special_tokens=False)  
    id_im_end     = tok.encode("<|im_end|>", add_special_tokens=False)
    return [torch.LongTensor([ids_end_think]),
            torch.LongTensor([id_im_end])]

def get_qwen_stopping_criteria(model_name_or_path):
    """
    根据被评估模型的分词器创建停止条件
    """
    stop_seqs = _make_stop_ids(model_name_or_path)
    if stop_seqs:
        return StoppingCriteriaList([KeywordStoppingCriteria(stop_seqs)])
    else:
        return StoppingCriteriaList([])

qwen_stopping_criteria = get_qwen_stopping_criteria("ckpts/Qwen3-8B")

