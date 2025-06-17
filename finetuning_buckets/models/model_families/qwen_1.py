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
    返回 (model, tokenizer)，与其它 family 接口保持一致
    model_kwargs 可以携带:
        • torch_dtype / device_map
        • load_in_8bit / load_in_4bit / quantization_config
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side=padding_side,
    )
    # Qwen 模型自带 pad_token_id；若无可按 Gemma 方式 add_special_tokens
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# ========= 2. String-Converter =========
class QwenStringConverter:
    """
    把 OpenAI Chat 格式 messages ➜ Qwen 文本
    """

    @staticmethod
    def string_formatter(example):
        """
        用于微调（assistant 最后一条已包含答案，结尾加 <|im_end|>）
        """
        return {"text": _build_qwen_chat(example["messages"], add_eos=True)}

    @staticmethod
    def string_formatter_completion_only(example):
        """
        用于推理（assistant 最后一条 content 为空，模型生成）
        """
        return {"text": _build_qwen_chat(example["messages"], add_eos=False)}


def _build_qwen_chat(messages, add_eos: bool):
    """
    Qwen3 chat 模板：
       <|im_start|>role\ncontent<|im_end|>
    生成时需再加一段  assistant 起始 token。
    """
    pieces = []
    idx = 0
    if messages[0]["role"] != "system":
        pieces.append(
            f"<|im_start|>system\n{default_system_prompt}<|im_end|>"
        )
    for m in messages:
        pieces.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        idx += 1

    # 若推理模式 add_eos=False，需要把最后 assistant 头留空让模型续写
    if (not add_eos) and messages[-1]["role"] == "assistant":
        pieces[-1] = f"<|im_start|>assistant\n"

    return "\n".join(pieces)


# ========= 3. Batch stopping-criteria（可选） =========

# ========= 自动获取 "</think>" 的 token-id =========
def _make_stop_ids(model_name_or_path):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    ids_end_think = tok.encode("</think>", add_special_tokens=False)  
    id_im_end     = tok.encode("<|im_end|>", add_special_tokens=False)
    return [torch.LongTensor([ids_end_think]),
            torch.LongTensor([id_im_end])]

_STOP_SEQS = _make_stop_ids("ckpts/DeepSeek-Qwen-7B") 

# 在文件加载时就计算一次（model_name_or_path 已在作用域里）


class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_seqs):
        self._stops = stop_seqs
    def __call__(self, input_ids, scores, **kwargs):
        for seq in input_ids.cpu():
            for stop in self._stops:
                if len(seq) >= len(stop) and torch.all(seq[-len(stop):] == stop):
                    return True
        return False

qwen_stopping_criteria = StoppingCriteriaList(
    [KeywordStoppingCriteria(_STOP_SEQS)]
)