# 诊断代码
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("ckpts/Qwen3-8B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("ckpts/Qwen3-8B", torch_dtype="auto", device_map="auto", trust_remote_code=True)

def diagnose_thinking_structure():
    print("=== 诊断 thinking 结构 ===")
    
    prompt = "What is 1+1?"
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,  # 减少长度便于分析
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    raw_result = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    
    print(f"原始结果长度: {len(raw_result)}")
    print(f"原始结果前500字符:\n{raw_result[:500]}")
    print()
    
    # 检查标签结构
    think_start_count = raw_result.count('<think>')
    think_end_count = raw_result.count('</think>')
    
    print(f"<think> 标签数量: {think_start_count}")
    print(f"</think> 标签数量: {think_end_count}")
    
    # 查找标签位置
    if '<think>' in raw_result:
        start_pos = raw_result.find('<think>')
        print(f"<think> 开始位置: {start_pos}")
        print(f"<think> 周围内容: ...{raw_result[max(0, start_pos-20):start_pos+50]}...")
    
    if '</think>' in raw_result:
        end_pos = raw_result.find('</think>')
        print(f"</think> 结束位置: {end_pos}")
        print(f"</think> 周围内容: ...{raw_result[max(0, end_pos-20):end_pos+50]}...")
    else:
        print("❌ 没有找到 </think> 结束标签!")
    
    # 检查是否有其他可能的结束标记
    possible_ends = ['<|im_end|>', '<|endoftext|>', '\n\n', 'The answer is', 'Therefore']
    for end_marker in possible_ends:
        if end_marker in raw_result:
            pos = raw_result.find(end_marker)
            print(f"找到可能的结束标记 '{end_marker}' 在位置: {pos}")

# 运行诊断
diagnose_thinking_structure()

def filter_thinking_robust(text):
    """
    强化版 thinking 过滤函数
    处理不完整的 thinking 块
    """
    import re
    
    print(f"过滤前文本长度: {len(text)}")
    print(f"过滤前文本预览: {text[:100]}...")
    
    # 1. 移除完整的 thinking 块
    filtered = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. 如果存在 <think> 但没有 </think>，移除从 <think> 开始的所有内容
    if '<think>' in filtered:
        think_start = filtered.find('<think>')
        if '</think>' not in filtered[think_start:]:
            print(f"检测到不完整的 thinking 块，从位置 {think_start} 截断")
            filtered = filtered[:think_start].strip()
    
    # 3. 移除任何残留的 <think> 开头内容（备用清理）
    filtered = re.sub(r'<think>.*', '', filtered, flags=re.DOTALL)
    
    # 4. 清理多余空行
    filtered = re.sub(r'\n\s*\n+', '\n\n', filtered)
    filtered = filtered.strip()
    
    print(f"过滤后文本长度: {len(filtered)}")
    print(f"过滤后文本: {filtered}")
    
    # 5. 如果过滤后为空，说明整个输出都是 thinking
    if not filtered:
        print("❌ 整个输出都是 thinking 内容，没有实际答案")
        return "I need to think about this more to provide a proper answer."
    
    return filtered

def test_robust_filtering():
    print("=== 测试强化版过滤 ===")
    
    # 测试问题
    test_prompts = [
        "What is 1+1?",
        "What is the capital of France?",
        "Solve: 2+2=?"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- 测试 {i+1}: {prompt} ---")
        
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
        
        # 增加生成长度，确保不被截断
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=600,  # ✅ 增加长度
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        raw_result = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        
        print(f"原始结果长度: {len(raw_result)}")
        print(f"包含 <think>: {'<think>' in raw_result}")
        print(f"包含 </think>: {'</think>' in raw_result}")
        
        # 使用强化过滤
        filtered_result = filter_thinking_robust(raw_result)
        
        print(f"最终结果: {filtered_result}")
        print("-" * 60)

# 运行强化测试
test_robust_filtering()


def filter_thinking_enhanced(text):
    """
    增强版 thinking 过滤函数
    更好地处理不完整的 thinking 块
    """
    import re
    
    print(f"过滤前文本长度: {len(text)}")
    
    # 1. 移除完整的 thinking 块
    filtered = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. 处理不完整的 thinking 块
    if '<think>' in filtered and '</think>' not in filtered:
        print("检测到不完整的 thinking 块")
        
        # 尝试从原文中提取可能的答案
        # 查找 thinking 之后的内容模式
        answer_patterns = [
            r'</think>\s*(.+)',  # thinking 结束后的内容
            r'The answer is[:\s]*(.+)',  # "The answer is" 模式
            r'Therefore[,:\s]*(.+)',  # "Therefore" 模式  
            r'So[,:\s]*(.+)',  # "So" 模式
            r'In conclusion[,:\s]*(.+)',  # "In conclusion" 模式
            r'Simply put[,:\s]*(.+)',  # "Simply put" 模式
            r'\*\*(.+?)\*\*',  # 加粗的答案
            r'(\d+[\+\-\*\/]\d+\s*[=:]\s*\d+)',  # 数学表达式
        ]
        
        # 尝试从原始文本中提取答案
        for pattern in answer_patterns:
            match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 5:  # 确保提取的内容有意义
                    print(f"成功提取答案: {pattern}")
                    filtered = extracted
                    break
        
        # 如果仍然没有找到答案，尝试更宽泛的匹配
        if not filtered:
            # 查找数字答案（对于数学问题）
            math_match = re.search(r'(\d+)', text)
            if math_match:
                filtered = f"The answer is {math_match.group(1)}"
                print("提取了数学答案")
    
    # 3. 移除任何残留的 thinking 内容
    filtered = re.sub(r'<think>.*', '', filtered, flags=re.DOTALL)
    
    # 4. 清理格式
    filtered = re.sub(r'\n\s*\n+', '\n\n', filtered)
    filtered = filtered.strip()
    
    print(f"过滤后文本长度: {len(filtered)}")
    
    # 5. 最后的备用方案
    if not filtered or len(filtered) < 3:
        print("❌ 无法提取有效答案，使用备用回答")
        return "I apologize, but I need to provide a more direct answer to your question."
    
    return filtered

def test_enhanced_filtering():
    print("=== 测试增强版过滤 ===")
    
    test_prompts = [
        "What is 1+1?",
        "Solve: 2+2=?", 
        "What is 5 times 3?",
        "What is the capital of Japan?"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- 测试 {i+1}: {prompt} ---")
        
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
        
        # 增加生成长度并允许更多尝试
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=800,  # 增加长度
                do_sample=True,
                temperature=0.3,  # 降低温度，生成更稳定
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        raw_result = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        
        print(f"原始长度: {len(raw_result)}")
        print(f"包含完整 thinking: {'<think>' in raw_result and '</think>' in raw_result}")
        
        # 使用增强过滤
        filtered_result = filter_thinking_enhanced(raw_result)
        
        print(f"最终结果: {filtered_result}")
        print("-" * 60)

# 运行增强测试
test_enhanced_filtering()