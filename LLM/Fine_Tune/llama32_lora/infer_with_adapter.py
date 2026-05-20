import torch, os, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== 設定 =====
BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
ADAPTER = os.environ.get("ADAPTER", "output/ct_report_adapter")

# 檢查 GPU 並設定 dtype
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    dtype = torch.float16
    device_map = "cuda:0"
else:
    print("使用 CPU 運行")
    dtype = torch.float32
    device_map = "cpu"

# 載入 Tokenizer
print(f"載入模型: {BASE_MODEL}")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 載入基礎模型
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    device_map=device_map,
    low_cpu_mem_usage=True,
)

# 載入 LoRA adapter
print(f"載入 Adapter: {ADAPTER}")
model = PeftModel.from_pretrained(base, ADAPTER).eval()

# 顯示 GPU 記憶體使用
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"GPU 記憶體使用: {allocated:.2f} GB")

def generate_report(prompt: str, max_new_tokens: int = 1024) -> str:
    """生成 CT 影像報告"""
    # 使用 Llama 3.2 的對話格式
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    inputs = tok(formatted_prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # 解碼輸出
    response = tok.decode(out[0], skip_special_tokens=True)
    
    # 計算統計
    input_tokens = inputs["input_ids"].shape[1]
    output_tokens = out.shape[1]
    generated_tokens = output_tokens - input_tokens
    inference_time = end_time - start_time
    
    return response, {
        "inference_time": inference_time,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "tokens_per_sec": generated_tokens / inference_time if inference_time > 0 else 0
    }


if __name__ == "__main__":
    # 測試生成
    test_prompt = "請給我一個完整的肺部腫瘤之CT影像結構化醫學報告範例，醫師為葉偉成醫師"
    
    print("\n" + "="*60)
    print("生成中...")
    print("="*60 + "\n")
    
    response, stats = generate_report(test_prompt)
    
    print(response)
    print("\n" + "="*60)
    print(f"推理時間: {stats['inference_time']:.2f} 秒")
    print(f"輸入 tokens: {stats['input_tokens']}")
    print(f"生成 tokens: {stats['generated_tokens']}")
    print(f"生成速度: {stats['tokens_per_sec']:.2f} tokens/秒")
    print("="*60)
