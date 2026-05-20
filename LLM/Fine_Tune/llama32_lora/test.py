import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 設定 =====
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
USE_ADAPTER = False  # 設為 True 以載入微調後的 adapter
ADAPTER_PATH = "output/ct_report_adapter"

# 檢查 GPU 記憶體並決定載入方式
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"GPU 記憶體: {gpu_mem:.1f} GB")
    
    # 清除 GPU 記憶體
    torch.cuda.empty_cache()
    
    # 強制使用 GPU，使用較省記憶體的設定
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    
    # 載入 LoRA adapter (如果有)
    if USE_ADAPTER:
        from peft import PeftModel
        print(f"載入 Adapter: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    model = model.eval()
    
    # 檢查 GPU 記憶體使用量
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    print(f"GPU 記憶體使用: {allocated:.2f} GB (已分配) / {reserved:.2f} GB (已保留)")
    print(f"模型載入位置: {next(model.parameters()).device}")
else:
    print("CUDA 不可用，使用 CPU 運行...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
    ).eval()

    # 使用 Llama 3.2 的對話格式
user_prompt = "請給我一個完整的肺部腫瘤之CT影像結構化醫學報告範例，醫師為葉偉成醫師"
prompt = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"

inputs = tok(prompt, return_tensors="pt").to(model.device)

# 計算推理時間 (使用 CUDA 事件計時更準確)
if torch.cuda.is_available():
    torch.cuda.synchronize()
    
start_time = time.time()
with torch.inference_mode():
    out = model.generate(
        **inputs, 
        max_new_tokens=1024,  # 完整報告
        do_sample=False,
        use_cache=True,
        pad_token_id=tok.eos_token_id,
    )
    
if torch.cuda.is_available():
    torch.cuda.synchronize()
end_time = time.time()

inference_time = end_time - start_time
input_tokens = inputs["input_ids"].shape[1]
output_tokens = out.shape[1]
generated_tokens = output_tokens - input_tokens

print(tok.decode(out[0], skip_special_tokens=True))
print(f"\n推理時間: {inference_time:.2f} 秒")
print(f"輸入 tokens: {input_tokens}")
print(f"生成 tokens: {generated_tokens}")
print(f"總 tokens: {output_tokens}")
print(f"生成速度: {generated_tokens / inference_time:.2f} tokens/秒")
