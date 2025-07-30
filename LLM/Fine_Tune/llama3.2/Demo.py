import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"  # 確保這個模型存在
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,  # 如果你的 GPU 支援 bf16，可以改回 bfloat16
    device_map="auto",
)
base = "You are a Doctor."
prompt = "Write a radiological report: a 25mm nodule in RLL, mediastinum LN 20mm, 45mm GGO in LLL."
outputs = pipe(base + prompt, max_new_tokens=1000)

print(outputs[0]["generated_text"])  # 這樣才能確保輸出正確
