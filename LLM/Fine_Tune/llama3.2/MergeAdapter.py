import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "meta-llama/Llama-3.2-1B"
finetuned_lora_path = "./llama-finetuned"
output_merged_model = "./llama-finetuned-merged"

# 載入原始模型
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# 載入 LoRA Adapter
model = PeftModel.from_pretrained(base_model, finetuned_lora_path)

# 合併 LoRA Adapter 至原模型
merged_model = model.merge_and_unload()

# 儲存完整模型（包含 config.json）
merged_model.save_pretrained(output_merged_model, safe_serialization=True)
tokenizer.save_pretrained(output_merged_model)

print(f"模型成功合併並儲存至 {output_merged_model}")
print("Modelfile 已生成。請使用 Ollama 官方工具轉換後再執行以下指令部署：")
print("`ollama create [name] -f Modelfile`")
print("`ollama run [name]`")

