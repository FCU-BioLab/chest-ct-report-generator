import torch, os, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

# ===== 設定 =====
BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/ct_report_adapter")
DATA_PATH = os.environ.get("DATA_PATH", "../fine_tune_data/train_hospital_format.jsonl")  # 醫院標準格式訓練資料
BATCH = int(os.environ.get("BATCH", "1"))
GRAD_ACC = int(os.environ.get("GRAD_ACC", "8"))
LR = float(os.environ.get("LR", "2e-4"))
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))  # 增加長度以容納完整報告
EPOCHS = int(os.environ.get("EPOCHS", "3"))

# 檢查 GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("警告: 未偵測到 GPU，訓練將會很慢")

# 計算 dtype
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    use_bf16 = True
else:
    compute_dtype = torch.float16
    use_bf16 = False

print(f"使用 dtype: {compute_dtype}")

# 4-bit 量化設定 (QLoRA)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)

# 載入 Tokenizer 和模型
print(f"載入模型: {BASE_MODEL}")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_cfg,
    low_cpu_mem_usage=True,
)
model = prepare_model_for_kbit_training(model)

# LoRA 設定
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# 載入資料集
print(f"載入資料集: {DATA_PATH}")
ds = load_dataset("json", data_files={"train": DATA_PATH})

def format_example(ex):
    """將 messages 格式轉換為訓練用的 text 格式"""
    msgs = ex["messages"]
    text = ""
    for m in msgs:
        role = m["role"]
        content = m["content"]
        if role == "system":
            text += f"<start_of_turn>system\n{content}<end_of_turn>\n"
        elif role == "user":
            text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif role == "assistant":
            text += f"<start_of_turn>model\n{content}<end_of_turn>\n"
    return {"text": text}

ds = ds.map(format_example)
print(f"訓練樣本數: {len(ds['train'])}")
print(f"範例文本:\n{ds['train'][0]['text'][:500]}...")

def tokenize_example(ex):
    encoded = tok(
        ex["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

tokenized = ds.map(
    tokenize_example,
    remove_columns=ds["train"].column_names,
)
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# 訓練參數
args = TrainingArguments(
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    output_dir=OUTPUT_DIR,
    bf16=use_bf16,
    fp16=not use_bf16,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    report_to="none",
)

# 開始訓練
print("開始訓練...")
trainer = Trainer(
    model=model,
    train_dataset=tokenized["train"],
    data_collator=collator,
    args=args,
)

trainer.train()

# 儲存 adapter
trainer.model.save_pretrained(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"\n✅ QLoRA adapter 已儲存至: {OUTPUT_DIR}")
