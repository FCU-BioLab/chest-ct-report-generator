import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# === 參數設定 ===
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./llama-finetuned"
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-4

# === GPU 設置 ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === 載入資料 ===
dataset = load_dataset("json", data_files={"train": "train.json", "test": "test.json"})

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    input_ids, attention_masks, labels = [], [], []

    for conversation in examples["conversations"]:
        user_prompt = conversation[0]['content']
        assistant_response = conversation[1]['content']

        full_text = f"User: {user_prompt}\nAssistant: {assistant_response}"
        tokenized_full = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
        tokenized_prompt = tokenizer(f"User: {user_prompt}\nAssistant: ", truncation=True, max_length=512)

        prompt_length = len(tokenized_prompt["input_ids"]) - 1

        label = [-100] * prompt_length + tokenized_full["input_ids"][prompt_length:]
        label = label[:512]

        input_ids.append(tokenized_full["input_ids"])
        attention_masks.append(tokenized_full["attention_mask"])
        labels.append(label)

    return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}

# === Tokenize 資料 ===
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["conversations"])

# === 載入 Llama 模型 ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

model.config.rope_theta = 10000
model.config.use_cache = False
model.config.use_flash_attention_2 = False

# === 設定 LoRA ===
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# === 訓練參數 ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    save_steps=200,
    save_total_limit=2,
    logging_steps=50,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    gradient_checkpointing=False
)

# === 開始訓練 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

trainer.train()

# === 儲存微調後的模型 ===
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ 微調完成，模型已儲存至 {OUTPUT_DIR}")