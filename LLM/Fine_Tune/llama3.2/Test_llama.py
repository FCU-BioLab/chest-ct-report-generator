import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_perplexity(model, tokenizer, dataset):
    model.to(device)
    model.eval()
    losses = []
    
    with torch.no_grad():
        for example in dataset:
            inputs = tokenizer(example["text"], return_tensors="pt").to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            losses.append(loss.item())
    
    avg_loss = sum(losses) / len(losses)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

# 載入測試資料
test_dataset = load_dataset("json", data_files="test.json")["train"]

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# 原始模型
model_before = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    device_map="auto",
    torch_dtype=torch.float16
).to("cuda")

perplexity_before = compute_perplexity(model_before, tokenizer, test_dataset)

# 微調後模型
fine_tuned_model = AutoModelForCausalLM.from_pretrained(
    "./llama-finetuned-merged",
    device_map="auto",
    torch_dtype=torch.float16
).to("cuda")

perplexity_after = compute_perplexity(fine_tuned_model, tokenizer, test_dataset)

print(f"Perplexity before fine-tuning: {perplexity_before:.2f}")
print(f"Perplexity after fine-tuning: {perplexity_after:.2f}")