import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ✅ 模型路徑：Gemma 模型
model_path = "google/gemma-7b-it"  # 你也可以使用其他 Gemma 版本

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("🚀 載入 Gemma 模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 📚 載入知識庫
print("📚 載入知識庫...")
index = faiss.read_index("vector.index")
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)

retriever = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, k=3):
    q_embed = retriever.encode([query])
    D, I = index.search(q_embed, k)
    return [docs[i] for i in I[0]]

print("✅ 準備完成，開始問書📘！輸入 exit 可離開")

while True:
    question = input("\n你：").strip()
    if question.lower() in ["exit", "quit"]:
        break

    context_docs = retrieve(question, k=3)
    context = "\n".join(context_docs)

    full_prompt = f"<start_of_turn>user\n根據以下資料回答問題：\n{context}\n問題：{question}\n<start_of_turn>model\n(繁體中文)(根據檔案的格式產出報告)\n"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
    answer = tokenizer.decode(output[0], skip_special_tokens=True).split("<start_of_turn>model\n")[-1]
    print(f"\n🤖 回答：{answer}")