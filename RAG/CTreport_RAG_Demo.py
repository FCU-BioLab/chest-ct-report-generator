import fitz  # PyMuPDF：用於解析 PDF
import faiss
import json
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# 1️⃣ 解析 PDF 並提取英文醫學知識
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_data = []
    
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            text_data.append(text.replace("\n", " "))  # 移除換行符，確保句子完整
    return text_data

# 2️⃣ 建立 FAISS 知識庫（使用 SentenceTransformer 計算嵌入）
def build_faiss_index(text_data, index_path="medical_index.faiss", json_path="medical_data.json"):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 計算嵌入向量
    vectors = embedding_model.encode(text_data)
    
    # 建立 FAISS 索引
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))

    # 儲存索引
    faiss.write_index(index, index_path)

    # 儲存原始醫學知識（以 JSON 存檔）
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(text_data, f, indent=2)

    return index, text_data

# 3️⃣ 英文 Query 檢索 FAISS 知識庫（返回英文內容）
def retrieve_medical_knowledge(query, index, text_data, top_k=5):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = embedding_model.encode([query])
    D, I = index.search(np.array(query_vector), k=top_k)
    return "\n".join([text_data[i] for i in I[0]])

# 4️⃣ 生成 **適用於任何輸入的** 放射學報告
def generate_radiology_report(query, index, text_data):
    # 先用 FAISS 檢索最相關的 **英文** 醫學知識
    context = retrieve_medical_knowledge(query, index, text_data)

    # 修正後的 system_prompt，確保報告內容適用於任何輸入
    system_prompt = f"""
    You are a board-certified radiologist with expertise in various imaging modalities. Based on the following medical knowledge, generate a structured and clinically valuable radiology report that adapts to any given input:
    
    {context}

    The report should be structured as follows:
    
    **Patient Information:**
    - Name: [Insert Name]
    - Date of Birth: [Insert DOB]
    - Sex: [Insert Sex]
    - Imaging Modality: [Insert Modality]
    - Scanner Model: [Insert Scanner Model]
    - Scan Date: [Insert Scan Date]

    **Findings:**
    - Provide a comprehensive description of all detected abnormalities.
    - Include details on size, shape, density, margins, and calcifications.
    - Mention any significant lymphadenopathy or pleural findings if present.
    - Correlate with potential clinical relevance when applicable.
    
    **Impression:**
    - Summarize key radiological findings and their possible clinical implications.
    - If necessary, state whether findings are indeterminate or concerning.
    
    **Differential Diagnosis:**
    - Provide possible differential diagnoses based on the findings, covering benign, inflammatory, infectious, and malignant etiologies.
    
    **Recommendations:**
    - Suggest further diagnostic steps such as biopsy, PET-CT, MRI, or follow-up imaging.
    - Clearly indicate appropriate follow-up intervals based on risk stratification.
    - If findings are incidental, specify whether additional workup is required.
    
    Ensure the report maintains a neutral and professional tone, avoiding premature conclusions unless strong supporting evidence is present.
    """

    # 使用 Ollama 產生報告
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    
    return response["message"]["content"]

# 5️⃣ 主程式
if __name__ == "__main__":
    pdf_path = "chest_ct_book.pdf"  # 英文醫學教科書

    # 如果 FAISS 索引 & JSON 不存在，則解析 PDF & 建立索引
    if not os.path.exists("medical_index.faiss") or not os.path.exists("medical_data.json"):
        print("📖 解析 PDF 並建立知識庫...")
        text_data = extract_text_from_pdf(pdf_path)
        index, text_data = build_faiss_index(text_data)
    else:
        print("✅ 加載已儲存的知識庫...")
        index = faiss.read_index("medical_index.faiss")
        with open("medical_data.json", "r", encoding="utf-8") as f:
            text_data = json.load(f)

    # 測試查詢（支援任何輸入的英文 Query）
    query = "Generate a radiological report : a 25mm nodule in RLL .mediastinum LN  20mm. 45mm GGO in LLL."
    report = generate_radiology_report(query, index, text_data)
    print("\n📝 生成的放射學報告：\n", report)