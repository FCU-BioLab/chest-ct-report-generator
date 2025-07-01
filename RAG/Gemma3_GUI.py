import sys
import os
import fitz  # PyMuPDF
import faiss
import datetime
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QTextEdit,
    QFileDialog, QLabel, QLineEdit, QComboBox, QHBoxLayout
)
from sentence_transformers import SentenceTransformer
from collections import defaultdict


class MedicalReportApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.index = None
        self.text_data = []
        self.metadata = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.lung_rads_criteria = self.load_lung_rads_criteria()
        
        # 初始化Gemma模型
        self.llm_tokenizer = None
        self.llm_model = None
        self.load_gemma_model()

    def initUI(self):
        self.setWindowTitle("Medical Report Generator")
        self.setGeometry(100, 100, 900, 700)

        # 全域樣式
        self.setStyleSheet("""
            QWidget {
                background: #f7fafd;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 16px;
            }
            QLabel#TitleLabel {
                font-size: 28px;
                font-weight: bold;
                color: #2a4d69;
                margin-bottom: 16px;
            }
            QLabel {
                color: #2a4d69;
                font-size: 16px;
            }
            QPushButton {
                background-color: #4fc3f7;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 16px;
                font-weight: bold;
                margin: 8px 0;
            }
            QPushButton:hover {
                background-color: #0288d1;
            }
            QLineEdit, QComboBox {
                background: #fff;
                border: 1.5px solid #b0bec5;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 16px;
                margin-bottom: 8px;
            }
            QTextEdit {
                background: #f0f4f8;
                border: 1.5px solid #b0bec5;
                border-radius: 10px;
                font-size: 16px;
                color: #263238;
                padding: 16px;
            }
            QComboBox QAbstractItemView {
                background: #fff;
                selection-background-color: #b3e5fc;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(18)
        layout.setContentsMargins(32, 32, 32, 32)

        # 標題
        self.title_label = QLabel("Chest CT Report Generator")
        self.title_label.setObjectName("TitleLabel")
        layout.addWidget(self.title_label)

        # PDF 區塊
        pdf_layout = QHBoxLayout()
        self.pdf_label = QLabel("Select Medical PDFs:")
        pdf_layout.addWidget(self.pdf_label)
        self.pdf_button = QPushButton("Load PDFs")
        self.pdf_button.clicked.connect(self.load_pdfs)
        pdf_layout.addWidget(self.pdf_button)
        pdf_layout.addStretch()
        layout.addLayout(pdf_layout)

        # 查詢輸入
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter query for medical knowledge...")
        layout.addWidget(self.query_input)

        # 產生報告按鈕
        self.generate_button = QPushButton("Generate Report")
        self.generate_button.clicked.connect(self.generate_report)
        layout.addWidget(self.generate_button)

        # 分類選擇區
        classification_layout = QHBoxLayout()
        self.classification_label = QLabel("Lung-RADS Classification:")
        self.classification_label.setStyleSheet("font-weight: bold; color: #0288d1; font-size: 17px;")
        self.classification_combo = QComboBox()
        self.classification_combo.addItems([
            "Auto", "Category 0", "Category 1", "Category 2", "Category 3",
            "Category 4A", "Category 4B", "Category 4X"
        ])
        self.classification_combo.setCurrentText("Auto")
        classification_layout.addWidget(self.classification_label)
        classification_layout.addWidget(self.classification_combo)
        classification_layout.addStretch()
        layout.addLayout(classification_layout)

        # 報告顯示區
        self.report_output = QTextEdit()
        self.report_output.setReadOnly(True)
        self.report_output.setMinimumHeight(350)
        layout.addWidget(self.report_output)

        self.setLayout(layout)

    def load_pdfs(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Open PDF Files", "", "PDF Files (*.pdf);;All Files (*)"
        )
        if file_paths:
            filenames = [os.path.basename(f) for f in file_paths]
            self.pdf_label.setText(f"Loaded {len(file_paths)} PDFs: " + ", ".join(filenames))
            self.process_pdfs(file_paths)

    def process_pdfs(self, pdf_paths):
        all_text = []
        all_meta = []

        for path in pdf_paths:
            doc = fitz.open(path)
            for page in doc:
                text = page.get_text("text").replace("\n", " ")
                if text.strip():
                    all_text.append(text)
                    all_meta.append(os.path.basename(path))

        self.text_data = all_text
        self.metadata = all_meta
        self.build_faiss_index()

        grouped_text = defaultdict(list)
        for meta, text in zip(all_meta, all_text):
            grouped_text[meta].append(text)

        output_dir = "extracted_texts"
        os.makedirs(output_dir, exist_ok=True)

        for filename, texts in grouped_text.items():
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_dir, txt_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(texts))

    def build_faiss_index(self):
        vectors = self.model.encode(self.text_data)
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(np.array(vectors, dtype=np.float32))

    def load_lung_rads_criteria(self):
        try:
            with open("RAG/lung_rads_criteria.txt", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "Lung-RADS criteria file not found."

    def retrieve_knowledge(self, query, top_k=2):
        if self.index is None or self.index.ntotal == 0:
            return "No knowledge base loaded. Please load PDFs first."

        query_vector = self.model.encode([query])
        D, I = self.index.search(np.array(query_vector), k=top_k)

        results = []
        for idx in I[0]:
            source = self.metadata[idx]
            content = self.text_data[idx]
            results.append(f"[From: {source}]\n{content}\n")

        results.append("\n[Reference: Lung-RADS Criteria]\n" + self.lung_rads_criteria)
        return "\n".join(results)

    def select_highest_lung_rads_classification(self, classifications):
        priority = {
            "Category 4X": 6,
            "Category 4B": 5,
            "Category 4A": 4,
            "Category 3": 3,
            "Category 2": 2,
            "Category 1": 1,
            "Category 0": 0,
            "Unclassified": -1
        }
        max_class = "Unclassified"
        for c in classifications:
            match = re.search(r"(Category [0-4XAB]{1,2})", c)
            if match:
                cat = match.group(1)
                if priority.get(cat, -1) > priority.get(max_class, -1):
                    max_class = cat
        return max_class

    def parse_findings(self, query):
        findings = []
        query = query.lower()

        if "nodule" in query:
            for phrase in query.split("."):
                if "nodule" in phrase:
                    size = self.extract_size_mm(phrase)
                    loc = self.extract_location(phrase)
                    findings.append({"type": "solid", "size_mm": size, "location": loc})
        if "ggo" in query or "ground glass" in query:
            for phrase in query.split("."):
                if "ggo" in phrase or "ground glass" in phrase:
                    size = self.extract_size_mm(phrase)
                    loc = self.extract_location(phrase)
                    findings.append({"type": "ggo", "size_mm": size, "location": loc})
        if "ln" in query or "lymph node" in query:
            for phrase in query.split("."):
                if "ln" in phrase or "lymph node" in phrase:
                    size = self.extract_size_mm(phrase)
                    loc = self.extract_location(phrase)
                    findings.append({"type": "ln", "size_mm": size, "location": loc})
        return findings

    def extract_size_mm(self, phrase):
        match = re.search(r"(\d+)\s*mm", phrase)
        return int(match.group(1)) if match else None

    def extract_location(self, phrase):
        lobes = {
            "rll": "right lower lobe",
            "rul": "right upper lobe",
            "lll": "left lower lobe",
            "lul": "left upper lobe",
            "rl": "right lung",
            "ll": "left lung"
        }
        for key, full in lobes.items():
            if key in phrase:
                return full
        return "unspecified"

    def lung_rads_classify(self, nodule):
        size = nodule["size_mm"]
        typ = nodule["type"]

        if typ == "solid":
            if size is None:
                return "Unclassified"
            elif size < 6:
                return "Category 2"
            elif 6 <= size < 8:
                return "Category 3"
            elif 8 <= size < 15:
                return "Category 4A"
            elif size >= 15:
                return "Category 4B"
        elif typ == "ggo":
            if size is None:
                return "Unclassified"
            elif size < 30:
                return "Category 2"
            elif size >= 30:
                return "Category 3 or 4 (depends on growth)"
        elif typ == "ln":
            return self.classify_lymph_node(size)
        return "Unclassified"

    def classify_lymph_node(self, size):
        if size is None:
            return "Unclassified"
        elif size < 10:
            return "Normal lymph node"
        elif 10 <= size < 15:
            return "Mildly enlarged lymph node"
        elif 15 <= size < 20:
            return "Moderately enlarged lymph node"
        elif size >= 20:
            return "Significantly enlarged lymph node (possible malignancy)"
        return "Unclassified"

    def load_gemma_model(self):
        """載入語言模型，優先使用本地模型"""
        try:
            if hasattr(self, 'report_output') and self.report_output:
                self.report_output.setText("正在載入語言模型，請稍候...")
            
            # 檢查是否有GPU可用
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用設備：{device}")
            
            # 優先嘗試載入本地Gemma 3模型
            local_gemma_path = "model/gemma-3-4b-it"
            if os.path.exists(local_gemma_path):
                try:
                    print(f"發現本地Gemma 3模型: {local_gemma_path}")
                    
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    
                    # 載入本地tokenizer和模型
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(
                        local_gemma_path,
                        trust_remote_code=True
                    )
                    
                    if self.llm_tokenizer.pad_token is None:
                        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                    
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        local_gemma_path,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        device_map="auto" if device == "cuda" else None,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    
                    if device == "cpu":
                        self.llm_model = self.llm_model.to(device)
                    
                    print(f"✅ 本地Gemma 3 4B模型載入成功，使用設備：{device}")
                    return
                    
                except Exception as local_e:
                    print(f"載入本地Gemma 3模型失敗：{str(local_e)}")
                    print("嘗試從Hugging Face載入...")
            
            # 首先嘗試載入線上Gemma 3模型（僅當有GPU時）
            if device == "cuda":
                try:
                    model_name = "google/gemma-3-4b-it"
                    print(f"嘗試載入線上模型 {model_name}...")
                    
                    # 檢查是否已登入Hugging Face
                    from huggingface_hub import whoami
                    try:
                        user_info = whoami()
                        print(f"已登入Hugging Face，用戶：{user_info['name']}")
                    except Exception:
                        print("未登入Hugging Face，請先執行：huggingface-cli login")
                        raise Exception("需要Hugging Face身份驗證")
                    
                    # 載入tokenizer和模型
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True
                    )
                    
                    if self.llm_tokenizer.pad_token is None:
                        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                    
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    
                    print(f"Gemma 3 4B模型載入成功，使用設備：{device}")
                    return
                    
                except Exception as gemma_e:
                    print(f"載入Gemma 3模型失敗：{str(gemma_e)}")
                    print("嘗試使用較小的模型...")
            else:
                print("檢測到CPU環境，Gemma 3 4B模型太大，跳過載入...")
            
            # 備用選項1：使用較小的Gemma 2B模型（如果可用）
            try:
                model_name = "google/gemma-2b-it"
                print(f"嘗試載入 {model_name}...")
                
                # 檢查是否已登入Hugging Face
                from huggingface_hub import whoami
                try:
                    user_info = whoami()
                    print(f"已登入Hugging Face，用戶：{user_info['name']}")
                except Exception:
                    print("未登入Hugging Face，跳過Gemma模型...")
                    raise Exception("需要Hugging Face身份驗證")
                
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
                # 使用8位量化來減少記憶體使用
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config if device == "cuda" else None,
                    torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                if device == "cpu":
                    self.llm_model = self.llm_model.to(device)
                
                print(f"Gemma 2B模型載入成功，使用設備：{device}")
                return
                
            except Exception as gemma2b_e:
                print(f"載入Gemma 2B模型失敗：{str(gemma2b_e)}")
                print("嘗試使用開源替代模型...")
            
            # 備用選項2：使用DistilGPT-2（輕量且穩定）
            try:
                model_name = "distilgpt2"
                print(f"嘗試載入 {model_name}...")
                
                self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
                if device == "cpu":
                    self.llm_model = self.llm_model.to(device)
                
                print(f"DistilGPT-2模型載入成功，使用設備：{device}")
                return
                
            except Exception as distilgpt2_e:
                print(f"載入DistilGPT-2失敗：{str(distilgpt2_e)}")
            
            # 備用選項3：使用更小的GPT-2模型
            try:
                model_name = "gpt2"
                print(f"嘗試載入 {model_name}...")
                
                self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
                if device == "cpu":
                    self.llm_model = self.llm_model.to(device)
                
                print(f"GPT-2模型載入成功，使用設備：{device}")
                return
                
            except Exception as gpt2_e:
                print(f"載入GPT-2失敗：{str(gpt2_e)}")
            
            # 最後備用：簡單文本生成
            print("所有模型載入失敗，將使用簡單文本模板")
            self.llm_tokenizer = None
            self.llm_model = None
            
        except Exception as e:
            print(f"載入語言模型時發生錯誤：{str(e)}")
            self.llm_tokenizer = None
            self.llm_model = None

    def generate_with_gemma(self, prompt, max_length=1024):
        """使用語言模型生成文本"""
        if self.llm_model is None or self.llm_tokenizer is None:
            # 如果沒有模型，使用簡單的模板生成
            return self.generate_template_report(prompt)
        
        try:
            # 編碼輸入，限制輸入長度避免記憶體問題
            max_input_length = 512  # 限制輸入長度
            inputs = self.llm_tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=max_input_length, 
                truncation=True
            )
            
            # 移到正確的設備
            device = next(self.llm_model.parameters()).device
            inputs = inputs.to(device)
            
            # 計算合理的最大生成長度
            input_length = inputs.shape[1]
            max_new_tokens = min(max_length - input_length, 512)  # 限制新生成的token數量
            
            # 生成文本
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,  # 使用max_new_tokens而不是max_length
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3  # 避免重複
                )
            
            # 解碼輸出
            generated_text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 移除原始prompt，只返回生成的部分
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text if generated_text else self.generate_template_report(prompt)
            
        except Exception as e:
            print(f"模型生成失敗：{str(e)}")
            return self.generate_template_report(prompt)
    
    def generate_template_report(self, prompt):
        """使用模板生成基本報告"""
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        template_report = f"""
**Patient Information:**
- Name: [Patient Name]
- Date of Birth: [DOB]
- Sex: [Sex]
- Imaging Modality: Chest CT
- Scanner Model: [Scanner Model]
- Scan Date: {current_date}

**Findings:**
Based on the query provided, the following observations are noted:
{self.extract_template_findings(prompt)}

**Impression:**
The findings suggest further evaluation may be warranted. This corresponds to an appropriate Lung-RADS classification based on the specific findings.

**Differential Diagnosis:**
- Benign: Inflammatory changes, granulomatous disease
- Infectious: Pneumonia, tuberculosis
- Malignant: Primary lung carcinoma, metastatic disease

**Recommendations:**
- Follow-up imaging as clinically indicated
- Correlation with clinical symptoms and laboratory findings
- Consider further diagnostic workup if clinically appropriate

**Note:** This is a template-generated report. For accurate medical diagnosis, please consult with a qualified radiologist and use appropriate diagnostic tools.
        """
        return template_report.strip()
    
    def extract_template_findings(self, prompt):
        """從prompt中提取關鍵發現用於模板"""
        findings_text = "Multiple findings noted requiring radiological assessment."
        
        # 簡單的關鍵詞檢測
        if "nodule" in prompt.lower():
            findings_text += "\n- Pulmonary nodule identified requiring classification and follow-up."
        if "ground glass" in prompt.lower() or "ggo" in prompt.lower():
            findings_text += "\n- Ground glass opacity observed."
        if "lymph node" in prompt.lower():
            findings_text += "\n- Lymph node changes noted."
        
        return findings_text

    def generate_report(self):
        query = self.query_input.text().strip()
        if not query:
            self.report_output.setText("Please enter a query.")
            return

        context = self.retrieve_knowledge(query)
        findings = self.parse_findings(query)
        classifications = [
            f"{f['type'].capitalize()} nodule ({f['size_mm']}mm, {f['location']}): {self.lung_rads_classify(f)}"
            for f in findings
        ]
        auto_classification_text = "\n".join(classifications)

        overall_class = self.select_highest_lung_rads_classification(classifications)

        # 使用者覆蓋自動分類
        selected_class = self.classification_combo.currentText()
        if selected_class != "Auto":
            overall_class = selected_class
        self.classification_label.setText(f"Lung-RADS Classification: {overall_class}")

        system_prompt = f"""
        [EXAMPLE]
        Nodule: 25mm solid nodule in the right lower lobe.
        → Lung-RADS Classification: Category 4B

        GGO: 45mm ground glass opacity in the left lower lobe.
        → Lung-RADS Classification: Category 3 or 4 (depends on growth)

        Lymph Node: 20mm mediastinal lymph node.
        → Considered significantly enlarged; possible malignancy.
        [END EXAMPLE]

        You are a board-certified radiologist with expertise in various imaging modalities.
        Based on the following medical knowledge, generate a structured and clinically valuable radiology report
        that adapts to any given input.

        Overall Lung-RADS Classification:
        {overall_class}

        Automatically parsed and classified findings:
        {auto_classification_text}

        {context}

        The report must strictly follow the structure below:

        **Patient Information:**
        - Name: [Insert Name]
        - Date of Birth: [Insert DOB]
        - Sex: [Insert Sex]
        - Imaging Modality: [Insert Modality]
        - Scanner Model: [Insert Scanner Model]
        - Scan Date: {datetime.datetime.now().strftime("%Y-%m-%d")}

        **Findings:**
        - Describe relevant abnormalities with emphasis on size, location, and shape.
        - ⚠️ At the end of each pulmonary nodule or GGO finding, explicitly add:
        "This corresponds to Lung-RADS Category X."
        - Do not omit this statement.

        **Impression:**
        - Summarize key radiological findings.
        - ⚠️ Clearly state the Lung-RADS classification for the most suspicious finding using:
        "This corresponds to Lung-RADS Category X."

        **Differential Diagnosis:**
        - Provide possible differential diagnoses based on the findings, covering benign, inflammatory, infectious, and malignant etiologies.

        **Recommendations:**
        - Suggest further diagnostic steps such as biopsy, PET-CT, MRI, or follow-up imaging.
        - Clearly indicate appropriate follow-up intervals based on Lung-RADS classification.
        - If findings are incidental, specify whether additional workup is required.

        **Lung Rads Criteria:**
        - Include the Lung-RADS criteria for reference.

        ⚠️ It is mandatory to include the Lung-RADS classification in the body of the report.
        Use the format "This corresponds to Lung-RADS Category X" wherever applicable.

        Ensure the report maintains a neutral and professional tone, avoids premature conclusions unless strongly supported, and strictly adheres to medical accuracy.
        Only output the report. Do not include extra comments or disclaimers.
        """

        try:
            # 使用語言模型生成報告
            if self.llm_model is None:
                self.report_output.setText("使用模板生成報告（語言模型未載入）\n\n")
            
            # 構建完整的prompt
            full_prompt = f"{system_prompt}\n\n用戶查詢：{query}\n\n請生成專業的胸部CT報告："
            
            # 使用模型生成報告
            generated_report = self.generate_with_gemma(full_prompt, max_length=2048)
            
            # 顯示報告
            if "錯誤：" in generated_report or "生成文本時發生錯誤：" in generated_report:
                self.report_output.setText(f"警告：{generated_report}\n\n使用模板生成基本報告...")
            else:
                self.report_output.setText(generated_report)
                
        except Exception as e:
            error_msg = f"報告生成過程中發生錯誤：{str(e)}\n\n使用模板生成基本報告..."
            self.report_output.setText(error_msg)
            # 嘗試使用模板生成
            try:
                template_report = self.generate_template_report(f"用戶查詢：{query}")
                self.report_output.setText(error_msg + "\n\n" + template_report)
            except Exception as template_e:
                self.report_output.setText(f"嚴重錯誤：無法生成報告 - {str(template_e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MedicalReportApp()
    window.show()
    sys.exit(app.exec())