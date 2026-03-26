# CT 報告生成流程（ct_report_pipeline）

此模組負責胸部 CT 的後處理流程，包含：

- 結節分割（MedSAM2）
- 特徵萃取（大小、體積、HU 等）
- 微調資料生成與 LLM 報告生成

## 目錄重點

- `config/`：流程設定與載入器
- `features/`：特徵萃取邏輯
- `scripts/`：資料生成、微調、工具腳本
- `segmentation/`：分割相關訓練/推論模組
- `quick_start.py`：快速示範入口
- `report_generator.py`：報告生成主程式

## 環境安裝

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 快速執行

```bash
python quick_start.py
```

## 互動式分割

```bash
python scripts/interactive_segmentation.py
```

## 分割模型微調（MedSAM2）

```bash
cd segmentation/finetune_medsam2
python main.py --data_root <LNDB_PATH> --epochs 50
```

## LLM 微調資料準備

```bash
python scripts/generate_finetune_data.py
python scripts/generate_real_report_data.py --reports_dir <REPORTS_PATH>
```

## LLM 微調

```bash
python scripts/finetune_llama.py --epochs 5 --batch_size 4
```

常用參數：

- `--model_name`：基礎模型名稱（預設通常為 Llama 系列指令模型）
- `--epochs`：訓練回合數
- `--lora_r`：LoRA rank
- `--use_8bit`：啟用 8-bit 載入（降低記憶體使用）

## 建議流程

1. 完成分割與特徵萃取
2. 產生 finetune JSONL
3. 訓練 LoRA 權重
4. 用 `report_generator.py` 產生最終報告

## 備註

- 詳細參數可用 `python <script> --help` 查詢。
- 若要串接整體案例流程，請搭配專案根目錄的 n8n runner。
