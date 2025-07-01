# Chest CT Report Generator / 胸部 CT 報告生成器

[English](#english) | [中文](#chinese)

## English

### Overview
This project is an AI-powered chest CT report generator that helps medical professionals generate standardized reports for chest CT scans. It utilizes advanced language models and RAG (Retrieval-Augmented Generation) techniques to produce accurate and consistent medical reports.

The system now uses **Hugging Face Transformers** with local model support, eliminating dependencies on external services like Ollama, providing faster performance and better offline capabilities.

### Features
- 🤖 **Automated chest CT report generation** with AI assistance
- 📊 **Integration with Lung-RADS criteria** for standardized reporting
- 🖥️ **User-friendly GUI interface** with real-time processing
- 🌐 **Local model support** - runs completely offline after setup
- ⚡ **Smart model fallback** - automatic degradation for different hardware
- 🔧 **CPU/GPU optimization** - adapts to available resources
- 📝 **Multi-language support** for medical terminology

### Project Structure
```
chest-ct-report-generator/
├── RAG/                           # RAG implementation and GUI
│   ├── Gemma3_GUI.py             # Main GUI (GPU optimized)
│   ├── Gemma3_GUI_CPU.py         # CPU optimized version
│   ├── download_model_simple.py   # Model download utility
│   ├── system_check.py           # System diagnostics
│   ├── requirements.txt          # Python dependencies
│   ├── install.bat               # Windows installer
│   ├── uninstall.bat            # Windows uninstaller
│   └── model/                    # Local model storage
│       ├── gemma-3-4b-it/       # Gemma 3 4B model (~8.6GB)
│       └── sentence_transformer/ # Embedding model (~90MB)
├── Fine_Tune/                    # Model fine-tuning files
├── data/                         # Medical data and reports
└── README.md                     # This documentation
```

### System Requirements

#### Hardware Requirements
- **GPU Environment (Recommended)**: 
  - NVIDIA GPU with 16GB+ VRAM
  - 32GB+ System RAM
  - 15GB+ available disk space
  
- **CPU Environment (Minimum)**:
  - 32GB+ System RAM
  - 15GB+ available disk space
  - Multi-core processor (8+ cores recommended)

#### Software Requirements
- Python 3.8 or higher
- Windows/Linux/macOS
- Hugging Face account (for model access)

### Quick Start

#### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/chest-ct-report-generator.git
cd chest-ct-report-generator/RAG
pip install -r requirements.txt
```

#### 2. System Check (Optional)
```bash
python system_check.py
```

#### 3. Model Download
```bash
# Download all models (recommended)
python download_model_simple.py --model all

# Or download specific models
python download_model_simple.py --model gemma      # Gemma 3 4B only
python download_model_simple.py --model sentence   # Embedding model only
```

#### 4. Launch Application
```bash
# GPU environment (recommended)
python Gemma3_GUI.py

# CPU environment (lightweight)
python Gemma3_GUI_CPU.py
```

### Model Setup Guide

#### Hugging Face Setup
1. Create account at https://huggingface.co
2. Request access to https://huggingface.co/google/gemma-3-4b-it
3. Login locally:
   ```bash
   huggingface-cli login
   ```

#### Model Download Options
The system supports multiple download methods:

```bash
# Method 1: Download all models
python download_model_simple.py all

# Method 2: Using parameter flags
python download_model_simple.py --model all

# Method 3: Specific models
python download_model_simple.py gemma          # Gemma 3 4B model
python download_model_simple.py sentence      # Sentence transformer
```

### Performance Optimization

#### Local vs Online Models
| Mode | Loading Time | Memory Usage | Network Required |
|------|-------------|--------------|------------------|
| Local Model | 30-60 seconds | 8-16GB | None |
| Online Model | 5-10 minutes | 8-16GB | High-speed |

#### Smart Model Fallback
1. **Primary**: Local Gemma 3 4B model
2. **Fallback 1**: Online Gemma 3 4B
3. **Fallback 2**: Gemma 2B model
4. **Fallback 3**: DistilGPT-2
5. **Final**: Template-based generation

### Troubleshooting

#### Common Issues

**Issue 1: Model download fails**
```bash
❌ Error: 401 Client Error: Unauthorized
```
Solution: Check Hugging Face access permissions and re-login

**Issue 2: CUDA out of memory**
```bash
❌ CUDA out of memory
```
Solution: Use CPU version or close other GPU applications

**Issue 3: Slow performance**
```bash
⚠️ Using CPU mode - performance may be slow
```
Solution: Upgrade to GPU hardware or use optimized CPU version

#### Diagnostic Commands
```bash
# Check system compatibility
python system_check.py

# Verify model downloads
ls -la model/

# Test basic functionality
python Gemma3_GUI.py  # Should load without errors
```

### License
[Add your license information here]

---

## Chinese

### 概述
本專案是一個基於人工智慧的胸部 CT 報告生成器，旨在協助醫療專業人員生成標準化的胸部 CT 掃描報告。該系統利用先進的語言模型和 RAG（檢索增強生成）技術，以產生準確且一致的醫療報告。

系統現在使用 **Hugging Face Transformers** 並支援本地模型，完全移除了對 Ollama 等外部服務的依賴，提供更快的性能和更好的離線能力。

### 功能特點
- 🤖 **自動化胸部 CT 報告生成** - AI 智能輔助
- 📊 **整合 Lung-RADS 標準** - 標準化報告格式
- 🖥️ **使用者友善的圖形介面** - 即時處理顯示
- 🌐 **本地模型支援** - 設定完成後可完全離線運行
- ⚡ **智能模型降級** - 根據硬體自動調整
- 🔧 **CPU/GPU 優化** - 適應可用資源
- 📝 **多語言支援** - 醫學術語本地化

### 專案結構
```
chest-ct-report-generator/
├── RAG/                           # RAG 實現和圖形介面
│   ├── Gemma3_GUI.py             # 主圖形介面 (GPU 優化)
│   ├── download_model_simple.py   # 模型下載工具
│   ├── system_check.py           # 系統診斷
│   ├── requirements.txt          # Python 依賴套件
│   ├── install.bat               # Windows 安裝程式
│   ├── uninstall.bat            # Windows 解除安裝程式
│   └── model/                    # 本地模型儲存
│       ├── gemma-3-4b-it/       # Gemma 3 4B 模型 (~8.6GB)
│       └── sentence_transformer/ # 嵌入模型 (~90MB)
├── Fine_Tune/                    # 模型微調檔案
├── data/                         # 醫療資料和報告
└── README.md                     # 本說明文件
```

### 系統需求

#### 硬體需求
- **GPU 環境（推薦）**：
  - NVIDIA GPU，16GB+ 顯存
  - 32GB+ 系統記憶體
  - 15GB+ 可用硬碟空間
  
- **CPU 環境（最低要求）**：
  - 32GB+ 系統記憶體
  - 15GB+ 可用硬碟空間
  - 多核心處理器（建議 8+ 核心）

#### 軟體需求
- Python 3.8 或更高版本
- Windows/Linux/macOS
- Hugging Face 帳號（用於模型存取）

### 快速開始

#### 1. 複製和設定
```bash
git clone https://github.com/yourusername/chest-ct-report-generator.git
cd chest-ct-report-generator/RAG
pip install -r requirements.txt
```

#### 2. 系統檢測（可選）
```bash
python system_check.py
```

#### 3. 模型下載
```bash
# 下載所有模型（推薦）
python download_model_simple.py --model all

# 或下載特定模型
python download_model_simple.py --model gemma      # 僅 Gemma 3 4B
python download_model_simple.py --model sentence   # 僅嵌入模型
```

#### 4. 啟動應用程式
```bash
# GPU 環境（推薦）
python Gemma3_GUI.py

# CPU 環境（輕量版）
python Gemma3_GUI_CPU.py
```

### 模型設定指南

#### Hugging Face 設定
1. 在 https://huggingface.co 建立帳號
2. 申請存取權限：https://huggingface.co/google/gemma-3-4b-it
3. 本地登入：
   ```bash
   huggingface-cli login
   ```

#### 模型下載選項
系統支援多種下載方式：

```bash
# 方法1：下載所有模型
python download_model_simple.py all

# 方法2：使用參數標誌
python download_model_simple.py --model all

# 方法3：特定模型
python download_model_simple.py gemma          # Gemma 3 4B 模型
python download_model_simple.py sentence      # 句子轉換器
```

### 性能優化

#### 本地 vs 線上模型
| 模式 | 載入時間 | 記憶體使用 | 網路需求 |
|------|----------|------------|----------|
| 本地模型 | 30-60秒 | 8-16GB | 無 |
| 線上模型 | 5-10分鐘 | 8-16GB | 高速網路 |

#### 智能模型降級
1. **主要**：本地 Gemma 3 4B 模型
2. **備用1**：線上 Gemma 3 4B
3. **備用2**：Gemma 2B 模型
4. **備用3**：DistilGPT-2
5. **最終**：模板式生成

### 故障排除

#### 常見問題

**問題1：模型下載失敗**
```bash
❌ 錯誤：401 Client Error: Unauthorized
```
解決方案：檢查 Hugging Face 存取權限並重新登入

**問題2：CUDA 記憶體不足**
```bash
❌ CUDA out of memory
```
解決方案：使用 CPU 版本或關閉其他 GPU 應用程式

**問題3：性能緩慢**
```bash
⚠️ 使用 CPU 模式 - 性能可能較慢
```
解決方案：升級 GPU 硬體或使用優化的 CPU 版本

#### 診斷命令
```bash
# 檢查系統相容性
python system_check.py

# 驗證模型下載
ls -la model/

# 測試基本功能
python Gemma3_GUI.py  # 應該無錯誤載入
```

### 完整使用流程

#### 步驟1：環境準備
```bash
# 1. 系統檢測
python system_check.py

# 2. 安裝依賴
pip install -r requirements.txt

# 3. Hugging Face 登入
huggingface-cli login
```

#### 步驟2：模型下載
```bash
# 下載所有模型（約 8.7GB）
python download_model_simple.py --model all

# 檢查下載狀態
ls -la model/
```

您應該看到：
```
model/
├── sentence_transformer/    # ~90MB
└── gemma-3-4b-it/          # ~8.6GB
```

#### 步驟3：啟動和測試
```bash
# 啟動程式
python Gemma3_GUI.py

# 測試輸入（在查詢框中）
"10mm solid nodule in right lower lobe"
```

成功訊息：
```
✅ 本地Gemma 3 4B模型載入成功，使用設備：cuda
```

### 進階配置

#### GPU 記憶體優化
```python
# 在 Gemma3_GUI.py 中已自動配置
torch_dtype = torch.float16 if device == "cuda" else torch.float32
```

#### CPU 環境專用版本
```bash
# 使用專為 CPU 優化的版本
python Gemma3_GUI_CPU.py
```

特色：
- 本地 sentence-transformers 優先
- 關鍵詞備用檢索
- 記憶體使用優化
- 智能降級機制

### 授權條款
[在此加入授權資訊]