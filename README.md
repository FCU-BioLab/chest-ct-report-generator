# Chest CT Report Generator / 胸部 CT 報告生成器

[English](#english) | [中文](#chinese)

## English

### Overview
This project is an AI-powered chest CT report generator that helps medical professionals generate standardized reports for chest CT scans. It utilizes advanced language models and RAG (Retrieval-Augmented Generation) techniques to produce accurate and consistent medical reports.

### Features
- Automated chest CT report generation
- Integration with Lung-RADS criteria
- User-friendly GUI interface
- Support for English report generation
- Fine-tuned language models for medical report generation

### Project Structure
```
chest-ct-report-generator/
├── RAG/                    # RAG implementation and GUI
│   ├── Gemma3_GUI.py      # Main GUI application
│   ├── requirements.txt    # Python dependencies
│   ├── install.bat        # Installation script
│   └── uninstall.bat      # Uninstallation script
├── Fine_Tune/             # Fine-tuning related files
│   └── llama3.2/         # Fine-tuned model files
└── data/                  # Data directory
```

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/chest-ct-report-generator.git
   cd chest-ct-report-generator
   ```

2. Run the installation script:
   - Windows: Double-click `RAG/install.bat`
   - Or manually install dependencies:
     ```bash
     cd RAG
     pip install -r requirements.txt
     ```

### Usage
1. Launch the application:
   ```bash
   cd RAG
   python Gemma3_GUI.py
   ```

2. Follow the on-screen instructions to:
   - Input prompt information
   - Generate and review reports

### Requirements
- Python 3.8 or higher
- Windows operating system
- Sufficient disk space for model files
- Internet connection for initial setup

### License
[Add your license information here]

---

## Chinese

### 概述
本專案是一個基於人工智慧的胸部 CT 報告生成器，旨在協助醫療專業人員生成標準化的胸部 CT 掃描報告。該系統利用先進的語言模型和 RAG（檢索增強生成）技術，以產生準確且一致的醫療報告。

### 功能特點
- 自動化胸部 CT 報告生成
- 整合 Lung-RADS 標準
- 使用者友善的圖形介面
- 支援英文報告生成
- 針對醫療報告生成進行微調的語言模型

### 專案結構
```
chest-ct-report-generator/
├── RAG/                    # RAG 實現和圖形介面
│   ├── Gemma3_GUI.py      # 主圖形介面應用程式
│   ├── requirements.txt    # Python 依賴套件
│   ├── install.bat        # 安裝腳本
│   └── uninstall.bat      # 解除安裝腳本
├── Fine_Tune/             # 模型微調相關檔案
│   └── llama3.2/         # 微調模型檔案
└── data/                  # 資料目錄
```

### 安裝步驟
1. 複製此儲存庫：
   ```bash
   git clone https://github.com/yourusername/chest-ct-report-generator.git
   cd chest-ct-report-generator
   ```

2. 執行安裝腳本：
   - Windows：雙擊 `RAG/install.bat`
   - 或手動安裝依賴套件：
     ```bash
     cd RAG
     pip install -r requirements.txt
     ```

### 使用方式
1. 啟動應用程式：
   ```bash
   cd RAG
   python Gemma3_GUI.py
   ```

2. 依照螢幕指示：
   - 輸入prompt資訊
   - 生成並檢視報告

### 系統需求
- Python 3.8 或更高版本
- Windows 作業系統
- 足夠的硬碟空間用於模型檔案
- 網路連線（用於初始設定）

### 授權條款
[在此加入授權資訊]