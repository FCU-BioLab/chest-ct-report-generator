# 外接硬碟 Repo + 內接硬碟 venv 使用說明

本文件說明如何將本專案的 `repo` 放在外接硬碟，並將 Python `venv` 建立在電腦內接硬碟上使用。

這是目前比較建議的配置，原因如下：

- repo 與 dataset 可以跟著外接硬碟移動。
- `venv` 與大型 Python 套件放在內接硬碟，啟動與安裝通常比較穩。
- `torch` / `torchvision` 可依每台電腦的 CUDA 版本個別安裝，不會被 repo 綁死。

## 建議配置

```text
外接硬碟
E:/chest-ct-report-generator/
E:/chest-ct-report-generator/dataset/LNDb/
E:/chest-ct-report-generator/dataset/MSD/Task06_Lung/
E:/chest-ct-report-generator/dataset/LUNA16/

內接硬碟
C:/venvs/chest-ct-report-generator/
```

如果 dataset 不想放在 repo 內，也可以放在其他磁碟位置，再透過環境變數指定。

## 磁碟格式建議

- `NTFS`：Windows 開發與訓練較建議，穩定性通常較好。
- `exFAT`：可以使用，但比較適合跨平台共用。若長時間大量讀寫，通常不如 `NTFS` 穩。

如果你主要只在 Windows 上使用，優先選 `NTFS`。

## 為什麼不建議把 venv 放在 repo 裡

不建議直接沿用 repo 內原本的 `venv/`，也不建議把 `venv` 跟 repo 一起搬到外接硬碟後繼續使用，原因如下：

- `venv` 常包含舊機器或舊路徑的絕對路徑。
- `torch` / `torchvision` 版本要跟當前電腦的 CUDA 對齊。
- 外接硬碟上的大量套件讀寫，速度與穩定性通常不如內接硬碟。

## 建立流程

### 1. 將 repo 複製到外接硬碟

範例：

```powershell
E:/chest-ct-report-generator
```

### 2. 在內接硬碟建立新的虛擬環境

```powershell
py -3.10 -m venv C:/venvs/chest-ct-report-generator
```

或使用你實際需要的 Python 版本。

### 3. 啟動 venv

```powershell
C:/venvs/chest-ct-report-generator/Scripts/Activate.ps1
```

### 4. 先手動安裝對應 CUDA 版本的 PyTorch

本專案的 `requirements.txt` 已刻意不直接安裝 `torch` / `torchvision`。

請先依你的電腦 CUDA 版本安裝，例如：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

實際請改成你那台電腦對應的版本。

### 5. 安裝其餘套件

```powershell
pip install -r E:/chest-ct-report-generator/requirements.txt
```

`requirements.txt` 內另外保留了一段「PyTorch-dependent packages」註解區。若你需要對應功能，再在 PyTorch 安裝完成後手動安裝：

- `monai`
- `segmentation-models-pytorch`
- `peft`
- `sentence-transformers`
- `trl`
- 其他註解中的套件

### 6. 準備 dataset

你有兩種方式。

方式 A：直接放在 repo 內

```text
E:/chest-ct-report-generator/dataset/LNDb
E:/chest-ct-report-generator/dataset/MSD/Task06_Lung
E:/chest-ct-report-generator/dataset/LUNA16
```

方式 B：放在其他位置，使用環境變數指定

```powershell
$env:LNDB_ROOT = 'E:/medical-data/LNDb'
$env:MSD_LUNG_ROOT = 'E:/medical-data/MSD/Task06_Lung'
$env:LUNA16_ROOT = 'E:/medical-data/LUNA16'
```

## 專案目前已支援的可攜設定

目前專案已調整為可跟著 repo 根目錄移動，包含：

- `llm/ct_report_pipeline/config/config.yaml`
- `llm/ct_report_pipeline/config/pipeline_config.yaml`
- `llm/ct_report_pipeline/config/config_loader.py`
- `n8n/workflows/chest_ct_pipeline_5_stages.json`

也就是說，repo 從 `C:` 移到 `E:` 後，程式會優先以新的 repo 根目錄推導路徑；若有設定環境變數，則以環境變數覆蓋。

## 執行範例

### 快速檢查設定

```powershell
C:/venvs/chest-ct-report-generator/Scripts/python.exe E:/chest-ct-report-generator/llm/ct_report_pipeline/quick_start.py
```

### 執行 n8n pipeline CLI

```powershell
C:/venvs/chest-ct-report-generator/Scripts/python.exe E:/chest-ct-report-generator/n8n/run_case_pipeline.py --help
```

### 執行任意 Python 模組

```powershell
C:/venvs/chest-ct-report-generator/Scripts/python.exe -m detection.retinanet.main --help
```

注意：上面這種 `-m` 用法，請先 `cd` 到 repo 根目錄。

```powershell
cd E:/chest-ct-report-generator
C:/venvs/chest-ct-report-generator/Scripts/python.exe -m detection.retinanet.main --help
```

## n8n 設定

若你有使用 `n8n`，建議設定：

```powershell
$env:CHEST_CT_REPO_ROOT = 'E:/chest-ct-report-generator'
$env:CHEST_CT_PYTHON = 'C:/venvs/chest-ct-report-generator/Scripts/python.exe'
```

這樣 workflow 就不需要寫死 `C:/GitHub/...`。

## 驗證清單

完成搬遷後，至少確認以下兩件事：

1. `quick_start.py` 可正常讀到 config。
2. `n8n/run_case_pipeline.py --help` 可正常執行。

建議命令：

```powershell
cd E:/chest-ct-report-generator
C:/venvs/chest-ct-report-generator/Scripts/python.exe llm/ct_report_pipeline/quick_start.py
C:/venvs/chest-ct-report-generator/Scripts/python.exe n8n/run_case_pipeline.py --help
```

## 常見問題

### Q1. Repo 放外接硬碟、venv 放內接硬碟，可以嗎？

可以，而且這是建議做法。

### Q2. repo 目前可以直接搬去外接硬碟嗎？

可以，但建議不要直接沿用舊的 `venv`。請在內接硬碟重建新的 `venv`。

### Q3. `exFAT` 可以嗎？

可以，但若主要在 Windows 開發與訓練，仍建議 `NTFS`。

### Q4. requirements 為什麼沒有直接放 `torch`？

因為每台電腦的 CUDA 版本不同，若直接放進 `requirements.txt`，很容易裝到不相容版本。