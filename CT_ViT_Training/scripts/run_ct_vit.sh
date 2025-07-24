#!/bin/bash
# CT-ViT 訓練系統啟動腳本 (Linux/Mac)
# 用於快速啟動訓練、驗證或推理

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}      CT-ViT 訓練系統啟動腳本      ${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# 檢查是否在正確目錄
if [ ! -f "train.py" ]; then
    echo -e "${RED}錯誤: 請在 CT_ViT_Training 目錄下運行此腳本${NC}"
    exit 1
fi

# 檢查Python
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}錯誤: 未找到Python，請安裝Python 3.8+${NC}"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo -e "${YELLOW}檢查Python環境...${NC}"
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "發現: $PYTHON_VERSION"

# 主選單函數
show_menu() {
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}      CT-ViT 訓練系統啟動腳本      ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
    echo -e "${GREEN}請選擇要執行的功能:${NC}"
    echo "1. 安裝依賴套件"
    echo "2. 訓練模型"
    echo "3. 評估模型"
    echo "4. 單張影像推理"
    echo "5. 批次推理"
    echo "6. 檢查系統環境"
    echo "7. 清理快取和日誌"
    echo "0. 退出"
    echo
    read -p "請輸入選項 (0-7): " choice
}

# 安裝依賴
install_deps() {
    echo
    echo -e "${YELLOW}安裝依賴套件...${NC}"
    $PYTHON_CMD -m pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}依賴套件安裝完成！${NC}"
    else
        echo -e "${RED}安裝失敗！${NC}"
        read -p "按任意鍵繼續..."
        return 1
    fi
    read -p "按任意鍵繼續..."
}

# 訓練模型
train_model() {
    echo
    echo -e "${YELLOW}開始訓練模型...${NC}"
    read -p "請輸入配置文件路徑 (默認: configs/default_config.yaml): " config_path
    config_path=${config_path:-configs/default_config.yaml}
    
    if [ ! -f "$config_path" ]; then
        echo -e "${RED}錯誤: 配置文件不存在: $config_path${NC}"
        read -p "按任意鍵繼續..."
        return 1
    fi
    
    echo -e "${BLUE}使用配置文件: $config_path${NC}"
    $PYTHON_CMD train.py --config "$config_path"
    echo
    echo -e "${GREEN}訓練完成！${NC}"
    read -p "按任意鍵繼續..."
}

# 評估模型
evaluate_model() {
    echo
    echo -e "${YELLOW}評估模型...${NC}"
    read -p "請輸入模型路徑: " model_path
    read -p "請輸入測試資料集路徑 (默認: ../dataset_splits/test): " dataset_path
    dataset_path=${dataset_path:-../dataset_splits/test}
    
    if [ ! -d "$model_path" ] && [ ! -f "$model_path" ]; then
        echo -e "${RED}錯誤: 模型路徑不存在: $model_path${NC}"
        read -p "按任意鍵繼續..."
        return 1
    fi
    
    $PYTHON_CMD inference.py --mode evaluate --model_path "$model_path" --input "$dataset_path" --output "./evaluation_results"
    echo
    echo -e "${GREEN}評估完成！結果保存在 evaluation_results 目錄${NC}"
    read -p "按任意鍵繼續..."
}

# 單張影像推理
single_inference() {
    echo
    echo -e "${YELLOW}單張影像推理...${NC}"
    read -p "請輸入模型路徑: " model_path
    read -p "請輸入影像路徑: " image_path
    
    if [ ! -d "$model_path" ] && [ ! -f "$model_path" ]; then
        echo -e "${RED}錯誤: 模型路徑不存在: $model_path${NC}"
        read -p "按任意鍵繼續..."
        return 1
    fi
    
    if [ ! -f "$image_path" ]; then
        echo -e "${RED}錯誤: 影像文件不存在: $image_path${NC}"
        read -p "按任意鍵繼續..."
        return 1
    fi
    
    $PYTHON_CMD inference.py --mode single --model_path "$model_path" --input "$image_path"
    read -p "按任意鍵繼續..."
}

# 批次推理
batch_inference() {
    echo
    echo -e "${YELLOW}批次推理...${NC}"
    read -p "請輸入模型路徑: " model_path
    read -p "請輸入影像目錄或影像列表文件: " input_path
    
    if [ ! -d "$model_path" ] && [ ! -f "$model_path" ]; then
        echo -e "${RED}錯誤: 模型路徑不存在: $model_path${NC}"
        read -p "按任意鍵繼續..."
        return 1
    fi
    
    if [ ! -e "$input_path" ]; then
        echo -e "${RED}錯誤: 輸入路徑不存在: $input_path${NC}"
        read -p "按任意鍵繼續..."
        return 1
    fi
    
    $PYTHON_CMD inference.py --mode batch --model_path "$model_path" --input "$input_path" --output "./batch_results"
    echo
    echo -e "${GREEN}批次推理完成！結果保存在 batch_results 目錄${NC}"
    read -p "按任意鍵繼續..."
}

# 檢查環境
check_environment() {
    echo
    echo -e "${YELLOW}檢查系統環境...${NC}"
    echo
    
    echo -e "${BLUE}Python版本:${NC}"
    $PYTHON_CMD --version
    
    echo
    echo -e "${BLUE}PyTorch版本:${NC}"
    $PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "未安裝PyTorch"
    
    echo
    echo -e "${BLUE}GPU信息:${NC}"
    $PYTHON_CMD -c "import torch; print(f'GPU數量: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null || echo "無GPU或PyTorch未安裝"
    
    echo
    echo -e "${BLUE}Transformers版本:${NC}"
    $PYTHON_CMD -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "未安裝Transformers"
    
    echo
    echo -e "${BLUE}其他套件:${NC}"
    $PYTHON_CMD -c "import cv2, pydicom, matplotlib, seaborn; print('OpenCV, pydicom, matplotlib, seaborn: 已安裝')" 2>/dev/null || echo "部分套件未安裝"
    
    read -p "按任意鍵繼續..."
}

# 清理快取
cleanup() {
    echo
    echo -e "${YELLOW}清理快取和日誌...${NC}"
    
    # 清理Python快取
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    echo "已清理 __pycache__"
    
    # 清理日誌文件
    if [ -d "logs" ]; then
        rm -rf logs
        echo "已清理 logs 目錄"
    fi
    
    if [ -d "runs" ]; then
        rm -rf runs
        echo "已清理 runs 目錄"
    fi
    
    # 清理臨時文件
    rm -f *.tmp *.log 2>/dev/null
    
    echo -e "${GREEN}清理完成！${NC}"
    read -p "按任意鍵繼續..."
}

# 主循環
while true; do
    show_menu
    
    case $choice in
        1)
            install_deps
            ;;
        2)
            train_model
            ;;
        3)
            evaluate_model
            ;;
        4)
            single_inference
            ;;
        5)
            batch_inference
            ;;
        6)
            check_environment
            ;;
        7)
            cleanup
            ;;
        0)
            echo
            echo -e "${GREEN}感謝使用 CT-ViT 訓練系統！${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}無效選項，請重新選擇${NC}"
            read -p "按任意鍵繼續..."
            ;;
    esac
done
