@echo off
:: CT-ViT 訓練系統啟動腳本
:: 用於快速啟動訓練、驗證或推理

setlocal enabledelayedexpansion

:: 顏色定義
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "NC=[0m"

echo %BLUE%========================================%NC%
echo %BLUE%      CT-ViT 訓練系統啟動腳本      %NC%
echo %BLUE%========================================%NC%
echo.

:: 檢查是否在正確目錄
if not exist "train.py" (
    echo %RED%錯誤: 請在 CT_ViT_Training 目錄下運行此腳本%NC%
    pause
    exit /b 1
)

:: 檢查Python和虛擬環境
echo %YELLOW%檢查Python環境...%NC%
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%錯誤: 未找到Python，請安裝Python 3.8+%NC%
    pause
    exit /b 1
)

:: 顯示功能選單
echo.
echo %GREEN%請選擇要執行的功能:%NC%
echo 1. 安裝依賴套件
echo 2. 訓練模型
echo 3. 評估模型
echo 4. 單張影像推理
echo 5. 批次推理
echo 6. 檢查系統環境
echo 7. 清理快取和日誌
echo 0. 退出
echo.

set /p choice="請輸入選項 (0-7): "

if "%choice%"=="1" goto install_deps
if "%choice%"=="2" goto train_model
if "%choice%"=="3" goto evaluate_model
if "%choice%"=="4" goto single_inference
if "%choice%"=="5" goto batch_inference
if "%choice%"=="6" goto check_environment
if "%choice%"=="7" goto cleanup
if "%choice%"=="0" goto exit
goto invalid_choice

:install_deps
echo.
echo %YELLOW%安裝依賴套件...%NC%
pip install -r requirements.txt
if errorlevel 1 (
    echo %RED%安裝失敗！%NC%
    pause
    exit /b 1
)
echo %GREEN%依賴套件安裝完成！%NC%
pause
goto menu

:train_model
echo.
echo %YELLOW%開始訓練模型...%NC%
set /p config_path="請輸入配置文件路徑 (默認: configs/default_config.yaml): "
if "%config_path%"=="" set config_path=configs/default_config.yaml

if not exist "%config_path%" (
    echo %RED%錯誤: 配置文件不存在: %config_path%%NC%
    pause
    goto menu
)

echo %BLUE%使用配置文件: %config_path%%NC%
python train.py --config "%config_path%"
echo.
echo %GREEN%訓練完成！%NC%
pause
goto menu

:evaluate_model
echo.
echo %YELLOW%評估模型...%NC%
set /p model_path="請輸入模型路徑: "
set /p dataset_path="請輸入測試資料集路徑 (默認: ../dataset_splits/test): "
if "%dataset_path%"=="" set dataset_path=../dataset_splits/test

if not exist "%model_path%" (
    echo %RED%錯誤: 模型路徑不存在: %model_path%%NC%
    pause
    goto menu
)

python inference.py --mode evaluate --model_path "%model_path%" --input "%dataset_path%" --output "./evaluation_results"
echo.
echo %GREEN%評估完成！結果保存在 evaluation_results 目錄%NC%
pause
goto menu

:single_inference
echo.
echo %YELLOW%單張影像推理...%NC%
set /p model_path="請輸入模型路徑: "
set /p image_path="請輸入影像路徑: "

if not exist "%model_path%" (
    echo %RED%錯誤: 模型路徑不存在: %model_path%%NC%
    pause
    goto menu
)

if not exist "%image_path%" (
    echo %RED%錯誤: 影像文件不存在: %image_path%%NC%
    pause
    goto menu
)

python inference.py --mode single --model_path "%model_path%" --input "%image_path%"
pause
goto menu

:batch_inference
echo.
echo %YELLOW%批次推理...%NC%
set /p model_path="請輸入模型路徑: "
set /p input_path="請輸入影像目錄或影像列表文件: "

if not exist "%model_path%" (
    echo %RED%錯誤: 模型路徑不存在: %model_path%%NC%
    pause
    goto menu
)

if not exist "%input_path%" (
    echo %RED%錯誤: 輸入路徑不存在: %input_path%%NC%
    pause
    goto menu
)

python inference.py --mode batch --model_path "%model_path%" --input "%input_path%" --output "./batch_results"
echo.
echo %GREEN%批次推理完成！結果保存在 batch_results 目錄%NC%
pause
goto menu

:check_environment
echo.
echo %YELLOW%檢查系統環境...%NC%
echo.

echo %BLUE%Python版本:%NC%
python --version

echo.
echo %BLUE%PyTorch版本:%NC%
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>nul || echo 未安裝PyTorch

echo.
echo %BLUE%GPU信息:%NC%
python -c "import torch; print(f'GPU數量: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>nul || echo 無GPU或PyTorch未安裝

echo.
echo %BLUE%Transformers版本:%NC%
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>nul || echo 未安裝Transformers

echo.
echo %BLUE%其他套件:%NC%
python -c "import cv2, pydicom, matplotlib, seaborn; print('OpenCV, pydicom, matplotlib, seaborn: 已安裝')" 2>nul || echo 部分套件未安裝

pause
goto menu

:cleanup
echo.
echo %YELLOW%清理快取和日誌...%NC%

:: 清理Python快取
if exist "__pycache__" (
    rmdir /s /q "__pycache__"
    echo 已清理 __pycache__
)
if exist "src\__pycache__" (
    rmdir /s /q "src\__pycache__"
    echo 已清理 src\__pycache__
)

:: 清理日誌文件
if exist "logs" (
    rmdir /s /q "logs"
    echo 已清理 logs 目錄
)
if exist "runs" (
    rmdir /s /q "runs"
    echo 已清理 runs 目錄
)

:: 清理臨時文件
del /q *.tmp 2>nul
del /q *.log 2>nul

echo %GREEN%清理完成！%NC%
pause
goto menu

:invalid_choice
echo %RED%無效選項，請重新選擇%NC%
pause

:menu
cls
echo %BLUE%========================================%NC%
echo %BLUE%      CT-ViT 訓練系統啟動腳本      %NC%
echo %BLUE%========================================%NC%
echo.
echo %GREEN%請選擇要執行的功能:%NC%
echo 1. 安裝依賴套件
echo 2. 訓練模型
echo 3. 評估模型
echo 4. 單張影像推理
echo 5. 批次推理
echo 6. 檢查系統環境
echo 7. 清理快取和日誌
echo 0. 退出
echo.

set /p choice="請輸入選項 (0-7): "

if "%choice%"=="1" goto install_deps
if "%choice%"=="2" goto train_model
if "%choice%"=="3" goto evaluate_model
if "%choice%"=="4" goto single_inference
if "%choice%"=="5" goto batch_inference
if "%choice%"=="6" goto check_environment
if "%choice%"=="7" goto cleanup
if "%choice%"=="0" goto exit
goto invalid_choice

:exit
echo.
echo %GREEN%感謝使用 CT-ViT 訓練系統！%NC%
pause
exit /b 0
