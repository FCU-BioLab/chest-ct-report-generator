@echo off
setlocal enabledelayedexpansion

:: Initial Check: Winget Availability
echo Checking for winget...
winget --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: winget not found. Please update Windows or install manually.
    pause
    exit /b
)
echo Winget is available.

:: Check: Python Availability
echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Installing Python...
    winget install --id Python.Python.3.10 --accept-source-agreements --accept-package-agreements

    :: Manually update PATH after installation (assuming default install path)
    set "PATH=%LocalAppData%\Programs\Python\Python310;%LocalAppData%\Programs\Python\Python310\Scripts;%PATH%"
)

:: Verify Python Installation (5 attempts)
set attempts=0
:check_python
where python >nul 2>&1
if %errorlevel% neq 0 (
    set /a attempts+=1
    if !attempts! geq 5 (
        echo ERROR: Python installation failed.
        pause
        exit /b
    )
    timeout /t 5 >nul
    goto check_python
)

echo Python installation confirmed:
python --version

:: Ensure pip is available
echo Ensuring pip is available...
python -m ensurepip --upgrade
if %errorlevel% neq 0 (
    echo ERROR: ensurepip failed.
    pause
    exit /b
)

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b
)

:: Remove problematic pathlib (if installed)
echo Checking pathlib compatibility...
pip list | findstr /I pathlib >nul 2>&1
if %errorlevel% equ 0 (
    echo pathlib detected. Removing pathlib...
    pip uninstall pathlib -y
)

:: Install requirements
echo Installing Python dependencies...
if not exist requirements.txt (
    echo ERROR: requirements.txt not found!
    pause
    exit /b
)
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed installing dependencies.
    pause
    exit /b
)

:: Check Ollama installation
echo Checking Ollama...
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo Ollama not found. Installing Ollama...
    winget install Ollama --accept-source-agreements --accept-package-agreements
)

:: Verify Ollama Installation (5 attempts)
set attempts=0
:check_ollama
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    set /a attempts+=1
    if !attempts! geq 5 (
        echo ERROR: Ollama installation failed.
        pause
        exit /b
    )
    timeout /t 5 >nul
    goto check_ollama
)

echo Ollama installed successfully.
ollama --version

:: Download Ollama Model (3 attempts)
set attempts=0
set model=gemma3:4b
:pull_model
echo Downloading Ollama model !model!...
ollama pull !model!
if %errorlevel% neq 0 (
    set /a attempts+=1
    if !attempts! geq 3 (
        echo ERROR: Failed to download Ollama model.
        pause
        exit /b
    )
    echo Retry attempt !attempts!...
    timeout /t 5 >nul
    goto pull_model
)
echo Ollama model downloaded successfully!

:: PyInstaller packaging
echo Building executable with PyInstaller...
python -m PyInstaller --noconfirm ^
    --name MedicalReportAppByGemma3 ^
    --onefile ^
    --windowed ^
    --add-data "lung_rads_criteria.txt;RAG" ^
    --collect-all sentence_transformers ^
    --collect-all fitz ^
    --collect-all faiss ^
    GUI.py

if %errorlevel% neq 0 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b
)

echo Installation and build complete!
pause