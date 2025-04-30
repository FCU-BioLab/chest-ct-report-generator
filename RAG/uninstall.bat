@echo off
setlocal enabledelayedexpansion

:: Confirm uninstallation
choice /M "This will uninstall MedicalReportApp and its dependencies. Continue?"
if errorlevel 2 (
    echo Uninstallation canceled.
    pause
    exit /b
)

:: Delete PyInstaller build directories and executable
echo Removing PyInstaller build files...
rd /s /q build >nul 2>&1
rd /s /q dist >nul 2>&1
del /f /q MedicalReportApp.spec >nul 2>&1
del /f /q MedicalReportApp.exe >nul 2>&1

:: Uninstall Python packages from requirements.txt
@REM if exist requirements.txt (
@REM     echo Uninstalling Python dependencies...
@REM     for /f "delims=" %%i in (requirements.txt) do (
@REM         pip uninstall %%i -y
@REM     )
@REM )

:: Optional: Uninstall Ollama
@REM choice /M "Do you want to uninstall Ollama?"
@REM if errorlevel 1 (
@REM     echo Uninstalling Ollama...
@REM     winget uninstall Ollama
@REM )

:: Optional: Uninstall Python
@REM choice /M "Do you want to uninstall Python?"
@REM if errorlevel 1 (
@REM     echo Uninstalling Python...
@REM     winget uninstall Python
@REM )

echo Uninstallation complete!
pause