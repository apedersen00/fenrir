@echo off
REM Batch script to create a local Conda environment and run a test script

REM Set environment path and files
SET ENV_PATH=%CD%\.snnenv
SET YAML_FILE=%CD%\environment.yml
SET TEST_SCRIPT=%CD%\test_env.py

REM Check if Conda is installed
WHERE conda >nul 2>nul
IF ERRORLEVEL 1 (
    echo [ERROR] Conda is not installed or not in PATH!
    echo Please install Miniconda or add it to PATH.
    pause
    exit /b 1
)

REM Manually activate Conda base
CALL "%ProgramData%\Miniconda3\condabin\conda.bat" activate base

REM Check if .snnenv already exists
IF EXIST "%ENV_PATH%" (
    echo [INFO] Removing existing environment...
    CALL conda remove --prefix "%ENV_PATH%" --all -y
)

REM Create the new environment
echo [INFO] Creating Conda environment in "%ENV_PATH%"...
CALL conda env create --prefix "%ENV_PATH%" --file "%YAML_FILE%"

IF ERRORLEVEL 1 (
    echo [ERROR] Failed to create Conda environment!
    pause
    exit /b 1
)

REM Activate the environment
echo [INFO] Activating Conda environment...
CALL "%ProgramData%\Miniconda3\condabin\conda.bat" activate "%ENV_PATH%"

REM Install PyTorch manually with CUDA 12.6
echo [INFO] Installing PyTorch with CUDA 12.6...
CALL python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

IF ERRORLEVEL 1 (
    echo [ERROR] Failed to install PyTorch!
    pause
    exit /b 1
)

REM Verify installation
echo [INFO] Verifying installation...
CALL python --version
CALL python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA Available:', torch.cuda.is_available())"

REM Run the test script
IF EXIST "%TEST_SCRIPT%" (
    echo [INFO] Running test script...
    CALL python "%TEST_SCRIPT%"
) ELSE (
    echo [WARNING] Test script not found! Skipping.
)

echo [SUCCESS] Environment setup complete! 🎉
pause
exit /b 0
