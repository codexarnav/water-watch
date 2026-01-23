@echo off
echo Starting Water Watch Backend...
echo.

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_venv.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
