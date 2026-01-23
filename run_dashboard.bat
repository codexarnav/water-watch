@echo off
echo Starting Water Watch Dashboard...
echo.

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_venv.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
cd dashboard
streamlit run app.py
