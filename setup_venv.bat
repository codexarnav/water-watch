@echo off
echo ========================================
echo Water Watch - Virtual Environment Setup
echo ========================================
echo.

echo Step 1: Creating virtual environment...
python -m venv venv

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing backend dependencies...
cd backend
pip install -r requirements.txt

echo.
echo Step 5: Installing dashboard dependencies...
cd ..\dashboard
pip install -r requirements.txt

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To activate the virtual environment:
echo   venv\Scripts\activate
echo.
echo To start the backend:
echo   cd backend
echo   uvicorn main:app --reload
echo.
echo To start the dashboard:
echo   cd dashboard
echo   streamlit run app.py
echo ========================================
echo.

pause
