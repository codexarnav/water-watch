@echo off
echo ========================================
echo Water Watch System - Startup Script
echo ========================================
echo.

echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not running
    echo Please install Docker Desktop and start it
    pause
    exit /b 1
)

echo Docker is running!
echo.

echo Starting services with Docker Compose...
docker-compose up -d

echo.
echo Waiting for services to start...
timeout /t 10 /nobreak >nul

echo.
echo ========================================
echo Services Status:
echo ========================================
docker-compose ps

echo.
echo ========================================
echo Access Points:
echo ========================================
echo Dashboard:    http://localhost:8501
echo API Docs:     http://localhost:8000/docs
echo Qdrant UI:    http://localhost:6333/dashboard
echo.

echo ========================================
echo To view logs:
echo   docker-compose logs -f
echo.
echo To stop services:
echo   docker-compose down
echo ========================================
echo.

pause
