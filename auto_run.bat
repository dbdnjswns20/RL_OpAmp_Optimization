@echo off
title [Auto-RL System] 2-Stage OpAmp Optimization

echo ==================================================
echo 1. Cleaning up previous TensorBoard instances...
echo ==================================================
:: 기존에 실행 중인 텐서보드가 있다면 강제 종료 (충돌 방지)
taskkill /IM tensorboard.exe /F 2>NUL

echo.
echo ==================================================
echo 2. Starting TensorBoard (Port 6007)...
echo ==================================================
:: 백그라운드(/B)에서 텐서보드 실행. 로그 폴더 경로 확인 필수!
start /B tensorboard --logdir=".\ppo_2stage_logs" --port 6007

:: 텐서보드가 켜질 때까지 3초 대기 (바로 브라우저 켜면 연결 거부될 수 있음)
timeout /t 3 >nul

:: 브라우저 자동 실행
start http://localhost:6007

echo.
echo ==================================================
echo 3. Starting Automated Training Loop...
echo    Running: run_loop.py
echo ==================================================

:: [핵심] 반복 실행을 담당하는 파이썬 스크립트를 실행합니다.
python run_loop.py

echo.
echo ==================================================
echo 🎉 All Training Loops Finished!
echo ==================================================
pause