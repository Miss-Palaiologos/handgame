@echo off
echo ============================================
echo  HandGame - Build Standalone EXE
echo ============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install torch numpy pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple -q
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo [2/3] Building exe...
pyinstaller --onefile --name handgame --console handgame_pack.py
if errorlevel 1 (
    echo ERROR: PyInstaller failed.
    pause
    exit /b 1
)

echo [3/3] Copying model...
if not exist dist\output mkdir dist\output
copy output\dqn_mixed.pt dist\output\dqn_mixed.pt >nul

echo.
echo ============================================
echo  Done! Share the entire dist\ folder:
echo    dist\handgame.exe
echo    dist\output\dqn_mixed.pt
echo ============================================
pause
