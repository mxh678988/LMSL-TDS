@echo off
REM 太极调度层 TDS - 对话测试台快捷启动脚本
REM 使用方法：双击此文件即可启动

setlocal enabledelayedexpansion

REM 检测 Python 路径
set "PYTHON_CMD="
set "PYTHON_PATHS="

REM 尝试常见 Python 安装路径
for %%i in (
    "C:\Program Files\Python312\python.exe"
    "C:\Program Files\Python311\python.exe"
    "C:\Program Files\Python310\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe"
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe"
) do (
    if exist "%%i" (
        set "PYTHON_CMD=%%i"
        goto :found_python
    )
)

REM 尝试使用系统 PATH 中的 python
for %%i in (python python3) do (
    for /f "delims=" %%p in ('where %%i 2^>nul') do (
        if not defined PYTHON_CMD (
            set "PYTHON_CMD=%%p"
            goto :found_python
        )
    )
)

:found_python
if not defined PYTHON_CMD (
    echo 错误：未找到 Python 解释器
    echo 请安装 Python 3.10+ 并添加到系统 PATH
    echo.
    echo 下载地址：https://www.python.org/downloads/
    echo 安装时请勾选"Add Python to PATH"选项
    pause
    exit /b 1
)

echo ========================================
echo 太极调度层 TDS 对话测试台
echo ========================================
echo Python: !PYTHON_CMD!
echo.

REM 设置工作目录
set "SCRIPT_DIR=%~dp0"
cd /d "!SCRIPT_DIR!"

REM 检查必要文件
if not exist "start_ui.bat" (
    echo 错误：未找到 start_ui.bat 文件
    echo 请确保在正确的目录下运行此脚本
    pause
    exit /b 1
)

echo 正在启动测试台...
echo 访问地址：http://127.0.0.1:8770
echo.
echo 按 Ctrl+C 停止服务
echo ========================================
echo.

REM 启动测试台
start "TDS 测试台" "!PYTHON_CMD!" -m tds.ui.cli serve

echo 测试台已在后台启动
echo.
echo 如果浏览器没有自动打开，请手动访问：
echo http://127.0.0.1:8770
echo.

REM 保持窗口打开以便查看可能的错误信息
timeout /t 3 /nobreak > nul
echo 测试台启动完成
pause