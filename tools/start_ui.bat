@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
echo ==========================================
echo  TDS 对话测试台（知识分支，不计入可售 IP）
echo ==========================================
echo  模型后端默认 http://127.0.0.1:11434
echo  面板地址默认 http://127.0.0.1:8770/
echo  关闭本窗口即停止面板。
echo.
python -m tds.ui.cli serve --port 8770 --open
if errorlevel 1 (
  echo.
  echo [失败] 启动未成功。请确认：
  echo   1. 已安装 Python 3.9+ 且在 PATH 中；
  echo   2. 本脚本位于 TDS 工程的 tools 目录内（需保证能找到 tds 包）；
  echo   3. 显式指定后端：python -m tds.ui.cli serve --backend http://127.0.0.1:11434
  pause
)
endlocal
