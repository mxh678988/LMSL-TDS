@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
echo ==========================================
echo  TDS 用例批量回归（知识分支，不计入可售 IP）
echo ==========================================
echo  默认：套件 full、自动挑 2 个本地多模态优先模型
echo  报告输出：bench_out 目录（Markdown + JSON）
echo.
echo  常用变体：
echo    python -m tds.ui.cli bench --list
echo    python -m tds.ui.cli bench --suite basic --models qwen2.5:3b --show
echo    python -m tds.ui.cli bench --suite full --models a,b --repeat 2 --out bench_out
echo.
python -m tds.ui.cli bench --suite full --repeat 1 --out bench_out
if errorlevel 1 (
  echo.
  echo [失败] 回归未完成。请确认：
  echo   1. Ollama 已启动（默认 http://127.0.0.1:11434）；
  echo   2. 本脚本位于 TDS 工程的 tools 目录内（需保证能找到 tds 包）；
  echo   3. 可用 --models 显式指定模型名，或用 --backend 指定其他后端。
  pause
)
endlocal
