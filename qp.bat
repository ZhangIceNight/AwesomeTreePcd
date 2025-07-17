@echo off
setlocal

:: 获取当前时间作为默认提交信息
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set commit_message=Update at %datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2% %datetime:~8,2%:%datetime:~10,2%:%datetime:~12,2%

:: 如果提供了命令行参数，则使用该参数作为提交信息
if not "%~1"=="" (
    set commit_message=%*
)

:: 执行git命令
git add .
git commit -m "%commit_message%"
git push

echo 已完成推送！提交信息：%commit_message%

pause