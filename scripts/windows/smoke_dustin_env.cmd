@echo off
setlocal EnableExtensions
if "%TTS_ROOT%"=="" (
  echo Set TTS_ROOT to the relocated tts directory.
  exit /b 1
)
for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
set "VENV=%TTS_ROOT%\venv-dustin-py312"
if not exist "%VENV%\Scripts\python.exe" (
  echo The Dustin virtual environment is missing. Run new_dustin_venv.cmd. Do not use venv-cbft.
  exit /b 1
)
set "TTS_PROFILE=dustin"
set "TTS_CONFIRM_TRAIN="
"%VENV%\Scripts\python.exe" "%REPO%\deploy\smoke_check.py" --packages --gpu
exit /b %ERRORLEVEL%
