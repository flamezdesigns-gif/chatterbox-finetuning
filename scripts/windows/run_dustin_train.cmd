@echo off
setlocal EnableExtensions
if /I not "%~1"=="train" (
  echo This launcher does not train unless you pass the train argument.
  echo It never starts the Lewis job.
  echo.
  echo   run_dustin_train.cmd train
  echo   smoke_dustin_env.cmd
  exit /b 1
)
if "%TTS_ROOT%"=="" (
  echo Set TTS_ROOT to the relocated tts directory.
  exit /b 1
)
if "%TTS_MODEL_DIR%"=="" (
  echo TTS_MODEL_DIR is unset. Dustin reads pretrained_models inside this checkout.
) else (
  echo Using read-only TTS_MODEL_DIR. Setup will not modify it.
)
for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
set "VENV=%TTS_ROOT%\venv-dustin-py312"
if not exist "%VENV%\Scripts\python.exe" (
  echo The Dustin virtual environment is missing. Run new_dustin_venv.cmd first. Do not use venv-cbft.
  exit /b 1
)
if not exist "%TTS_ROOT%\dustin" mkdir "%TTS_ROOT%\dustin"
if not exist "%TTS_ROOT%\dustin\data" mkdir "%TTS_ROOT%\dustin\data"
if not exist "%TTS_ROOT%\dustin\data\wavs" mkdir "%TTS_ROOT%\dustin\data\wavs"
if not exist "%TTS_ROOT%\dustin\data\preprocess" mkdir "%TTS_ROOT%\dustin\data\preprocess"
if not exist "%TTS_ROOT%\dustin\output" mkdir "%TTS_ROOT%\dustin\output"
if not exist "%TTS_ROOT%\dustin\logs" mkdir "%TTS_ROOT%\dustin\logs"
set "TTS_PROFILE=dustin"
set "TTS_ALLOW_LEWIS="
set "TTS_CONFIRM_TRAIN=1"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
pushd "%REPO%"
"%VENV%\Scripts\python.exe" -u train.py > "%TTS_ROOT%\dustin\logs\train.log" 2>&1
set "RC=%ERRORLEVEL%"
popd
echo Training process exited %RC%. The log is under the Dustin logs directory.
exit /b %RC%
