@echo off
setlocal EnableExtensions
rem Creates venv-dustin-py312. Does not modify any other virtual environment.

if "%TTS_ROOT%"=="" (
  echo Set TTS_ROOT to the relocated tts directory, then run this script again.
  exit /b 1
)
if "%TTS_MODEL_DIR%"=="" (
  echo TTS_MODEL_DIR is unset. Dustin reads pretrained_models inside this checkout.
  echo A sibling checkout must set TTS_MODEL_DIR to the existing standard base weights.
) else (
  echo Using read-only TTS_MODEL_DIR. Setup will not modify it.
)

for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
set "VENV=%TTS_ROOT%\venv-dustin-py312"

if exist "%TTS_ROOT%\venv-cbft" (
  echo Found venv-cbft. It will not be modified.
)
if exist "%VENV%" (
  echo %VENV% already exists. Refusing to recreate or alter it.
  exit /b 1
)

set "BASEPY="
if defined PYTHON312 set "BASEPY=%PYTHON312%"
if not defined BASEPY (
  for /f "delims=" %%I in ('py -3.12 -c "import sys; print(sys.executable)"') do set "BASEPY=%%I"
)
if not defined BASEPY (
  echo Python 3.12 was not found. Install it, or set PYTHON312 to the base interpreter.
  echo Do not point PYTHON312 at a moved virtual environment.
  exit /b 1
)

echo "%BASEPY%" | findstr /I /C:"\venv-cbft\" /C:"\venv-dustin-py312\" >nul
if not errorlevel 1 (
  echo Refusing to bootstrap from an existing training environment.
  exit /b 1
)

"%BASEPY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3, 12) else 1)"
if errorlevel 1 (
  echo The selected interpreter is not Python 3.12.
  exit /b 1
)

echo Creating the new virtual environment.
"%BASEPY%" -m venv "%VENV%"
if errorlevel 1 exit /b 1

"%VENV%\Scripts\python.exe" "%REPO%\deploy\create_dustin_venv.py" --apply
if errorlevel 1 exit /b 1

echo Running the smoke check. This does not train and does not download TTS weights.
set "TTS_PROFILE=dustin"
set "TTS_CONFIRM_TRAIN="
"%VENV%\Scripts\python.exe" "%REPO%\deploy\smoke_check.py" --packages --gpu
exit /b %ERRORLEVEL%
