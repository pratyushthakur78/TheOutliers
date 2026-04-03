@echo off
title The Outliers — launching…
set "ROOT=D:\OneDrive - Biz2X Platform Pvt Ltd\Documents\Hackathon"

REM --- Find Google Chrome (common install locations) ---
set "CHROME=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" set "CHROME=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" set "CHROME=%LocalAppData%\Google\Chrome\Application\chrome.exe"

cd /d "%ROOT%"
echo Starting Streamlit server…
echo The app will open in Google Chrome when ready.
echo.

REM Run Streamlit without opening the default browser; UI opens in Chrome below.
start "The Outliers — Streamlit" /D "%ROOT%" cmd /k py -3 -m streamlit run app.py --server.headless true

REM Wait until the app responds (up to ~45s), then open Chrome.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$url='http://127.0.0.1:8501'; $ok=$false; for($i=0;$i -lt 45;$i++) { try { $r=Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop; if($r.StatusCode -ge 200){ $ok=$true; break } } catch {} Start-Sleep -Seconds 1 }"

if exist "%CHROME%" (
  start "" "%CHROME%" "http://127.0.0.1:8501/"
) else (
  echo Google Chrome was not found in the usual folders.
  echo Opening the app in your default browser instead.
  start "" "http://127.0.0.1:8501/"
)

echo.
echo Keep the other window titled "The Outliers — Streamlit" open while you use the app.
echo Close that window to stop the server.
echo.
pause
