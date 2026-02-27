@echo off
REM ============================================================
REM  install.bat – Rendszergazdai jog NELKUL is mukodik
REM  A csomagok ide kerulnek: C:\ai\  (rovid utvonal!)
REM ============================================================

echo ============================================================
echo   Magyar Sentiment Elemzo - Windows Telepito
echo   (Rendszergazdai jog nem szukseges)
echo ============================================================
echo.

REM ── Rövid alap útvonal – ez oldja meg a Long Path problémát ─────────────
set AI_DIR=C:\ai
set VENV_DIR=%AI_DIR%\venv

echo A program ide lesz telepitve: %VENV_DIR%
echo (Rovid utvonal = nincs Long Path problema)
echo.

REM ── 1. Célmappa létrehozása ──────────────────────────────────────────────
echo [1/6] Celkonyvtar letrehozasa: %AI_DIR%
if not exist "%AI_DIR%" mkdir "%AI_DIR%"
if %errorlevel% neq 0 (
    echo HIBA: Nem sikerult letrehozni: %AI_DIR%
    echo Probald meg a D:\ meghajtora: allitsd at az AI_DIR sort
    pause
    exit /b 1
)

REM ── 2. Virtuális környezet létrehozása ──────────────────────────────────
echo [2/6] Virtualis kornyezet letrehozasa...
if exist "%VENV_DIR%" (
    echo      Mar letezik, kihagyva.
) else (
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo HIBA: Virtualis kornyezet letrehozasa sikertelen!
        pause
        exit /b 1
    )
    echo      Letrehozva: %VENV_DIR%
)
echo.

REM ── 3. pip frissítése a venv-ben ────────────────────────────────────────
echo [3/6] pip frissitese...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet
echo      Kesz.
echo.

REM ── 4. PyTorch CPU telepítése (rövidebb útvonalakkal!) ──────────────────
echo [4/6] PyTorch telepitese (CPU valtozat, ~200 MB)...
echo      (Ez eltart par percig...)
"%VENV_DIR%\Scripts\pip.exe" install torch --index-url https://download.pytorch.org/whl/cpu --quiet
if %errorlevel% neq 0 (
    echo HIBA: PyTorch telepites sikertelen!
    pause
    exit /b 1
)
echo      PyTorch kesz.
echo.

REM ── 5. Többi csomag telepítése ──────────────────────────────────────────
echo [5/6] NLP csomagok telepitese (spacy, transformers, bertopic, stb.)...
echo      (Ez is eltart par percig...)
"%VENV_DIR%\Scripts\pip.exe" install ^
    spacy ^
    huspacy ^
    transformers ^
    bertopic ^
    scikit-learn ^
    pandas ^
    openpyxl ^
    chardet ^
    matplotlib ^
    wordcloud ^
    --quiet
if %errorlevel% neq 0 (
    echo HIBA: Csomag telepites sikertelen!
    pause
    exit /b 1
)
echo      Csomagok keszen.
echo.

REM ── 6. HuSpaCy magyar modell telepítése helyi fájlból ────────────────────
echo [6/6] HuSpaCy magyar modell telepitese helyi fajlbol...
"%VENV_DIR%\Scripts\pip.exe" install "%~dp0hu_core_news_lg-3.8.1-py3-none-any.whl"
if %errorlevel% neq 0 (
    echo HIBA: A helyi HuSpaCy modell telepitese sikertelen!
    echo Ellenorizd, hogy a .whl fajl pontosan az install.bat mellett van-e!
    pause
    exit /b 1
)
echo      Modell kesz.
echo.

REM ── Indítószkriptek generálása ───────────────────────────────────────────
echo Inditoszkriptek letrehozasa...

(
echo @echo off
echo cd /d "%%~dp0"
echo "%VENV_DIR%\Scripts\python.exe" gui_app.py
echo pause
) > "%~dp0run_gui.bat"

(
echo @echo off
echo cd /d "%%~dp0"
echo "%VENV_DIR%\Scripts\python.exe" main.py
echo pause
) > "%~dp0run_cli.bat"

echo.
echo ============================================================
echo   Telepites BEFEJEZVE!
echo ============================================================
echo.
echo   Grafikus felulet:  run_gui.bat  (dupla klikk)
echo   Parancssor:        run_cli.bat  (dupla klikk)
echo.
echo   A Python kornyezet helye: %VENV_DIR%
echo.
pause
