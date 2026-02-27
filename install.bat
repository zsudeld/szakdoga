@echo off
setlocal
echo ============================================================
echo   Magyar Sentiment Elemzo - Windows Telepito
echo   (Teljes offline modell elokeszitessel)
echo ============================================================

:: [1/7] Celkonyvtar letrehozasa
if not exist "C:\ai" mkdir "C:\ai"

:: [2/7] Virtualis kornyezet letrehozasa
if not exist "C:\ai\venv" (
    echo [2/7] Virtualis kornyezet letrehozasa...
    python -m venv "C:\ai\venv"
)

:: [3/7] pip frissitese
echo [3/7] pip frissitese...
"C:\ai\venv\Scripts\python.exe" -m pip install --upgrade pip

:: [4/7] PyTorch telepitese (CPU valtozat)
echo [4/7] PyTorch telepitese (CPU)...
"C:\ai\venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu

:: [5/7] NLP csomagok telepitese
echo [5/7] NLP csomagok telepitese...
"C:\ai\venv\Scripts\python.exe" -m pip install -r requirements.txt

:: [6/7] HuSpaCy magyar modell letoltese
echo [6/7] HuSpaCy hu_core_news_lg modell letoltese es telepitese...
curl.exe -s -L -o hu_core_news_lg-3.8.1-py3-none-any.whl https://huggingface.co/huspacy/hu_core_news_lg/resolve/main/hu_core_news_lg-any-py3-none-any.whl
"C:\ai\venv\Scripts\python.exe" -m pip install hu_core_news_lg-3.8.1-py3-none-any.whl
del hu_core_news_lg-3.8.1-py3-none-any.whl

:: [7/7] NYTK Sentiment modell letoltese a fallback hiba elkerulesere
echo [7/7] NYTK AI modell letoltese az offline mukodeshez (~450 MB)...

:: Ideiglenes letolto szkript generalasa
echo import os > dl_model.py
echo from transformers import AutoModelForSequenceClassification, AutoTokenizer >> dl_model.py
echo m = "NYTK/sentiment-ohb3-hubert-hungarian" >> dl_model.py
echo p = os.path.join(os.getcwd(), "models", "sentiment-alap") >> dl_model.py
echo os.makedirs(p, exist_ok=True) >> dl_model.py
echo print("   - Tokenizer letoltese...") >> dl_model.py
echo AutoTokenizer.from_pretrained(m).save_pretrained(p) >> dl_model.py
echo print("   - Sulyok letoltese (model.safetensors)...") >> dl_model.py
echo AutoModelForSequenceClassification.from_pretrained(m).save_pretrained(p) >> dl_model.py
echo print("   - Letoltes kesz, fajlok a helyukon!") >> dl_model.py

:: Szkript futtatasa es takaritas
"C:\ai\venv\Scripts\python.exe" dl_model.py
del dl_model.py

echo.
echo === Minden telepites es letoltes sikeresen befejezve! ===
echo A szoftver mostantol 100%% offline is hiba nelkul hasznalhato.
pause