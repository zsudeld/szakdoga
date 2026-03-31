#!/usr/bin/env bash
# ============================================================
# install.sh – Linux/macOS telepítő szkript
# Futtasd: bash install.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HOME/.sentiment-venv"

echo "=== Magyar Sentiment Elemzo telepitese ==="
echo

# 1. Virtuális környezet
echo "[1/5] Virtualis kornyezet letrehozasa: $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# 2. Python csomagok
echo "[2/5] Python csomagok telepitese..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# 3. HuSpaCy modell
echo
echo "[3/5] HuSpaCy hu_core_news_lg modell letoltese..."
pip install huspacy
python -m spacy download hu_core_news_lg

# 4. NYTK Sentiment modell
echo
echo "[4/5] NYTK Sentiment modell letoltese (~450 MB)..."
python - <<EOF
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
m = "NYTK/sentiment-ohb3-hubert-hungarian"
p = os.path.join("$SCRIPT_DIR", "models", "sentiment-alap")
if os.path.exists(p):
    print("   - Mar letoltve, kihagyva.")
else:
    os.makedirs(p, exist_ok=True)
    print("   - Tokenizer letoltese...")
    AutoTokenizer.from_pretrained(m).save_pretrained(p)
    print("   - Sulyok letoltese...")
    AutoModelForSequenceClassification.from_pretrained(m).save_pretrained(p)
    print("   - Kesz!")
EOF

# 5. BERTopic embedding modell
echo
echo "[5/5] BERTopic embedding modell letoltese (~120 MB)..."
python - <<EOF
import os
from sentence_transformers import SentenceTransformer
m = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
p = os.path.join("$SCRIPT_DIR", "models", "bertopic-embedding")
if os.path.exists(p):
    print("   - Mar letoltve, kihagyva.")
else:
    os.makedirs(p, exist_ok=True)
    SentenceTransformer(m).save(p)
    print("   - Kesz!")
EOF

echo
echo "=== Telepites befejezve! ==="
echo "Aktivald a kornyezetet: source $VENV_DIR/bin/activate"
echo "Inditas GUI-val:  python gui_app.py"
echo "Inditas CLI-vel:  python main.py"
