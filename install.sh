#!/usr/bin/env bash
# ============================================================
# install.sh – Linux/macOS telepítő szkript
# Futtasd: bash install.sh
# ============================================================

set -e

echo "=== Magyar Sentiment Elemzo telepitese ==="
echo

# 1. Python csomagok
pip install -r requirements.txt

# 2. HuSpaCy modell
echo
echo "HuSpaCy hu_core_news_lg modell letoltese..."
pip install huspacy
python -m spacy download hu_core_news_lg

# 3. NYTK Sentiment modell
echo
echo "NYTK Sentiment modell letoltese (~450 MB)..."
python - <<'EOF'
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
m = "NYTK/sentiment-ohb3-hubert-hungarian"
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sentiment-alap")
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

# 4. BERTopic embedding modell
echo
echo "BERTopic embedding modell letoltese (~120 MB)..."
python - <<'EOF'
import os
from sentence_transformers import SentenceTransformer
m = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bertopic-embedding")
if os.path.exists(p):
    print("   - Mar letoltve, kihagyva.")
else:
    os.makedirs(p, exist_ok=True)
    SentenceTransformer(m).save(p)
    print("   - Kesz!")
EOF

echo
echo "=== Telepites befejezve! ==="
echo "Inditas GUI-val:  python gui_app.py"
echo "Inditas CLI-vel:  python main.py"
