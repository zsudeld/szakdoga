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

echo
echo "=== Telepites befejezve! ==="
echo "Inditas GUI-val:  python gui_app.py"
echo "Inditas CLI-vel:  python main.py"
