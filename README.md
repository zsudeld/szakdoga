# Magyar Hibrid Sentiment Elemző

Professzionális NLP eszköz magyar nyelvű szövegek automatikus sentiment elemzéséhez.

## Technológiák

| Komponens | Szerepe |
|-----------|---------|
| **HuSpaCy** (`hu_core_news_lg`) | Morfológiai elemzés, NER, szintaktikai fa |
| **HunBERT** (`SZTAKI-HLT/hubert-base-cc`) | Transzformer-alapú sentiment |
| **BERTopic** | Automatikus témafeltárás |
| **openpyxl** | Professzionális Excel riport |

## Fájlstruktúra

```
sentiment_elemzo/
├── main.py              # CLI belépési pont
├── gui_app.py           # Tkinter grafikus felület
├── huspacy_elemzo.py    # Fő NLP elemző modul
├── riport_generator.py  # Excel riport generátor
├── utils.py             # CSV beolvasás, segédfüggvények
├── requirements.txt     # Python függőségek (referencia)
├── install.bat          # Windows telepítő (rendszergazda NEM kell)
├── install.sh           # Linux/macOS telepítő
├── run_gui.bat          # GUI indítása (install után)
└── run_cli.bat          # CLI indítása (install után)
```

## Telepítés – Windows (rendszergazdai jog NEM szükséges)

1. Töltsd le és csomagold ki a projektet
2. Futtasd az `install.bat` fájlt (dupla klikk)
3. A telepítő létrehozza a `C:\ai\venv` virtuális környezetet
   – ez egy **rövid útvonal**, elkerüli a Windows 260 karakteres korlátját
4. Telepítés után kész az indítás:
   - `run_gui.bat` → grafikus felület
   - `run_cli.bat` → parancssori mód

> **Megjegyzés:** A HunBERT modell (~400 MB) az első elemzéskor töltődik le automatikusan.

### Ha nem a C:\ meghajtóra szeretnéd telepíteni

Nyisd meg az `install.bat` fájlt szövegszerkesztővel, és módosítsd ezt a sort:
```bat
set AI_DIR=C:\ai
```
például:
```bat
set AI_DIR=D:\ai
```

## Telepítés – Linux / macOS

```bash
bash install.sh
```

## Rendszerkövetelmények

- Python 3.10+
- RAM: minimum 8 GB
- Lemez: ~2 GB szabad hely (modellek + venv)
- Internet: első futtatáshoz (modellek letöltése)

## CSV formátum

A program automatikusan felismeri:
- **Elválasztó:** `;` `,` vagy TAB
- **Kódolás:** UTF-8, UTF-8 BOM, CP1250, ISO-8859-2
- **Szöveg oszlop:** automatikus (leghosszabb szövegeket tartalmazó oszlop)

## Kimeneti Excel lapok

| Lap | Tartalom |
|-----|---------|
| **Elemzési Eredmények** | Minden szöveg elemzése, színkódolt sentiment |
| **Statisztikai Összefoglaló** | Pie chart, bar chart, átlagok |
| **Témaelemzés** | BERTopic témakörök |
| **Nyelvészeti Részletek** | POS tagek, dependency kapcsolatok |
| **Entitások** | Named Entity Recognition eredmények |

## Hibaelhárítás

| Hiba | Megoldás |
|------|---------|
| `OSError: [Errno 2]` Long Path hiba | Használd az `install.bat`-ot – rövid útvonalra telepít |
| `OSError: hu_core_news_lg` | Futtasd újra az `install.bat`-ot |
| HunBERT timeout | Ellenőrizd az internetkapcsolatot |
| PermissionError CSV | Zárd be a fájlt Excelben, majd próbáld újra |
