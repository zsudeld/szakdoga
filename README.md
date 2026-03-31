# Magyar Hibrid Sentiment Elemző – v1.0

Professzionális NLP eszköz magyar nyelvű szövegek automatikus sentiment elemzéséhez.  
A program hibrid architektúrát alkalmaz: szótári módszer, fine-tuned transzformer modell és témafeltárás kombinációjával dolgozik, minden feldolgozás **lokálisan, internet nélkül** fut (GDPR-kompatibilis).

---

## Technológiai stack

| Komponens | Verzió | Szerepe |
|-----------|--------|---------|
| **HuSpaCy** (`hu_core_news_lg`) | ≥ 3.7 | Morfológiai elemzés, NER, szintaktikai fa |
| **NYTK HuBERT** (`sentiment-ohb3-hubert-hungarian`) | – | Fine-tuned magyar sentiment transzformer |
| **BERTopic** | ≥ 0.16 | Automatikus korpusz-szintű témafeltárás |
| **scikit-learn** | ≥ 1.3 | Per-dokumentum TF-IDF kulcsszó-kinyerés |
| **sentence-transformers** | ≥ 2.2 | BERTopic embedding modell |
| **pandas** | ≥ 2.0 | Adatkezelés és CSV beolvasás |
| **openpyxl** | ≥ 3.1 | Professzionális Excel riport generálás |
| **chardet** | ≥ 5.0 | Automatikus CSV kódolás-detektálás |

---

## Fájlstruktúra

```
sentiment_elemzo/
├── main.py               # CLI belépési pont
├── gui_app.py            # Tkinter grafikus felület
├── elemzo_pipeline.py    # Közös elemzési pipeline (CLI + GUI)
├── huspacy_elemzo.py     # Hibrid NLP elemző modul
├── riport_generator.py   # Excel riport generátor
├── utils.py              # CSV beolvasás, segédfüggvények, függőség-ellenőrzés
├── requirements.txt      # Python függőségek
├── install.bat           # Windows telepítő (rendszergazda NEM kell)
├── install.sh            # Linux/macOS telepítő
├── run_gui.bat           # GUI indítása Windows alatt
├── run_cli.bat           # CLI indítása Windows alatt
├── models/               # Letöltött modellek helyi cache-e (offline mód)
│   ├── sentiment-alap/
│   └── bertopic-embedding/
└── tests/
    └── test_elemzo.py    # Egységtesztek (pytest)
```

---

## Telepítés

### Windows (rendszergazdai jog nem szükséges)

1. Csomagold ki a projektet tetszőleges mappába.
2. Futtasd az `install.bat` fájlt (dupla klikk).
3. A telepítő létrehozza a `C:\nlp\venv` virtuális környezetet.  
   A rövid útvonal szándékos – elkerüli a Windows 260 karakteres path-korlátját.
4. Az első futtatáskor a program letölti a modelleket (~570 MB), és a `models/` mappába menti őket. Utána internet nélkül is fut.

**A programot az alábbi launcher fájlokkal indíthatod (telepítés után):**

| Fájl | Indítás módja |
|------|---------------|
| `run_gui.bat` | Grafikus felület |
| `run_cli.bat` | Parancssori mód |

> **Ha nem a `C:\` meghajtóra szeretnéd telepíteni:** nyisd meg az `install.bat` fájlt szövegszerkesztővel, és módosítsd a `set APP_DIR=C:\nlp` sort a kívánt elérési útra. A launcher fájlok automatikusan az új útvonallal generálódnak.

### Linux / macOS

```bash
bash install.sh
```

A telepítő `~/.sentiment-venv` alatt hoz létre virtuális környezetet.

---

## Rendszerkövetelmények

| | Minimum |
|-|---------|
| **Python** | 3.10+ |
| **RAM** | 8 GB |
| **Lemezterület** | ~2 GB (modellek + venv) |
| **Internet** | Csak az első futtatáshoz (modell letöltés) |

---

## Használat

### Grafikus felület (GUI)

Indítás: `run_gui.bat` (Windows), illetve `python gui_app.py`

1. Kattints a **CSV Fájl Kiválasztása** gombra.
2. Kattints az **Elemzés Indítása** gombra.
3. Az elemzés végeztével a program megmutatja az összefoglalót, és elmenti az Excel riportot.

### Parancssori mód (CLI)

```bash
python main.py                                   # interaktív mód
python main.py -f adatok.csv                     # fájl megadásával
python main.py -f adatok.csv -c velemeny         # oszlop megadásával
python main.py -f adatok.csv -o riport.xlsx      # kimeneti fájlnév megadásával
```

---

## CSV bemeneti formátum

A program automatikusan felismeri a leggyakoribb CSV változatokat:

| Jellemző | Támogatott értékek |
|----------|--------------------|
| **Elválasztó** | `;` &nbsp; `,` &nbsp; TAB |
| **Kódolás** | UTF-8, UTF-8 BOM, CP1250, ISO-8859-2, Windows-1250 |
| **Szöveg oszlop** | Automatikus (leghosszabb átlagos szöveghossz alapján), vagy manuálisan megadható |

---

## Kimeneti Excel riport

Az elemzés eredménye egy timestampes `.xlsx` fájl, amely az alábbi lapokat tartalmazza:

| Lap | Tartalom |
|-----|----------|
| **Elemzési Eredmények** | Minden szöveg teljes elemzése – hibrid kategória, pontszámok, kulcsszavak, NER, szintaxis |
| **Statisztikai Összefoglaló** | Kategória-eloszlás, pie chart, bar chart, átlagos pontszámok |
| **Témaelemzés** | BERTopic korpusz-szintű témakörök előfordulással és százalékos aránnyal |
| **Nyelvészeti Részletek** | POS-statisztikák, dependency kapcsolatok, pozitív/negatív elemek |
| **Entitások** | Named Entity Recognition (NER) összesítés típus szerint |

---

## Hibrid elemzési logika

Az elemzés két forrást kombinál soronként:

1. **HuSpaCy szótári módszer** – morfológiai elemzésen alapuló pontszám (`–1.0 … +1.0`), scope-alapú negáció- és fokozókezeléssel.
2. **NYTK HuBERT transzformer** – fine-tuned magyar sentiment modell konfidencia-értékkel.

A végső kategóriát a két forrás prioritásos összevetése adja: ha az eredmények egyeznek, az a döntés; ha nem, a magasabb konfidenciájú forrás érvényesül.

---

## Tesztek

```bash
pytest tests/
```

Az egységtesztek lefedik a hibrid döntési logikát, az NYTK modell eredmény-feldolgozását, a szótári pontszámítást, a kulcsszó-kinyerést és a CSV beolvasást.

---

## Hibaelhárítás

| Hiba | Megoldás |
|------|----------|
| `OSError: [Errno 2]` / Long Path hiba | Használd az `install.bat`-ot – rövid útvonalra (`C:\nlp`) telepít |
| `OSError: hu_core_news_lg not found` | Futtasd újra az `install.bat`-ot |
| Modell letöltési hiba | Ellenőrizd az internetkapcsolatot az első futtatáskor |
| `PermissionError` CSV megnyitáskor | Zárd be a fájlt Excelben, majd próbáld újra |
| `ModuleNotFoundError` | Győződj meg róla, hogy a virtuális környezet aktív |

---

## Changelog

### v1.0
- Hibrid sentiment pipeline: HuSpaCy szótári elemzés + NYTK HuBERT transzformer
- BERTopic témafeltárás és per-dokumentum TF-IDF kulcsszó-kinyerés
- Tkinter grafikus felület progressbar-ral és szálbiztos frissítéssel
- Parancssori mód (CLI) argumentumokkal és interaktív módban is
- Ötlapos, formázott Excel riport chartokkal, feltételes formázással és NER összesítéssel
- Automatikus CSV kódolás-detektálás (chardet + fallback lista)
- Scope-alapú negáció- és fokozókezelés a szótári elemzőben
- Offline modell cache (`models/` mappa, GDPR-kompatibilis)
- Windows és Linux/macOS telepítő szkriptek
- Egységtesztek (pytest)
