"""
utils.py
========
Segédfüggvények: CSV beolvasás, oszlopválasztás, függőség-ellenőrzés.

Javítások (v2):
  - ellenorizd_fuggosegeket() visszaadja a hiányzók listáját (gui_app.py-hoz)
  - huspacy csomag ellenőrzése hozzáadva
  - visszaad_listaban=True paraméter a GUI dinamikus hibaüzenetéhez
"""

import sys
import pandas as pd
import chardet


def intelligens_csv_beolvasas(fajlnev: str):
    """
    Hibatűrő CSV beolvasás automatikus encoding-detektálással.

    Visszatér: pandas DataFrame, vagy None ha sikertelen.
    """
    # 1. Encoding detektálás
    detected_encoding = None
    try:
        with open(fajlnev, 'rb') as f:
            raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding']
        print(f"Eszlelt kodolas: {detected_encoding} (megbizhatosag: {result['confidence']:.0%})")
    except Exception as e:
        print(f"Kodolas detektálas sikertelen: {e}")

    # 2. Beállítások prioritási sorrendben
    beallitasok = []
    if detected_encoding:
        beallitasok.extend([
            {'sep': ';', 'encoding': detected_encoding, 'decimal': ','},
            {'sep': ',', 'encoding': detected_encoding, 'decimal': '.'},
        ])
    beallitasok.extend([
        {'sep': ';',  'encoding': 'utf-8-sig',    'decimal': ','},
        {'sep': ';',  'encoding': 'cp1250',        'decimal': ','},
        {'sep': ';',  'encoding': 'windows-1250',  'decimal': ','},
        {'sep': ';',  'encoding': 'iso-8859-2',    'decimal': ','},
        {'sep': ',',  'encoding': 'utf-8-sig',     'decimal': '.'},
        {'sep': ',',  'encoding': 'utf-8',          'decimal': '.'},
        {'sep': ',',  'encoding': 'cp1250',         'decimal': ','},
        {'sep': '\t', 'encoding': 'utf-8-sig',      'decimal': ','},
    ])

    utolso_hiba = None

    for idx, params in enumerate(beallitasok, 1):
        try:
            df = pd.read_csv(fajlnev, on_bad_lines='skip', low_memory=False, **params)

            if len(df) > 0 and len(df.columns) > 0:
                df.columns = df.columns.str.strip()
                df         = df.dropna(how='all')

                if len(df) > 0:
                    print(f"Sikeres beolvasas: {len(df)} sor, {len(df.columns)} oszlop")
                    print(f"   Beallitas: {params}")
                    return df

        except PermissionError:
            raise PermissionError(
                "A fajl zarolt – zard be (pl. Excelben) es probald ujra."
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            utolso_hiba = e
            continue
        except Exception as e:
            print(f"  Hiba ({idx}/{len(beallitasok)}): {type(e).__name__}: {str(e)[:80]}")
            utolso_hiba = e
            continue

    print("\nA fajlt nem sikerult beolvasni egyik kiserlettel sem.")
    if utolso_hiba:
        print(f"   Utolso hiba: {type(utolso_hiba).__name__}: {str(utolso_hiba)[:200]}")
    print("\nLehetseges megoldasok:")
    print("   1. Ellenorizd, hogy valóban CSV fajl-e")
    print("   2. Nyisd meg Excelben es mentsd el UTF-8 CSV formatumban")
    print("   3. Ellenorizd, hogy van-e adat a fajlban")
    return None


def optimalis_oszlop(df: pd.DataFrame) -> str:
    """
    Kiválasztja azt az oszlopot, amelynek szövegei valószínűleg
    a véleményeket tartalmazzák (leghosszabb átlagos szöveghossz,
    legalább átlagosan 3 szó).
    """
    max_len        = 0
    legjobb_oszlop = None
    oszlop_statisztikak = []

    for col in df.columns:
        try:
            sorozat = df[col].dropna().astype(str)
            szurt   = sorozat[(sorozat.str.len() >= 5) & (sorozat.str.len() <= 10000)]

            if len(szurt) == 0:
                continue

            atlag_hossz  = szurt.str.len().mean()
            median_hossz = szurt.str.len().median()
            atlag_szavak = szurt.str.count(' ').mean()

            oszlop_statisztikak.append({
                'oszlop':       col,
                'atlag_hossz':  atlag_hossz,
                'median_hossz': median_hossz,
                'atlag_szavak': atlag_szavak,
            })

            if atlag_hossz > max_len and atlag_szavak > 3:
                max_len        = atlag_hossz
                legjobb_oszlop = col

        except Exception:
            continue

    if oszlop_statisztikak:
        print("\nOszlop statisztikak:")
        for stat in sorted(oszlop_statisztikak, key=lambda x: x['atlag_hossz'], reverse=True)[:5]:
            print(
                f"   '{stat['oszlop']}': atlag={stat['atlag_hossz']:.0f} kar, "
                f"median={stat['median_hossz']:.0f} kar, szavak~{stat['atlag_szavak']:.0f}"
            )

    if legjobb_oszlop:
        print(f"Kivalasztott oszlop: '{legjobb_oszlop}' (atlag {max_len:.0f} karakter)")
        return legjobb_oszlop

    print(f"Nem talaltunk megfelelő oszlopot, elso oszlop hasznalata: '{df.columns[0]}'")
    return df.columns[0]


def ellenorizd_fuggosegeket(visszaad_listaban: bool = False):
    """
    Ellenőrzi, hogy minden szükséges Python csomag és SpaCy modell
    telepítve van-e.

    Args:
        visszaad_listaban: Ha True, (bool, list) tuple-t ad vissza,
                           ahol a lista a hiányzó csomagokat tartalmazza.
                           Ha False (alapértelmezett), csak bool-t ad vissza.

    Visszatér: True ha minden rendben, False ha valami hiányzik.
               visszaad_listaban=True esetén: (bool, list_of_missing)
    """
    szukseges_csomagok = {
        'spacy':        'spacy',
        'huspacy':      'huspacy',
        'transformers': 'transformers',
        'torch':        'torch',
        'bertopic':     'bertopic',
        'pandas':       'pandas',
        'openpyxl':     'openpyxl',
        'matplotlib':   'matplotlib',
        'wordcloud':    'wordcloud',
        'chardet':      'chardet',
        'sklearn':      'sklearn',
    }

    hianyzo = []
    for csomag_neve, import_nev in szukseges_csomagok.items():
        try:
            __import__(import_nev)
        except ImportError:
            hianyzo.append(csomag_neve)

    modell_hiba = False
    if not hianyzo:
        # SpaCy modell ellenőrzés
        try:
            import spacy
            spacy.load("hu_core_news_lg")
        except OSError:
            print("\nHuSpaCy modell (hu_core_news_lg) nincs telepitve!")
            print("Telepites: pip install huspacy && python -m spacy download hu_core_news_lg")
            hianyzo.append("hu_core_news_lg (SpaCy modell)")
            modell_hiba = True

    if hianyzo and not modell_hiba:
        print(f"\nHianyzo csomagok: {', '.join(hianyzo)}")
        print(f"Telepitsd: pip install {' '.join(h for h in hianyzo if '(' not in h)}")

    ok = len(hianyzo) == 0

    if ok:
        print("Minden fuggoseg telepitve van!")

    if visszaad_listaban:
        return ok, hianyzo
    return ok
