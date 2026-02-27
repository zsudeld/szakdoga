"""
main.py
=======
Parancssori (CLI) belépési pont a hibrid sentiment elemzőhöz.
Indítás: python main.py

Javítások (v2):
  - Riport fájlneve a generalj_riportot visszatérési értékéből jön (timestamp)
  - Korábbi riportok nem íródnak felül
"""

import sys

import pandas as pd

from huspacy_elemzo import FejlettSentimentElemzo, generalj_temakat
from riport_generator import generalj_riportot
from utils import ellenorizd_fuggosegeket, intelligens_csv_beolvasas, optimalis_oszlop

BANNER = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   PROFESSZIONALIS SENTIMENT ELEMZO  v2                    ║
║                                                           ║
║   HuSpaCy + HunBERT + BERTopic                           ║
║   Magyar NLP Pipeline Teljes Kapacitassal                 ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""


def inditas():
    """Fő CLI indítófüggvény."""
    print(BANNER)

    # 1. Függőségek
    print("Fuggosegek ellenorzese...\n")
    ok, hianyzo = ellenorizd_fuggosegeket(visszaad_listaban=True)
    if not ok:
        print("\nHianyzo fuggosegek:")
        for h in hianyzo:
            print(f"  - {h}")
        print("\nTelepitsd oket es probald ujra.")
        sys.exit(1)

    print("\n" + "=" * 60)

    # 2. Fájl bekérése
    fajl = input("CSV fajl neve (vagy teljes utvonal): ").strip()
    if not fajl:
        print("Nem adtal meg fajlnevet!")
        return

    print(f"\nFajl beolvasasa: {fajl}\n")

    # 3. CSV beolvasás
    try:
        df_raw = intelligens_csv_beolvasas(fajl)
    except PermissionError as e:
        print(f"Hozzaferesi hiba: {e}")
        return

    if df_raw is None:
        print("A fajlt nem sikerult beolvasni.")
        return

    print(f"Sikeres beolvasas: {len(df_raw)} sor, {len(df_raw.columns)} oszlop\n")

    # 4. Oszlop kiválasztás
    oszlop = optimalis_oszlop(df_raw)

    # 5. NLP motor
    print("\n" + "=" * 60)
    print("NLP MOTOROK BETOLTESE")
    print("=" * 60 + "\n")
    motor = FejlettSentimentElemzo()

    # 6. Szövegek
    texts = [str(t) for t in df_raw[oszlop].dropna().tolist()]
    print(f"Elemenezendo szovegek: {len(texts)} db\n")

    # 7. Hibrid elemzés (batch)
    print("HuSpaCy + HunBERT hibrid elemzes folyamatban...")
    print("(Ez eltarthat par percig a szovegek szamatol fuggoen)\n")

    eredmenyek = []
    for i, res in enumerate(motor.elemzes_batch(texts), 1):
        eredmenyek.append(res)
        if i % 10 == 0 or i == len(texts):
            progress = int(i / len(texts) * 50)
            bar      = "█" * progress + "░" * (50 - progress)
            print(f"\r   [{bar}] {i}/{len(texts)} ({i / len(texts) * 100:.1f}%)", end="", flush=True)

    print("\n\nSentiment elemzes befejezve!")
    df_final = pd.DataFrame(eredmenyek)

    # 8. BERTopic
    print("\n" + "=" * 60)
    print("TEMAMODELLEZES (BERTopic)")
    print("=" * 60 + "\n")

    lemmatizalt = df_final.get('lemmatizalt_szoveg', pd.Series(texts)).tolist()
    temak       = generalj_temakat(lemmatizalt)

    if len(temak) == len(df_final):
        df_final['tema_kulcsszavak'] = temak
        print("Temak sikeresen generalva!\n")
    else:
        df_final['tema_kulcsszavak'] = "Tema hiba"
        print("Temak reszben generalhatok\n")

    # 9. Statisztikák
    print("=" * 60)
    print("STATISZTIKAK")
    print("=" * 60 + "\n")

    stats = {
        'Pozitív (Hibrid)':     int((df_final['hibrid_kategoria'] == 'pozitív').sum()),
        'Negatív (Hibrid)':     int((df_final['hibrid_kategoria'] == 'negatív').sum()),
        'Semleges (Hibrid)':    int((df_final['hibrid_kategoria'] == 'semleges').sum()),
        'Átlagos HuSpaCy pont': round(float(df_final['pontszam'].mean()), 2),
    }

    total = len(df_final)
    for kat in ['Pozitív (Hibrid)', 'Negatív (Hibrid)', 'Semleges (Hibrid)']:
        db = stats[kat]
        print(f"{kat}: {db} ({db / total * 100:.1f}%)")
    print(f"Atlag pontszam: {stats['Átlagos HuSpaCy pont']}")

    # 10. Excel riport (timestampes fájlnév)
    print("\n" + "=" * 60)
    print("EXCEL RIPORT GENERALASA")
    print("=" * 60 + "\n")
    riport_fajl = generalj_riportot(df_final, stats)

    print("\n" + "=" * 60)
    print("ELEMZES SIKERESEN BEFEJEZVE!")
    print("=" * 60 + "\n")
    print(f"Riport fajl: {riport_fajl}")
    print("Lapok: Elemzesi Eredmenyek | Statisztikai Osszefoglalo | Temaelemzes | Nyelveszeti Reszletek | Entitasok\n")


if __name__ == "__main__":
    try:
        inditas()
    except KeyboardInterrupt:
        print("\n\nElemzes megszakitva.")
        sys.exit(0)
    except Exception as e:
        print(f"\nVaratlan hiba: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
