"""
main.py
=======
Parancssori (CLI) belépési pont a hibrid sentiment elemzőhöz.
Indítás: python main.py
"""

import sys
import argparse
import traceback

from elemzo_pipeline import elemez
from utils import ellenorizd_fuggosegeket

BANNER = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   PROFESSZIONALIS SENTIMENT ELEMZO                        ║
║                                                           ║
║   HuSpaCy + HuBERT + BERTopic                            ║
║   Magyar NLP Pipeline Teljes Kapacitassal                 ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""


def _parse_args():
    """Parancssori argumentumok feldolgozása. Üres meghívásnál interaktív mód."""
    parser = argparse.ArgumentParser(
        description="Professzionális Magyar Sentiment Elemző",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file",   "-f", help="CSV fájl elérési útja")
    parser.add_argument("--column", "-c", help="Elemzendő szöveg oszlop neve (elhagyható, auto-detect)")
    parser.add_argument("--output", "-o", help="Kimeneti Excel fájl neve (elhagyható, auto-timestamp)")
    return parser.parse_args()


def _progress(pct: int, uzenet: str) -> None:
    bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
    print(f"\r   [{bar}] {pct}%  {uzenet:<40}", end="", flush=True)


def inditas():
    """Fő CLI indítófüggvény."""
    print(BANNER)

    args = _parse_args()

    print("Fuggosegek ellenorzese...\n")
    ok, hianyzo = ellenorizd_fuggosegeket(visszaad_listaban=True)
    if not ok:
        print("\nHianyzo fuggosegek:")
        for h in hianyzo:
            print(f"  - {h}")
        print("\nTelepitsd oket es probald ujra.")
        sys.exit(1)

    print("\n" + "=" * 60)

    fajl = args.file
    if not fajl:
        fajl = input("CSV fajl neve (vagy teljes utvonal): ").strip()
    if not fajl:
        print("Nem adtal meg fajlnevet!")
        return

    print(f"\nFajl: {fajl}\n")

    try:
        riport_fajl, stats = elemez(
            fajl_ut=fajl,
            oszlop_nev=args.column or None,
            kimenet=args.output or None,
            progress_cb=_progress,
        )
    except PermissionError as e:
        print(f"\nHozzaferesi hiba: {e}")
        return
    except ValueError as e:
        print(f"\nHiba: {e}")
        return

    print(f"\n\n{'=' * 60}")
    print("ELEMZES SIKERESEN BEFEJEZVE!")
    print("=" * 60)
    poz = stats["Pozitív (Hibrid)"]
    neg = stats["Negatív (Hibrid)"]
    sem = stats["Semleges (Hibrid)"]
    atl = stats["Átlagos HuSpaCy pont"]
    total = poz + neg + sem
    print(f"\n  Pozitiv:  {poz:>5} db  ({poz/total*100:.1f}%)" if total else "")
    print(f"  Negativ:  {neg:>5} db  ({neg/total*100:.1f}%)" if total else "")
    print(f"  Semleges: {sem:>5} db  ({sem/total*100:.1f}%)" if total else "")
    print(f"  Atlag pontszam: {atl}")
    print(f"\nRiport: {riport_fajl}\n")


if __name__ == "__main__":
    try:
        inditas()
    except KeyboardInterrupt:
        print("\n\nElemzes megszakitva.")
        sys.exit(0)
    except Exception as e:
        print(f"\nVaratlan hiba: {e}")
        traceback.print_exc()
        sys.exit(1)
