"""
elemzo_pipeline.py
==================
Közös elemzési pipeline – CLI és GUI egyaránt ezt használja.
"""

from collections.abc import Callable

import pandas as pd

from huspacy_elemzo import FejlettSentimentElemzo, generalj_bertopic, generalj_temakat
from riport_generator import generalj_riportot
from utils import intelligens_csv_beolvasas, optimalis_oszlop


def elemez(
    fajl_ut: str,
    oszlop_nev: str | None = None,
    kimenet: str | None = None,
    progress_cb: Callable[[int, str], None] | None = None,
) -> tuple[str, dict]:
    """
    Teljes sentiment elemzési pipeline egy CSV fájlon.

    Paraméterek:
        fajl_ut:     CSV fájl elérési útja.
        oszlop_nev:  Elemzendő oszlop neve; None esetén auto-detect.
        kimenet:     Kimeneti Excel fájlnév; None esetén timestampes auto-név.
        progress_cb: Opcionális callback(percent: int, uzenet: str) –
                     a GUI arra használja, hogy frissítse a progress bart.

    Visszatér: (riport_fajlnev, stats) tuple.
    Kivételt dob, ha a CSV nem olvasható vagy üres.
    """

    def _cb(pct: int, uzenet: str) -> None:
        if progress_cb:
            progress_cb(pct, uzenet)

    # 1. CSV beolvasás
    _cb(0, "CSV beolvasása...")
    df_raw = intelligens_csv_beolvasas(fajl_ut)
    if df_raw is None or df_raw.empty:
        raise ValueError("A CSV üres vagy nem olvasható.")

    # 2. Oszlop kiválasztás
    if oszlop_nev and oszlop_nev in df_raw.columns:
        oszlop = oszlop_nev
    else:
        if oszlop_nev:
            print(f"Figyelem: '{oszlop_nev}' oszlop nem található, auto-detect fut.")
        oszlop = optimalis_oszlop(df_raw)

    valid_sorok = df_raw[oszlop].dropna()
    eredeti_index = valid_sorok.index.tolist()
    texts = [str(t) for t in valid_sorok.tolist()]
    if not texts:
        raise ValueError(f"Nincs feldolgozható szöveg a(z) '{oszlop}' oszlopban.")

    _cb(10, f"{len(texts)} szöveg feldolgozásra vár...")

    # 3. NLP motor betöltése
    _cb(12, "HuSpaCy + HuBERT betöltése...")
    motor = FejlettSentimentElemzo()

    # 4. Sentiment elemzés (batch)
    _cb(25, "Hibrid sentiment elemzés...")
    eredmenyek = []
    for i, res in enumerate(motor.elemzes_batch(texts), 1):
        eredmenyek.append(res)
        pct = 25 + int(i / len(texts) * 40)
        _cb(pct, f"Sentiment: {i}/{len(texts)}")

    df_final = pd.DataFrame(eredmenyek, index=eredeti_index)
    df_final.index.name = "csv_sor"

    # 5a. Per-dokumentum TF-IDF kulcsszavak
    _cb(66, "Per-dokumentum kulcsszó-kinyerés (TF-IDF)...")
    lemmatizalt = df_final.get("lemmatizalt_szoveg", pd.Series(texts)).tolist()
    per_doc_temak = generalj_temakat(lemmatizalt)
    df_final["tema_kulcsszavak"] = (
        per_doc_temak
        if len(per_doc_temak) == len(df_final)
        else ["Téma hiba"] * len(df_final)
    )

    # 5b. BERTopic korpusz-szintű klaszterezés
    _cb(75, "BERTopic klaszterezés...")
    bertopic_tema_lista = generalj_bertopic(lemmatizalt)["tema_lista"]
    if bertopic_tema_lista and len(bertopic_tema_lista) == len(df_final):
        df_final["bertopic_tema"] = bertopic_tema_lista
    else:
        df_final["bertopic_tema"] = df_final["tema_kulcsszavak"]

    # 6. Statisztikák
    stats = {
        "Pozitív (Hibrid)":     int((df_final["hibrid_kategoria"] == "pozitív").sum()),
        "Negatív (Hibrid)":     int((df_final["hibrid_kategoria"] == "negatív").sum()),
        "Semleges (Hibrid)":    int((df_final["hibrid_kategoria"] == "semleges").sum()),
        "Átlagos HuSpaCy pont": round(float(df_final["pontszam"].mean()), 2),
    }

    # 7. Excel riport
    _cb(90, "Excel riport generálása...")
    riport_fajl = generalj_riportot(df_final, stats, kimenet=kimenet)

    _cb(100, "Elemzés befejezve!")
    return riport_fajl, stats
