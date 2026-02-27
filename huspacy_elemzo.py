"""
huspacy_elemzo.py
=================
HuSpaCy + NYTK Magyar Sentiment + BERTopic hibrid elemző.

Technológiák:
  - HuSpaCy (hu_core_news_lg)                         -> morfológia, NER, szintaxis
  - NYTK/sentiment-hts5-hubert-hungarian               -> fine-tuned magyar sentiment
  - NYTK/sentiment-hts5-xlm-roberta-hungarian          -> tartalék (pontosabb, nagyobb)
  - BERTopic                                            -> témafeltárás

GDPR: minden feldolgozás lokálisan történik.
      Az offline_mod=True beállítással internet-hozzáférés sem szükséges
      (a modellek egyszer letöltődnek a models/ mappába).

Javítások (v2):
  - top_k=None pipeline: 3-osztályos ÉS 5-osztályos NYTK modellt kezel helyesen
  - Sqrt-alapú normalizáció: hosszú szövegek sem torzítanak semleges felé
  - Kibővített sentiment szótár
  - Szimmetrikusabb negáció-kezelés
  - Csökkentett hibrid küszöbök: drámaian kevesebb téves semleges besorolás
  - nlp.pipe() + NYTK batch: gyorsabb feldolgozás
  - Egységes hibakezelés
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Generator, List

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Konfiguráció
# ---------------------------------------------------------------------------

SENTIMENT_MODELL_ALAP = "NYTK/sentiment-hts5-hubert-hungarian"
SENTIMENT_MODELL_NAGY = "NYTK/sentiment-hts5-xlm-roberta-hungarian"
MODELL_MAPPA = Path(__file__).parent / "models"

# Hibrid döntési küszöbök
_NYTK_MAGAS_KUSZOOB   = 0.55   # Felett: NYTK dönt (volt: 0.70)
_NYTK_MERESEKELT_KUSZOOB = 0.50  # Felett: NYTK gyenge jel is számít
_HUSPACY_EROS_KUSZOOB  = 0.25   # Felett: HuSpaCy dönt (volt: 0.45)
_HUSPACY_GYENGE_KUSZOOB = 0.10   # Felett: HuSpaCy gyenge jel is számít
_HUSPACY_KAT_KUSZOOB   = 0.10   # Kategória-határvonal (volt: 0.12)


# ---------------------------------------------------------------------------
# Bővített magyar sentiment szótár
# (OpinHuBank + HuSent korpusz + oktatási értékelések)
# ---------------------------------------------------------------------------

POZITIV_SZAVAK: set = {
    # Általános pozitív
    "jó", "kiváló", "remek", "fantasztikus", "nagyszerű", "kitűnő",
    "csodálatos", "tökéletes", "szuper", "briliáns", "örömteli",
    "hasznos", "segítőkész", "szakszerű", "igényes", "precíz",
    "innovatív", "lelkes", "ügyes", "tehetséges", "inspiráló",
    "motiváló", "érdekes", "érthető", "logikus", "átlátható",
    "konstruktív", "fejlesztő", "hatékony", "sikeres", "professzionális",
    # Értékelési kifejezések
    "ajánlom", "ajánlanám", "elégedett", "elégedettség", "tetszik",
    "szeretem", "imádom", "szép", "kellemes", "barátságos", "örülök",
    "boldog", "izgalmas", "lenyűgöző", "meglepő", "pozitív", "optimista",
    # Oktatási kontextus
    "szemléletes", "közérthető", "gyakorlatias", "naprakész", "alapos",
    "felkészült", "lelkiismeretes", "türelmes", "empatikus", "dinamikus",
    "strukturált", "interaktív", "élvezetes", "tanulságos",
    # Fokozók önállóan is pozitív értékkel
    "kimagasló", "kiemelkedő", "elsőrangú", "méltányolandó",
    # Kibővítés
    "pontos", "részletes", "átfogó", "érdekes", "informatív",
    "megbízható", "stabil", "tiszta", "egyértelmű", "gördülékeny",
    "kényelmes", "szervezett", "következetes", "rugalmas", "támogató",
    "ösztönző", "figyelmes", "gondos", "kreatív", "eredeti",
    "friss", "modern", "naprakész", "releváns", "megfelelő",
    "gyors", "hatásos", "eredményes", "teljes", "komplex",
    "különleges", "kiemelten", "magas", "erős", "szilárd",
    "élő", "pezsgő", "aktív", "bevonó", "öszintén", "igaz",
}

NEGATIV_SZAVAK: set = {
    # Általános negatív
    "rossz", "gyenge", "silány", "szörnyű", "borzasztó", "rettenetes",
    "katasztrofális", "értelmetlen", "összefüggéstelen",
    "érthetetlen", "zavaros", "kaotikus", "rendezetlen", "elavult",
    "irreleváns", "felesleges", "hasznontalan", "bosszantó", "frusztráló",
    "idegesítő", "lassú", "hiányos", "hibás", "problémás",
    # Értékelési kifejezések
    "csalódottság", "csalódott", "utálom", "szegényes", "kifogásolható",
    "pontatlan", "felületes", "sablonos", "unalmas", "demotiváló",
    # Oktatási kontextus
    "száraz", "monoton", "élettelen", "elmaradott", "korszerűtlen",
    "nehézkes", "áttekinthetetlen", "következetlen", "kapkodó",
    "felkészületlen", "türelmetlen", "lekezelő",
    # Fokozók önállóan is negatív értékkel
    "elfogadhatatlan", "tarthatatlan", "siralmas",
    # Kibővítés
    "rossz", "nehéz", "bonyolult", "zavaró", "kellemetlen",
    "homályos", "kusza", "megbízhatatlan", "hiányos", "töredékes",
    "felületes", "sekélyes", "sivár", "érdektelen", "ósdi",
    "lassú", "akadozó", "bugos", "törött", "nem_működik",
    "értéktelen", "haszontalan", "kárba", "veszteség",
    "nehézkes", "körülményes", "macerás", "fárasztó", "kimerítő",
    "unszolós", "tolakodó", "zavaró", "elviselhetetlen", "terhes",
    "csalódás", "cserbenhagyott", "becsapott", "félrevezető",
}

FOKOZOK: set = {
    "nagyon", "igen", "rendkívül", "kifejezetten", "különösen",
    "eléggé", "igazán", "teljesen", "abszolút", "végtelenül",
    "meglehetősen", "elég", "igencsak", "alaposan",
    "rendkívüli", "kivételesen", "határozottan", "erősen", "mélyen",
}

NEGACIOK: set = {
    "nem", "se", "sem", "soha", "semmit", "sehol", "senki",
    "semmi", "semmilyen", "egyáltalán", "korántsem", "távolról",
    "sehogy", "semmiképpen", "semmiféle",
}

NEGACIO_HATKORE = 5


# ---------------------------------------------------------------------------
# Lazy model betöltés
# ---------------------------------------------------------------------------

_spacy_nlp           = None
_sentiment_pipeline  = None
_modell_neve         = None
_nytk_n_classes      = None   # 3 vagy 5, futás közben derül ki


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        print("   -> hu_core_news_lg betöltése...")
        _spacy_nlp = spacy.load("hu_core_news_lg")
    return _spacy_nlp


def _get_sentiment_pipeline():
    """
    Betölti az NYTK fine-tuned magyar sentiment modellt.
    top_k=None beállítással MINDEN osztály pontszámát visszaadja,
    így 3-osztályos és 5-osztályos modell egyaránt helyesen kezelhető.
    """
    global _sentiment_pipeline, _modell_neve

    if _sentiment_pipeline is not None:
        return _sentiment_pipeline

    from transformers import pipeline as hf_pipeline

    helyi_alap = MODELL_MAPPA / "sentiment-alap"

    if helyi_alap.exists():
        print("   -> Helyi modell betöltése (offline mód)...")
        modell_utvonal = str(helyi_alap)
        _modell_neve = "NYTK HuBERT (helyi cache)"
    else:
        print(f"   -> NYTK magyar sentiment modell letöltése: {SENTIMENT_MODELL_ALAP}")
        print("      (Első indítás: ~400 MB, ezután offline is fut)")
        modell_utvonal = SENTIMENT_MODELL_ALAP
        _modell_neve = SENTIMENT_MODELL_ALAP

    try:
        _sentiment_pipeline = hf_pipeline(
            task="text-classification",
            model=modell_utvonal,
            tokenizer=modell_utvonal,
            truncation=True,
            max_length=512,
            device=-1,         # CPU – GDPR: nem küld adatot GPU cloud-ba
            top_k=None,        # MINDEN osztály pontszámát visszaadja (3 VAGY 5 osztály)
        )

        if not helyi_alap.exists() and modell_utvonal != str(helyi_alap):
            _menti_modellt(_sentiment_pipeline, helyi_alap)

        print(f"   -> Sentiment modell kész: {_modell_neve}")
        return _sentiment_pipeline

    except Exception as e:
        print(f"   Figyelem: Alap modell sikertelen ({e}), fallback módban fut")
        _sentiment_pipeline = None
        return None


def _menti_modellt(pipeline_obj, cel_mappa: Path):
    """Helyi cache-be menti a modellt a GDPR-biztos offline futáshoz."""
    try:
        cel_mappa.mkdir(parents=True, exist_ok=True)
        pipeline_obj.model.save_pretrained(str(cel_mappa))
        pipeline_obj.tokenizer.save_pretrained(str(cel_mappa))
        print(f"   -> Modell elmentve offline használathoz: {cel_mappa}")
    except Exception as e:
        print(f"   Figyelem: Modell mentés sikertelen: {e}")


# ---------------------------------------------------------------------------
# NYTK eredmény feldolgozás – 3-osztályos ÉS 5-osztályos modell
# ---------------------------------------------------------------------------

def _feldolgoz_nytk_eredmeny(raw) -> tuple:
    """
    Robust NYTK modell eredmény feldolgozás.

    top_k=None esetén a pipeline list of dict-et ad vissza:
        [{"label": "LABEL_0", "score": 0.05}, {"label": "LABEL_2", "score": 0.92}, ...]

    Kezeli:
      - Explicit string labeleket (POSITIVE, NEGATIVE, NEUTRAL, stb.)
      - 3-osztályos numerikus mappinget: LABEL_0=neg, LABEL_1=neu, LABEL_2=pos
      - 5-osztályos numerikus mappinget: LABEL_0=very_neg, ..., LABEL_4=very_pos
    """
    global _nytk_n_classes

    if raw is None:
        return "semleges", 0.5

    # Ha dict (nem top_k=None futás maradványa): csomagold listába
    if isinstance(raw, dict):
        raw = [raw]

    if not raw:
        return "semleges", 0.5

    scores_by_label = {r["label"].upper(): float(r["score"]) for r in raw}

    # --- 1. Explicit string label-ek kezelése ---
    poz = sum(v for k, v in scores_by_label.items()
              if any(p in k for p in ("POSITIV", "POSITIVE", "POS", "POZIT")))
    neg_e = sum(v for k, v in scores_by_label.items()
                if any(n in k for n in ("NEGATIV", "NEGATIVE", "NEG")))
    neu = sum(v for k, v in scores_by_label.items()
              if any(n in k for n in ("NEUTRAL", "NEU", "SEMLEGES")))

    # Ha van bármilyen explicit egyezés
    if poz + neg_e + neu > 0.05:
        best = max(poz, neg_e, neu)
        if best == poz and poz > 0:
            return "pozitív", poz
        elif best == neg_e and neg_e > 0:
            return "negatív", neg_e
        else:
            return "semleges", neu if neu > 0 else 0.5

    # --- 2. Numerikus LABEL_N mapping ---
    numeric = {}
    for r in raw:
        label = r["label"].upper().replace("LABEL_", "")
        try:
            numeric[int(label)] = float(r["score"])
        except ValueError:
            pass

    if not numeric:
        return "semleges", 0.5

    n_classes = len(numeric)

    # Osztályszám detektálás és cache
    if _nytk_n_classes is None:
        _nytk_n_classes = n_classes

    if n_classes == 3:
        # 3-osztályos: LABEL_0=negatív, LABEL_1=semleges, LABEL_2=pozitív
        neg_s  = numeric.get(0, 0.0)
        neu_s  = numeric.get(1, 0.0)
        poz_s  = numeric.get(2, 0.0)

    elif n_classes == 5:
        # 5-osztályos: 0=erősen_neg, 1=gyengén_neg, 2=semleges, 3=gyengén_poz, 4=erősen_poz
        neg_s  = numeric.get(0, 0.0) + numeric.get(1, 0.0)
        neu_s  = numeric.get(2, 0.0)
        poz_s  = numeric.get(3, 0.0) + numeric.get(4, 0.0)

    else:
        # Általános eset: alsó harmad negatív, felső harmad pozitív
        max_label = max(numeric.keys())
        low  = max_label / 3
        high = max_label * 2 / 3
        neg_s  = sum(v for k, v in numeric.items() if k <= low)
        neu_s  = sum(v for k, v in numeric.items() if low < k < high)
        poz_s  = sum(v for k, v in numeric.items() if k >= high)

    best = max(poz_s, neg_s, neu_s)
    if best == poz_s and poz_s > 0:
        return "pozitív", poz_s
    elif best == neg_s and neg_s > 0:
        return "negatív", neg_s
    else:
        return "semleges", neu_s if neu_s > 0 else 0.5


def _modell_elemez(szoveg: str, pipeline_obj) -> tuple:
    """
    NYTK modell hívás egyetlen szövegre.
    Visszatér: (kategória_str, confidence_float)
    """
    if pipeline_obj is None:
        return "semleges", 0.5
    try:
        raw = pipeline_obj(szoveg[:512])
        # top_k=None esetén raw = [[{...}, {...}, ...]]
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        return _feldolgoz_nytk_eredmeny(raw)
    except Exception:
        return "semleges", 0.5


# ---------------------------------------------------------------------------
# Negáció kezelés (scope-alapú)
# ---------------------------------------------------------------------------

def _negalt_e(token, doc) -> bool:
    """
    Megvizsgálja, hogy a token negatív hatókörben van-e.

    Három szinten keres:
    1. Közvetlen dependency gyermek (pl. "nem jó")
    2. Szülő dependency (pl. "nem volt jó")
    3. Megelőző N token (hatókör-alapú, agglutináló nyelvekhez)
    """
    if any(child.lemma_.lower() in NEGACIOK for child in token.children):
        return True
    if token.head.lemma_.lower() in NEGACIOK:
        return True
    tok_idx = token.i
    start = max(0, tok_idx - NEGACIO_HATKORE)
    for i in range(start, tok_idx):
        if doc[i].lemma_.lower() in NEGACIOK:
            if doc[i].sent == token.sent:
                return True
    return False


# ---------------------------------------------------------------------------
# Szótári pontszámítás – sqrt-alapú normalizáció
# ---------------------------------------------------------------------------

def _szotari_pontszam(doc) -> float:
    """
    Bővített szótár-alapú pontszámítás HuSpaCy Doc objektumon.
    Scope-alapú negációkezeléssel és fokozók figyelembevételével.

    Normalizáció: sqrt(token_szam) – hosszú szövegek sem torzítanak semleges felé.
    Visszatér: [-1.0 … +1.0]
    """
    pont = 0.0

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma not in POZITIV_SZAVAK and lemma not in NEGATIV_SZAVAK:
            continue

        negalt = _negalt_e(token, doc)

        fokozott = (
            any(child.lemma_.lower() in FOKOZOK for child in token.children)
            or token.head.lemma_.lower() in FOKOZOK
        )
        szorzo = 1.4 if fokozott else 1.0

        if lemma in POZITIV_SZAVAK:
            # Pozitív szó: +0.7 / negált esetén -0.5
            # ("nem jó" gyengébb negatív mint "rossz")
            pont += (-0.5 if negalt else 0.7) * szorzo
        elif lemma in NEGATIV_SZAVAK:
            # Negatív szó: -0.8 / negált esetén +0.4
            # ("nem rossz" mérsékelt pozitív, gyengébb mint "jó")
            pont += (0.4 if negalt else -0.8) * szorzo

    # sqrt-alapú normalizáció: sokkal igazságosabb hosszú szövegeknél
    token_szam = max(len([t for t in doc if not t.is_space]), 1)
    norm = max(math.sqrt(token_szam), 1.0)
    return max(-1.0, min(1.0, pont / norm * 1.8))


# ---------------------------------------------------------------------------
# Mondatszintű elemzés és aggregáció
# ---------------------------------------------------------------------------

def _mondatszintu_sentimentek(doc, pipeline_obj) -> list:
    """
    Mondatonkénti sentiment elemzés.
    Visszatér: [{mondat, huspacy_pont, modell_kat, modell_conf}]
    """
    mondatok = []
    for sent in doc.sents:
        szoveg = sent.text.strip()
        if len(szoveg) < 5:
            continue
        huspacy_pont = _szotari_pontszam(sent.as_doc())
        modell_kat, modell_conf = _modell_elemez(szoveg, pipeline_obj)
        mondatok.append({
            "mondat":       szoveg,
            "huspacy_pont": huspacy_pont,
            "modell_kat":   modell_kat,
            "modell_conf":  modell_conf,
        })
    return mondatok


def _mondatszintu_sentimentek_cached(doc, doc_idx: int, nytk_cache: dict) -> list:
    """
    Mondatonkénti sentiment – előre kiszámított NYTK eredmények használatával.
    Batch feldolgozáshoz.
    """
    mondatok = []
    for sent_idx, sent in enumerate(doc.sents):
        szoveg = sent.text.strip()
        if len(szoveg) < 5:
            continue
        huspacy_pont = _szotari_pontszam(sent.as_doc())
        modell_kat, modell_conf = nytk_cache.get((doc_idx, sent_idx), ("semleges", 0.5))
        mondatok.append({
            "mondat":       szoveg,
            "huspacy_pont": huspacy_pont,
            "modell_kat":   modell_kat,
            "modell_conf":  modell_conf,
        })
    return mondatok


def _aggregalt_sentiment(mondatok: list) -> tuple:
    """
    Mondatszintű sentimentek súlyozott aggregálása dokumentum-szintű értékké.
    Hosszabb mondatok nagyobb súlyt kapnak.
    """
    if not mondatok:
        return 0.0, "semleges", 0.5

    ossz_suly = 0.0
    suly_pont = 0.0
    kat_szamlalo: Counter = Counter()

    for m in mondatok:
        suly = len(m["mondat"].split())
        ossz_suly += suly
        suly_pont += m["huspacy_pont"] * suly
        kat_szamlalo[m["modell_kat"]] += m["modell_conf"] * suly

    atlag_pont    = suly_pont / ossz_suly if ossz_suly > 0 else 0.0
    legjobb_kat   = kat_szamlalo.most_common(1)[0][0] if kat_szamlalo else "semleges"
    legjobb_suly  = kat_szamlalo.most_common(1)[0][1] if kat_szamlalo else 0.0
    atlag_conf    = min(legjobb_suly / ossz_suly, 1.0) if ossz_suly > 0 else 0.5

    return atlag_pont, legjobb_kat, atlag_conf


# ---------------------------------------------------------------------------
# Hibrid döntés – csökkentett küszöbök, kevesebb téves semleges
# ---------------------------------------------------------------------------

def _hibrid_kategoria(
    huspacy_pont: float,
    modell_kat: str,
    modell_conf: float,
) -> tuple:
    """
    Kombinálja a szótári és transzformer eredményt.

    Döntési sorrend (legegyértelműbbtől a legbizonytalanabbig):
      1. Mindkét modell egyezik → egyértelműen az a kategória
      2. NYTK magas konfidencia (≥ 0.55) → NYTK dönt
      3. HuSpaCy erős szótári jel (|pont| ≥ 0.25) → HuSpaCy dönt
      4. NYTK mérsékelt jel (> 0.50, nem semleges) → NYTK dönt
      5. HuSpaCy gyenge jel (|pont| > 0.10) → HuSpaCy dönt
      6. Valóban semleges (mindkét modell gyenge/ellentmondó)
    """
    # HuSpaCy kategória meghatározása
    if huspacy_pont > _HUSPACY_KAT_KUSZOOB:
        huspacy_kat = "pozitív"
    elif huspacy_pont < -_HUSPACY_KAT_KUSZOOB:
        huspacy_kat = "negatív"
    else:
        huspacy_kat = "semleges"

    # 1. Egyezés
    if huspacy_kat == modell_kat:
        return huspacy_kat, (
            f"Egyezés: {huspacy_kat} "
            f"(HuSpaCy: {huspacy_pont:.3f}, NYTK conf: {modell_conf:.2f})"
        )

    # 2. NYTK magas konfidencia
    if modell_conf >= _NYTK_MAGAS_KUSZOOB and modell_kat != "semleges":
        return modell_kat, (
            f"NYTK magabiztos ({modell_conf:.2f}) döntött. "
            f"HuSpaCy: {huspacy_pont:.3f}"
        )

    # 3. HuSpaCy erős szótári jel
    if abs(huspacy_pont) >= _HUSPACY_EROS_KUSZOOB and huspacy_kat != "semleges":
        return huspacy_kat, (
            f"Erős szótári jel ({huspacy_pont:.3f}). "
            f"NYTK: {modell_kat} ({modell_conf:.2f})"
        )

    # 4. NYTK mérsékelt jel (conf > 0.50, nem semleges)
    if modell_conf > _NYTK_MERESEKELT_KUSZOOB and modell_kat != "semleges":
        return modell_kat, (
            f"NYTK mérsékelt jel ({modell_conf:.2f}). "
            f"HuSpaCy: {huspacy_pont:.3f}"
        )

    # 5. HuSpaCy gyenge de valós jel
    if abs(huspacy_pont) > _HUSPACY_GYENGE_KUSZOOB and huspacy_kat != "semleges":
        return huspacy_kat, (
            f"Mérsékelt szótári jel ({huspacy_pont:.3f}). "
            f"NYTK: {modell_kat} ({modell_conf:.2f})"
        )

    # 6. Valóban semleges
    return "semleges", (
        f"Gyenge/ellentmondó jelek → semleges "
        f"(HuSpaCy: {huspacy_pont:.3f}, NYTK: {modell_kat} {modell_conf:.2f})"
    )


# ---------------------------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------------------------

def _tisztit(szoveg: str) -> str:
    szoveg = re.sub(r"<[^>]+>", " ", szoveg)
    szoveg = re.sub(r"https?://\S+", " ", szoveg)
    szoveg = re.sub(r"\s+", " ", szoveg)
    return szoveg.strip()


def _pos_statisztika(doc) -> str:
    counter = Counter(t.pos_ for t in doc if not t.is_space)
    return ", ".join(f"{p}:{n}" for p, n in counter.most_common(5))


def _dep_statisztika(doc) -> str:
    counter = Counter(t.dep_ for t in doc if not t.is_space)
    return ", ".join(f"{d}:{n}" for d, n in counter.most_common(5))


def _ner_entitasok(doc) -> str:
    return ", ".join(f"{e.text} ({e.label_})" for e in doc.ents) if doc.ents else ""


def _nyelveszeti_kapcsolatok(doc) -> str:
    kapcsolatok = [
        f"{t.head.text}->{t.dep_}->{t.text}"
        for t in doc
        if t.dep_ in ("nsubj", "dobj", "nmod", "amod") and not t.is_stop
    ]
    return " | ".join(kapcsolatok[:8])


def _pozitiv_elemek(doc) -> str:
    return ", ".join(
        t.text for t in doc if t.lemma_.lower() in POZITIV_SZAVAK and not t.is_stop
    )


def _negativ_elemek(doc) -> str:
    return ", ".join(
        t.text for t in doc if t.lemma_.lower() in NEGATIV_SZAVAK and not t.is_stop
    )


def _lemmatizal(doc) -> str:
    return " ".join(
        t.lemma_.lower()
        for t in doc
        if not t.is_stop and not t.is_punct and not t.is_space and len(t.lemma_) > 2
    )


def _mondatszintu_osszefoglalo(mondatok: list) -> str:
    if not mondatok:
        return ""
    reszek = []
    for i, m in enumerate(mondatok, 1):
        kat  = m["modell_kat"]
        pont = m["huspacy_pont"]
        reszek.append(f"M{i}:{kat}({pont:+.2f})")
    return " | ".join(reszek)


def _mondatszintu_osszefoglalo_doc(doc) -> str:
    """
    Tájékoztató mondatszintű bontás kizárólag az adott sor doc-jából.
    Csak HuSpaCy szótári pontot használ – szomszéd sorok adatai
    semmilyen körülmények között nem kerülnek ide.
    """
    reszek = []
    for i, sent in enumerate(doc.sents, 1):
        szoveg = sent.text.strip()
        if len(szoveg) < 5:
            continue
        pont = _szotari_pontszam(sent.as_doc())
        if pont > _HUSPACY_KAT_KUSZOOB:
            kat = "poz"
        elif pont < -_HUSPACY_KAT_KUSZOOB:
            kat = "neg"
        else:
            kat = "sem"
        reszek.append(f"M{i}:{kat}({pont:+.2f})")
    return " | ".join(reszek)


def _hiba_eredmeny(szoveg: str, hiba: str) -> Dict:
    """Hibás feldolgozás esetén visszaadott alapértelmezett eredmény."""
    return {
        "eredeti_szoveg":          str(szoveg)[:200],
        "lemmatizalt_szoveg":      "",
        "kategoria":               "hiba",
        "pontszam":                0.0,
        "hunbert_kategoria":       "hiba",
        "hunbert_confidence":      0.0,
        "hibrid_kategoria":        "semleges",
        "hibrid_megalapozas":      f"Hiba: {hiba[:100]}",
        "mondatszintu_sentiment":  "",
        "pozitiv_elemek":          "",
        "negativ_elemek":          "",
        "emlitett_nevek":          "",
        "mondatok_szama":          0,
        "token_szam":              0,
        "pos_statisztika":         "",
        "dep_statisztika":         "",
        "nyelveszeti_kapcsolatok": "",
    }


# ---------------------------------------------------------------------------
# Fő elemző osztály
# ---------------------------------------------------------------------------

class FejlettSentimentElemzo:
    """
    Hibrid NLP sentiment elemző: HuSpaCy + NYTK fine-tuned magyar modell.

    GDPR: minden adat lokálisan marad. Offline módhoz futtasd egyszer
          online, hogy a modellek letöltődjenek a models/ mappába.

    Példa:
        motor = FejlettSentimentElemzo()
        for eredmeny in motor.elemzes_batch(szoveg_lista):
            print(eredmeny['hibrid_kategoria'])
    """

    def __init__(self):
        print("NLP motorok betöltése...")
        self.nlp       = _get_spacy()
        self.sentiment = _get_sentiment_pipeline()
        if self.sentiment is None:
            print("   Figyelem: NYTK modell nem elérhető, csak szótár-alapú elemzés fut")
        print("Elemző kész!\n")

    def elemez_egyet(self, szoveg: str) -> Dict:
        """Egyetlen szöveg (sor) teljes NLP elemzése. Az elemzés egysége maga a sor."""
        tiszta = _tisztit(str(szoveg))
        doc    = self.nlp(tiszta)

        # HuSpaCy: teljes sor pontszáma (nem mondatonként)
        huspacy_pont = _szotari_pontszam(doc)

        # NYTK: teljes sor szövege
        modell_kat, modell_conf = _modell_elemez(tiszta, self.sentiment)

        if huspacy_pont > _HUSPACY_KAT_KUSZOOB:
            huspacy_kat = "pozitív"
        elif huspacy_pont < -_HUSPACY_KAT_KUSZOOB:
            huspacy_kat = "negatív"
        else:
            huspacy_kat = "semleges"

        hibrid_kat, megalapozas = _hibrid_kategoria(huspacy_pont, modell_kat, modell_conf)

        return {
            "eredeti_szoveg":          tiszta,
            "lemmatizalt_szoveg":      _lemmatizal(doc),
            "kategoria":               huspacy_kat,
            "pontszam":                round(huspacy_pont, 4),
            "hunbert_kategoria":       modell_kat,
            "hunbert_confidence":      round(modell_conf, 4),
            "hibrid_kategoria":        hibrid_kat,
            "hibrid_megalapozas":      megalapozas,
            "mondatszintu_sentiment":  _mondatszintu_osszefoglalo_doc(doc),
            "pozitiv_elemek":          _pozitiv_elemek(doc),
            "negativ_elemek":          _negativ_elemek(doc),
            "emlitett_nevek":          _ner_entitasok(doc),
            "mondatok_szama":          len(list(doc.sents)),
            "token_szam":              len([t for t in doc if not t.is_space]),
            "pos_statisztika":         _pos_statisztika(doc),
            "dep_statisztika":         _dep_statisztika(doc),
            "nyelveszeti_kapcsolatok": _nyelveszeti_kapcsolatok(doc),
        }

    def elemzes_batch(
        self,
        szovegek: List[str],
        batch_meret: int = 16,
    ) -> Generator[Dict, None, None]:
        """
        Batch elemzés generátorként – sor-szintű feldolgozással.

        Az elemzés egysége a CSV sor (nem a mondat).
        Egy sor akárhány mondatot tartalmazhat – az NYTK és a HuSpaCy
        is mindig a teljes sort kapja inputként, sosem szomszéd sorok
        mondatait.

        1. HuSpaCy nlp.pipe(): soronként külön Doc (nincs átnyúlás)
        2. NYTK batch: teljes sor szövegét kapja (nem mondatokat)
        3. Eredmények összeállítása – minden mező csak az adott sorból
        """
        if not szovegek:
            return

        tisztitott = [_tisztit(str(s)) for s in szovegek]

        # --- 1. HuSpaCy batch (nlp.pipe) – soronként önálló Doc ---
        try:
            docs = list(self.nlp.pipe(tisztitott, batch_size=32))
        except Exception as e:
            print(f"   Figyelem: nlp.pipe hiba, egyesével folytatom: {e}")
            docs = []
            for t in tisztitott:
                try:
                    docs.append(self.nlp(t))
                except Exception:
                    docs.append(self.nlp(""))

        # --- 2. NYTK batch – teljes sor szövege, NEM mondatok ---
        # Kulcs: doc_idx (= sor sorszáma), értéke: (kategória, konfidencia)
        nytk_sor_cache: dict = {}  # doc_idx -> (kat, conf)

        if self.sentiment is not None:
            try:
                raw_batch: list = []
                for i in range(0, len(tisztitott), batch_meret):
                    chunk     = [t[:512] for t in tisztitott[i : i + batch_meret]]
                    raw_chunk = self.sentiment(chunk)
                    # top_k=None: [[{...}, ...], [{...}, ...]]
                    if raw_chunk and isinstance(raw_chunk[0], list):
                        raw_batch.extend(raw_chunk)
                    else:
                        raw_batch.extend([[r] for r in raw_chunk])

                for doc_idx, raw in enumerate(raw_batch):
                    kat, conf = _feldolgoz_nytk_eredmeny(raw)
                    nytk_sor_cache[doc_idx] = (kat, conf)

            except Exception as e:
                print(f"   Figyelem: NYTK batch hiba ({e}), szövegenkénti fallback")
                for doc_idx, szoveg in enumerate(tisztitott):
                    kat, conf = _modell_elemez(szoveg, self.sentiment)
                    nytk_sor_cache[doc_idx] = (kat, conf)

        # --- 3. Eredmények összeállítása – minden mező kizárólag az adott sorból ---
        for doc_idx, (szoveg, tiszta, doc) in enumerate(zip(szovegek, tisztitott, docs)):
            try:
                # HuSpaCy pontszám: teljes sor doc-ján (nem sent.as_doc())
                huspacy_pont = _szotari_pontszam(doc)

                # NYTK eredmény: az adott sor indexéhez tartozó cache-bejegyzés
                modell_kat, modell_conf = nytk_sor_cache.get(doc_idx, ("semleges", 0.5))

                if huspacy_pont > _HUSPACY_KAT_KUSZOOB:
                    huspacy_kat = "pozitív"
                elif huspacy_pont < -_HUSPACY_KAT_KUSZOOB:
                    huspacy_kat = "negatív"
                else:
                    huspacy_kat = "semleges"

                hibrid_kat, megalapozas = _hibrid_kategoria(huspacy_pont, modell_kat, modell_conf)

                # Mondatszintű bontás (tájékoztató jellegű, csak az adott sor mondatai)
                mondatszintu = _mondatszintu_osszefoglalo_doc(doc)

                yield {
                    "eredeti_szoveg":          tiszta,
                    "lemmatizalt_szoveg":      _lemmatizal(doc),
                    "kategoria":               huspacy_kat,
                    "pontszam":                round(huspacy_pont, 4),
                    "hunbert_kategoria":       modell_kat,
                    "hunbert_confidence":      round(modell_conf, 4),
                    "hibrid_kategoria":        hibrid_kat,
                    "hibrid_megalapozas":      megalapozas,
                    "mondatszintu_sentiment":  mondatszintu,
                    "pozitiv_elemek":          _pozitiv_elemek(doc),
                    "negativ_elemek":          _negativ_elemek(doc),
                    "emlitett_nevek":          _ner_entitasok(doc),
                    "mondatok_szama":          len(list(doc.sents)),
                    "token_szam":              len([t for t in doc if not t.is_space]),
                    "pos_statisztika":         _pos_statisztika(doc),
                    "dep_statisztika":         _dep_statisztika(doc),
                    "nyelveszeti_kapcsolatok": _nyelveszeti_kapcsolatok(doc),
                }

            except Exception as e:
                print(f"\n   Figyelem – szöveg kihagyva: {str(e)[:80]}")
                yield _hiba_eredmeny(szoveg, str(e))

    def modell_cache_letoltes(self):
        """
        Letölti és helyi cache-be menti a sentiment modellt.
        Futtasd egyszer internet-hozzáféréssel, utána offline is működik.
        """
        helyi_alap = MODELL_MAPPA / "sentiment-alap"

        if helyi_alap.exists():
            print(f"Modell már mentve: {helyi_alap}")
            return

        # Ha a pipeline még nincs betöltve, most töltjük be
        if self.sentiment is None:
            print("Modell betöltése letöltéshez...")
            self.sentiment = _get_sentiment_pipeline()

        if self.sentiment is None:
            print("Modell betöltése sikertelen, letöltés nem lehetséges.")
            return

        _menti_modellt(self.sentiment, helyi_alap)


# ---------------------------------------------------------------------------
# BERTopic témafeltárás
# ---------------------------------------------------------------------------

BERTOPIC_EMBEDDING_MODELL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BERTOPIC_EMBEDDING_MAPPA  = MODELL_MAPPA / "bertopic-embedding"


def _get_bertopic_embedding():
    """
    Betölti a BERTopic embedding modellt helyi cache-ből.
    Ha a cache nem létezik, letölti és elmenti – utána offline is fut.
    """
    from sentence_transformers import SentenceTransformer

    if BERTOPIC_EMBEDDING_MAPPA.exists():
        print("   -> BERTopic embedding modell betöltése (helyi cache)...")
        return SentenceTransformer(str(BERTOPIC_EMBEDDING_MAPPA))

    print(f"   -> BERTopic embedding modell letöltése: {BERTOPIC_EMBEDDING_MODELL}")
    print("      (Első indítás: ~120 MB, ezután offline is fut)")
    modell = SentenceTransformer(BERTOPIC_EMBEDDING_MODELL)
    try:
        BERTOPIC_EMBEDDING_MAPPA.mkdir(parents=True, exist_ok=True)
        modell.save(str(BERTOPIC_EMBEDDING_MAPPA))
        print(f"   -> BERTopic embedding elmentve: {BERTOPIC_EMBEDDING_MAPPA}")
    except Exception as e:
        print(f"   Figyelem: BERTopic embedding mentés sikertelen: {e}")
    return modell


def generalj_temakat(szovegek: List[str], min_topic_size: int = 5) -> List[str]:
    """
    BERTopic témamodellezés automatikus fallback-kel.
    Teljesen lokális futás – az embedding modell helyi cache-ből töltődik,
    nem küld adatot külső szerverre.
    """
    if not szovegek:
        return []
    if len(szovegek) < min_topic_size:
        return _egyszeru_tema_fallback(szovegek)

    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer

        embedding_model = _get_bertopic_embedding()

        vectorizer = CountVectorizer(
            min_df=1,
            ngram_range=(1, 2),
            max_features=5000,
        )
        topic_model = BERTopic(
            embedding_model=embedding_model,   # helyi modell, nem HF letöltés
            min_topic_size=min(min_topic_size, max(2, len(szovegek) // 5)),
            vectorizer_model=vectorizer,
            verbose=False,
            calculate_probabilities=False,
        )

        topics, _ = topic_model.fit_transform(szovegek)

        tema_lista = []
        for topic_id in topics:
            if topic_id == -1:
                tema_lista.append("Vegyes / Besorolatlan")
            else:
                try:
                    top_words  = topic_model.get_topic(topic_id)
                    kulcsszavak = (
                        ", ".join(w for w, _ in top_words[:5])
                        if top_words else f"Téma {topic_id}"
                    )
                    tema_lista.append(kulcsszavak)
                except Exception:
                    tema_lista.append(f"Téma {topic_id}")

        unique = len(set(t for t in topics if t != -1))
        print(f"   BERTopic: {unique} témakör feltárva")
        return tema_lista

    except ImportError:
        return _egyszeru_tema_fallback(szovegek)
    except Exception as e:
        print(f"   BERTopic hiba: {e}")
        return _egyszeru_tema_fallback(szovegek)


def _egyszeru_tema_fallback(szovegek: List[str]) -> List[str]:
    """Szógyakoriság-alapú téma-hozzárendelés BERTopic nélkül."""
    STOPSZAVAK = {
        "a", "az", "és", "hogy", "is", "de", "meg", "egy", "el", "van",
        "nem", "ezt", "azt", "ez", "én", "te", "mi", "ti", "ők", "ami",
        "aki", "volt", "lett", "csak", "már", "még", "sem", "se", "ha",
        "ki", "be", "fel", "le", "on", "ön", "őt", "ezt", "azt", "itt",
    }
    eredmeny = []
    for szoveg in szovegek:
        szavak = re.findall(r'\b\w{4,}\b', szoveg.lower())
        szurt  = [s for s in szavak if s not in STOPSZAVAK]
        top    = [w for w, _ in Counter(szurt).most_common(5)]
        eredmeny.append(", ".join(top) if top else "Általános")
    return eredmeny
