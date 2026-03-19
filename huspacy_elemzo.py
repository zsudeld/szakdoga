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
_NYTK_MAGAS_KUSZOOB      = 0.55
_NYTK_MERESEKELT_KUSZOOB = 0.50
_HUSPACY_EROS_KUSZOOB    = 0.25
_HUSPACY_GYENGE_KUSZOOB  = 0.10
_HUSPACY_KAT_KUSZOOB     = 0.10

# ---------------------------------------------------------------------------
# Bővített magyar sentiment szótár
# (OpinHuBank + HuSent + hétköznapi vélemény-szövegek, értékelések)
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
    "boldog", "izgalmas", "lenyűgöző", "pozitív", "optimista",
    # Oktatási kontextus
    "szemléletes", "közérthető", "gyakorlatias", "naprakész", "alapos",
    "felkészült", "lelkiismeretes", "türelmes", "empatikus", "dinamikus",
    "strukturált", "interaktív", "élvezetes", "tanulságos",
    # Fokozók önállóan is pozitív értékkel
    "kimagasló", "kiemelkedő", "elsőrangú", "méltányolandó",
    # Pontosság és részletesség
    "pontos", "részletes", "átfogó", "informatív",
    "megbízható", "stabil", "tiszta", "egyértelmű", "gördülékeny",
    "kényelmes", "szervezett", "következetes", "rugalmas", "támogató",
    "ösztönző", "figyelmes", "gondos", "kreatív", "eredeti",
    "friss", "modern", "releváns", "megfelelő",
    "gyors", "hatásos", "eredményes", "teljes", "komplex",
    "különleges", "kiemelten", "magas", "erős", "szilárd",
    "aktív", "bevonó", "igaz",
    # Műszaki/oktatási kontextus
    "működőképes", "működik", "helyes", "intuitív", "célszerű",
    "értékes", "ismeretgazdagító", "szisztematikus",
    "felépített", "metodikus", "felfogható",
    # Étel/szolgáltatás/vendéglátás
    "finom", "ízletes", "ízlős", "zamatos", "omlós", "ropogós",
    "frissen", "frissítő", "illatos", "ínycsiklandó", "mennyei",
    "pompás", "elegáns", "hangulatos", "kényelmes", "tiszta",
    "gyorsan", "udvarias", "előzékeny", "figyelmes", "mosolygós",
    "segít", "segített", "megoldotta", "megoldás", "ajánlom",
    "visszamegyek", "visszajövök", "érdemes",
    "klassz", "király", "fasza",  # szleng pozitív
    "örömmel", "szívesen", "köszönet", "hálás", "elégedve",
    # Általános pozitív igék/főnevek
    "siker", "fejlődés", "haladás", "növekedés", "javulás", "áttörés",
    "megoldódott", "rendeződött", "megvalósult", "megvalósul",
    "nyert", "győzött", "elért", "teljesített", "rekord",
    # Vélemény és ajánlás
    "ajánlott", "javasolt", "bevált", "megbízható", "kipróbált",
    "elismerés", "dicséret", "gratulálok", "gratulál", "brávó",
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
    # Egyéb problémák
    "bonyolult", "zavaró", "kellemetlen",
    "homályos", "kusza", "megbízhatatlan", "töredékes",
    "sekélyes", "sivár", "érdektelen", "ósdi",
    "akadozó", "bugos", "törött",
    "értéktelen", "kárba", "veszteség",
    "körülményes", "macerás", "fárasztó", "kimerítő",
    "tolakodó", "elviselhetetlen", "terhes",
    "csalódás", "cserbenhagyott", "becsapott", "félrevezető",
    # Műszaki kontextus
    "hiba",
    # Étel/szolgáltatás/vendéglátás
    "hideg", "kihűlt", "langyos", "nyers", "égett", "savanyú",
    "állott", "régi", "büdös", "undorító", "ehetetetlen",
    "ízetlen", "sótlan", "túlsózott",  # száraz: általános negatív szekcióban már szerepel
    "kövér", "nehéz",  # étel kontextusban negatív
    "piszok", "piszkos", "koszos", "rovar",
    "fázós",
    "drága", "túlárazott", "megvezetett", "nem ér annyit",
    "flegma", "mogorva", "barátságtalan", "udvariatlan", "arrogáns",
    "szemtelen", "goromba", "durva",
    "vár", "várat", "várakozás", "sokáig", "késő", "késik", "késett",
    "megvárattuk", "hagyta", "magára hagyott",
    "botrány", "botrányos", "szégyenteljes", "felháborító",
    "panasz", "reklamáció", "visszaküldtem", "visszaküldtük",
    "sajnos", "sajnálatos", "sajnálom", "sajnálattal",
    "nem ajánlom", "nem ajánlanám",
    "soha többet", "utoljára", "utolsó",
    "rettentő", "ijesztő", "veszélyes",
    # Általános negatív igék/főnevek
    "megbukott", "kudarcot", "kudarc", "elveszett", "elveszített",
    "meghiúsult", "visszalépett", "lemondott", "megszűnt",
    "bajba", "veszélybe", "fenyeget", "aggaszt", "aggodalmat",
    # Vélemény
    "sikeretlen", "sikertelen", "nem_teljesített",
    "kifogás", "hiányosság", "mulasztás", "hanyagság",
}

FOKOZOK: set = {
    "nagyon", "igen", "rendkívül", "kifejezetten", "különösen",
    "eléggé", "igazán", "teljesen", "abszolút", "végtelenül",
    "meglehetősen", "elég", "igencsak", "alaposan",
    "rendkívüli", "kivételesen", "határozottan", "erősen", "mélyen",
    "érdemben", "valóban", "lényegesen", "jelentősen",
    "sokkal", "jóval", "drámaian", "nyilvánvalóan",
    "túlzottan", "extrém", "brutálisan", "rettentően",
}

NEGACIOK: set = {
    "nem", "se", "sem", "soha", "semmit", "sehol", "senki",
    "semmi", "semmilyen", "egyáltalán", "korántsem", "távolról",
    "sehogy", "semmiképpen", "semmiféle",
}

NEGACIO_HATKORE = 6


# ---------------------------------------------------------------------------
# Lazy model betöltés
# ---------------------------------------------------------------------------

_spacy_nlp           = None
_sentiment_pipeline  = None
_modell_neve         = None
_nytk_n_classes      = None


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
    top_k=None beállítással MINDEN osztály pontszámát visszaadja.
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
            device=-1,
            top_k=None,
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
    """Helyi cache-be menti a modellt GDPR-biztos offline futáshoz."""
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

    if isinstance(raw, dict):
        raw = [raw]

    if not raw:
        return "semleges", 0.5

    scores_by_label = {r["label"].upper(): float(r["score"]) for r in raw}

    # --- 1. Explicit string label-ek ---
    poz  = sum(v for k, v in scores_by_label.items()
               if any(p in k for p in ("POSITIV", "POSITIVE", "POS", "POZIT")))
    neg_e = sum(v for k, v in scores_by_label.items()
                if any(n in k for n in ("NEGATIV", "NEGATIVE", "NEG")))
    neu  = sum(v for k, v in scores_by_label.items()
               if any(n in k for n in ("NEUTRAL", "NEU", "SEMLEGES")))

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
    if _nytk_n_classes is None:
        _nytk_n_classes = n_classes

    if n_classes == 3:
        neg_s = numeric.get(0, 0.0)
        neu_s = numeric.get(1, 0.0)
        poz_s = numeric.get(2, 0.0)

    elif n_classes == 5:
        neg_s = numeric.get(0, 0.0) + numeric.get(1, 0.0)
        neu_s = numeric.get(2, 0.0)
        poz_s = numeric.get(3, 0.0) + numeric.get(4, 0.0)

    else:
        max_label = max(numeric.keys())
        low  = max_label / 3
        high = max_label * 2 / 3
        neg_s = sum(v for k, v in numeric.items() if k <= low)
        neu_s = sum(v for k, v in numeric.items() if low < k < high)
        poz_s = sum(v for k, v in numeric.items() if k >= high)

    best = max(poz_s, neg_s, neu_s)
    if best == poz_s and poz_s > 0:
        return "pozitív", poz_s
    elif best == neg_s and neg_s > 0:
        return "negatív", neg_s
    else:
        return "semleges", neu_s if neu_s > 0 else 0.5


def _modell_elemez(szoveg: str, pipeline_obj) -> tuple:
    """NYTK modell hívás egyetlen szövegre."""
    if pipeline_obj is None:
        return "semleges", 0.5
    try:
        raw = pipeline_obj(szoveg[:512])
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
    Három szinten keres: közvetlen gyermek, szülő, megelőző N token.
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
# Szótárban keresés – lemma ÉS szóalak alapján
# ---------------------------------------------------------------------------

def _token_talal_pozitiv(token) -> bool:
    """Igaz, ha a token lemma-ja VAGY szóalakja szerepel a pozitív szótárban."""
    return (
        token.lemma_.lower() in POZITIV_SZAVAK
        or token.text.lower() in POZITIV_SZAVAK
    )


def _token_talal_negativ(token) -> bool:
    """Igaz, ha a token lemma-ja VAGY szóalakja szerepel a negatív szótárban."""
    return (
        token.lemma_.lower() in NEGATIV_SZAVAK
        or token.text.lower() in NEGATIV_SZAVAK
    )


# ---------------------------------------------------------------------------
# Szótári pontszámítás – sqrt-alapú normalizáció
# ---------------------------------------------------------------------------

def _szotari_pontszam(doc) -> float:
    """
    Bővített szótár-alapú pontszámítás HuSpaCy Doc objektumon.
    Scope-alapú negációkezeléssel és fokozók figyelembevételével.

    FIX: Lemma ÉS szóalak alapú keresés (HuSpaCy lemmatizáció hiányosságainak
    kompenzálásához).

    Normalizáció:
      - Rövid szöveg (≤5 tartalmas szó): nincs normalizáció
      - Közepes szöveg (6-15 szó): félig normalizálunk
      - Hosszú szöveg (>15 szó): teljes normalizáció

    Visszatér: [-1.0 … +1.0]
    """
    pont = 0.0

    for token in doc:
        if not _token_talal_pozitiv(token) and not _token_talal_negativ(token):
            continue

        negalt = _negalt_e(token, doc)

        fokozott = (
            any(child.lemma_.lower() in FOKOZOK for child in token.children)
            or token.head.lemma_.lower() in FOKOZOK
        )
        szorzo = 1.4 if fokozott else 1.0

        if _token_talal_pozitiv(token):
            pont += (-0.5 if negalt else 0.7) * szorzo
        elif _token_talal_negativ(token):
            pont += (0.4 if negalt else -0.8) * szorzo

    tartalmas_szavak = [
        t for t in doc
        if not t.is_space and t.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV')
    ]
    tartalmas_szam = len(tartalmas_szavak)

    if tartalmas_szam == 0:
        return 0.0

    if tartalmas_szam <= 5:
        norm = 1.0
        szorzo = 1.5
    elif tartalmas_szam <= 15:
        norm = max(1.0, math.sqrt(tartalmas_szam / 2))
        szorzo = 1.2
    else:
        norm = max(1.0, math.sqrt(tartalmas_szam))
        szorzo = 1.0

    final_pont = pont / norm * szorzo
    return max(-1.0, min(1.0, final_pont * 1.8))


# ---------------------------------------------------------------------------
# Hibrid döntés
# ---------------------------------------------------------------------------

def _hibrid_kategoria(
    huspacy_pont: float,
    modell_kat: str,
    modell_conf: float,
) -> tuple:
    """
    Kombinálja a szótári és transzformer eredményt.

    Döntési sorrend:
      1. Mindkét modell egyezik → egyértelműen az a kategória
      2. NYTK magas konfidencia (≥ 0.55) → NYTK dönt
      3. HuSpaCy erős szótári jel (|pont| ≥ 0.25) → HuSpaCy dönt
      4. NYTK mérsékelt jel (> 0.50, nem semleges) → NYTK dönt
      5. HuSpaCy gyenge jel (|pont| > 0.10) → HuSpaCy dönt
      6. Valóban semleges
    """
    if huspacy_pont > _HUSPACY_KAT_KUSZOOB:
        huspacy_kat = "pozitív"
    elif huspacy_pont < -_HUSPACY_KAT_KUSZOOB:
        huspacy_kat = "negatív"
    else:
        huspacy_kat = "semleges"

    if huspacy_kat == modell_kat:
        return huspacy_kat, (
            f"Egyezés: {huspacy_kat} "
            f"(HuSpaCy: {huspacy_pont:.3f}, NYTK conf: {modell_conf:.2f})"
        )

    if modell_conf >= _NYTK_MAGAS_KUSZOOB and modell_kat != "semleges":
        return modell_kat, (
            f"NYTK magabiztos ({modell_conf:.2f}) döntött. "
            f"HuSpaCy: {huspacy_pont:.3f}"
        )

    if abs(huspacy_pont) >= _HUSPACY_EROS_KUSZOOB and huspacy_kat != "semleges":
        return huspacy_kat, (
            f"Erős szótári jel ({huspacy_pont:.3f}). "
            f"NYTK: {modell_kat} ({modell_conf:.2f})"
        )

    if modell_conf > _NYTK_MERESEKELT_KUSZOOB and modell_kat != "semleges":
        return modell_kat, (
            f"NYTK mérsékelt jel ({modell_conf:.2f}). "
            f"HuSpaCy: {huspacy_pont:.3f}"
        )

    if abs(huspacy_pont) > _HUSPACY_GYENGE_KUSZOOB and huspacy_kat != "semleges":
        return huspacy_kat, (
            f"Mérsékelt szótári jel ({huspacy_pont:.3f}). "
            f"NYTK: {modell_kat} ({modell_conf:.2f})"
        )

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
    """
    FIX: Lemma ÉS szóalak alapú keresés.
    """
    return ", ".join(
        t.text for t in doc
        if _token_talal_pozitiv(t) and not t.is_stop
    )


def _negativ_elemek(doc) -> str:
    """
    FIX: Lemma ÉS szóalak alapú keresés.
    """
    return ", ".join(
        t.text for t in doc
        if _token_talal_negativ(t) and not t.is_stop
    )


def _lemmatizal(doc) -> str:
    return " ".join(
        t.lemma_.lower()
        for t in doc
        if not t.is_stop and not t.is_punct and not t.is_space and len(t.lemma_) > 2
    )


def _mondatszintu_osszefoglalo_doc(doc) -> str:
    """
    Mondatszintű bontás kizárólag az adott sor doc-jából.
    Csak HuSpaCy szótári pontot használ.
    """
    reszek = []
    for i, sent in enumerate(doc.sents, 1):
        szoveg = sent.text.strip()
        if len(szoveg) < 5:
            continue
        try:
            pont = _szotari_pontszam(sent.as_doc())
        except Exception:
            pont = 0.0
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

    GDPR: minden adat lokálisan marad.

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

    def elemzes_batch(
        self,
        szovegek: List[str],
        batch_meret: int = 16,
    ) -> Generator[Dict, None, None]:
        """
        Batch elemzés generátorként – sor-szintű feldolgozással.

        Az elemzés egysége a CSV sor (nem a mondat).
        NYTK és HuSpaCy is mindig a teljes sort kapja inputként.

        1. HuSpaCy nlp.pipe(): soronként külön Doc
        2. NYTK batch: teljes sor szövegét kapja
        3. Eredmények összeállítása – minden mező csak az adott sorból
        """
        if not szovegek:
            return

        tisztitott = [_tisztit(str(s)) for s in szovegek]

        # --- 1. HuSpaCy batch ---
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

        # --- 2. NYTK batch ---
        nytk_sor_cache: dict = {}

        if self.sentiment is not None:
            try:
                raw_batch: list = []
                for i in range(0, len(tisztitott), batch_meret):
                    chunk     = [t[:512] for t in tisztitott[i: i + batch_meret]]
                    raw_chunk = self.sentiment(chunk)
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

        # --- 3. Eredmények összeállítása ---
        for doc_idx, (szoveg, tiszta, doc) in enumerate(zip(szovegek, tisztitott, docs)):
            try:
                huspacy_pont = _szotari_pontszam(doc)
                modell_kat, modell_conf = nytk_sor_cache.get(doc_idx, ("semleges", 0.5))

                if huspacy_pont > _HUSPACY_KAT_KUSZOOB:
                    huspacy_kat = "pozitív"
                elif huspacy_pont < -_HUSPACY_KAT_KUSZOOB:
                    huspacy_kat = "negatív"
                else:
                    huspacy_kat = "semleges"

                hibrid_kat, megalapozas = _hibrid_kategoria(huspacy_pont, modell_kat, modell_conf)

                yield {
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

            except Exception as e:
                print(f"\n   Figyelem – szöveg kihagyva: {str(e)[:80]}")
                yield _hiba_eredmeny(szoveg, str(e))


# ---------------------------------------------------------------------------
# Témafeltárás – KÉT elkülönített szint
#
# 1. generalj_temakat()      → per-dokumentum TF-IDF kulcsszavak
#                               (az "Elemzési Eredmények" lap Téma oszlopához)
#                               Minden sor SAJÁT tartalmát tükrözi – nincs áthallás.
#
# 2. generalj_bertopic()     → BERTopic klaszterezés a teljes korpuszon
#                               (a "Témaelemzés" lap összesítő táblájához)
#                               Klaszter-szintű témacsoportok – helyes felhasználás.
#
# MIÉRT VOLT PROBLÉMA? BERTopic klaszter-kulcsszavakat írt minden sorba:
#   ha 17 éttermi szöveg kerül egy klaszterbe, mind a 17 sor kapja
#   a "hús, hideg, rántott, étel" kulcsszavakat – még a "Fantasztikus volt
#   a kiszolgálás" sor is, holott abban nincs szó húsról.
#   A klaszter-kulcsszó a CSOPORT reprezentatív szava, nem az adott soré.
# ---------------------------------------------------------------------------

# Magyar stopszavak az egyedi kulcsszó-kinyeréshez
_HU_STOPSZAVAK: set = {
    "a", "az", "és", "hogy", "is", "de", "meg", "egy", "el", "van",
    "nem", "ezt", "azt", "ez", "én", "te", "mi", "ti", "ők", "ami",
    "aki", "volt", "lett", "csak", "már", "még", "sem", "se", "ha",
    "ki", "be", "fel", "le", "on", "ön", "őt", "itt", "ott", "úgy",
    "így", "mint", "mert", "igen",
    "vele", "neki", "abban", "ebben", "azon", "ezen", "amiért", "amely",
    "amelyet", "amelynek", "amelyben", "amellyel", "amelyre", "ezért",
    "azért", "ezzel", "azzal", "erről", "arról", "ebből", "abból",
    "ehhez", "ahhoz", "ettől", "attól", "erre", "arra", "ide", "oda",
    "innen", "onnan", "addig", "eddig", "majd", "talán", "szinte",
    "persze", "hiszen", "tehát", "viszont", "azonban", "ám", "bár",
    "mégis", "ugye", "vajon", "egyébként", "valóban", "tényleg",
    "minden", "mindenki", "mindig", "sehol", "soha", "sose", "sosem",
    "nála", "náluk", "tőle", "tőlük", "rá", "rájuk", "tőlem",
}

BERTOPIC_EMBEDDING_MODELL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BERTOPIC_EMBEDDING_MAPPA  = MODELL_MAPPA / "bertopic-embedding"


def _get_bertopic_embedding():
    """Betölti a BERTopic embedding modellt helyi cache-ből."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "A sentence-transformers csomag hiányzik. "
            "Telepítsd: pip install sentence-transformers"
        ) from exc

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


def generalj_temakat(szovegek: List[str]) -> List[str]:
    """
    Per-dokumentum kulcsszó-kinyerés TF-IDF alapon.

    Minden sor KIZÁRÓLAG A SAJÁT szövegének kulcsszavait kapja meg.
    Nincs klaszter-szintű áthallás: "Fantasztikus kiszolgálás" nem kapja
    meg a "hús, hideg" kulcsszavakat csak azért, mert étterem témájú.

    Algoritmus:
      1. TF-IDF vectorizer az egész korpuszon (IDF = ritkaság súlyozás)
      2. Minden dokumentumhoz a saját TF-IDF legmagasabb értékű 5 szava
      3. Fallback: szógyakoriság, ha TF-IDF nem elérhető

    Visszatér: ['kulcsszó1, kulcsszó2, ...'] per sor
    """
    if not szovegek:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        # TF-IDF: min_df=1 (minden szót figyelembe vesz),
        #         ngram_range=(1,1) – egyedi szavak per dokumentum
        vectorizer = TfidfVectorizer(
            min_df=1,
            max_df=0.95,          # Túl gyakori (szinte mindenben lévő) szavak kizárva
            ngram_range=(1, 1),
            max_features=10000,
            token_pattern=r'\b[a-záéíóöőúüű]{4,}\b',  # Magyar betűk, min 4 karakter
            stop_words=list(_HU_STOPSZAVAK),
        )

        tfidf_matrix = vectorizer.fit_transform(szovegek)
        feature_names = vectorizer.get_feature_names_out()

        eredmeny = []
        for i in range(tfidf_matrix.shape[0]):
            sor = tfidf_matrix[i].toarray().flatten()
            # Top 5 szó az adott dokumentumban TF-IDF szerint
            top_idx = sor.argsort()[::-1][:5]
            top_szavak = [feature_names[j] for j in top_idx if sor[j] > 0.0]
            if top_szavak:
                eredmeny.append(", ".join(top_szavak))
            else:
                # Fallback: egyszerű szógyakoriság ebből a szövegből
                eredmeny.append(_egy_szoveg_kulcsszavak(szovegek[i]))

        print(f"   TF-IDF kulcsszó-kinyerés: {len(eredmeny)} sor feldolgozva")
        return eredmeny

    except ImportError:
        print("   sklearn nem elérhető, egyszerű kulcsszó-kinyerés fut")
        return [_egy_szoveg_kulcsszavak(sz) for sz in szovegek]
    except Exception as e:
        print(f"   TF-IDF hiba ({e}), egyszerű fallback")
        return [_egy_szoveg_kulcsszavak(sz) for sz in szovegek]


def _egy_szoveg_kulcsszavak(szoveg: str) -> str:
    """Egyszerű szógyakoriság-alapú kulcsszó-kinyerés egyetlen szövegből."""
    szavak = re.findall(r'\b[a-záéíóöőúüű]{4,}\b', szoveg.lower())
    szurt  = [s for s in szavak if s not in _HU_STOPSZAVAK]
    top    = [w for w, _ in Counter(szurt).most_common(5)]
    return ", ".join(top) if top else "Általános"


def generalj_bertopic(szovegek: List[str], min_topic_size: int = 3) -> dict:
    """
    BERTopic klaszterezés a TELJES korpuszon – a Témaelemzés lap számára.

    Ez a függvény KLASZTER-SZINTŰ elemzést végez: meghatározza a korpusz
    fő témacsoportjait és azok kulcsszavait. A visszatérési értéke dict,
    amelyet a riport_generator.py a Témaelemzés lapra ír.

    FONTOS: Ez az eredmény NEM kerül a sorok mellé – csak az összesítő tabba.

    Visszatér: {
        'tema_lista':   [str, ...],   # klaszter neve (kulcsszavak)
        'topic_ids':    [int, ...],   # topic_id per dokumentum
        'topic_info':   DataFrame,    # BERTopic topic_info táblája
        'n_topics':     int,
    }
    """
    if not szovegek or len(szovegek) < 3:
        return {'tema_lista': [], 'topic_ids': [], 'topic_info': None, 'n_topics': 0}

    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer

        embedding_model = _get_bertopic_embedding()

        n = len(szovegek)
        if n < 20:
            min_size = 2
        elif n < 50:
            min_size = 3
        else:
            min_size = min(min_topic_size, max(2, n // 10))

        vectorizer = CountVectorizer(
            min_df=1,
            ngram_range=(1, 2),
            max_features=5000,
        )
        topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=min_size,
            vectorizer_model=vectorizer,
            verbose=False,
            # calculate_probabilities eltávolítva: BERTopic >= 0.16-ban
            # a konstruktorból kivonták, alapértelmezetten False
        )

        topics, _ = topic_model.fit_transform(szovegek)

        # Klaszter-szintű témanevek
        tema_nevek = {}
        for tid in set(topics):
            if tid == -1:
                tema_nevek[tid] = "Vegyes / Besorolatlan"
            else:
                try:
                    top_words = topic_model.get_topic(tid)
                    tema_nevek[tid] = (
                        ", ".join(w for w, _ in top_words[:5])
                        if top_words else f"Téma {tid}"
                    )
                except Exception:
                    tema_nevek[tid] = f"Téma {tid}"

        tema_lista = [tema_nevek.get(tid, "Ismeretlen") for tid in topics]

        try:
            topic_info = topic_model.get_topic_info()
        except Exception:
            topic_info = None

        unique = len(set(t for t in topics if t != -1))
        print(f"   BERTopic: {unique} témakör feltárva a korpuszban")

        return {
            'tema_lista': tema_lista,
            'topic_ids':  topics,
            'topic_info': topic_info,
            'n_topics':   unique,
        }

    except ImportError:
        print("   BERTopic nem elérhető, Témaelemzés lap üres lesz")
        return {'tema_lista': [], 'topic_ids': [], 'topic_info': None, 'n_topics': 0}
    except Exception as e:
        print(f"   BERTopic hiba: {e}")
        return {'tema_lista': [], 'topic_ids': [], 'topic_info': None, 'n_topics': 0}