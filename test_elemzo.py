"""
tests/test_elemzo.py
====================
Egységtesztek a hibrid sentiment elemző alapfüggvényeihez.

Futtatás: pytest tests/
"""

import pytest


# ---------------------------------------------------------------------------
# _hibrid_kategoria
# ---------------------------------------------------------------------------

from huspacy_elemzo import _hibrid_kategoria


class TestHibridKategoria:

    def test_egyezes_pozitiv(self):
        kat, _ = _hibrid_kategoria(0.5, "pozitív", 0.8)
        assert kat == "pozitív"

    def test_egyezes_negativ(self):
        kat, _ = _hibrid_kategoria(-0.5, "negatív", 0.8)
        assert kat == "negatív"

    def test_egyezes_semleges(self):
        kat, _ = _hibrid_kategoria(0.0, "semleges", 0.9)
        assert kat == "semleges"

    def test_nytk_magas_konfidencia_dont(self):
        """NYTK magas konfidenciával dönt HuSpaCy gyenge jelével szemben."""
        kat, _ = _hibrid_kategoria(0.05, "negatív", 0.80)
        assert kat == "negatív"

    def test_huspacy_eros_jel_dont(self):
        """Erős szótári jel dönt mérsékelt NYTK-val szemben."""
        kat, _ = _hibrid_kategoria(0.40, "semleges", 0.52)
        assert kat == "pozitív"

    def test_gyenge_jelek_semleges(self):
        """Mindkét jel tényleg gyenge → semleges. mc=0.49 < 0.50, abs(hp)=0.05 < 0.15"""
        kat, _ = _hibrid_kategoria(0.05, "pozitív", 0.49)
        assert kat == "semleges"

    def test_visszateres_tuple(self):
        eredmeny = _hibrid_kategoria(0.3, "pozitív", 0.7)
        assert isinstance(eredmeny, tuple)
        assert len(eredmeny) == 2
        assert isinstance(eredmeny[1], str)

    def test_megalapozas_nem_ures(self):
        _, megalapozas = _hibrid_kategoria(0.3, "pozitív", 0.7)
        assert len(megalapozas) > 0


# ---------------------------------------------------------------------------
# _feldolgoz_nytk_eredmeny
# ---------------------------------------------------------------------------

from huspacy_elemzo import _feldolgoz_nytk_eredmeny


class TestFeldolgozNytkEredmeny:

    def test_none_input(self):
        kat, conf = _feldolgoz_nytk_eredmeny(None)
        assert kat == "semleges"
        assert conf == 0.5

    def test_ures_lista(self):
        kat, conf = _feldolgoz_nytk_eredmeny([])
        assert kat == "semleges"

    def test_harom_osztalyos_pozitiv(self):
        raw = [
            {"label": "LABEL_0", "score": 0.05},
            {"label": "LABEL_1", "score": 0.10},
            {"label": "LABEL_2", "score": 0.85},
        ]
        kat, conf = _feldolgoz_nytk_eredmeny(raw)
        assert kat == "pozitív"
        assert conf == pytest.approx(0.85)

    def test_harom_osztalyos_negativ(self):
        raw = [
            {"label": "LABEL_0", "score": 0.90},
            {"label": "LABEL_1", "score": 0.05},
            {"label": "LABEL_2", "score": 0.05},
        ]
        kat, conf = _feldolgoz_nytk_eredmeny(raw)
        assert kat == "negatív"
        assert conf == pytest.approx(0.90)

    def test_harom_osztalyos_semleges(self):
        raw = [
            {"label": "LABEL_0", "score": 0.10},
            {"label": "LABEL_1", "score": 0.80},
            {"label": "LABEL_2", "score": 0.10},
        ]
        kat, conf = _feldolgoz_nytk_eredmeny(raw)
        assert kat == "semleges"

    def test_ot_osztalyos_pozitiv(self):
        raw = [
            {"label": "LABEL_0", "score": 0.02},
            {"label": "LABEL_1", "score": 0.03},
            {"label": "LABEL_2", "score": 0.05},
            {"label": "LABEL_3", "score": 0.30},
            {"label": "LABEL_4", "score": 0.60},
        ]
        kat, conf = _feldolgoz_nytk_eredmeny(raw)
        assert kat == "pozitív"
        assert conf == pytest.approx(0.90)

    def test_explicit_positive_label(self):
        raw = [{"label": "POSITIVE", "score": 0.92}]
        kat, conf = _feldolgoz_nytk_eredmeny(raw)
        assert kat == "pozitív"
        assert conf == pytest.approx(0.92)

    def test_explicit_negative_label(self):
        raw = [{"label": "NEGATIVE", "score": 0.88}]
        kat, conf = _feldolgoz_nytk_eredmeny(raw)
        assert kat == "negatív"

    def test_dict_input_wrapped(self):
        """Egyetlen dict-et listába kell csomagolni."""
        raw = {"label": "LABEL_2", "score": 0.75}
        kat, _ = _feldolgoz_nytk_eredmeny(raw)
        assert kat == "pozitív"


# ---------------------------------------------------------------------------
# _szotari_pontszam – SpaCy nélkül, mock doc-cal tesztelve
# ---------------------------------------------------------------------------

from huspacy_elemzo import (
    _szotari_pontszam,
    _token_talal_pozitiv,
    _token_talal_negativ,
)


class MockToken:
    """Minimális SpaCy token mock a szótári függvények teszteléséhez."""

    def __init__(self, text: str, lemma: str = "", pos: str = "NOUN",
                 is_stop: bool = False, is_space: bool = False,
                 is_punct: bool = False, children=None, head_lemma: str = ""):
        self.text = text
        self.lemma_ = lemma or text
        self.pos_ = pos
        self.is_stop = is_stop
        self.is_space = is_space
        self.is_punct = is_punct
        self.children = children or []
        self._head_lemma = head_lemma
        self.i = 0

    @property
    def head(self):
        class _Head:
            def __init__(self, lemma):
                self.lemma_ = lemma
        return _Head(self._head_lemma)

    @property
    def sent(self):
        return self


class MockDoc:
    def __init__(self, tokens):
        self._tokens = tokens
        for i, t in enumerate(tokens):
            t.i = i

    def __iter__(self):
        return iter(self._tokens)

    def __getitem__(self, idx):
        return self._tokens[idx]


class TestTokenTalal:

    def test_pozitiv_szoalak(self):
        tok = MockToken("jó")
        assert _token_talal_pozitiv(tok)

    def test_pozitiv_lemma(self):
        tok = MockToken("jók", lemma="jó")
        assert _token_talal_pozitiv(tok)

    def test_negativ_szoalak(self):
        tok = MockToken("rossz")
        assert _token_talal_negativ(tok)

    def test_semleges_szo(self):
        tok = MockToken("asztal")
        assert not _token_talal_pozitiv(tok)
        assert not _token_talal_negativ(tok)

    def test_tobbszavas_nem_illeszkedik(self):
        """Többszavas kifejezések (ha maradt volna) soha nem illeszkednek."""
        tok = MockToken("nem ajánlom")
        assert not _token_talal_negativ(tok)


class TestSzotariPontszam:

    def test_pozitiv_szoveg_pozitiv_pontot_ad(self):
        tokens = [
            MockToken("kiváló", pos="ADJ"),
            MockToken("remek", pos="ADJ"),
            MockToken("jó", pos="ADJ"),
        ]
        doc = MockDoc(tokens)
        pont = _szotari_pontszam(doc)
        assert pont > 0

    def test_negativ_szoveg_negativ_pontot_ad(self):
        tokens = [
            MockToken("rossz", pos="ADJ"),
            MockToken("borzasztó", pos="ADJ"),
        ]
        doc = MockDoc(tokens)
        pont = _szotari_pontszam(doc)
        assert pont < 0

    def test_ures_doc_nulla(self):
        doc = MockDoc([])
        assert _szotari_pontszam(doc) == 0.0

    def test_eredmeny_minus1_es_1_kozott(self):
        tokens = [MockToken("kiváló", pos="ADJ")] * 50
        doc = MockDoc(tokens)
        pont = _szotari_pontszam(doc)
        assert -1.0 <= pont <= 1.0

    def test_csak_stopszavak_nulla(self):
        tokens = [
            MockToken("a", is_stop=True, pos="DET"),
            MockToken("az", is_stop=True, pos="DET"),
        ]
        doc = MockDoc(tokens)
        assert _szotari_pontszam(doc) == 0.0


# ---------------------------------------------------------------------------
# generalj_temakat
# ---------------------------------------------------------------------------

from huspacy_elemzo import generalj_temakat


class TestGeneraljTemakat:

    def test_ures_lista(self):
        assert generalj_temakat([]) == []

    def test_visszater_lista(self):
        szovegek = ["Fantasztikus kiszolgálás volt", "Rossz étel hideg volt"]
        eredmeny = generalj_temakat(szovegek)
        assert isinstance(eredmeny, list)
        assert len(eredmeny) == len(szovegek)

    def test_minden_sor_kap_kulcsszavat(self):
        szovegek = ["alma körte barack", "autó motor kerék", "ház ablak ajtó"]
        eredmeny = generalj_temakat(szovegek)
        for e in eredmeny:
            assert isinstance(e, str)
            assert len(e) > 0


# ---------------------------------------------------------------------------
# intelligens_csv_beolvasas
# ---------------------------------------------------------------------------

import pandas as pd
import tempfile
import os

from utils import intelligens_csv_beolvasas, optimalis_oszlop


class TestCsvBeolvasas:

    def setup_method(self):
        self._temp_files: list[str] = []

    def _tmp_csv(self, tartalom: str, encoding: str = "utf-8") -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding=encoding,
            delete=False, newline=""
        )
        f.write(tartalom)
        f.close()
        self._temp_files.append(f.name)
        return f.name

    def teardown_method(self):
        for path in self._temp_files:
            try:
                os.chmod(path, 0o644)
                os.unlink(path)
            except OSError:
                pass
        self._temp_files.clear()

    def test_pontosvesszo_elvalaszto(self):
        path = self._tmp_csv("nev;velemeny\nKati;Nagyon jo volt\nPeti;Rossz volt\n")
        df = intelligens_csv_beolvasas(path)
        assert df is not None
        assert "velemeny" in df.columns

    def test_vesszos_elvalaszto(self):
        path = self._tmp_csv("name,review\nAnn,Great service\nBob,Terrible food\n")
        df = intelligens_csv_beolvasas(path)
        assert df is not None
        assert len(df) == 2

    def test_nem_letezo_fajl_none(self):
        df = intelligens_csv_beolvasas("/nem/letezik/fajl.csv")
        assert df is None

    def test_permission_error(self):
        if os.getuid() == 0:
            pytest.skip("Root felhasználóként a chmod 000 nem okoz PermissionError-t")
        path = self._tmp_csv("a;b\n1;2\n")
        os.chmod(path, 0o000)
        with pytest.raises(PermissionError):
            intelligens_csv_beolvasas(path)


class TestOptimalisOszlop:

    def test_leghosszabb_szoveg_oszlop(self):
        df = pd.DataFrame({
            "id": ["1", "2", "3"],
            "szoveg": [
                "Ez egy hosszabb véleményes szöveg",
                "Ez is egy hosszabb szöveg lesz",
                "Szintén hosszabb vélemény van itt",
            ],
        })
        oszlop = optimalis_oszlop(df)
        assert oszlop == "szoveg"

    def test_egyszeru_fallback(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        oszlop = optimalis_oszlop(df)
        assert oszlop == "a"
