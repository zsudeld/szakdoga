"""
gui_app.py
==========
Tkinter grafikus felület a hibrid sentiment elemzőhöz.
Indítás: python gui_app.py

Javítások (v2):
  - Függőség-hiány üzenet dinamikus (pontos lista)
  - Riport fájlneve a riport_generator.generalj_riportot visszatérési értékéből jön
  - Eredmény-ablak mutatja a tényleges fájlnevet
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from huspacy_elemzo import FejlettSentimentElemzo, generalj_temakat
from riport_generator import generalj_riportot
from utils import ellenorizd_fuggosegeket, intelligens_csv_beolvasas, optimalis_oszlop


class SentimentGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Professzionális Hibrid NLP Analizátor")
        self.root.geometry("700x550")
        self.root.resizable(False, False)
        self.selected_path: str | None = None
        self.is_running = False
        self._build_ui()
        # Függőségek ellenőrzése 500ms késleltetéssel (GUI megjelenik előbb)
        self.root.after(500, self._check_dependencies)

    # ------------------------------------------------------------------
    # UI felépítés
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Fejléc
        frame_title = tk.Frame(self.root, bg="#1F4E78", height=80)
        frame_title.pack(fill="x")
        tk.Label(
            frame_title,
            text="Hibrid Sentiment Elemző",
            font=("Segoe UI", 18, "bold"),
            bg="#1F4E78",
            fg="white",
        ).pack(pady=25)

        # Fájlválasztó
        frame_file = tk.Frame(self.root, pady=20)
        frame_file.pack()
        self.btn_select = ttk.Button(
            frame_file, text="CSV Fajl Kivalasztasa",
            command=self.select_file, width=30,
        )
        self.btn_select.pack(pady=10)
        self.label_file = ttk.Label(
            frame_file, text="Nincs fajl kivalasztva", font=("Segoe UI", 10)
        )
        self.label_file.pack()

        # Progress
        frame_progress = tk.Frame(self.root, pady=20)
        frame_progress.pack(fill="x", padx=50)
        self.progress = ttk.Progressbar(
            frame_progress, orient="horizontal", length=500, mode="determinate"
        )
        self.progress.pack(pady=10)
        self.label_status = ttk.Label(
            frame_progress, text="Varakozas fajlra...", font=("Segoe UI", 10)
        )
        self.label_status.pack()

        # Indítás gomb
        self.btn_run = ttk.Button(
            self.root,
            text="AI Elemzes Inditasa",
            command=self.start_thread,
            state="disabled",
            width=30,
        )
        self.btn_run.pack(pady=20)

        ttk.Label(
            self.root,
            text="HuSpaCy + HunBERT + BERTopic technologia",
            font=("Segoe UI", 9, "italic"),
            foreground="gray",
        ).pack(pady=10)

    # ------------------------------------------------------------------
    # Függőség-ellenőrzés (háttérszálban)
    # ------------------------------------------------------------------

    def _check_dependencies(self):
        self._update_status("Fuggosegek ellenorzese...")

        def check():
            ok, hianyzo_lista = ellenorizd_fuggosegeket(visszaad_listaban=True)
            if not ok:
                # Dinamikus üzenet: pontosan mutatja, mi hiányzik
                if hianyzo_lista:
                    lista_szoveg = "\n".join(f"  • {cs}" for cs in hianyzo_lista)
                    uzenet = f"Hiányzó függőségek ({len(hianyzo_lista)} db):\n{lista_szoveg}\n\nLásd a konzolt a telepítési utasításokhoz."
                else:
                    uzenet = "Függőségi hiba – lásd a konzolt a részletekért."
                messagebox.showerror("Hiányzo Fuggosegek", uzenet)
                self._update_status("Fuggosegi hiba – telepites szukseges")
                self.btn_select.config(state="disabled")
            else:
                self._update_status("Minden fuggoseg OK")

        threading.Thread(target=check, daemon=True).start()

    # ------------------------------------------------------------------
    # Fájlválasztás
    # ------------------------------------------------------------------

    def select_file(self):
        path = filedialog.askopenfilename(
            title="Valassz CSV fajlt",
            filetypes=[("CSV fajlok", "*.csv"), ("Minden fajl", "*.*")],
        )
        if path:
            self.selected_path = path
            filename = path.replace("\\", "/").split("/")[-1]
            self.label_file.config(text=f"Kivalasztva: {filename}")
            self.btn_run.config(state="normal")
            self._update_status("Keszen all az elemzesre")

    # ------------------------------------------------------------------
    # Elemzés indítása (háttérszálban)
    # ------------------------------------------------------------------

    def start_thread(self):
        if self.is_running:
            messagebox.showwarning("Figyelem", "Mar fut egy elemzes!")
            return
        self.is_running = True
        self.btn_run.config(state="disabled")
        self.btn_select.config(state="disabled")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        riport_fajl = None
        try:
            # 1. CSV beolvasás
            self._update_status("CSV beolvasasa...")
            self._set_progress(0)
            df_raw = intelligens_csv_beolvasas(self.selected_path)
            if df_raw is None or df_raw.empty:
                raise ValueError("A CSV ures vagy nem olvasható!")

            self._set_progress(10)

            # 2. Szöveg oszlop
            oszlop = optimalis_oszlop(df_raw)
            texts  = [str(t) for t in df_raw[oszlop].dropna().tolist()]
            if not texts:
                raise ValueError("Nincs feldolgozható szoveg a kivalasztott oszlopban!")

            self._update_status(f"{len(texts)} szoveg feldolgozasra var...")
            self._set_progress(15)

            # 3. NLP motor
            self._update_status("HuSpaCy + HunBERT betoltese...")
            motor = FejlettSentimentElemzo()
            self._set_progress(25)

            # 4. Sentiment elemzés (batch)
            self._update_status("Hibrid sentiment elemzes...")
            eredmenyek = []
            for i, res in enumerate(motor.elemzes_batch(texts), 1):
                eredmenyek.append(res)
                if i % 5 == 0 or i == len(texts):
                    prog = 25 + int((i / len(texts)) * 40)
                    self._set_progress(prog)
                    self._update_status(f"Sentiment: {i}/{len(texts)}")

            df_final = pd.DataFrame(eredmenyek)
            self._set_progress(65)

            # 5. BERTopic
            self._update_status("BERTopic temamodellezes...")
            self._set_progress_indeterminate()
            lemmatizalt = df_final.get('lemmatizalt_szoveg', pd.Series(texts)).tolist()
            temak       = generalj_temakat(lemmatizalt)
            df_final['tema_kulcsszavak'] = (
                temak if len(temak) == len(df_final)
                else ["Tema nem generalható"] * len(df_final)
            )
            self._set_progress_determinate()
            self._set_progress(85)

            # 6. Statisztikák
            stats = {
                'Pozitív (Hibrid)':     int((df_final['hibrid_kategoria'] == 'pozitív').sum()),
                'Negatív (Hibrid)':     int((df_final['hibrid_kategoria'] == 'negatív').sum()),
                'Semleges (Hibrid)':    int((df_final['hibrid_kategoria'] == 'semleges').sum()),
                'Átlagos HuSpaCy pont': round(float(df_final['pontszam'].mean()), 3),
            }

            # 7. Excel riport (fájlnév a visszatérési értékből)
            self._update_status("Excel riport generalasa...")
            self._set_progress(90)
            riport_fajl = generalj_riportot(df_final, stats)
            self._set_progress(100)
            self._update_status("Elemzes befejezve!")

            poz = stats['Pozitív (Hibrid)']
            neg = stats['Negatív (Hibrid)']
            sem = stats['Semleges (Hibrid)']
            atl = stats['Átlagos HuSpaCy pont']
            messagebox.showinfo(
                "Elemzés Befejezve",
                f"Elemzes kesz!\n\n"
                f"Pozitiv: {poz} db\n"
                f"Negativ: {neg} db\n"
                f"Semleges: {sem} db\n"
                f"Atlag pontszam: {atl}\n\n"
                f"Riport: {riport_fajl}",
            )

        except Exception as e:
            messagebox.showerror("Hiba", f"Hiba az elemzes soran:\n{e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup_progress()
            self.is_running = False
            self.btn_run.config(state="normal")
            self.btn_select.config(state="normal")

    # ------------------------------------------------------------------
    # Progress bar segédek (szálbiztos)
    # ------------------------------------------------------------------

    def _update_status(self, text: str):
        self.root.after(0, lambda: self.label_status.config(text=text))

    def _set_progress(self, value: int):
        def update():
            self.progress["maximum"] = 100
            self.progress["value"]   = value
        self.root.after(0, update)

    def _set_progress_indeterminate(self):
        def update():
            self.progress.config(mode="indeterminate")
            self.progress.start(10)
        self.root.after(0, update)

    def _set_progress_determinate(self):
        def update():
            self.progress.stop()
            self.progress.config(mode="determinate")
        self.root.after(0, update)

    def _cleanup_progress(self):
        def update():
            if self.progress.cget('mode') == 'indeterminate':
                self.progress.stop()
            self.progress.config(mode="determinate")
            self.progress["value"] = 0
        self.root.after(0, update)


# ---------------------------------------------------------------------------
# Belépési pont
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app  = SentimentGui(root)
    root.mainloop()
