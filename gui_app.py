"""
gui_app.py
==========
Tkinter grafikus felület a hibrid sentiment elemzőhöz.
Indítás: python gui_app.py
"""

import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from elemzo_pipeline import elemez
from utils import ellenorizd_fuggosegeket


class SentimentGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Professzionális Hibrid NLP Analizátor")
        self.root.geometry("700x550")
        self.root.minsize(600, 480)
        self.root.resizable(True, True)
        self.selected_path: str | None = None
        self.is_running = False
        self._run_lock = threading.Lock()
        self._deps_ok = False          # ← True csak ha az ellenőrzés sikeresen lefutott
        self._build_ui()
        self.root.after(500, self._check_dependencies)

    # ------------------------------------------------------------------
    # UI felépítés
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        frame_title = tk.Frame(self.root, bg="#1F4E78", height=80)
        frame_title.grid(row=0, column=0, sticky="ew")
        frame_title.columnconfigure(0, weight=1)
        tk.Label(
            frame_title,
            text="Hibrid Sentiment Elemző",
            font=("Segoe UI", 18, "bold"),
            bg="#1F4E78",
            fg="white",
        ).grid(row=0, column=0, pady=25)

        frame_file = tk.Frame(self.root, pady=20)
        frame_file.grid(row=1, column=0)
        self.btn_select = ttk.Button(
            frame_file, text="CSV Fajl Kivalasztasa",
            command=self.select_file, width=30,
        )
        self.btn_select.pack(pady=10)
        self.label_file = ttk.Label(
            frame_file, text="Nincs fajl kivalasztva", font=("Segoe UI", 10)
        )
        self.label_file.pack()

        frame_progress = tk.Frame(self.root, pady=20)
        frame_progress.grid(row=2, column=0, sticky="ew", padx=50)
        frame_progress.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(
            frame_progress, orient="horizontal", mode="determinate"
        )
        self.progress.grid(row=0, column=0, sticky="ew", pady=10)
        self.label_status = ttk.Label(
            frame_progress, text="Varakozas fajlra...", font=("Segoe UI", 10)
        )
        self.label_status.grid(row=1, column=0)

        self.btn_run = ttk.Button(
            self.root,
            text="Elemzes Inditasa",
            command=self.start_thread,
            state="disabled",
            width=30,
        )
        self.btn_run.grid(row=3, column=0, pady=20)

        ttk.Label(
            self.root,
            text="HuSpaCy + HuBERT + BERTopic technologia",
            font=("Segoe UI", 9, "italic"),
            foreground="gray",
        ).grid(row=4, column=0, pady=10)

    # ------------------------------------------------------------------
    # Függőség-ellenőrzés (háttérszálban)
    # ------------------------------------------------------------------

    def _check_dependencies(self):
        self._update_status("Fuggosegek ellenorzese...")

        def check():
            ok, hianyzo_lista = ellenorizd_fuggosegeket(visszaad_listaban=True)
            if not ok:
                if hianyzo_lista:
                    lista_szoveg = "\n".join(f"  • {cs}" for cs in hianyzo_lista)
                    uzenet = (
                        f"Hiányzó függőségek ({len(hianyzo_lista)} db):\n"
                        f"{lista_szoveg}\n\nLásd a konzolt a telepítési utasításokhoz."
                    )
                else:
                    uzenet = "Függőségi hiba – lásd a konzolt a részletekért."
                self.root.after(0, lambda: messagebox.showerror("Hiányzo Fuggosegek", uzenet))
                self._update_status("Fuggosegi hiba – telepites szukseges")
                self.root.after(0, lambda: self.btn_select.config(state="disabled"))
            else:
                self._deps_ok = True
                self._update_status("Minden fuggoseg OK")
                # Ha már van fájl kiválasztva, csak most engedélyezzük a gombot
                if self.selected_path:
                    self.root.after(0, lambda: self.btn_run.config(state="normal"))

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
            if self._deps_ok:
                self.btn_run.config(state="normal")
                self._update_status("Keszen all az elemzesre")
            else:
                self._update_status("Fuggosegek ellenorzese folyamatban...")

    # ------------------------------------------------------------------
    # Elemzés indítása (háttérszálban)
    # ------------------------------------------------------------------

    def start_thread(self):
        with self._run_lock:
            if self.is_running:
                self.root.after(0, lambda: messagebox.showwarning("Figyelem", "Mar fut egy elemzes!"))
                return
            self.is_running = True
        self.btn_run.config(state="disabled")
        self.btn_select.config(state="disabled")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            self._set_indeterminate("CSV beolvasása...")

            def progress_cb(pct: int, uzenet: str) -> None:
                if pct < 25:
                    self._update_status(uzenet)
                else:
                    self._set_determinate(pct, uzenet)

            riport_fajl, stats = elemez(
                fajl_ut=self.selected_path,
                progress_cb=progress_cb,
            )

            poz   = stats["Pozitív (Hibrid)"]
            neg   = stats["Negatív (Hibrid)"]
            sem   = stats["Semleges (Hibrid)"]
            atl   = stats["Átlagos HuSpaCy pont"]
            total = poz + neg + sem
            uzenet = (
                f"Elemzes kesz!\n\n"
                f"Pozitiv:  {poz} db  ({poz/total*100:.1f}%)\n"
                f"Negativ:  {neg} db  ({neg/total*100:.1f}%)\n"
                f"Semleges: {sem} db  ({sem/total*100:.1f}%)\n"
                f"Atlag pontszam: {atl}\n\n"
                f"Riport: {riport_fajl}"
            ) if total else f"Elemzes kesz!\n\nRiport: {riport_fajl}"

            def _befejez():
                self.progress.stop()
                self.progress.config(mode="determinate")
                self.progress["value"] = 100
                self.label_status.config(text="Kész.")
                self.root.update_idletasks()
                messagebox.showinfo("Elemzés Befejezve", uzenet)

            self.root.after(0, _befejez)

        except Exception as e:
            traceback.print_exc()
            hiba = str(e)
            def _hiba_dialog():
                self.progress.stop()
                self.progress.config(mode="determinate")
                self.progress["value"] = 0
                self.label_status.config(text="Hiba.")
                messagebox.showerror("Hiba", f"Hiba az elemzes soran:\n{hiba}")
            self.root.after(0, _hiba_dialog)

        finally:
            with self._run_lock:
                self.is_running = False
            self.root.after(0, lambda: self.btn_run.config(state="normal"))
            self.root.after(0, lambda: self.btn_select.config(state="normal"))

    # ------------------------------------------------------------------
    # Progress bar segédek (szálbiztos)
    # ------------------------------------------------------------------

    def _update_status(self, text: str):
        self.root.after(0, lambda: self.label_status.config(text=text))

    def _set_indeterminate(self, uzenet: str = ""):
        def update():
            self.progress.stop()
            self.progress.config(mode="indeterminate")
            self.progress.start(10)
            self.label_status.config(text=uzenet)
        self.root.after(0, update)

    def _set_determinate(self, value: int, uzenet: str = ""):
        def update():
            self.progress.stop()
            self.progress.config(mode="determinate")
            self.progress["maximum"] = 100
            self.progress["value"] = value
            self.label_status.config(text=uzenet)
        self.root.after(0, update)


# ---------------------------------------------------------------------------
# Belépési pont
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app  = SentimentGui(root)
    root.mainloop()
