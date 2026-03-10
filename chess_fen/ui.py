import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from .capture import capture_full_screen
from .constants import DEFAULT_ENGINE_ELO
from .engine import get_best_move_from_fen
from .service import _load_predictor, analyze_screen_image


class ChessAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Screen Assistant")
        self.root.resizable(False, False)

        self.predictor = None
        self.is_busy = False

        self.stockfish_var = tk.StringVar(value=os.environ.get("STOCKFISH_PATH", ""))
        self.side_to_move_var = tk.StringVar(value="w")
        self.status_var = tk.StringVar(
            value="Pret. Clique sur le bouton pour capturer l'ecran et analyser le plateau."
        )
        self.fen_var = tk.StringVar(value="-")
        self.best_move_var = tk.StringVar(value="-")
        self.eval_var = tk.StringVar(value="-")
        self.metrics_var = tk.StringVar(value="-")
        self.engine_var = tk.StringVar(value="-")

        self._build_layout()

    def _build_layout(self):
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)

        ttk.Label(main_frame, text="Stockfish").grid(row=0, column=0, sticky="w", pady=(0, 8))
        stockfish_entry = ttk.Entry(
            main_frame,
            textvariable=self.stockfish_var,
            width=52,
        )
        stockfish_entry.grid(row=0, column=1, sticky="ew", pady=(0, 8))
        ttk.Button(
            main_frame,
            text="Parcourir",
            command=self._browse_stockfish,
        ).grid(row=0, column=2, padx=(8, 0), pady=(0, 8))

        ttk.Label(main_frame, text="Trait").grid(row=1, column=0, sticky="w", pady=(0, 12))
        side_frame = ttk.Frame(main_frame)
        side_frame.grid(row=1, column=1, columnspan=2, sticky="w", pady=(0, 12))
        ttk.Radiobutton(
            side_frame,
            text="Blancs",
            value="w",
            variable=self.side_to_move_var,
        ).grid(row=0, column=0, padx=(0, 12))
        ttk.Radiobutton(
            side_frame,
            text="Noirs",
            value="b",
            variable=self.side_to_move_var,
        ).grid(row=0, column=1)

        self.analyze_button = ttk.Button(
            main_frame,
            text="Capturer l'ecran et recommander",
            command=self._start_analysis,
        )
        self.analyze_button.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 12))

        ttk.Separator(main_frame, orient="horizontal").grid(
            row=3,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(0, 12),
        )

        ttk.Label(main_frame, text="Statut").grid(row=4, column=0, sticky="nw", pady=(0, 8))
        ttk.Label(
            main_frame,
            textvariable=self.status_var,
            wraplength=520,
            justify="left",
        ).grid(row=4, column=1, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(main_frame, text="FEN").grid(row=5, column=0, sticky="nw", pady=(0, 8))
        ttk.Label(
            main_frame,
            textvariable=self.fen_var,
            wraplength=520,
            justify="left",
        ).grid(row=5, column=1, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(main_frame, text="Meilleur coup").grid(row=6, column=0, sticky="nw", pady=(0, 8))
        ttk.Label(
            main_frame,
            textvariable=self.best_move_var,
            wraplength=520,
            justify="left",
        ).grid(row=6, column=1, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(main_frame, text="Evaluation").grid(row=7, column=0, sticky="nw", pady=(0, 8))
        ttk.Label(
            main_frame,
            textvariable=self.eval_var,
            wraplength=520,
            justify="left",
        ).grid(row=7, column=1, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(main_frame, text="Qualite").grid(row=8, column=0, sticky="nw", pady=(0, 8))
        ttk.Label(
            main_frame,
            textvariable=self.metrics_var,
            wraplength=520,
            justify="left",
        ).grid(row=8, column=1, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(main_frame, text="Moteur").grid(row=9, column=0, sticky="nw")
        ttk.Label(
            main_frame,
            textvariable=self.engine_var,
            wraplength=520,
            justify="left",
        ).grid(row=9, column=1, columnspan=2, sticky="w")

    def _browse_stockfish(self):
        selected_path = filedialog.askopenfilename(
            title="Choisir stockfish.exe",
            filetypes=[("Executables", "*.exe"), ("Tous les fichiers", "*.*")],
        )
        if selected_path:
            self.stockfish_var.set(selected_path)

    def _set_busy(self, is_busy):
        self.is_busy = is_busy
        state = "disabled" if is_busy else "normal"
        self.analyze_button.config(state=state)

    def _set_status(self, message):
        self.root.after(0, lambda: self.status_var.set(message))

    def _start_analysis(self):
        if self.is_busy:
            return

        self._set_busy(True)
        self.status_var.set("Initialisation de l'analyse...")
        self.best_move_var.set("-")
        self.eval_var.set("-")
        self.engine_var.set("-")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _ensure_predictor(self):
        if self.predictor is None:
            self._set_status("Chargement du modele de reconnaissance...")
            self.predictor = _load_predictor()
        return self.predictor

    def _run_analysis(self):
        try:
            predictor = self._ensure_predictor()
            self._set_status("Capture de l'ecran en cours...")
            screen_image = capture_full_screen()
            self._set_status("Detection du plateau et extraction du FEN...")
            analysis = analyze_screen_image(
                screen_image,
                predictor=predictor,
                use_filters=True,
                source_label="full_screen_capture.png",
            )

            engine_message = "Stockfish non configure"
            move_message = "Aucun coup"
            eval_message = "Aucune evaluation"

            fen = analysis["fen"]
            if fen:
                try:
                    self._set_status("Calcul du meilleur coup...")
                    move_info = get_best_move_from_fen(
                        fen,
                        stockfish_path=self.stockfish_var.get().strip() or None,
                        side_to_move=self.side_to_move_var.get(),
                        think_time=0.20,
                        engine_elo=DEFAULT_ENGINE_ELO,
                    )
                    move_message = (
                        f"{move_info['best_move_san']} "
                        f"({move_info['best_move_uci']})"
                    )
                    if move_info["mate_in"] is not None:
                        eval_message = f"Mat en {move_info['mate_in']}"
                    elif move_info["score_cp"] is not None:
                        eval_message = f"{move_info['score_cp']} centipions"
                    else:
                        eval_message = "Evaluation indisponible"

                    engine_path = move_info.get("stockfish_path") or ""
                    engine_elo = move_info.get("engine_elo")
                    if engine_elo is None:
                        engine_message = engine_path or "Stockfish actif"
                    else:
                        engine_message = f"{engine_path} | Elo {engine_elo}"
                except Exception as error:
                    engine_message = str(error)
                    move_message = "Impossible de calculer le coup"
                    eval_message = "Moteur indisponible"
            else:
                engine_message = "Aucun FEN detecte, moteur non lance"

            self.root.after(
                0,
                lambda: self._finish_analysis(
                    analysis=analysis,
                    move_message=move_message,
                    eval_message=eval_message,
                    engine_message=engine_message,
                ),
            )
        except Exception as error:
            self.root.after(0, lambda: self._show_error(str(error)))

    def _finish_analysis(self, analysis, move_message, eval_message, engine_message):
        self.fen_var.set(analysis["fen"] or "Aucun FEN detecte")
        self.best_move_var.set(move_message)
        self.eval_var.set(eval_message)
        self.metrics_var.set(
            "qualite={quality} | confiance={confidence:.3f} | stabilite={stability}".format(
                quality=analysis["quality_score"],
                confidence=analysis["avg_confidence"],
                stability=analysis["stability_count"],
            )
        )
        self.engine_var.set(engine_message)
        self.status_var.set("Analyse terminee.")
        self._set_busy(False)

    def _show_error(self, message):
        self.status_var.set(f"Erreur: {message}")
        self._set_busy(False)


def launch_desktop_app():
    root = tk.Tk()
    ChessAssistantApp(root)
    root.mainloop()
