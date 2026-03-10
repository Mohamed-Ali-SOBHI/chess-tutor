import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from .capture import capture_full_screen
from .constants import DEFAULT_ENGINE_ELO
from .engine import close_cached_engines
from .service import (
    _load_predictor,
    analyze_screen_image_and_suggest_move,
)


class ChessAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Screen Assistant")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.predictor = None
        self.is_busy = False

        self.stockfish_var = tk.StringVar(value=os.environ.get("STOCKFISH_PATH", ""))
        self.side_to_move_var = tk.StringVar(value="w")
        self.status_var = tk.StringVar(
            value="Pret. Clique sur le bouton pour capturer l'ecran et analyser le plateau."
        )
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="0%")
        self.fen_var = tk.StringVar(value="-")
        self.best_move_var = tk.StringVar(value="-")
        self.eval_var = tk.StringVar(value="-")
        self.metrics_var = tk.StringVar(value="-")
        self.engine_var = tk.StringVar(value="-")
        self.runtime_var = tk.StringVar(value="-")

        self._build_layout()

    def _build_layout(self):
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)

        ttk.Label(main_frame, text="Stockfish").grid(row=0, column=0, sticky="w", pady=(0, 4))
        stockfish_entry = ttk.Entry(
            main_frame,
            textvariable=self.stockfish_var,
            width=52,
        )
        stockfish_entry.grid(row=0, column=1, sticky="ew", pady=(0, 4))
        ttk.Button(
            main_frame,
            text="Parcourir",
            command=self._browse_stockfish,
        ).grid(row=0, column=2, padx=(8, 0), pady=(0, 4))
        ttk.Label(
            main_frame,
            text="Optionnel. Laisse vide pour installer Stockfish automatiquement.",
        ).grid(row=1, column=1, columnspan=2, sticky="w", pady=(0, 12))

        ttk.Label(main_frame, text="Trait").grid(row=2, column=0, sticky="w", pady=(0, 12))
        side_frame = ttk.Frame(main_frame)
        side_frame.grid(row=2, column=1, columnspan=2, sticky="w", pady=(0, 12))
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
        self.analyze_button.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 12))

        self.progress_bar = ttk.Progressbar(
            main_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
        )
        self.progress_bar.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Label(
            main_frame,
            textvariable=self.progress_text_var,
            width=6,
            anchor="e",
        ).grid(row=4, column=2, sticky="e", pady=(0, 4))

        ttk.Separator(main_frame, orient="horizontal").grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(0, 12),
        )

        ttk.Label(main_frame, text="Statut").grid(row=6, column=0, sticky="nw", pady=(0, 8))
        ttk.Label(
            main_frame,
            textvariable=self.status_var,
            wraplength=520,
            justify="left",
        ).grid(row=6, column=1, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(main_frame, text="FEN").grid(row=7, column=0, sticky="nw", pady=(0, 8))
        ttk.Label(
            main_frame,
            textvariable=self.fen_var,
            wraplength=520,
            justify="left",
        ).grid(row=7, column=1, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(main_frame, text="Meilleur coup").grid(row=8, column=0, sticky="nw", pady=(0, 8))
        ttk.Label(
            main_frame,
            textvariable=self.best_move_var,
            wraplength=520,
            justify="left",
        ).grid(row=8, column=1, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(main_frame, text="Evaluation").grid(row=9, column=0, sticky="nw", pady=(0, 8))
        ttk.Label(
            main_frame,
            textvariable=self.eval_var,
            wraplength=520,
            justify="left",
        ).grid(row=9, column=1, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(main_frame, text="Qualite").grid(row=10, column=0, sticky="nw", pady=(0, 8))
        ttk.Label(
            main_frame,
            textvariable=self.metrics_var,
            wraplength=520,
            justify="left",
        ).grid(row=10, column=1, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(main_frame, text="Moteur").grid(row=11, column=0, sticky="nw", pady=(0, 8))
        ttk.Label(
            main_frame,
            textvariable=self.engine_var,
            wraplength=520,
            justify="left",
        ).grid(row=11, column=1, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(main_frame, text="Inference").grid(row=12, column=0, sticky="nw")
        ttk.Label(
            main_frame,
            textvariable=self.runtime_var,
            wraplength=520,
            justify="left",
        ).grid(row=12, column=1, columnspan=2, sticky="w")

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

    def _set_progress(self, value, text=None):
        bounded_value = max(0.0, min(100.0, float(value)))

        def _apply():
            self.progress_var.set(bounded_value)
            if text is None:
                self.progress_text_var.set(f"{bounded_value:.0f}%")
            else:
                self.progress_text_var.set(text)

        self.root.after(0, _apply)

    def _handle_fen_progress(self, payload):
        progress_value = payload.get("progress_percent")
        if progress_value is None:
            total = max(1, int(payload.get("total") or 1))
            current = max(0, int(payload.get("current") or 0))
            ratio = min(1.0, current / total)
            progress_value = 20.0 + 65.0 * ratio
        message = payload.get("message") or "Analyse du FEN..."
        self._set_status(message)
        self._set_progress(progress_value)

    def _start_analysis(self):
        if self.is_busy:
            return

        self._set_busy(True)
        self.status_var.set("Initialisation de l'analyse...")
        self.progress_var.set(0.0)
        self.progress_text_var.set("0%")
        self.best_move_var.set("-")
        self.eval_var.set("-")
        self.engine_var.set("-")
        if self.predictor is None:
            self.runtime_var.set("-")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _ensure_predictor(self):
        if self.predictor is None:
            self._set_status("Chargement du modele de reconnaissance...")
            self._set_progress(0.0)
            self.predictor = _load_predictor()
            runtime_label = getattr(
                self.predictor,
                "runtime_device_label",
                getattr(self.predictor, "device", "CPU"),
            )
            self.root.after(0, lambda: self.runtime_var.set(str(runtime_label)))
        return self.predictor

    def _run_analysis(self):
        try:
            predictor = self._ensure_predictor()
            self._set_status("Capture de l'ecran en cours...")
            screen_image = capture_full_screen()
            self._set_status("Detection du plateau...")
            analysis = analyze_screen_image_and_suggest_move(
                screen_image,
                stockfish_path=self.stockfish_var.get().strip() or None,
                side_to_move=self.side_to_move_var.get(),
                think_time=0.20,
                engine_elo=DEFAULT_ENGINE_ELO,
                predictor=predictor,
                use_filters=True,
                source_label="full_screen_capture.png",
                progress_callback=self._handle_fen_progress,
                engine_progress_callback=self._set_status,
            )

            engine_message = "Stockfish non configure"
            move_message = "Aucun coup"
            eval_message = "Aucune evaluation"
            final_status = "Analyse terminee."

            fen = analysis["fen"]
            is_reliable = bool(analysis.get("is_reliable"))
            if fen and is_reliable:
                if analysis.get("phase_name") != "exhaustive":
                    self._set_progress(95.0)
                self._set_status("Calcul du meilleur coup...")
                side_label = "Blancs" if analysis["side_to_move"] == "w" else "Noirs"
                move_message = (
                    f"{side_label}: {analysis['best_move_san']} "
                    f"({analysis['best_move_uci']})"
                )
                if analysis["mate_in"] is not None:
                    eval_message = f"Mat en {analysis['mate_in']}"
                elif analysis["score_cp"] is not None:
                    eval_message = f"{analysis['score_cp']} centipions"
                else:
                    eval_message = "Evaluation indisponible"

                engine_path = analysis.get("stockfish_path") or ""
                engine_elo = analysis.get("engine_elo")
                if engine_path and not self.stockfish_var.get().strip():
                    self.root.after(0, lambda: self.stockfish_var.set(engine_path))
                if engine_elo is None:
                    engine_message = engine_path or "Stockfish actif"
                else:
                    engine_message = f"{engine_path} | Elo {engine_elo}"
            elif fen:
                move_message = "Aucun coup: FEN non fiable"
                eval_message = "Verification manuelle requise"
                engine_message = "Moteur non lance"
                final_status = analysis.get(
                    "message",
                    "Extraction du FEN non fiable, aucun coup calcule.",
                )
            else:
                engine_message = "Aucun FEN detecte, moteur non lance"
                final_status = analysis.get(
                    "message",
                    "Aucun FEN detecte, moteur non lance.",
                )

            self.root.after(
                0,
                lambda: self._finish_analysis(
                    analysis=analysis,
                    move_message=move_message,
                    eval_message=eval_message,
                    engine_message=engine_message,
                    final_status=final_status,
                ),
            )
        except Exception as error:
            self.root.after(0, lambda: self._show_error(str(error)))

    def _finish_analysis(
        self,
        analysis,
        move_message,
        eval_message,
        engine_message,
        final_status,
    ):
        fen = (analysis.get("fen") or "").strip()
        if fen and not analysis.get("is_reliable"):
            fen_display = f"A verifier: {fen}"
        else:
            fen_display = fen or "Aucun FEN detecte"

        self.fen_var.set(fen_display)
        self.best_move_var.set(move_message)
        self.eval_var.set(eval_message)
        self.metrics_var.set(
            "phase={phase} | qualite={quality} | confiance={confidence:.3f} | stabilite={stability} | fiable={reliable}".format(
                phase=analysis.get("phase_name") or "-",
                quality=analysis["quality_score"],
                confidence=analysis["avg_confidence"],
                stability=analysis["stability_count"],
                reliable="oui" if analysis.get("is_reliable") else "non",
            )
        )
        self.engine_var.set(engine_message)
        self.status_var.set(final_status)
        self.progress_var.set(100.0)
        self.progress_text_var.set("100%")
        self._set_busy(False)

    def _show_error(self, message):
        self.status_var.set(f"Erreur: {message}")
        self.progress_var.set(0.0)
        self.progress_text_var.set("0%")
        self._set_busy(False)

    def _on_close(self):
        close_cached_engines()
        self.root.destroy()


def launch_desktop_app():
    root = tk.Tk()
    ChessAssistantApp(root)
    root.mainloop()
