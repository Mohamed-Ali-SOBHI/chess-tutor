import os
import textwrap
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from .capture import capture_full_screen
from .constants import DEFAULT_ENGINE_ELO
from .engine import close_cached_engines
from .service import _load_predictor, analyze_screen_image_and_suggest_move


class ChessAssistantApp:
    def __init__(self, root):
        self.root = root
        self.predictor = None
        self.is_busy = False
        self.side_buttons = {}
        self.status_badge_label = None
        self.canvas = None
        self.canvas_window = None
        self.content_frame = None
        self.scrollbar = None
        self.scrollbar_visible = False
        self.dynamic_wrap_targets = []

        self.colors = {
            "window": "#f3f4f6",
            "surface": "#ffffff",
            "surface_alt": "#f8faf9",
            "border": "#e5e7eb",
            "text": "#111111",
            "muted": "#6b7280",
            "accent": "#34c759",
            "accent_pressed": "#248a3d",
            "accent_soft": "#eaf9ee",
            "accent_soft_strong": "#dff5e6",
            "success": "#248a3d",
            "success_soft": "#eaf9ee",
            "warning": "#a16207",
            "warning_soft": "#fff7e8",
            "danger": "#c2410c",
            "danger_soft": "#fff1eb",
            "segment": "#eef2f7",
            "disabled": "#cbd5e1",
            "line": "#edf0f3",
        }

        self.title_font = None
        self.label_font = None
        self.body_font = None
        self.small_font = None
        self.move_font = None
        self.mono_font = None

        self.side_to_move_var = tk.StringVar(value="w")
        self.status_var = tk.StringVar(value="Pret")
        self.status_badge_var = tk.StringVar(value="Pret")
        self.stage_var = tk.StringVar(value="En attente")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="0%")
        self.fen_var = tk.StringVar(value="-")
        self.best_move_var = tk.StringVar(value="-")
        self.eval_var = tk.StringVar(value="-")
        self.metrics_var = tk.StringVar(value="-")
        self.engine_var = tk.StringVar(value="Stockfish auto")
        self.runtime_var = tk.StringVar(value="Modele non charge")
        self.system_var = tk.StringVar(value="Stockfish auto | Modele non charge")
        self.pin_var = tk.StringVar(value="Epinglee")

        self._configure_window()
        self._configure_typography()
        self._configure_styles()
        self._build_layout()
        self._apply_status_message(self.status_var.get())
        self._update_side_toggle_styles()
        self._update_primary_button_state()
        self._update_pin_button()
        self._refresh_system_summary()
        self.root.after_idle(self._refresh_layout)

    def _configure_window(self):
        self.root.title("Chess Tutor")
        self.root.geometry("392x640")
        self.root.minsize(344, 500)
        self.root.configure(bg=self.colors["window"])
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def _pick_font_family(self, candidates):
        try:
            available_fonts = set(tkfont.families(self.root))
        except tk.TclError:
            available_fonts = set()

        for candidate in candidates:
            if candidate in available_fonts:
                return candidate
        return "TkDefaultFont"

    def _configure_typography(self):
        display_family = self._pick_font_family(
            ["SF Pro Display", "Aptos Display", "Segoe UI Variable Display", "Segoe UI"]
        )
        body_family = self._pick_font_family(
            ["SF Pro Text", "Aptos", "Segoe UI Variable Text", "Segoe UI"]
        )
        mono_family = self._pick_font_family(
            ["SF Mono", "Cascadia Mono", "Consolas", "Courier New"]
        )

        self.title_font = (display_family, 15, "bold")
        self.label_font = (body_family, 9, "bold")
        self.body_font = (body_family, 10)
        self.small_font = (body_family, 9)
        self.move_font = (display_family, 14, "bold")
        self.mono_font = (mono_family, 9)

    def _configure_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Compact.Horizontal.TProgressbar",
            background=self.colors["accent"],
            troughcolor=self.colors["segment"],
            bordercolor=self.colors["segment"],
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
            thickness=6,
        )

    def _build_layout(self):
        shell = tk.Frame(self.root, bg=self.colors["window"], padx=10, pady=10)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.grid_rowconfigure(0, weight=1)
        shell.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            shell,
            bg=self.colors["window"],
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(shell, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.content_frame = tk.Frame(self.canvas, bg=self.colors["window"])
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.content_frame.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        self._build_controls(self.content_frame)
        self._build_results(self.content_frame)

    def _build_controls(self, parent):
        card = self._create_card(parent, row=0, pady=(0, 10))
        body = tk.Frame(card, bg=self.colors["surface"], padx=14, pady=14)
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)

        top_row = tk.Frame(body, bg=self.colors["surface"])
        top_row.grid(row=0, column=0, sticky="ew")
        top_row.grid_columnconfigure(0, weight=1)
        top_row.grid_columnconfigure(1, weight=0)
        top_row.grid_columnconfigure(2, weight=0)

        tk.Label(
            top_row,
            text="Trait",
            font=self.label_font,
            fg=self.colors["muted"],
            bg=self.colors["surface"],
        ).grid(row=0, column=0, sticky="w")

        self.pin_button = tk.Button(
            top_row,
            textvariable=self.pin_var,
            command=self._toggle_pin,
            font=self.label_font,
            bd=0,
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            highlightthickness=0,
        )
        self.pin_button.grid(row=0, column=1, sticky="e", padx=(8, 8))

        self.status_badge_label = tk.Label(
            top_row,
            textvariable=self.status_badge_var,
            font=self.label_font,
            padx=10,
            pady=4,
            bg=self.colors["accent_soft"],
            fg=self.colors["accent"],
        )
        self.status_badge_label.grid(row=0, column=2, sticky="e")

        self._build_segmented_control(body).grid(row=1, column=0, sticky="ew", pady=(10, 10))

        self.analyze_button = tk.Button(
            body,
            text="Capturer et recommander",
            command=self._start_analysis,
            font=self.body_font,
            bd=0,
            relief="flat",
            padx=14,
            pady=12,
            cursor="hand2",
            highlightthickness=0,
            activeforeground="#ffffff",
        )
        self.analyze_button.grid(row=2, column=0, sticky="ew")

        stage_row = tk.Frame(body, bg=self.colors["surface"])
        stage_row.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        stage_row.grid_columnconfigure(0, weight=1)

        tk.Label(
            stage_row,
            textvariable=self.stage_var,
            font=self.body_font,
            fg=self.colors["text"],
            bg=self.colors["surface"],
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            stage_row,
            textvariable=self.progress_text_var,
            font=self.small_font,
            fg=self.colors["muted"],
            bg=self.colors["surface"],
        ).grid(row=0, column=1, sticky="e")

        self.progress_bar = ttk.Progressbar(
            body,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
            style="Compact.Horizontal.TProgressbar",
        )
        self.progress_bar.grid(row=4, column=0, sticky="ew", pady=(6, 6))

        status_label = tk.Label(
            body,
            textvariable=self.status_var,
            font=self.small_font,
            fg=self.colors["muted"],
            bg=self.colors["surface"],
            justify="left",
            anchor="w",
        )
        status_label.grid(row=5, column=0, sticky="ew")
        self._register_dynamic_wrap(status_label, padding=72, min_wrap=170)

    def _build_segmented_control(self, parent):
        segment = tk.Frame(
            parent,
            bg=self.colors["segment"],
            padx=4,
            pady=4,
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            bd=0,
        )
        segment.grid_columnconfigure(0, weight=1)
        segment.grid_columnconfigure(1, weight=1)

        for column, (side, label) in enumerate((("w", "Blancs"), ("b", "Noirs"))):
            button = tk.Button(
                segment,
                text=label,
                command=lambda selected_side=side: self._set_side_to_move(selected_side),
                font=self.body_font,
                bd=0,
                relief="flat",
                padx=0,
                pady=9,
                cursor="hand2",
                highlightthickness=0,
            )
            button.grid(row=0, column=column, sticky="ew")
            self.side_buttons[side] = button

        return segment

    def _build_results(self, parent):
        card = self._create_card(parent, row=1)
        body = tk.Frame(card, bg=self.colors["surface"], padx=14, pady=14)
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)

        move_panel = tk.Frame(
            body,
            bg=self.colors["accent_soft"],
            highlightbackground=self.colors["accent_soft_strong"],
            highlightthickness=1,
            bd=0,
            padx=12,
            pady=12,
        )
        move_panel.grid(row=0, column=0, sticky="ew")
        move_panel.grid_columnconfigure(0, weight=1)

        tk.Label(
            move_panel,
            text="Recommandation",
            font=self.label_font,
            fg=self.colors["accent"],
            bg=self.colors["accent_soft"],
        ).grid(row=0, column=0, sticky="w")

        move_label = tk.Label(
            move_panel,
            textvariable=self.best_move_var,
            font=self.move_font,
            fg=self.colors["text"],
            bg=self.colors["accent_soft"],
            justify="left",
            anchor="w",
        )
        move_label.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self._register_dynamic_wrap(move_label, padding=42, min_wrap=210)

        eval_label = tk.Label(
            move_panel,
            textvariable=self.eval_var,
            font=self.body_font,
            fg=self.colors["muted"],
            bg=self.colors["accent_soft"],
            justify="left",
            anchor="w",
        )
        eval_label.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self._register_dynamic_wrap(eval_label, padding=42, min_wrap=190)

        tk.Label(
            body,
            text="FEN",
            font=self.label_font,
            fg=self.colors["muted"],
            bg=self.colors["surface"],
        ).grid(row=1, column=0, sticky="w", pady=(12, 0))

        fen_frame = tk.Frame(
            body,
            bg=self.colors["surface_alt"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            bd=0,
            padx=10,
            pady=10,
        )
        fen_frame.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        fen_frame.grid_columnconfigure(0, weight=1)

        fen_label = tk.Label(
            fen_frame,
            textvariable=self.fen_var,
            font=self.mono_font,
            fg=self.colors["text"],
            bg=self.colors["surface_alt"],
            justify="left",
            anchor="w",
        )
        fen_label.grid(row=0, column=0, sticky="ew")
        self._register_dynamic_wrap(fen_label, padding=42, min_wrap=210)

        tk.Label(
            body,
            text="Details",
            font=self.label_font,
            fg=self.colors["muted"],
            bg=self.colors["surface"],
        ).grid(row=3, column=0, sticky="w", pady=(12, 0))

        details_label = tk.Label(
            body,
            textvariable=self.metrics_var,
            font=self.small_font,
            fg=self.colors["text"],
            bg=self.colors["surface"],
            justify="left",
            anchor="w",
        )
        details_label.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        self._register_dynamic_wrap(details_label, padding=72, min_wrap=180)

        system_label = tk.Label(
            body,
            textvariable=self.system_var,
            font=self.small_font,
            fg=self.colors["muted"],
            bg=self.colors["surface"],
            justify="left",
            anchor="w",
        )
        system_label.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        self._register_dynamic_wrap(system_label, padding=72, min_wrap=170)

    def _create_card(self, parent, row, pady=(0, 0)):
        card = tk.Frame(
            parent,
            bg=self.colors["surface"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            bd=0,
        )
        card.grid(row=row, column=0, sticky="ew", pady=pady)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=1)
        return card

    def _bind_mousewheel(self, _event=None):
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None):
        self.root.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        if not self.scrollbar_visible:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=event.width)
        self._refresh_wraplengths(event.width)
        self.root.after_idle(self._update_scrollbar_visibility)

    def _on_content_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.root.after_idle(self._update_scrollbar_visibility)

    def _register_dynamic_wrap(self, widget, padding, min_wrap=150):
        self.dynamic_wrap_targets.append((widget, padding, min_wrap))

    def _refresh_wraplengths(self, width):
        content_width = max(220, int(width))
        for widget, padding, min_wrap in self.dynamic_wrap_targets:
            widget.configure(wraplength=max(min_wrap, content_width - padding))

    def _update_scrollbar_visibility(self):
        if self.canvas is None or self.scrollbar is None or self.content_frame is None:
            return

        needs_scroll = self.content_frame.winfo_reqheight() > self.canvas.winfo_height() + 4
        if needs_scroll and not self.scrollbar_visible:
            self.scrollbar.grid(row=0, column=1, sticky="ns", padx=(8, 0))
            self.scrollbar_visible = True
            self.root.after_idle(self._refresh_layout)
        elif not needs_scroll and self.scrollbar_visible:
            self.scrollbar.grid_remove()
            self.scrollbar_visible = False
            self.root.after_idle(self._refresh_layout)

    def _refresh_layout(self):
        if self.canvas is None:
            return
        self.canvas.update_idletasks()
        width = self.canvas.winfo_width()
        if width > 1:
            self.canvas.itemconfigure(self.canvas_window, width=width)
            self._refresh_wraplengths(width)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._update_scrollbar_visibility()

    def _status_palette(self, message):
        lowered = (message or "").lower()
        if "erreur" in lowered:
            return "Erreur", self.colors["danger_soft"], self.colors["danger"]
        if "non fiable" in lowered or "verifier" in lowered:
            return "Verifier", self.colors["warning_soft"], self.colors["warning"]
        if "terminee" in lowered:
            return "Pret", self.colors["success_soft"], self.colors["success"]
        if "fiable" in lowered:
            return "Fiable", self.colors["success_soft"], self.colors["success"]
        if "moteur" in lowered or "coup" in lowered:
            return "Moteur", self.colors["accent_soft"], self.colors["accent"]
        if "chargement" in lowered or "initialisation" in lowered:
            return "Chargement", self.colors["accent_soft"], self.colors["accent"]
        if "capture" in lowered:
            return "Capture", self.colors["accent_soft"], self.colors["accent"]
        if "detection" in lowered:
            return "Detection", self.colors["accent_soft"], self.colors["accent"]
        if "analyse" in lowered or "verification" in lowered:
            return "Analyse", self.colors["accent_soft"], self.colors["accent"]
        return "Pret", self.colors["accent_soft"], self.colors["accent"]

    def _apply_status_message(self, message):
        self.status_var.set(message)
        badge_text, badge_bg, badge_fg = self._status_palette(message)
        self.status_badge_var.set(badge_text)
        if self.status_badge_label is not None:
            self.status_badge_label.configure(bg=badge_bg, fg=badge_fg)

    def _format_phase_name(self, phase_name):
        return {
            "quick": "Rapide",
            "deep": "Approfondie",
            "exhaustive": "Exhaustive",
        }.get(phase_name or "", "Analyse")

    def _format_engine_message(self, engine_path, engine_elo):
        if not engine_path:
            return "Stockfish auto"

        engine_name = os.path.basename(engine_path) or "Stockfish"
        if engine_elo is None:
            return engine_name
        return f"{engine_name} | Elo {engine_elo}"

    def _format_display_text(self, value, width):
        text = (value or "").strip()
        if not text:
            return "-"
        return "\n".join(
            textwrap.wrap(
                text,
                width=width,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )

    def _refresh_system_summary(self):
        engine_text = (self.engine_var.get() or "").strip() or "Stockfish auto"
        runtime_text = (self.runtime_var.get() or "").strip() or "Modele non charge"
        self.system_var.set(f"{engine_text} | {runtime_text}")

    def _set_side_to_move(self, selected_side):
        if self.is_busy:
            return
        self.side_to_move_var.set(selected_side)
        self._update_side_toggle_styles()

    def _update_side_toggle_styles(self):
        selected_side = self.side_to_move_var.get()
        for side, button in self.side_buttons.items():
            is_selected = side == selected_side
            button.configure(
                bg=self.colors["accent_soft"] if is_selected else self.colors["segment"],
                fg=self.colors["accent"] if is_selected else self.colors["muted"],
                activebackground=self.colors["accent_soft"] if is_selected else self.colors["segment"],
                activeforeground=self.colors["accent"] if is_selected else self.colors["text"],
                state="disabled" if self.is_busy else "normal",
                cursor="arrow" if self.is_busy else "hand2",
            )

    def _update_primary_button_state(self):
        if self.is_busy:
            self.analyze_button.configure(
                text="Analyse en cours...",
                bg=self.colors["disabled"],
                fg="#ffffff",
                activebackground=self.colors["disabled"],
                state="disabled",
                cursor="arrow",
            )
            return

        self.analyze_button.configure(
            text="Capturer et recommander",
            bg=self.colors["accent"],
            fg="#ffffff",
            activebackground=self.colors["accent_pressed"],
            state="normal",
            cursor="hand2",
        )

    def _toggle_pin(self):
        is_topmost = bool(self.root.attributes("-topmost"))
        self.root.attributes("-topmost", not is_topmost)
        self._update_pin_button()

    def _update_pin_button(self):
        is_topmost = bool(self.root.attributes("-topmost"))
        self.pin_var.set("Epinglee" if is_topmost else "Libre")
        if not hasattr(self, "pin_button"):
            return
        self.pin_button.configure(
            bg=self.colors["accent_soft"] if is_topmost else self.colors["segment"],
            fg=self.colors["accent"] if is_topmost else self.colors["muted"],
            activebackground=self.colors["accent_soft"] if is_topmost else self.colors["segment"],
            activeforeground=self.colors["accent"] if is_topmost else self.colors["text"],
        )

    def _set_busy(self, is_busy):
        self.is_busy = is_busy
        self._update_primary_button_state()
        self._update_side_toggle_styles()
        self._update_pin_button()

    def _set_status(self, message):
        self.root.after(0, lambda current_message=message: self._apply_status_message(current_message))

    def _set_stage(self, stage):
        self.root.after(0, lambda current_stage=stage: self.stage_var.set(current_stage))

    def _set_progress(self, value, text=None):
        bounded_value = max(0.0, min(100.0, float(value)))

        def _apply():
            self.progress_var.set(bounded_value)
            self.progress_text_var.set(text or f"{bounded_value:.0f}%")

        self.root.after(0, _apply)

    def _handle_fen_progress(self, payload):
        progress_value = payload.get("progress_percent")
        if progress_value is None:
            total = max(1, int(payload.get("total") or 1))
            current = max(0, int(payload.get("current") or 0))
            ratio = min(1.0, current / total)
            progress_value = 20.0 + 65.0 * ratio

        phase_name = payload.get("phase_name")
        if phase_name:
            self._set_stage(self._format_phase_name(phase_name))

        self._set_status(payload.get("message") or "Analyse du FEN...")
        self._set_progress(progress_value)

    def _handle_engine_progress(self, message):
        self._set_stage("Calcul moteur")
        self._set_status(message)

    def _start_analysis(self):
        if self.is_busy:
            return

        self._set_busy(True)
        self._apply_status_message("Initialisation de l'analyse...")
        self.stage_var.set("Preparation")
        self.progress_var.set(0.0)
        self.progress_text_var.set("0%")
        self.fen_var.set("-")
        self.best_move_var.set("-")
        self.eval_var.set("-")
        self.metrics_var.set("-")
        self.engine_var.set("Stockfish auto")
        if self.predictor is None:
            self.runtime_var.set("Modele non charge")
        self._refresh_system_summary()
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _ensure_predictor(self):
        if self.predictor is None:
            self._set_stage("Chargement")
            self._set_status("Chargement du modele de reconnaissance...")
            self._set_progress(0.0)
            self.predictor = _load_predictor()
            runtime_label = getattr(
                self.predictor,
                "runtime_device_label",
                getattr(self.predictor, "device", "CPU"),
            )
            self.root.after(0, lambda value=str(runtime_label): self._set_runtime_value(value))
        return self.predictor

    def _set_runtime_value(self, value):
        self.runtime_var.set(value)
        self._refresh_system_summary()

    def _run_analysis(self):
        try:
            predictor = self._ensure_predictor()
            self._set_stage("Capture")
            self._set_status("Capture de l'ecran en cours...")
            screen_image = capture_full_screen()
            self._set_stage("Detection")
            self._set_status("Detection du plateau...")
            analysis = analyze_screen_image_and_suggest_move(
                screen_image,
                stockfish_path=None,
                side_to_move=self.side_to_move_var.get(),
                think_time=0.20,
                engine_elo=DEFAULT_ENGINE_ELO,
                predictor=predictor,
                use_filters=True,
                source_label="full_screen_capture.png",
                progress_callback=self._handle_fen_progress,
                engine_progress_callback=self._handle_engine_progress,
            )

            engine_message = "Stockfish auto"
            move_message = "Aucun coup"
            eval_message = "Aucune evaluation"
            final_status = "Analyse terminee."

            fen = analysis["fen"]
            is_reliable = bool(analysis.get("is_reliable"))
            if fen and is_reliable:
                if analysis.get("phase_name") != "exhaustive":
                    self._set_progress(95.0)
                self._set_stage("Calcul moteur")
                self._set_status("Calcul du meilleur coup...")
                side_label = "Blancs" if analysis["side_to_move"] == "w" else "Noirs"
                move_message = f"{side_label}: {analysis['best_move_san']} ({analysis['best_move_uci']})"

                if analysis["mate_in"] is not None:
                    eval_message = f"Mat en {analysis['mate_in']}"
                elif analysis["score_cp"] is not None:
                    eval_message = f"{analysis['score_cp']} centipions"
                else:
                    eval_message = "Evaluation indisponible"

                engine_message = self._format_engine_message(
                    analysis.get("stockfish_path") or "",
                    analysis.get("engine_elo"),
                )
            elif fen:
                move_message = "Aucun coup: FEN non fiable"
                eval_message = "Verification manuelle requise"
                engine_message = "Moteur non lance"
                final_status = analysis.get(
                    "message",
                    "Extraction du FEN non fiable, aucun coup calcule.",
                )
            else:
                engine_message = "Moteur non lance"
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

        confidence = analysis["avg_confidence"]
        stability = analysis["stability_count"]
        reliability_label = "fiable" if analysis.get("is_reliable") else "a verifier"

        self.fen_var.set(self._format_display_text(fen_display, 34))
        self.best_move_var.set(self._format_display_text(move_message, 26))
        self.eval_var.set(self._format_display_text(eval_message, 34))
        self.metrics_var.set(
            self._format_display_text(
                f"{self._format_phase_name(analysis.get('phase_name'))} | "
                f"Q{analysis['quality_score']} C{confidence:.3f} "
                f"S{stability} {reliability_label}",
                34,
            )
        )
        self.engine_var.set(engine_message)
        self._refresh_system_summary()
        self.stage_var.set("Termine")
        self._apply_status_message(final_status)
        self.progress_var.set(100.0)
        self.progress_text_var.set("100%")
        self._set_busy(False)
        self.root.after_idle(self._refresh_layout)

    def _show_error(self, message):
        self.stage_var.set("Erreur")
        self._apply_status_message(f"Erreur: {message}")
        self.progress_var.set(0.0)
        self.progress_text_var.set("0%")
        self._set_busy(False)
        self.root.after_idle(self._refresh_layout)

    def _on_close(self):
        self._unbind_mousewheel()
        close_cached_engines()
        self.root.destroy()


def launch_desktop_app():
    root = tk.Tk()
    ChessAssistantApp(root)
    root.mainloop()
