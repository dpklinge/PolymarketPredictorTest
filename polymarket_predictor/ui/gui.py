from __future__ import annotations

import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from ..datasets.data import fetch_snapshots, prepare_dataset
from ..ml.pipeline import predict_open_markets, train_models
from ..review.snapshotting import compare_prediction_snapshots, save_prediction_snapshots
from .gui_utils import close_distance_efficacy_frame, prediction_comparison_frame, snapshot_review_summary_frame, training_comparison_frame


FETCH_ORDER_OPTIONS = ["volume24hr", "updatedAt", "volume", "liquidity"]
BOOLEAN_OPTIONS = ["Yes", "No"]
SORT_DIRECTION_OPTIONS = ["Descending", "Ascending"]
MODEL_TYPE_OPTIONS = ["logistic", "boosted_trees", "prior"]
SNAPSHOT_STATUS_OPTIONS = ["All", "Pending", "Success", "Failure"]
EFFICACY_METRIC_OPTIONS = ["accuracy", "log_loss", "brier_score"]
PREDICTION_COLUMN_TOOLTIPS = {
    "market_id": "Internal Polymarket identifier for the market. Useful for matching the same market across model runs.",
    "slug": "Short text identifier used in Polymarket URLs and API responses.",
    "question": "Human-readable market question being priced.",
    "category": "Category inferred from market and event metadata, such as sports or politics.",
    "market_yes_probability": "The current market-implied YES probability from Polymarket itself.",
    "max_abs_edge": "Largest absolute model-vs-market difference across the compared runs for this market. Higher means the models disagree more strongly with the market or with each other.",
}
SNAPSHOT_REVIEW_COLUMN_TOOLTIPS = {
    "snapshot_time": "When the original prediction snapshot was saved.",
    "model_label": "Short label for the model run that produced the prediction.",
    "artifact_dir": "Artifact folder used to generate the original prediction.",
    "market_id": "Internal Polymarket identifier for the market.",
    "slug": "Short market identifier used in URLs and API responses.",
    "question": "Human-readable market question.",
    "category": "Category inferred for the market.",
    "predicted_side": "YES if the model's saved probability was 50% or higher, otherwise NO.",
    "predicted_yes_probability": "Saved YES probability from the model at the time the snapshot was taken.",
    "market_yes_probability_at_snapshot": "Polymarket's YES probability when the snapshot was saved.",
    "stake_cost_at_snapshot": "Cost to buy one share on the model's predicted side at the snapshot odds.",
    "max_profit_at_snapshot": "Best-case profit for one share bought on the predicted side at the snapshot odds.",
    "max_loss_at_snapshot": "Worst-case loss for one share bought on the predicted side at the snapshot odds.",
    "current_market_yes_probability": "Polymarket's current YES probability right now.",
    "current_closed": "Whether Polymarket currently marks this market as closed.",
    "actual_side": "Resolved winning side if the market is finished. Blank means the result is not final yet.",
    "verdict": "Success means the saved prediction picked the correct side, Failure means it picked the wrong side, and Pending means the market is not resolved yet.",
    "realized_payout": "Settlement payout for one share on the predicted side after resolution. Blank while the market is still pending.",
    "realized_pnl": "Realized profit or loss for one share bought on the predicted side at snapshot time. Blank while the market is still pending.",
    "current_edge_vs_snapshot_market": "How much the market's YES probability moved since the snapshot was taken.",
}
SNAPSHOT_SUMMARY_COLUMN_TOOLTIPS = {
    "model_label": "Short label for the model whose snapshot calls are being summarized.",
    "rows": "Total snapshot rows saved for this model in the selected file.",
    "resolved_rows": "How many rows now have a final outcome.",
    "pending_rows": "How many rows are still unresolved.",
    "wins": "Resolved rows where the predicted side matched the final outcome.",
    "losses": "Resolved rows where the prediction was wrong.",
    "win_rate": "Win percentage across resolved rows only.",
    "total_cost_at_snapshot": "Total amount spent if you bought one share on each predicted side at the snapshot odds.",
    "resolved_cost_at_snapshot": "Portion of that spend belonging to rows that are already resolved.",
    "resolved_payout": "Total settlement payout received so far from resolved rows.",
    "resolved_pnl": "Realized profit or loss so far from resolved rows.",
    "avg_resolved_pnl": "Average realized profit or loss per resolved row.",
    "resolved_roi": "Realized profit or loss divided by resolved cost.",
    "pending_max_profit": "Best-case remaining profit if every pending row resolves in the model's favor.",
    "pending_max_loss": "Worst-case remaining loss if every pending row resolves against the model.",
}


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window: tk.Toplevel | None = None
        self.widget.bind("<Enter>", self.show_tip, add="+")
        self.widget.bind("<Leave>", self.hide_tip, add="+")

    def show_tip(self, _event=None) -> None:
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            wraplength=340,
            background="#fff8d6",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
            padx=8,
            pady=6,
        )
        label.pack()

    def hide_tip(self, _event=None) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class TreeHeadingToolTip:
    def __init__(self, tree: ttk.Treeview, descriptions: dict[str, str]) -> None:
        self.tree = tree
        self.descriptions = descriptions
        self.tooltip = ToolTip(tree, "")
        self.tooltip.hide_tip()
        self.active_column = ""
        self.tree.bind("<Motion>", self._on_motion, add="+")
        self.tree.bind("<Leave>", self._on_leave, add="+")

    def _on_motion(self, event) -> None:
        region = self.tree.identify_region(event.x, event.y)
        if region != "heading":
            self._on_leave()
            return
        column_id = self.tree.identify_column(event.x)
        if not column_id:
            self._on_leave()
            return
        index = int(column_id.replace("#", "")) - 1
        columns = self.tree["columns"]
        if index < 0 or index >= len(columns):
            self._on_leave()
            return
        column_name = columns[index]
        description = self.descriptions.get(column_name)
        if not description:
            self._on_leave()
            return
        if self.active_column != column_name:
            self.tooltip.hide_tip()
            self.tooltip.text = description
            self.tooltip.show_tip()
            self.active_column = column_name

    def _on_leave(self, _event=None) -> None:
        self.active_column = ""
        self.tooltip.hide_tip()


class PredictorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Polymarket Predictor GUI")
        self.root.geometry("1460x940")

        self.output_queue: list[tuple[str, str]] = []
        self.last_prediction_frames: dict[str, pd.DataFrame] = {}
        self.tree_sort_state: dict[tuple[str, str], bool] = {}
        self.available_artifact_dirs: list[str] = []

        self._build_styles()
        self._build_layout()
        self._refresh_available_artifact_dirs()
        self._poll_queue()

    def _build_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Section.TLabelframe", padding=10)

    def _build_layout(self) -> None:
        container = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(container, padding=10)
        results = ttk.Frame(container, padding=10)
        container.add(controls, weight=3)
        container.add(results, weight=4)

        notebook = ttk.Notebook(controls)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.fetch_tab = ttk.Frame(notebook)
        self.prepare_tab = ttk.Frame(notebook)
        self.train_tab = ttk.Frame(notebook)
        self.predict_tab = ttk.Frame(notebook)
        self.snapshot_tab = ttk.Frame(notebook)
        notebook.add(self.fetch_tab, text="Fetch")
        notebook.add(self.prepare_tab, text="Prepare")
        notebook.add(self.train_tab, text="Train")
        notebook.add(self.predict_tab, text="Predict")
        notebook.add(self.snapshot_tab, text="Snapshot Review")

        self._build_fetch_tab()
        self._build_prepare_tab()
        self._build_train_tab()
        self._build_predict_tab()
        self._build_snapshot_tab()

        header = ttk.Label(results, text="Results", style="Header.TLabel")
        header.pack(anchor="w")
        ToolTip(header, "Output views for workflow logs, model metric comparisons, and side-by-side prediction comparisons.")

        result_notebook = ttk.Notebook(results)
        result_notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.log_text = tk.Text(result_notebook, wrap="word", font=("Consolas", 10))
        result_notebook.add(self.log_text, text="Log")
        ToolTip(
            self.log_text,
            "Running history and status messages. Use this to see what the app is doing, where files were written, and any errors that need attention.",
        )

        self.metrics_tree = self._create_tree_tab(
            result_notebook,
            "Model Comparison",
            "Comparison of saved training runs. Lower validation log loss, Brier score, and calibration error are generally better.",
        )
        self.predictions_tree = self._create_tree_tab(
            result_notebook,
            "Prediction Comparison",
            "Merged live prediction table. Each model run gets its own probability and edge columns so you can see where models agree or disagree.",
        )
        self.snapshot_review_tree = self._create_tree_tab(
            result_notebook,
            "Snapshot Review",
            "Evaluation of saved prediction snapshots versus the latest Polymarket market state.",
        )
        self.snapshot_summary_tree = self._create_tree_tab(
            result_notebook,
            "Snapshot Summary",
            "Per-model profit and loss summary based on buying one share on each predicted side at the snapshot odds.",
        )
        self.efficacy_canvas = self._create_canvas_tab(
            result_notebook,
            "Close-Distance Graph",
            "Graph of model validation efficacy as a function of how far the prediction was from market close.",
        )

    def _create_tree_tab(self, notebook: ttk.Notebook, title: str, tooltip_text: str) -> ttk.Treeview:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        tree = ttk.Treeview(frame, show="headings")
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        ToolTip(tree, tooltip_text)
        return tree

    def _create_canvas_tab(self, notebook: ttk.Notebook, title: str, tooltip_text: str) -> tk.Canvas:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        canvas = tk.Canvas(frame, background="white", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        ToolTip(canvas, tooltip_text)
        return canvas

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        label: str,
        row: int,
        *,
        default: str = "",
        width: int = 40,
        help_text: str,
    ) -> tk.StringVar:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        value = tk.StringVar(value=default)
        entry = ttk.Entry(parent, textvariable=value, width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)
        ToolTip(label_widget, help_text)
        ToolTip(entry, help_text)
        return value

    def _add_labeled_combobox(
        self,
        parent: ttk.Frame,
        label: str,
        row: int,
        *,
        values: list[str],
        default: str,
        help_text: str,
        width: int = 38,
    ) -> tk.StringVar:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        value = tk.StringVar(value=default)
        combo = ttk.Combobox(parent, textvariable=value, values=values, width=width, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)
        ToolTip(label_widget, help_text)
        ToolTip(combo, help_text)
        return value

    def _add_labeled_listbox(
        self,
        parent: ttk.Frame,
        label: str,
        row: int,
        *,
        values: list[str],
        selected: list[str],
        help_text: str,
        height: int = 4,
    ) -> tk.Listbox:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)
        listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, exportselection=False, height=height)
        listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.configure(yscrollcommand=scrollbar.set)
        for index, item in enumerate(values):
            listbox.insert(tk.END, item)
            if item in selected:
                listbox.selection_set(index)
        ToolTip(label_widget, help_text)
        ToolTip(listbox, help_text)
        return listbox

    def _set_listbox_values(self, listbox: tk.Listbox, values: list[str], *, selected: list[str] | None = None) -> None:
        selected_set = set(selected or [])
        listbox.delete(0, tk.END)
        for index, item in enumerate(values):
            listbox.insert(tk.END, item)
            if item in selected_set:
                listbox.selection_set(index)

    def _refresh_available_artifact_dirs(self) -> None:
        discovered: list[str] = []
        for bundle_path in Path.cwd().glob("artifacts/**/model_bundle.json"):
            parent = bundle_path.parent.resolve()
            if parent == (Path.cwd() / "artifacts").resolve():
                continue
            discovered.append(str(parent))
        self.available_artifact_dirs = sorted(set(discovered))
        if hasattr(self, "metrics_artifact_list"):
            previous_metrics = self._selected_metrics_artifact_dirs()
            selected_metrics = [item for item in previous_metrics if item in self.available_artifact_dirs]
            if not selected_metrics:
                selected_metrics = self.available_artifact_dirs[: min(2, len(self.available_artifact_dirs))]
            self._set_listbox_values(self.metrics_artifact_list, self.available_artifact_dirs, selected=selected_metrics)
        if hasattr(self, "predict_artifact_list"):
            previously_selected = self._selected_prediction_artifact_dirs()
            selected = [item for item in previously_selected if item in self.available_artifact_dirs]
            if not selected:
                selected = self.available_artifact_dirs[: min(2, len(self.available_artifact_dirs))]
            self._set_listbox_values(self.predict_artifact_list, self.available_artifact_dirs, selected=selected)

    def _selected_prediction_artifact_dirs(self) -> list[str]:
        if not hasattr(self, "predict_artifact_list"):
            return []
        return [self.predict_artifact_list.get(index) for index in self.predict_artifact_list.curselection()]

    def _selected_metrics_artifact_dirs(self) -> list[str]:
        if not hasattr(self, "metrics_artifact_list"):
            return []
        return [self.metrics_artifact_list.get(index) for index in self.metrics_artifact_list.curselection()]

    def _browse_file(self, variable: tk.StringVar, *, save: bool = False, multiple: bool = False) -> None:
        initial_dir = self._initial_dir_for_value(variable.get(), fallback=Path.cwd() / "artifacts")
        if multiple:
            paths = filedialog.askopenfilenames(initialdir=str(initial_dir))
            if paths:
                variable.set(";".join(paths))
            return
        if save:
            path = filedialog.asksaveasfilename(initialdir=str(initial_dir))
        else:
            path = filedialog.askopenfilename(initialdir=str(initial_dir))
        if path:
            variable.set(path)

    def _browse_directory(self, variable: tk.StringVar, *, fallback: str | Path | None = None) -> None:
        initial_dir = self._initial_dir_for_value(variable.get(), fallback=fallback or (Path.cwd() / "artifacts"))
        path = filedialog.askdirectory(initialdir=str(initial_dir))
        if path:
            variable.set(path)

    @staticmethod
    def _initial_dir_for_value(value: str, *, fallback: str | Path) -> Path:
        fallback_path = Path(fallback)
        if fallback_path.suffix:
            fallback_path = fallback_path.parent
        fallback_path = fallback_path if fallback_path.exists() else Path.cwd()

        parts = [item.strip() for item in value.split(";") if item.strip()]
        if not parts:
            return fallback_path

        candidate = Path(parts[0])
        if candidate.exists():
            return candidate if candidate.is_dir() else candidate.parent
        if candidate.suffix:
            parent = candidate.parent
            return parent if parent.exists() else fallback_path
        return candidate if candidate.exists() else fallback_path

    def _build_fetch_tab(self) -> None:
        frame = ttk.LabelFrame(self.fetch_tab, text="Fetch Snapshot", style="Section.TLabelframe")
        frame.pack(fill=tk.X, expand=False, padx=4, pady=4)

        self.fetch_output_var = self._add_labeled_entry(
            frame,
            "Output JSONL",
            0,
            default="artifacts/raw_open_history.jsonl",
            help_text="Where fetched market snapshots will be saved. JSONL means one market snapshot per line so the file can grow over time.",
        )
        browse = ttk.Button(frame, text="Browse", command=lambda: self._browse_file(self.fetch_output_var, save=True))
        browse.grid(row=0, column=2, padx=(8, 0))
        ToolTip(browse, "Choose where the fetched snapshot file should be written.")

        self.fetch_max_pages_var = self._add_labeled_entry(
            frame,
            "Max Pages",
            1,
            default="5",
            help_text="How many API pages to fetch. Higher values pull more markets but take longer and create larger files.",
        )
        self.fetch_page_size_var = self._add_labeled_entry(
            frame,
            "Page Size",
            2,
            default="200",
            help_text="How many markets to request per page from the API.",
        )
        self.fetch_order_var = self._add_labeled_combobox(
            frame,
            "Order",
            3,
            values=FETCH_ORDER_OPTIONS,
            default="volume24hr",
            help_text="Which field controls the fetch ranking. For example, `volume24hr` favors active markets, while `updatedAt` favors recently changed markets.",
        )
        self.fetch_market_type_var = self._add_labeled_combobox(
            frame,
            "Market Type",
            4,
            values=["Open markets", "Closed markets"],
            default="Open markets",
            help_text="Choose whether to fetch currently tradable markets or already closed markets used more often for training data.",
        )
        self.fetch_direction_var = self._add_labeled_combobox(
            frame,
            "Sort Direction",
            5,
            values=SORT_DIRECTION_OPTIONS,
            default="Descending",
            help_text="Descending usually gives the largest or newest items first. Ascending reverses the order.",
        )
        self.fetch_append_var = self._add_labeled_combobox(
            frame,
            "Append To File",
            6,
            values=BOOLEAN_OPTIONS,
            default="Yes",
            help_text="`Yes` adds new snapshots to the end of the existing file, which is useful for building market history over time. `No` replaces the file.",
        )
        run = ttk.Button(frame, text="Run Fetch", command=self.run_fetch)
        run.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ToolTip(run, "Fetch raw Polymarket data using the selected options and save it to the chosen JSONL file.")

    def _build_prepare_tab(self) -> None:
        frame = ttk.LabelFrame(self.prepare_tab, text="Prepare Dataset", style="Section.TLabelframe")
        frame.pack(fill=tk.X, expand=False, padx=4, pady=4)

        self.prepare_input_var = self._add_labeled_entry(
            frame,
            "Input JSONL Files",
            0,
            default="artifacts/raw_closed_small.jsonl",
            help_text="One or more raw snapshot files separated by semicolons. These are turned into a flat CSV dataset for training.",
        )
        browse = ttk.Button(frame, text="Browse", command=lambda: self._browse_file(self.prepare_input_var, multiple=True))
        browse.grid(row=0, column=2, padx=(8, 0))
        ToolTip(browse, "Pick one or more JSONL snapshot files to combine into a training dataset.")

        self.prepare_output_var = self._add_labeled_entry(
            frame,
            "Output CSV",
            1,
            default="artifacts/prepared_training_data.csv",
            help_text="Where the prepared training table should be saved. This CSV is what the training step reads.",
        )
        browse_out = ttk.Button(frame, text="Browse", command=lambda: self._browse_file(self.prepare_output_var, save=True))
        browse_out.grid(row=1, column=2, padx=(8, 0))
        ToolTip(browse_out, "Choose the destination CSV for the prepared dataset.")

        run = ttk.Button(frame, text="Run Prepare", command=self.run_prepare)
        run.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ToolTip(run, "Convert raw snapshots into model-ready rows with engineered features.")

    def _build_train_tab(self) -> None:
        frame = ttk.LabelFrame(self.train_tab, text="Train Models", style="Section.TLabelframe")
        frame.pack(fill=tk.X, expand=False, padx=4, pady=4)

        self.train_dataset_var = self._add_labeled_entry(
            frame,
            "Dataset CSV",
            0,
            default="artifacts/prepared_training_data.csv",
            help_text="Prepared feature table used for model training and validation.",
        )
        browse_dataset = ttk.Button(frame, text="Browse", command=lambda: self._browse_file(self.train_dataset_var))
        browse_dataset.grid(row=0, column=2, padx=(8, 0))
        ToolTip(browse_dataset, "Choose the prepared CSV that should be used for training.")

        self.train_artifact_base_var = self._add_labeled_entry(
            frame,
            "Artifact Base Dir",
            1,
            default="artifacts/gui_runs",
            help_text="Folder where each trained model run will be stored. The GUI creates one subfolder per selected model type.",
        )
        browse_dir = ttk.Button(
            frame,
            text="Browse",
            command=lambda: self._browse_directory(self.train_artifact_base_var, fallback=Path.cwd() / "artifacts" / "gui_runs"),
        )
        browse_dir.grid(row=1, column=2, padx=(8, 0))
        ToolTip(browse_dir, "Choose the parent folder for trained model artifacts and metrics.")

        self.train_model_types_list = self._add_labeled_listbox(
            frame,
            "Model Types",
            2,
            values=MODEL_TYPE_OPTIONS,
            selected=["logistic", "boosted_trees"],
            help_text="Pick one or more model families to train. `logistic` is the strongest baseline here, `boosted_trees` is a nonlinear alternative, and `prior` is a minimal baseline.",
        )
        self.train_validation_var = self._add_labeled_entry(
            frame,
            "Validation Fraction",
            3,
            default="0.2",
            help_text="Share of the newest rows reserved for validation. This is used for out-of-sample metrics and probability calibration.",
        )
        self.train_min_category_var = self._add_labeled_entry(
            frame,
            "Min Category Samples",
            4,
            default="40",
            help_text="Minimum number of training rows needed before the system trains a category-specific model instead of only using the global model.",
        )
        self.train_edge_threshold_var = self._add_labeled_entry(
            frame,
            "Edge Threshold",
            5,
            default="0.05",
            help_text="Minimum model-vs-market probability gap required before the backtest counts a trade.",
        )
        run = ttk.Button(frame, text="Train Selected Models", command=self.run_train)
        run.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ToolTip(run, "Train all selected model families, save their bundles and metrics, and refresh the comparison table.")

        compare_frame = ttk.LabelFrame(self.train_tab, text="Compare Existing Runs", style="Section.TLabelframe")
        compare_frame.pack(fill=tk.X, expand=False, padx=4, pady=4)
        self.metrics_artifact_list = self._add_labeled_listbox(
            compare_frame,
            "Existing Runs",
            0,
            values=[],
            selected=[],
            help_text="Select one or more existing trained model runs. Only artifact folders that actually contain saved model bundles are listed here.",
            height=5,
        )
        refresh_runs = ttk.Button(compare_frame, text="Refresh Runs", command=self.refresh_available_runs)
        refresh_runs.grid(row=0, column=2, padx=(8, 0), sticky="n")
        ToolTip(refresh_runs, "Rescan the artifacts folder for existing trained model runs.")
        self.efficacy_metric_var = self._add_labeled_combobox(
            compare_frame,
            "Graph Metric",
            1,
            values=EFFICACY_METRIC_OPTIONS,
            default="accuracy",
            help_text="Which validation metric to plot against time-to-close. Accuracy is easier to read, while log loss and Brier score emphasize probability quality.",
        )
        run_compare = ttk.Button(compare_frame, text="Load Metrics Comparison", command=self.load_metrics_comparison)
        run_compare.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ToolTip(run_compare, "Load and compare metric summaries from the listed model artifact folders.")
        run_graph = ttk.Button(compare_frame, text="Load Close-Distance Graph", command=self.load_close_distance_graph)
        run_graph.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ToolTip(run_graph, "Plot model validation performance versus how many hours before close each training row was anchored.")

    def _build_predict_tab(self) -> None:
        frame = ttk.LabelFrame(self.predict_tab, text="Compare Predictions", style="Section.TLabelframe")
        frame.pack(fill=tk.X, expand=False, padx=4, pady=4)

        self.predict_artifact_list = self._add_labeled_listbox(
            frame,
            "Trained Models",
            0,
            values=[],
            selected=[],
            help_text="Select one or more trained model artifact folders discovered under the artifacts directory. These runs will be scored side by side on the same live markets.",
            height=5,
        )
        refresh_models = ttk.Button(frame, text="Refresh Models", command=self.refresh_prediction_artifacts)
        refresh_models.grid(row=0, column=2, padx=(8, 0), sticky="n")
        ToolTip(refresh_models, "Rescan the artifacts folder for trained model runs and refresh the selectable list.")
        self.predict_history_var = self._add_labeled_entry(
            frame,
            "History JSONL Files (; separated)",
            1,
            default="artifacts/raw_open_history.jsonl",
            help_text="Optional prior open-market snapshots. If provided, live predictions can use temporal change features rather than only the current market state.",
        )
        browse = ttk.Button(frame, text="Browse", command=lambda: self._browse_file(self.predict_history_var, multiple=True))
        browse.grid(row=1, column=2, padx=(8, 0))
        ToolTip(browse, "Choose one or more snapshot-history files to supply live temporal context.")

        self.predict_max_pages_var = self._add_labeled_entry(
            frame,
            "Max Pages",
            2,
            default="1",
            help_text="How many live API pages to score. More pages means more markets in the comparison.",
        )
        self.predict_page_size_var = self._add_labeled_entry(
            frame,
            "Page Size",
            3,
            default="20",
            help_text="How many markets to request per page during prediction.",
        )
        self.predict_limit_var = self._add_labeled_entry(
            frame,
            "Limit",
            4,
            default="25",
            help_text="Maximum number of rows shown in the merged prediction comparison table after sorting by largest disagreement or edge.",
        )
        self.predict_category_var = self._add_labeled_entry(
            frame,
            "Category Filter",
            5,
            default="",
            help_text="Optional category name such as `sports` or `politics`. Leave blank to score all fetched markets.",
        )
        run = ttk.Button(frame, text="Run Prediction Comparison", command=self.run_prediction_comparison)
        run.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ToolTip(run, "Fetch live markets, score them with each selected model run, and merge the outputs into one comparison table.")

        self.snapshot_output_var = self._add_labeled_entry(
            frame,
            "Snapshot Output Folder",
            7,
            default="artifacts/prediction_snapshots",
            help_text="Base folder where prediction snapshots should be stored. The app will create a date folder and timestamped CSV file automatically.",
        )
        browse_snapshot = ttk.Button(
            frame,
            text="Browse",
            command=lambda: self._browse_directory(
                self.snapshot_output_var,
                fallback=Path.cwd() / "artifacts" / "prediction_snapshots",
            ),
        )
        browse_snapshot.grid(row=7, column=2, padx=(8, 0))
        ToolTip(browse_snapshot, "Choose the base folder for saved prediction snapshots.")
        save_snapshot = ttk.Button(frame, text="Save Current Predictions Snapshot", command=self.save_current_prediction_snapshot)
        save_snapshot.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ToolTip(save_snapshot, "Save the most recently generated prediction frames so they can be reviewed later against the eventual market outcome.")

    def _build_snapshot_tab(self) -> None:
        frame = ttk.LabelFrame(self.snapshot_tab, text="Review Saved Prediction Snapshots", style="Section.TLabelframe")
        frame.pack(fill=tk.X, expand=False, padx=4, pady=4)

        self.review_snapshot_input_var = self._add_labeled_entry(
            frame,
            "Snapshot CSV",
            0,
            default="artifacts/prediction_snapshots",
            help_text="CSV file containing saved model predictions from an earlier point in time.",
        )
        browse = ttk.Button(frame, text="Browse", command=lambda: self._browse_file(self.review_snapshot_input_var))
        browse.grid(row=0, column=2, padx=(8, 0))
        ToolTip(browse, "Choose a saved prediction snapshot file to review.")

        self.review_status_var = self._add_labeled_combobox(
            frame,
            "Status Filter",
            1,
            values=SNAPSHOT_STATUS_OPTIONS,
            default="All",
            help_text="Filter reviewed rows to only pending, successful, or failed predictions if desired.",
        )
        self.review_limit_var = self._add_labeled_entry(
            frame,
            "Limit",
            2,
            default="",
            help_text="Maximum number of reviewed snapshot rows to show in the table. Leave blank to show all rows.",
        )
        run = ttk.Button(frame, text="Compare Snapshot To Current Market", command=self.run_snapshot_review)
        run.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ToolTip(run, "Load the snapshot file, fetch the current market state from Polymarket, and label each prediction as Success, Failure, or Pending.")

    def _poll_queue(self) -> None:
        while self.output_queue:
            kind, message = self.output_queue.pop(0)
            if kind == "log":
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
            elif kind == "error":
                self.log_text.insert(tk.END, "[error] " + message + "\n")
                self.log_text.see(tk.END)
                messagebox.showerror("Polymarket Predictor GUI", message)
        self.root.after(150, self._poll_queue)

    def _run_background(self, description: str, callback) -> None:
        def worker() -> None:
            try:
                self.output_queue.append(("log", f"{description}..."))
                callback()
                self.output_queue.append(("log", f"{description} completed."))
            except Exception as exc:  # pragma: no cover
                details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                self.output_queue.append(("error", details))

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _combo_is_yes(value: str) -> bool:
        return value.strip().lower() == "yes"

    @staticmethod
    def _combo_is_ascending(value: str) -> bool:
        return value.strip().lower() == "ascending"

    def _selected_model_types(self) -> list[str]:
        return [self.train_model_types_list.get(index) for index in self.train_model_types_list.curselection()]

    def refresh_prediction_artifacts(self) -> None:
        self._refresh_available_artifact_dirs()
        self.output_queue.append(("log", f"Discovered {len(self.available_artifact_dirs)} trained model artifact folders."))

    def refresh_available_runs(self) -> None:
        self._refresh_available_artifact_dirs()
        self.output_queue.append(("log", f"Discovered {len(self.available_artifact_dirs)} existing trained runs."))

    def run_fetch(self) -> None:
        def callback() -> None:
            path = fetch_snapshots(
                output_path=self.fetch_output_var.get(),
                closed=self.fetch_market_type_var.get() == "Closed markets",
                max_pages=int(self.fetch_max_pages_var.get()),
                page_size=int(self.fetch_page_size_var.get()),
                order=self.fetch_order_var.get().strip() or None,
                ascending=self._combo_is_ascending(self.fetch_direction_var.get()),
                append=self._combo_is_yes(self.fetch_append_var.get()),
            )
            self.output_queue.append(("log", f"Saved snapshot file to {path}"))

        self._run_background("Fetch", callback)

    def run_prepare(self) -> None:
        def callback() -> None:
            inputs = [item.strip() for item in self.prepare_input_var.get().split(";") if item.strip()]
            path = prepare_dataset(snapshot_paths=inputs, output_path=self.prepare_output_var.get())
            self.output_queue.append(("log", f"Prepared dataset saved to {path}"))

        self._run_background("Prepare dataset", callback)

    def run_train(self) -> None:
        def callback() -> None:
            base_dir = Path(self.train_artifact_base_var.get())
            dataset = self.train_dataset_var.get().strip() or None
            model_types = self._selected_model_types()
            if not model_types:
                raise ValueError("Select at least one model type to train.")
            created_dirs: list[str] = []
            for model_type in model_types:
                artifact_dir = base_dir / model_type
                result = train_models(
                    artifact_dir=artifact_dir,
                    dataset_path=dataset,
                    min_category_samples=int(self.train_min_category_var.get()),
                    validation_fraction=float(self.train_validation_var.get()),
                    model_type=model_type,
                    edge_threshold=float(self.train_edge_threshold_var.get()),
                )
                created_dirs.append(str(artifact_dir))
                self.output_queue.append(("log", f"{model_type}: validation log loss={result.metrics['global_validation']['log_loss']:.6f}"))
            self._refresh_available_artifact_dirs()
            self._set_listbox_values(self.metrics_artifact_list, self.available_artifact_dirs, selected=created_dirs)
            self._set_listbox_values(self.predict_artifact_list, self.available_artifact_dirs, selected=created_dirs)
            self.load_metrics_comparison(from_background=True)

        self._run_background("Train models", callback)

    def load_metrics_comparison(self, *, from_background: bool = False) -> None:
        def callback() -> None:
            dirs = self._selected_metrics_artifact_dirs()
            if not dirs:
                raise ValueError("Select at least one existing run in the Train tab.")
            frame = training_comparison_frame(dirs)
            self._populate_tree(self.metrics_tree, frame)
            self.output_queue.append(("log", f"Loaded training comparison for {len(frame)} runs."))

        if from_background:
            callback()
        else:
            self._run_background("Load metrics comparison", callback)

    def load_close_distance_graph(self) -> None:
        def callback() -> None:
            dirs = self._selected_metrics_artifact_dirs()
            if not dirs:
                raise ValueError("Select at least one existing run in the Train tab.")
            dataset_path = self.train_dataset_var.get().strip()
            if not dataset_path:
                raise ValueError("Choose a dataset CSV in the Train tab before loading the close-distance graph.")
            frame = close_distance_efficacy_frame(dirs, dataset_path)
            self._draw_efficacy_chart(frame, metric=self.efficacy_metric_var.get())
            self.output_queue.append(("log", f"Loaded close-distance graph for {len(dirs)} runs.")) 

        self._run_background("Load close-distance graph", callback)

    def run_prediction_comparison(self) -> None:
        def callback() -> None:
            artifact_dirs = self._selected_prediction_artifact_dirs()
            if not artifact_dirs:
                raise ValueError("Select at least one trained model run in the Predict tab.")
            history_paths = [item.strip() for item in self.predict_history_var.get().split(";") if item.strip()]
            category = self.predict_category_var.get().strip() or None
            frames: dict[str, pd.DataFrame] = {}
            for artifact_dir in artifact_dirs:
                label = Path(artifact_dir).name or artifact_dir
                frame = predict_open_markets(
                    artifact_dir=artifact_dir,
                    max_pages=int(self.predict_max_pages_var.get()),
                    page_size=int(self.predict_page_size_var.get()),
                    limit=int(self.predict_limit_var.get()),
                    category_filter=category,
                    history_snapshot_paths=history_paths or None,
                )
                frames[label] = frame
                self.output_queue.append(("log", f"{label}: generated {len(frame)} predictions"))
            self.last_prediction_frames = frames
            comparison = prediction_comparison_frame(frames)
            self._populate_tree(self.predictions_tree, comparison)
            self.output_queue.append(("log", f"Loaded prediction comparison for {len(frames)} runs."))

        self._run_background("Prediction comparison", callback)

    def save_current_prediction_snapshot(self) -> None:
        def callback() -> None:
            if not self.last_prediction_frames:
                raise RuntimeError("Run a prediction comparison first so there is something to snapshot.")
            artifact_dirs = self._selected_prediction_artifact_dirs()
            artifact_lookup = {(Path(path).name or path): path for path in artifact_dirs}
            result = save_prediction_snapshots(
                self.last_prediction_frames,
                output_path=self.snapshot_output_var.get(),
                append=True,
                artifact_dir_lookup=artifact_lookup,
            )
            self.review_snapshot_input_var.set(str(result.output_path))
            self.output_queue.append(("log", f"Saved {result.records_written} prediction snapshot rows to {result.output_path}"))

        self._run_background("Save prediction snapshot", callback)

    def run_snapshot_review(self) -> None:
        def callback() -> None:
            limit_text = self.review_limit_var.get().strip()
            frame = compare_prediction_snapshots(
                self.review_snapshot_input_var.get(),
                limit=int(limit_text) if limit_text else None,
                status_filter=self.review_status_var.get(),
            )
            self._populate_tree(self.snapshot_review_tree, frame)
            self._populate_tree(self.snapshot_summary_tree, snapshot_review_summary_frame(frame))
            self.output_queue.append(("log", f"Loaded snapshot review with {len(frame)} rows."))

        self._run_background("Snapshot review", callback)

    def _populate_tree(self, tree: ttk.Treeview, frame: pd.DataFrame) -> None:
        columns = list(frame.columns)
        tree.delete(*tree.get_children())
        tree["columns"] = columns
        for column in columns:
            tree.heading(column, text=column, command=lambda c=column, t=tree: self._sort_tree_by_column(t, c))
            tree.column(column, width=150, anchor="w")
        for _, row in frame.iterrows():
            values = [self._format_cell(value) for value in row.tolist()]
            tree.insert("", tk.END, values=values)
        if tree is self.predictions_tree:
            descriptions = self._prediction_column_descriptions(columns)
            TreeHeadingToolTip(tree, descriptions)
        elif tree is self.snapshot_review_tree:
            TreeHeadingToolTip(tree, dict(SNAPSHOT_REVIEW_COLUMN_TOOLTIPS))
        elif tree is self.snapshot_summary_tree:
            TreeHeadingToolTip(tree, dict(SNAPSHOT_SUMMARY_COLUMN_TOOLTIPS))

    @staticmethod
    def _format_cell(value) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return "" if pd.isna(value) else str(value)

    @staticmethod
    def _prediction_column_descriptions(columns: list[str]) -> dict[str, str]:
        descriptions = dict(PREDICTION_COLUMN_TOOLTIPS)
        for column in columns:
            if column.endswith("_predicted_yes_probability"):
                model_name = column[: -len("_predicted_yes_probability")]
                descriptions[column] = (
                    f"Predicted YES probability from the `{model_name}` run. "
                    "This is the model's estimate after feature engineering, blending, and calibration."
                )
            elif column.endswith("_edge"):
                model_name = column[: -len("_edge")]
                descriptions[column] = (
                    f"Difference between the `{model_name}` prediction and the market YES probability. "
                    "Positive means the model is more bullish on YES than the market."
                )
            elif column.endswith("_model_scope"):
                model_name = column[: -len("_model_scope")]
                descriptions[column] = (
                    f"Which scoring path the `{model_name}` run used for this row, such as a global model or a blended category-specific model."
                )
        return descriptions

    def _sort_tree_by_column(self, tree: ttk.Treeview, column: str) -> None:
        sort_key = (str(tree), column)
        descending = self.tree_sort_state.get(sort_key, False)

        items: list[tuple[tuple[int, object], str]] = []
        columns = list(tree["columns"])
        column_index = columns.index(column)
        for item_id in tree.get_children(""):
            raw_value = tree.item(item_id, "values")[column_index]
            items.append((self._sortable_tree_value(raw_value), item_id))

        items.sort(key=lambda pair: pair[0], reverse=descending)
        for position, (_, item_id) in enumerate(items):
            tree.move(item_id, "", position)

        self.tree_sort_state[sort_key] = not descending

    @staticmethod
    def _sortable_tree_value(value: object) -> tuple[int, object]:
        text = str(value).strip()
        if text == "":
            return (2, "")
        try:
            return (0, float(text))
        except ValueError:
            return (1, text.lower())

    def _draw_efficacy_chart(self, frame: pd.DataFrame, *, metric: str) -> None:
        canvas = self.efficacy_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 900)
        height = max(canvas.winfo_height(), 420)
        if frame.empty:
            canvas.create_text(width / 2, height / 2, text="No close-distance data available for the selected runs and dataset.", font=("Segoe UI", 12))
            return

        margin_left = 80
        margin_right = 180
        margin_top = 40
        margin_bottom = 70
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom

        metric_values = frame[metric].astype(float)
        x_values = frame["lead_hours_midpoint"].astype(float)
        y_min = float(metric_values.min())
        y_max = float(metric_values.max())
        if abs(y_max - y_min) < 1e-9:
            y_min -= 0.01
            y_max += 0.01

        x_min = float(x_values.min())
        x_max = float(x_values.max())
        if abs(x_max - x_min) < 1e-9:
            x_min = max(0.0, x_min - 1.0)
            x_max += 1.0

        def x_to_canvas(value: float) -> float:
            return margin_left + ((value - x_min) / max(x_max - x_min, 1e-9)) * plot_width

        def y_to_canvas(value: float) -> float:
            return margin_top + plot_height - ((value - y_min) / max(y_max - y_min, 1e-9)) * plot_height

        canvas.create_text(width / 2, 18, text=f"Model efficacy vs hours before close ({metric})", font=("Segoe UI", 12, "bold"))
        canvas.create_line(margin_left, margin_top, margin_left, margin_top + plot_height, fill="#444")
        canvas.create_line(margin_left, margin_top + plot_height, margin_left + plot_width, margin_top + plot_height, fill="#444")

        for tick in range(5):
            ratio = tick / 4
            y_value = y_min + (y_max - y_min) * ratio
            y_pos = y_to_canvas(y_value)
            canvas.create_line(margin_left - 6, y_pos, margin_left, y_pos, fill="#666")
            canvas.create_text(margin_left - 10, y_pos, text=f"{y_value:.3f}", anchor="e", font=("Segoe UI", 9))

        unique_x = sorted(frame[["distance_label", "lead_hours_midpoint"]].drop_duplicates().to_dict("records"), key=lambda item: item["lead_hours_midpoint"])
        for point in unique_x:
            x_pos = x_to_canvas(float(point["lead_hours_midpoint"]))
            canvas.create_line(x_pos, margin_top + plot_height, x_pos, margin_top + plot_height + 6, fill="#666")
            canvas.create_text(x_pos, margin_top + plot_height + 22, text=point["distance_label"], anchor="n", font=("Segoe UI", 9))

        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]
        for index, (run_label, group) in enumerate(frame.groupby("run_label")):
            color = colors[index % len(colors)]
            ordered = group.sort_values(by="lead_hours_midpoint")
            points: list[float] = []
            for row in ordered.itertuples(index=False):
                x_pos = x_to_canvas(float(row.lead_hours_midpoint))
                y_pos = y_to_canvas(float(getattr(row, metric)))
                points.extend([x_pos, y_pos])
                canvas.create_oval(x_pos - 3, y_pos - 3, x_pos + 3, y_pos + 3, fill=color, outline=color)
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2, smooth=True)

            legend_x = margin_left + plot_width + 20
            legend_y = margin_top + 20 + (index * 22)
            canvas.create_line(legend_x, legend_y, legend_x + 18, legend_y, fill=color, width=3)
            canvas.create_text(legend_x + 24, legend_y, text=run_label, anchor="w", font=("Segoe UI", 9))

        canvas.create_text(margin_left + plot_width / 2, height - 20, text="Hours before close", font=("Segoe UI", 10))
        canvas.create_text(20, margin_top + plot_height / 2, text=metric, angle=90, font=("Segoe UI", 10))


def launch() -> None:
    root = tk.Tk()
    PredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
