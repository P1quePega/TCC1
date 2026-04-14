"""
modules_mdo.py — Módulo MDO / Algoritmo Genético para GUI T.O.C.A. v0.3
Última aba — executa pipeline completo em ordem:
  1. Schrenk → Cargas
  2. Entelagem → Verificação de deflexão
  3. Wingbox (com/sem) → GJ/EI
  4. Dimensionamento de longarinas
  5. Aeroelasticidade → Flutter check
  6. (Opcional) ANSYS → TopOpt por nervura
  7. Resultados MDO com gráficos expandidos

Gráficos: Indivíduo × ID, Pareto front, convergência, parallel coordinates.
"""

import numpy as np
import time
import threading

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QDoubleSpinBox, QSpinBox, QScrollArea,
    QComboBox, QTabWidget, QTextEdit, QProgressBar, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# Cores
BG = "#111318"; SURFACE = "#1C2030"; PANEL = "#21263C"
RAISED = "#282E48"; BORDER = "#30395A"; BORDER2 = "#3E4C70"
ACCENT = "#4D82D6"; TEXT_P = "#C8D4EC"; TEXT_S = "#6A7A9C"
TEXT_D = "#3A4562"; GREEN = "#4EC88A"; RED = "#D95252"; AMBER = "#D9963A"
PURPLE = "#7C5CBF"


def _spin(val, lo, hi, dec=2):
    if dec is None:
        w = QSpinBox(); w.setRange(int(lo), int(hi)); w.setValue(int(val))
    else:
        w = QDoubleSpinBox(); w.setDecimals(dec)
        w.setRange(float(lo), float(hi)); w.setValue(float(val))
    w.setMinimumWidth(120); return w


def _make_fig(w=8, h=5):
    fig = Figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(BG)
    canvas = FigureCanvas(fig)
    canvas.setMinimumHeight(int(h * 100))
    return fig, canvas


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_S, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if title: ax.set_title(title, color=TEXT_P, fontsize=10, fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_S, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_S, fontsize=9)
    ax.grid(True, alpha=0.15, color=TEXT_S)


# ═══════════════════════════════════════════════════════════════════════════════
#  Worker MDO
# ═══════════════════════════════════════════════════════════════════════════════

class MDOWorker(QThread):
    log_signal = pyqtSignal(str, str)
    progress_signal = pyqtSignal(int, str)
    gen_signal = pyqtSignal(int, dict)    # gen, stats
    done_signal = pyqtSignal(object)      # MDOResult

    def __init__(self, params):
        super().__init__()
        self.params = params
        self._stop = threading.Event()

    def stop(self): self._stop.set()

    def log(self, msg, nivel="INFO"): self.log_signal.emit(msg, nivel)

    def run(self):
        try:
            self._run_mdo()
        except Exception as e:
            import traceback
            self.log(f"ERRO: {e}", "ERROR")
            self.log(traceback.format_exc(), "ERROR")

    def _run_mdo(self):
        p = self.params
        self.log("=" * 60, "SECTION")
        self.log("MDO — PIPELINE INTEGRADO (NSGA-II)", "SECTION")
        self.log("=" * 60, "SECTION")

        from mdo_optimizer import MDOConfig, MDOEvaluator, NSGAIIOptimizer

        config = MDOConfig(
            pop_size=int(p.get("pop_size", 40)),
            n_gen=int(p.get("n_gen", 50)),
            crossover_prob=p.get("crossover_prob", 0.9),
            mutation_prob=p.get("mutation_prob", 0.1),
            run_ansys=p.get("run_ansys", False),
            run_flutter=p.get("run_flutter", True),
            run_covering=p.get("run_covering", True),
        )

        base_params = {
            "semi_span": p.get("semi_span", 750),
            "root_chord": p.get("root_chord", 300),
            "tip_chord": p.get("tip_chord", 200),
            "velocity": p.get("velocity", 15),
            "mass": p.get("mass", 5),
            "load_factor": p.get("load_factor", 4),
            "area_casca_root": p.get("area_casca_root", 1200),
            "area_otim_root": p.get("area_otim_root", 800),
            "use_wingbox": p.get("use_wingbox", True),
        }

        evaluator = MDOEvaluator(
            base_params, config,
            material_name=p.get("material", "Balsa C-grain"),
            log_callback=self.log
        )

        all_individuals = []
        gen_stats = []

        def on_gen(gen_obj):
            if self._stop.is_set(): return
            stats = {
                "gen": gen_obj.gen_number,
                "best_mass": gen_obj.best_mass,
                "best_stress": gen_obj.best_stress,
                "pop_size": len(gen_obj.population),
                "n_feasible": sum(1 for i in gen_obj.population if i.feasible),
                "pareto_size": len(gen_obj.pareto_front),
            }
            gen_stats.append(stats)
            for ind in gen_obj.population:
                all_individuals.append({
                    "id": len(all_individuals),
                    "gen": gen_obj.gen_number,
                    "mass_g": ind.objectives[0],
                    "stress_MPa": ind.objectives[1],
                    "feasible": ind.feasible,
                    "genes": ind.genes.tolist() if hasattr(ind.genes, 'tolist') else list(ind.genes),
                    "details": ind.details,
                })

            pct = int(gen_obj.gen_number / config.n_gen * 100)
            self.progress_signal.emit(pct,
                f"Gen {gen_obj.gen_number}/{config.n_gen} — "
                f"Melhor massa: {gen_obj.best_mass:.1f}g | "
                f"Factíveis: {stats['n_feasible']}/{stats['pop_size']}")
            self.gen_signal.emit(gen_obj.gen_number, stats)

        optimizer = NSGAIIOptimizer(evaluator, config, callback=on_gen)
        self.log("Iniciando NSGA-II...", "OK")
        result = optimizer.run()

        result_data = {
            "mdo_result": result,
            "all_individuals": all_individuals,
            "gen_stats": gen_stats,
            "config": config,
        }
        self.done_signal.emit(result_data)
        self.log(f"\nMDO concluído! {result.total_evaluations} avaliações em {result.elapsed_time_s:.1f}s", "OK")


# ═══════════════════════════════════════════════════════════════════════════════
#  Gráficos expandidos — Indivíduo × ID
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mdo_expanded(fig, all_individuals, gen_stats, best_individual=None):
    """
    Gráficos expandidos do MDO:
    1. Indivíduo × ID (massa)
    2. Indivíduo × ID (tensão)
    3. Frente de Pareto
    4. Convergência
    5. Parallel coordinates (genes)
    6. Factibilidade por geração
    """
    fig.clear()
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.08, right=0.96, top=0.95, bottom=0.06)

    if not all_individuals:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Aguardando resultados...", color=TEXT_D,
                ha='center', va='center', fontsize=14)
        ax.set_facecolor(BG); ax.axis('off')
        return

    ids = [i["id"] for i in all_individuals]
    masses = [i["mass_g"] for i in all_individuals]
    stresses = [i["stress_MPa"] for i in all_individuals]
    feasible = [i["feasible"] for i in all_individuals]
    gens = [i["gen"] for i in all_individuals]

    colors_f = [GREEN if f else RED for f in feasible]
    alpha_f = [0.7 if f else 0.25 for f in feasible]

    # 1. Indivíduo × ID — Massa
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, (x, y, c, a) in enumerate(zip(ids, masses, colors_f, alpha_f)):
        ax1.scatter(x, y, c=c, s=8, alpha=a, edgecolors='none')
    _style_ax(ax1, "Indivíduo × ID — Massa [g]", "ID do Indivíduo", "Massa [g]")

    # 2. Indivíduo × ID — Tensão
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, (x, y, c, a) in enumerate(zip(ids, stresses, colors_f, alpha_f)):
        ax2.scatter(x, y, c=c, s=8, alpha=a, edgecolors='none')
    _style_ax(ax2, "Indivíduo × ID — Tensão [MPa]", "ID do Indivíduo", "σ [MPa]")

    # 3. Frente de Pareto
    ax3 = fig.add_subplot(gs[1, 0])
    feas_m = [m for m, f in zip(masses, feasible) if f]
    feas_s = [s for s, f in zip(stresses, feasible) if f]
    infeas_m = [m for m, f in zip(masses, feasible) if not f]
    infeas_s = [s for s, f in zip(stresses, feasible) if not f]
    if infeas_m:
        ax3.scatter(infeas_m, infeas_s, c=RED, s=6, alpha=0.15, label="Infactível")
    if feas_m:
        ax3.scatter(feas_m, feas_s, c=ACCENT, s=12, alpha=0.6, label="Factível")
    if best_individual:
        ax3.scatter([best_individual.objectives[0]], [best_individual.objectives[1]],
                    c=GREEN, s=80, marker='*', zorder=5, label="Melhor")
    _style_ax(ax3, "Frente de Pareto", "Massa [g]", "Tensão [MPa]")
    ax3.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT_P)

    # 4. Convergência
    ax4 = fig.add_subplot(gs[1, 1])
    if gen_stats:
        gens_n = [s["gen"] for s in gen_stats]
        best_m = [s["best_mass"] for s in gen_stats]
        best_s = [s["best_stress"] for s in gen_stats]
        ax4.plot(gens_n, best_m, color=ACCENT, lw=2, label="Massa mín [g]")
        ax4b = ax4.twinx()
        ax4b.plot(gens_n, best_s, color=AMBER, lw=2, label="Tensão mín [MPa]")
        ax4b.tick_params(colors=AMBER, labelsize=8)
        ax4b.set_ylabel("σ mín [MPa]", color=AMBER, fontsize=8)
        lines1, l1 = ax4.get_legend_handles_labels()
        lines2, l2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, l1 + l2, fontsize=7,
                   facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT_P)
    _style_ax(ax4, "Convergência", "Geração", "Massa mín [g]")

    # 5. Genes scatter (n_ribs vs rib_thick)
    ax5 = fig.add_subplot(gs[2, 0])
    genes_nribs = []
    genes_thick = []
    for ind in all_individuals:
        g = ind.get("genes", [])
        if len(g) >= 3:
            genes_nribs.append(g[0])
            genes_thick.append(g[2])
    if genes_nribs:
        ax5.scatter(genes_nribs, genes_thick, c=[GREEN if f else RED for f in feasible[:len(genes_nribs)]],
                    s=8, alpha=0.5, edgecolors='none')
    _style_ax(ax5, "Espaço de Design: N_ribs × Esp. Nervura", "N nervuras", "Esp. nervura [mm]")

    # 6. Factibilidade por geração
    ax6 = fig.add_subplot(gs[2, 1])
    if gen_stats:
        gens_n = [s["gen"] for s in gen_stats]
        n_feas = [s["n_feasible"] for s in gen_stats]
        pops = [s["pop_size"] for s in gen_stats]
        pct_feas = [100 * f / p if p > 0 else 0 for f, p in zip(n_feas, pops)]
        ax6.fill_between(gens_n, pct_feas, alpha=0.3, color=GREEN)
        ax6.plot(gens_n, pct_feas, color=GREEN, lw=2)
    _style_ax(ax6, "% Factíveis por Geração", "Geração", "Factíveis [%]")
    ax6.set_ylim(0, 105)

    fig.patch.set_facecolor(BG)


# ═══════════════════════════════════════════════════════════════════════════════
#  Builder do módulo MDO
# ═══════════════════════════════════════════════════════════════════════════════

def build_mdo_module(parent) -> QWidget:
    """Constrói o módulo MDO completo com abas de configuração e resultados."""
    page = QWidget(); page.setStyleSheet(f"background:{SURFACE};")
    lay = QVBoxLayout(page); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

    # Header
    hdr = QWidget(); hdr.setFixedHeight(64)
    hdr.setStyleSheet(f"background:{PANEL};border-bottom:1px solid {BORDER};")
    hl = QHBoxLayout(hdr); hl.setContentsMargins(20, 0, 20, 0)
    col = QVBoxLayout(); col.setSpacing(2)
    cl = QLabel("MÓDULOS / OTIMIZAÇÃO")
    cl.setStyleSheet(f"color:{TEXT_D};font-size:9px;letter-spacing:.12em;background:transparent;")
    tl = QLabel("MDO — Algoritmo Genético (NSGA-II) — Pipeline Integrado")
    tl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
    tl.setStyleSheet(f"color:{TEXT_P};background:transparent;")
    col.addWidget(cl); col.addWidget(tl)
    hl.addLayout(col); hl.addStretch()
    lay.addWidget(hdr)

    tabs = QTabWidget(); tabs.setDocumentMode(True)
    parent.mdo_tabs = tabs

    # ── Aba 1: Configuração ──
    tabs.addTab(_build_mdo_config_tab(parent), "⚙  Configuração MDO")

    # ── Aba 2: Log ──
    w_log = QWidget(); w_log.setStyleSheet(f"background:{SURFACE};")
    ll = QVBoxLayout(w_log); ll.setContentsMargins(12, 12, 12, 12)
    parent.mdo_log = QTextEdit(); parent.mdo_log.setReadOnly(True)
    parent.mdo_log.setFont(QFont("Consolas", 9))
    parent.mdo_log.setStyleSheet(f"background:{BG};color:{TEXT_P};"
        f"border:1px solid {BORDER};border-radius:5px;padding:8px;")
    ll.addWidget(parent.mdo_log)
    tabs.addTab(w_log, "📋  Log")

    # ── Aba 3: Gráficos Expandidos ──
    w_charts = QWidget(); w_charts.setStyleSheet(f"background:{SURFACE};")
    cl2 = QVBoxLayout(w_charts); cl2.setContentsMargins(8, 8, 8, 8)
    parent.mdo_fig, parent.mdo_canvas = _make_fig(12, 9)
    cl2.addWidget(parent.mdo_canvas)
    tabs.addTab(w_charts, "📊  Gráficos Indivíduo×ID")

    # ── Aba 4: Resultados / Tabela ──
    w_res = QWidget(); w_res.setStyleSheet(f"background:{SURFACE};")
    rl = QVBoxLayout(w_res); rl.setContentsMargins(12, 12, 12, 12)
    parent.mdo_result_box = QTextEdit(); parent.mdo_result_box.setReadOnly(True)
    parent.mdo_result_box.setStyleSheet(f"background:{BG};color:{TEXT_P};"
        f"border:1px solid {BORDER};border-radius:5px;padding:8px;font-size:11px;")
    rl.addWidget(parent.mdo_result_box)

    parent.mdo_table = QTableWidget()
    parent.mdo_table.setStyleSheet(f"QTableWidget{{background:{BG};color:{TEXT_P};"
        f"border:1px solid {BORDER};gridline-color:{BORDER};}}")
    rl.addWidget(parent.mdo_table)
    tabs.addTab(w_res, "📋  Resultados")

    lay.addWidget(tabs, stretch=1)

    # Action bar
    bar = QWidget(); bar.setFixedHeight(58)
    bar.setStyleSheet(f"background:{PANEL};border-top:1px solid {BORDER};")
    bl = QHBoxLayout(bar); bl.setContentsMargins(16, 0, 16, 0); bl.setSpacing(8)

    parent.mdo_btn_run = QPushButton("▶  Executar MDO Pipeline")
    parent.mdo_btn_run.setFixedHeight(34)
    parent.mdo_btn_run.setStyleSheet(f"QPushButton{{background:{PURPLE};color:white;border:none;"
        f"border-radius:4px;font-size:12px;font-weight:bold;padding:0 18px;}}"
        f"QPushButton:hover{{background:#6A4DAD;}}")
    parent.mdo_btn_run.clicked.connect(lambda: _run_mdo(parent))

    parent.mdo_btn_stop = QPushButton("⏹  Parar")
    parent.mdo_btn_stop.setFixedHeight(34)
    parent.mdo_btn_stop.setEnabled(False)
    parent.mdo_btn_stop.setStyleSheet(f"QPushButton{{background:{RED};color:white;border:none;"
        f"border-radius:4px;font-weight:bold;padding:0 16px;}}"
        f"QPushButton:disabled{{background:{BORDER};color:{TEXT_D};}}")
    parent.mdo_btn_stop.clicked.connect(lambda: _stop_mdo(parent))

    bl.addWidget(parent.mdo_btn_run); bl.addWidget(parent.mdo_btn_stop)
    bl.addSpacing(16)

    pc = QVBoxLayout(); pc.setSpacing(3)
    parent.mdo_prog_lbl = QLabel("Pronto")
    parent.mdo_prog_lbl.setStyleSheet(f"color:{TEXT_S};font-size:10px;background:transparent;")
    parent.mdo_prog = QProgressBar(); parent.mdo_prog.setFixedHeight(5)
    parent.mdo_prog.setTextVisible(False)
    pc.addWidget(parent.mdo_prog_lbl); pc.addWidget(parent.mdo_prog)
    bl.addLayout(pc, stretch=1)
    lay.addWidget(bar)

    parent._mdo_worker = None
    parent._mdo_all_individuals = []
    parent._mdo_gen_stats = []

    return page


def _build_mdo_config_tab(parent) -> QWidget:
    scroll = QScrollArea(); scroll.setWidgetResizable(True)
    scroll.setStyleSheet(f"background:{SURFACE};")
    w = QWidget(); w.setStyleSheet(f"background:{SURFACE};")
    scroll.setWidget(w)
    lay = QVBoxLayout(w); lay.setSpacing(10); lay.setContentsMargins(16, 16, 16, 16)

    # Pipeline
    info = QLabel("Pipeline MDO: Schrenk → Entelagem → Wingbox → Longarina → Aeroelasticidade → (ANSYS)")
    info.setWordWrap(True)
    info.setStyleSheet(f"background:{PANEL};color:{ACCENT};font-size:10px;font-weight:600;"
                       f"border:1px solid {BORDER};border-radius:5px;padding:10px;")
    lay.addWidget(info)

    # Algoritmo genético
    g1 = QGroupBox("Algoritmo Genético (NSGA-II)")
    gl = QGridLayout(g1); gl.setSpacing(8)
    parent.mdo_pop_size = _spin(40, 10, 200, None)
    parent.mdo_n_gen = _spin(50, 5, 500, None)
    parent.mdo_cx_prob = _spin(0.9, 0.5, 1.0, 2)
    parent.mdo_mut_prob = _spin(0.1, 0.01, 0.5, 2)
    for i, (l, wgt) in enumerate([
        ("Tamanho da população:", parent.mdo_pop_size),
        ("Nº de gerações:", parent.mdo_n_gen),
        ("Prob. cruzamento:", parent.mdo_cx_prob),
        ("Prob. mutação:", parent.mdo_mut_prob),
    ]):
        gl.addWidget(QLabel(l), i, 0); gl.addWidget(wgt, i, 1)
    lay.addWidget(g1)

    # Opções do pipeline
    g2 = QGroupBox("Opções do Pipeline")
    gl2 = QVBoxLayout(g2); gl2.setSpacing(6)
    parent.mdo_use_wingbox = QCheckBox("Usar análise de Wingbox (Caixa de Torção)")
    parent.mdo_use_wingbox.setChecked(True)
    parent.mdo_use_wingbox.setStyleSheet(f"color:{TEXT_S};background:transparent;")
    parent.mdo_run_flutter = QCheckBox("Verificar Flutter / Divergência")
    parent.mdo_run_flutter.setChecked(True)
    parent.mdo_run_flutter.setStyleSheet(f"color:{TEXT_S};background:transparent;")
    parent.mdo_run_covering = QCheckBox("Verificar Entelagem (Deflexão)")
    parent.mdo_run_covering.setChecked(True)
    parent.mdo_run_covering.setStyleSheet(f"color:{TEXT_S};background:transparent;")
    parent.mdo_run_ansys = QCheckBox("Chamar ANSYS (TopOpt) para nervura crítica")
    parent.mdo_run_ansys.setChecked(False)
    parent.mdo_run_ansys.setStyleSheet(f"color:{TEXT_S};background:transparent;")
    gl2.addWidget(parent.mdo_use_wingbox)
    gl2.addWidget(parent.mdo_run_flutter)
    gl2.addWidget(parent.mdo_run_covering)
    gl2.addWidget(parent.mdo_run_ansys)
    lay.addWidget(g2)

    # Material
    g3 = QGroupBox("Material das Nervuras (variação no MDO)")
    gl3 = QGridLayout(g3); gl3.setSpacing(8)
    from materials import get_all_materials
    parent.mdo_material = QComboBox()
    parent.mdo_material.addItems(list(get_all_materials().keys()))
    idx_b = parent.mdo_material.findText("Balsa C-grain")
    if idx_b >= 0: parent.mdo_material.setCurrentIndex(idx_b)

    parent.mdo_vary_material = QCheckBox("Variar material por nervura na otimização")
    parent.mdo_vary_material.setStyleSheet(f"color:{TEXT_S};background:transparent;")
    gl3.addWidget(QLabel("Material base:"), 0, 0); gl3.addWidget(parent.mdo_material, 0, 1)
    gl3.addWidget(parent.mdo_vary_material, 1, 0, 1, 2)
    lay.addWidget(g3)

    lay.addStretch()
    return scroll


# ═══════════════════════════════════════════════════════════════════════════════
#  Ações
# ═══════════════════════════════════════════════════════════════════════════════

CLRS = {"INFO": TEXT_P, "OK": GREEN, "WARN": AMBER, "ERROR": RED, "SECTION": ACCENT}

def _run_mdo(parent):
    from datetime import datetime
    parent.mdo_log.clear()
    parent._mdo_all_individuals = []
    parent._mdo_gen_stats = []
    parent.mdo_prog.setValue(0)
    parent.mdo_btn_run.setEnabled(False)
    parent.mdo_btn_stop.setEnabled(True)
    parent.mdo_tabs.setCurrentIndex(1)  # Log

    # Coletar parâmetros
    params = {
        "pop_size": parent.mdo_pop_size.value(),
        "n_gen": parent.mdo_n_gen.value(),
        "crossover_prob": parent.mdo_cx_prob.value(),
        "mutation_prob": parent.mdo_mut_prob.value(),
        "run_ansys": parent.mdo_run_ansys.isChecked(),
        "run_flutter": parent.mdo_run_flutter.isChecked(),
        "run_covering": parent.mdo_run_covering.isChecked(),
        "use_wingbox": parent.mdo_use_wingbox.isChecked(),
        "material": parent.mdo_material.currentText(),
        "semi_span": parent.sch_semi_span.value() if hasattr(parent, 'sch_semi_span') else 750,
        "root_chord": parent.sch_root_chord.value() if hasattr(parent, 'sch_root_chord') else 300,
        "tip_chord": parent.sch_tip_chord.value() if hasattr(parent, 'sch_tip_chord') else 200,
        "velocity": parent.aero_vel.value() if hasattr(parent, 'aero_vel') else 15,
        "mass": parent.sch_mass.value() if hasattr(parent, 'sch_mass') else 5,
        "load_factor": parent.aero_carga.value() if hasattr(parent, 'aero_carga') else 4,
    }

    worker = MDOWorker(params)
    worker.log_signal.connect(lambda m, n: _mdo_log(parent, m, n))
    worker.progress_signal.connect(lambda p, l: (
        parent.mdo_prog.setValue(p), parent.mdo_prog_lbl.setText(l)))
    worker.gen_signal.connect(lambda g, s: _mdo_on_gen(parent, g, s))
    worker.done_signal.connect(lambda r: _mdo_on_done(parent, r))
    worker.finished.connect(lambda: (
        parent.mdo_btn_run.setEnabled(True),
        parent.mdo_btn_stop.setEnabled(False)))
    parent._mdo_worker = worker
    worker.start()


def _stop_mdo(parent):
    if parent._mdo_worker:
        parent._mdo_worker.stop()


def _mdo_log(parent, msg, nivel):
    from datetime import datetime
    color = CLRS.get(nivel, TEXT_P)
    ts = datetime.now().strftime("%H:%M:%S")
    parent.mdo_log.append(
        f'<span style="color:{TEXT_D}">[{ts}]</span> '
        f'<span style="color:{color}">{msg}</span>')
    sb = parent.mdo_log.verticalScrollBar(); sb.setValue(sb.maximum())


def _mdo_on_gen(parent, gen, stats):
    parent._mdo_gen_stats.append(stats)


def _mdo_on_done(parent, result_data):
    parent._mdo_all_individuals = result_data.get("all_individuals", [])
    parent._mdo_gen_stats = result_data.get("gen_stats", [])
    mdo_result = result_data.get("mdo_result")

    # Gráficos expandidos
    plot_mdo_expanded(
        parent.mdo_fig,
        parent._mdo_all_individuals,
        parent._mdo_gen_stats,
        mdo_result.best_individual if mdo_result else None,
    )
    parent.mdo_canvas.draw()
    parent.mdo_tabs.setCurrentIndex(2)  # Gráficos

    # Resultados HTML
    if mdo_result and mdo_result.best_individual:
        best = mdo_result.best_individual
        html = f"""
        <h3 style='color:{ACCENT}'>Resultado MDO — NSGA-II</h3>
        <p style='color:{GREEN if best.feasible else RED};font-weight:bold;font-size:14px'>
        {"✓ FACTÍVEL" if best.feasible else "✗ INFACTÍVEL"}</p>
        <table style='width:100%;border-collapse:collapse;font-size:11px;'>
        <tr style='background:{PANEL}'><td style='padding:4px 8px'>Massa mínima</td>
            <td style='color:{ACCENT};font-weight:bold'>{best.objectives[0]:.1f} g</td></tr>
        <tr><td style='padding:4px 8px'>Tensão mínima</td>
            <td style='color:{ACCENT};font-weight:bold'>{best.objectives[1]:.3f} MPa</td></tr>
        <tr style='background:{PANEL}'><td style='padding:4px 8px'>Total de avaliações</td>
            <td style='color:{TEXT_S}'>{mdo_result.total_evaluations}</td></tr>
        <tr><td style='padding:4px 8px'>Tempo</td>
            <td style='color:{TEXT_S}'>{mdo_result.elapsed_time_s:.1f}s</td></tr>
        <tr style='background:{PANEL}'><td style='padding:4px 8px'>Indivíduos na Pareto</td>
            <td style='color:{ACCENT}'>{len(mdo_result.pareto_front)}</td></tr>
        </table>
        <p style='color:{TEXT_D};font-size:10px;margin-top:8px'>
        Genes: {', '.join(f'{g:.2f}' for g in best.genes)}</p>
        """
        parent.mdo_result_box.setHtml(html)
    else:
        parent.mdo_result_box.setHtml(f"<p style='color:{RED}'>Nenhum resultado factível encontrado.</p>")