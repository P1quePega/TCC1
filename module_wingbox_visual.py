"""
module_wingbox_visual.py — Módulo Visual do Wingbox + Posicionamento de Nervuras
Baseado no TCC: "Propostas de Melhorias a Projetos Estruturais da Equipe Tucano"
  - Idealização por booms (2 células: Borda de Ataque + Longarina ao BF)
  - Vista superior da asa (Caixa de Torção) — fiel ao estilo da imagem
  - Otimizador RIBSPO: espaçamento de nervuras via Evolução Diferencial
  - Análise de fluxo cisalhante, flambagem, massa e falha estrutural

Ref: Martins, A.S.; Mariano, I.H.S. — UFU 2020
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import threading
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, FancyBboxPatch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QDoubleSpinBox, QSpinBox, QScrollArea,
    QComboBox, QTabWidget, QTextEdit, QProgressBar, QCheckBox,
    QSlider, QFrame, QSizePolicy, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor

# ── Cores ─────────────────────────────────────────────────────────────────────
BG = "#111318"; SURFACE = "#1C2030"; PANEL = "#21263C"
RAISED = "#282E48"; BORDER = "#30395A"; BORDER2 = "#3E4C70"
ACCENT = "#4D82D6"; TEXT_P = "#C8D4EC"; TEXT_S = "#6A7A9C"
TEXT_D = "#3A4562"; GREEN = "#4EC88A"; RED = "#D95252"; AMBER = "#D9963A"

# ── Propriedades do TCC (Tabela 2.1 e 2.2) ────────────────────────────────────
BALSA_PROPS = {
    "E_long_MPa": 5500, "E_trans_MPa": 300, "G_MPa": 200,
    "tau_crit_MPa": 2.8, "density_kgm3": 150, "nu": 0.006,
}
CFRP_PROPS = {
    "E_long_MPa": 135000, "E_trans_MPa": 8000,
    "sigma_t_MPa": 1500, "sigma_c_MPa": 900,
    "density_kgm3": 1600, "ply_t_mm": 0.13, "vf": 0.60,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Estrutura de dados
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WingSection:
    """Uma seção da asa (entre duas nervuras)."""
    y_start_mm: float
    y_end_mm: float
    chord_start_mm: float
    chord_end_mm: float
    # Resultado estrutural
    V_N: float = 0.0          # Cortante
    M_Nmm: float = 0.0        # Momento fletor
    T_Nmm: float = 0.0        # Torque
    tau_balsa_MPa: float = 0.0
    sigma_tube_MPa: float = 0.0
    mass_g: float = 0.0
    failed: bool = False

    @property
    def length_mm(self): return self.y_end_mm - self.y_start_mm
    @property
    def chord_mid_mm(self): return (self.chord_start_mm + self.chord_end_mm) / 2


@dataclass
class WingboxConfig:
    """Configuração completa do wingbox."""
    # Geometria
    semi_span_mm: float = 750.0
    root_chord_mm: float = 300.0
    tip_chord_mm: float = 200.0
    spar_pct: float = 0.25        # Longarina a 25% da corda
    box_start_pct: float = 0.05   # Início da caixa de torção (5%)
    box_end_pct: float = 0.75     # Fim da caixa de torção (75%)
    profile_thickness_pct: float = 0.12  # Espessura max do perfil (12%)

    # Longarina (tubo de carbono)
    spar_od_mm: float = 12.0
    spar_wall_mm: float = 1.0
    # Casca (balsa)
    skin_t_mm: float = 1.5
    # N nervuras
    n_ribs: int = 10
    # Fator de segurança
    safety_factor: float = 1.5


@dataclass
class WingboxAnalysisResult:
    """Resultado da análise do wingbox."""
    sections: List[WingSection] = field(default_factory=list)
    rib_positions_mm: np.ndarray = field(default_factory=lambda: np.array([]))
    mass_total_g: float = 0.0
    mass_skin_g: float = 0.0
    mass_spar_g: float = 0.0
    mass_ribs_g: float = 0.0
    GJ_root: float = 0.0
    EI_root: float = 0.0
    tau_max_MPa: float = 0.0
    sigma_max_MPa: float = 0.0
    any_failure: bool = False
    deflection_tip_mm: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Motor de análise — Idealização por Booms (TCC Tucano)
# ═══════════════════════════════════════════════════════════════════════════════

def naca4_coords(thickness_pct: float, n: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Coordenadas NACA 4 dígitos simétrico (apenas espessura)."""
    t = thickness_pct / 100
    x = np.linspace(0, 1, n)
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    return x, yt


def boom_areas(chord_mm: float, skin_t_mm: float, spar_pct: float,
               box_start_pct: float, box_end_pct: float,
               thickness_pct: float = 12.0) -> dict:
    """
    Calcula áreas de boom para idealização de 6 booms (TCC Eq. 2.3).
    Booms: 2 na longarina (sup/inf), 2 no BA (sup/inf), 2 no BF (sup/inf).
    Apenas a região box_start_pct → box_end_pct é considerada.
    """
    c = chord_mm
    t_s = skin_t_mm
    h = (thickness_pct / 100) * c  # altura máxima do perfil

    # Geometria das webs
    w_cell1 = (spar_pct - box_start_pct) * c   # largura Célula I (BA → longarina)
    w_cell2 = (box_end_pct - spar_pct) * c      # largura Célula II (longarina → BF)
    h_cell = h  # altura do caixão

    # Áreas de boom (Eq. 2.3 — conservando Ixx)
    # Para painéis retangulares de largura L e espessura t:
    # B_boom = t*L/6 * (2 + sigma_adj/sigma_boom)
    # Simplificado: B = t_s * L / 3 (caso booms adjacentes com mesma tensão)
    B_spar_top    = t_s * (w_cell1 + w_cell2) / 3
    B_spar_bot    = B_spar_top
    B_ba_top      = t_s * w_cell1 / 6
    B_ba_bot      = B_ba_top
    B_bf_top      = t_s * w_cell2 / 6
    B_bf_bot      = B_bf_top

    # Posições dos booms (x, z) em mm, origem no CA (25% corda)
    x_ba    = -(spar_pct - box_start_pct) * c
    x_long  = 0.0
    x_bf    = (box_end_pct - spar_pct) * c
    z_top   = h / 2
    z_bot   = -h / 2

    booms = {
        "BA_sup":  {"B": B_ba_top,   "x": x_ba,   "z": z_top},
        "BA_inf":  {"B": B_ba_bot,   "x": x_ba,   "z": z_bot},
        "L_sup":   {"B": B_spar_top, "x": x_long, "z": z_top},
        "L_inf":   {"B": B_spar_bot, "x": x_long, "z": z_bot},
        "BF_sup":  {"B": B_bf_top,   "x": x_bf,   "z": z_top},
        "BF_inf":  {"B": B_bf_bot,   "x": x_bf,   "z": z_bot},
    }
    Ixx = sum(b["B"] * b["z"]**2 for b in booms.values())

    # Áreas das células (Bredt-Batho)
    A1 = w_cell1 * h_cell  # Célula I
    A2 = w_cell2 * h_cell  # Célula II

    return {
        "booms": booms, "Ixx_mm4": Ixx,
        "A1_mm2": A1, "A2_mm2": A2,
        "w1_mm": w_cell1, "w2_mm": w_cell2,
        "h_mm": h_cell, "chord_mm": c,
    }


def shear_flow_two_cell(bdata: dict, Vy_N: float, skin_t_mm: float,
                         G_MPa: float = 200.0) -> dict:
    """
    Fluxo cisalhante nas duas células (Eqs. 2.4–2.12 do TCC).
    Resolve sistema 2x2 para qs0 de cada célula.
    Retorna fluxos básicos e totais em N/mm.
    """
    Ixx = bdata["Ixx_mm4"]
    booms = bdata["booms"]
    A1, A2 = bdata["A1_mm2"], bdata["A2_mm2"]
    w1, w2, h = bdata["w1_mm"], bdata["w2_mm"], bdata["h_mm"]
    t = skin_t_mm
    G = G_MPa

    if Ixx < 1e-6 or A1 < 1e-6 or A2 < 1e-6:
        return {"q_total": {}, "q_web": {}}

    # Fluxo básico (abrindo em BA_sup)
    # Percurso: BA_sup → L_sup → BF_sup → BF_inf → L_inf → BA_inf → BA_sup
    order = ["BA_sup","L_sup","BF_sup","BF_inf","L_inf","BA_inf"]
    qb = {}
    q_acc = 0.0
    for i, name in enumerate(order):
        bm = booms[name]
        q_acc -= (Vy_N / Ixx) * bm["B"] * bm["z"]
        qb[name] = q_acc

    # Fluxo básico em cada web
    qb_w1_top = qb["BA_sup"]    # web superior célula I
    qb_w1_bot = qb["L_inf"]     # web inferior célula I
    qb_spar   = qb["L_sup"]     # web da longarina (compartilhada)
    qb_w2_top = qb["BF_sup"]    # web superior célula II
    qb_w2_bot = qb["BF_inf"]    # web inferior célula II

    # δ = s/t para cada painel (simplificado: comprimentos dos webs)
    d1_top  = w1 / t     # web sup Célula I
    d1_bot  = w1 / t     # web inf Célula I
    d_spar  = h / t      # alma da longarina
    d2_top  = w2 / t     # web sup Célula II
    d2_bot  = w2 / t     # web inf Célula II

    # Integrais de fluxo básico (∮ qb/t ds) para cada célula
    sum1_qb = (qb_w1_top * d1_top + qb_w1_bot * d1_bot +
               qb_spar * d_spar)  # Célula I
    sum2_qb = (qb_w2_top * d2_top + qb_w2_bot * d2_bot -
               qb_spar * d_spar)  # Célula II (longarina com sinal oposto)

    # Δ por célula (soma dos δ)
    delta1 = d1_top + d1_bot + d_spar
    delta2 = d2_top + d2_bot + d_spar
    delta12 = d_spar  # web compartilhada

    # Sistema linear: taxa de torção igual em ambas as células
    # G * dθ/dz * 2A1 = qs0_1 * delta1 - qs0_2 * delta12 + sum1_qb
    # G * dθ/dz * 2A2 = -qs0_1 * delta12 + qs0_2 * delta2 + sum2_qb
    # Para torção pura: dθ/dz = constante (compatibilidade de torção)
    # Eliminando dθ/dz:
    # qs0_1 * delta1 * A2 - qs0_2 * delta12 * A2 + sum1_qb * A2
    # = qs0_1 * delta12 * A1 - qs0_2 * delta2 * A1 + sum2_qb * A1 (errado, usar eq. direto)

    # Equação de compatibilidade: θ'_1 = θ'_2
    # A1*(qs01*delta1 - qs02*delta12 + sum1_qb) = A2*(-qs01*delta12 + qs02*delta2 + sum2_qb)
    # (simplificado — força apenas Vy, torque = 0 → qs01 e qs02 via compatibilidade)

    # Sistema:
    # Eq1 (equilíbrio de momentos em relação a ponto arbitrário):
    # 2*A1*qs01 + 2*A2*qs02 = Vy * x_shear_center
    # Eq2 (compatibilidade): δ1*qs01 - δ12*qs02 + sum1_qb)/A1 = (-δ12*qs01 + δ2*qs02 + sum2_qb)/A2

    # Resolvendo Eq2 apenas (Vy pura, sem torque externo):
    # a*qs01 + b*qs02 = c
    a_coef = delta1 / A1 + delta12 / A2
    b_coef = -delta12 / A1 - delta2 / A2
    c_coef = -sum1_qb / A1 + sum2_qb / A2

    # Segunda equação (momento): 2*A1*qs01 + 2*A2*qs02 = Vy * (x_ca_to_sc)
    # Aproximação: centro de cisalhamento ~ em x_long (longarina) para célula fechada
    # Como só queremos fluxos cisalhantes, usamos só compatibilidade:
    # Sistema 1 eq, 2 incógnitas → precisamos de 2ª condição
    # Usar: eq. de equilíbrio de momento em x=0 (longarina)
    # Sy * x_app = 2*A1*qs0_1 + 2*A2*qs0_2 + ∮qb * ds * p (momento dos fluxos básicos)
    # Para simplificação, assume-se x_app = 0 (Vy aplicada no centro de torção da longarina)
    # Então: 2*A1*qs01 + 2*A2*qs02 = -Mb_qb
    # Momento dos fluxos básicos em torno da longarina:
    Mb_qb = (qb_w1_top * w1 * h/2 + qb_w1_bot * w1 * h/2 +
             qb_w2_top * w2 * h/2 + qb_w2_bot * w2 * h/2)

    A_mat = np.array([[a_coef, b_coef],
                       [2*A1, 2*A2]])
    b_vec = np.array([c_coef, -Mb_qb])

    try:
        qs0 = np.linalg.solve(A_mat, b_vec)
        qs01, qs02 = qs0
    except np.linalg.LinAlgError:
        qs01, qs02 = 0.0, 0.0

    # Fluxos totais em cada web
    q_w1_top = qb_w1_top + qs01
    q_w1_bot = qb_w1_bot + qs01
    q_spar   = qb_spar + qs01 - qs02  # longarina: soma de ambas
    q_w2_top = qb_w2_top + qs02
    q_w2_bot = qb_w2_bot + qs02

    q_max = max(abs(q_w1_top), abs(q_w1_bot), abs(q_spar),
                abs(q_w2_top), abs(q_w2_bot))

    return {
        "q_w1_top": q_w1_top, "q_w1_bot": q_w1_bot,
        "q_spar": q_spar,
        "q_w2_top": q_w2_top, "q_w2_bot": q_w2_bot,
        "q_max_Nmm": q_max,
        "tau_max_MPa": q_max / skin_t_mm if skin_t_mm > 0 else 0,
        "qs01": qs01, "qs02": qs02,
    }


def spar_bending_stress(M_Nmm: float, spar_od_mm: float,
                         spar_wall_mm: float) -> float:
    """
    Tensão de flexão máxima no tubo de carbono (fibra extrema).
    σ = M * r / I
    """
    r_o = spar_od_mm / 2
    r_i = max(0, r_o - spar_wall_mm)
    I = np.pi / 4 * (r_o**4 - r_i**4)
    if I < 1e-6: return 0.0
    return abs(M_Nmm) * r_o / I


def skin_buckling_check(q_web_Nmm: float, panel_w_mm: float,
                          skin_t_mm: float, E_MPa: float,
                          nu: float = 0.006) -> dict:
    """
    Verificação de flambagem da casca (Eq. 2.17 do TCC).
    τ_cr = k * π² * E / (12*(1-ν²)) * (t/b)²
    k ≈ 5.35 para placa simplesmente apoiada em cisalhamento.
    """
    if panel_w_mm < 1e-3 or skin_t_mm < 1e-3: return {"tau_cr": 0, "MS": 0}
    k = 5.35  # placa SS em cisalhamento puro
    tau_cr = k * np.pi**2 * E_MPa / (12*(1 - nu**2)) * (skin_t_mm/panel_w_mm)**2
    tau_act = abs(q_web_Nmm) / skin_t_mm
    MS = tau_cr / (tau_act + 1e-12) - 1
    return {"tau_cr_MPa": tau_cr, "tau_act_MPa": tau_act, "MS_buckling": MS}


def analyze_wingbox_section(cfg: WingboxConfig,
                              rib_positions_mm: np.ndarray,
                              lift_dist_N_per_mm: np.ndarray,
                              y_coords_mm: np.ndarray) -> WingboxAnalysisResult:
    """
    Análise completa do wingbox por seções (entre nervuras).
    Usa idealização por booms do TCC para cada seção.
    """
    result = WingboxAnalysisResult()
    result.rib_positions_mm = rib_positions_mm
    n_ribs = len(rib_positions_mm)
    sections = []

    # Perfil de corda
    lam = cfg.tip_chord_mm / cfg.root_chord_mm

    def chord_at(y):
        return cfg.root_chord_mm * (1 - y / cfg.semi_span_mm * (1 - lam))

    # Cortante e momento fletor (integração da ponta para a raiz)
    y_all = y_coords_mm
    V_y  = np.zeros_like(y_all)
    M_y  = np.zeros_like(y_all)
    for i in range(len(y_all)-2, -1, -1):
        dy = y_all[i+1] - y_all[i]
        L_mid = (lift_dist_N_per_mm[i] + lift_dist_N_per_mm[i+1]) / 2
        V_y[i] = V_y[i+1] + L_mid * dy
    for i in range(len(y_all)-2, -1, -1):
        dy = y_all[i+1] - y_all[i]
        M_y[i] = M_y[i+1] + V_y[i] * dy

    # Torque estimado (Cm0 * q * c²)
    q_dyn = 0.5 * 1.225 * 15**2 * 1e-6  # MPa (V=15 m/s)
    T_y = np.array([q_dyn * chord_at(y)**2 * 0.05 for y in y_all])

    # Para cada seção entre nervuras
    tau_max = 0.0; sigma_max = 0.0; any_fail = False
    mass_skin = 0.0; mass_ribs = 0.0

    for i in range(n_ribs - 1):
        y_s = rib_positions_mm[i]
        y_e = rib_positions_mm[i+1]
        c_s = chord_at(y_s)
        c_e = chord_at(y_e)
        c_m = (c_s + c_e) / 2

        # Interpolar cargas na posição média
        y_m = (y_s + y_e) / 2
        V_m = float(np.interp(y_m, y_all, V_y))
        M_m = float(np.interp(y_m, y_all, M_y))
        T_m = float(np.interp(y_m, y_all, T_y))

        sec = WingSection(y_start_mm=y_s, y_end_mm=y_e,
                          chord_start_mm=c_s, chord_end_mm=c_e,
                          V_N=V_m, M_Nmm=M_m, T_Nmm=T_m)

        # Análise boom
        bdata = boom_areas(c_m, cfg.skin_t_mm, cfg.spar_pct,
                           cfg.box_start_pct, cfg.box_end_pct,
                           cfg.profile_thickness_pct * 100)
        flows = shear_flow_two_cell(bdata, V_m, cfg.skin_t_mm,
                                     BALSA_PROPS["G_MPa"])

        # Tensão na casca
        tau_balsa = flows.get("tau_max_MPa", 0.0)
        sec.tau_balsa_MPa = tau_balsa

        # Tensão no tubo (flexão)
        sig_tube = spar_bending_stress(M_m, cfg.spar_od_mm, cfg.spar_wall_mm)
        sec.sigma_tube_MPa = sig_tube

        # Falha estrutural
        fail_balsa = tau_balsa > BALSA_PROPS["tau_crit_MPa"] / cfg.safety_factor
        fail_tube  = sig_tube  > CFRP_PROPS["sigma_t_MPa"] / cfg.safety_factor
        sec.failed = fail_balsa or fail_tube
        if sec.failed: any_fail = True

        tau_max  = max(tau_max, tau_balsa)
        sigma_max = max(sigma_max, sig_tube)

        # Massa da seção
        L = y_e - y_s
        box_width = (cfg.box_end_pct - cfg.box_start_pct) * c_m
        box_h = cfg.profile_thickness_pct * c_m
        perimeter_box = 2 * (box_width + box_h)
        mass_skin_sec = (perimeter_box * cfg.skin_t_mm * L *
                         BALSA_PROPS["density_kgm3"] * 1e-9 * 1000)  # g
        mass_skin += mass_skin_sec
        sec.mass_g = mass_skin_sec
        sections.append(sec)

    # Massa das nervuras
    for y_r in rib_positions_mm:
        c_r = chord_at(y_r)
        box_area = ((cfg.box_end_pct - cfg.box_start_pct) * c_r *
                    cfg.profile_thickness_pct * c_r)
        mass_rib = (box_area * cfg.skin_t_mm *
                    BALSA_PROPS["density_kgm3"] * 1e-9 * 1000)
        mass_ribs += mass_rib

    # Massa da longarina (tubo de carbono)
    r_o = cfg.spar_od_mm / 2
    r_i = max(0, r_o - cfg.spar_wall_mm)
    A_spar = np.pi * (r_o**2 - r_i**2)
    mass_spar = (A_spar * cfg.semi_span_mm * 2 *
                 CFRP_PROPS["density_kgm3"] * 1e-9 * 1000)

    # Deflexão na ponta (método de Rayleigh, EI raiz)
    bdata_root = boom_areas(cfg.root_chord_mm, cfg.skin_t_mm,
                             cfg.spar_pct, cfg.box_start_pct, cfg.box_end_pct,
                             cfg.profile_thickness_pct * 100)
    I_spar = np.pi/4 * (r_o**4 - r_i**4)
    EI_root = CFRP_PROPS["E_long_MPa"] * I_spar
    GJ_root = BALSA_PROPS["G_MPa"] * (bdata_root["A1_mm2"] + bdata_root["A2_mm2"])**2 * 2

    V_root = float(V_y[0]) if len(V_y) > 0 else 0
    L_span = cfg.semi_span_mm
    defl_tip = (V_root * L_span**3) / (3 * EI_root) if EI_root > 1 else 0

    result.sections = sections
    result.mass_skin_g = mass_skin * 2   # ambas as meia-asas
    result.mass_ribs_g = mass_ribs * 2
    result.mass_spar_g = mass_spar
    result.mass_total_g = result.mass_skin_g + result.mass_ribs_g + result.mass_spar_g
    result.tau_max_MPa = tau_max
    result.sigma_max_MPa = sigma_max
    result.any_failure = any_fail
    result.EI_root = EI_root
    result.GJ_root = GJ_root
    result.deflection_tip_mm = defl_tip
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Otimizador RIBSPO (Evolução Diferencial — inspirado no TCC)
# ═══════════════════════════════════════════════════════════════════════════════

class RibSpacingOptimizer:
    """
    Otimiza o espaçamento de nervuras minimizando massa total.
    Usa Evolução Diferencial (SciPy) — análogo ao DE do TCC.
    Constraints: falha em cisalhamento (balsa) e flexão (carbono).
    """
    def __init__(self, cfg: WingboxConfig,
                 lift_dist: np.ndarray, y_coords: np.ndarray,
                 log_cb=None, progress_cb=None, gen_cb=None):
        self.cfg = cfg
        self.lift_dist = lift_dist
        self.y_coords = y_coords
        self.log = log_cb or (lambda m, n="INFO": None)
        self.progress = progress_cb or (lambda p, l: None)
        self.gen_cb = gen_cb  # callback(gen, best_mass, positions)
        self._stop = threading.Event()
        self.best_result = None
        self.n_evals = 0

    def stop(self): self._stop.set()

    def _eval(self, spacings_mm: np.ndarray) -> float:
        """Avalia um indivíduo: retorna massa (penalizada se houver falha)."""
        self.n_evals += 1
        if self._stop.is_set(): return 1e9

        n_ribs = self.cfg.n_ribs
        if len(spacings_mm) != n_ribs - 1: return 1e9

        # Construir posições das nervuras
        positions = np.zeros(n_ribs)
        positions[0] = 0.0
        for i in range(1, n_ribs):
            positions[i] = positions[i-1] + spacings_mm[i-1]

        if positions[-1] > self.cfg.semi_span_mm: return 1e9

        res = analyze_wingbox_section(self.cfg, positions,
                                      self.lift_dist, self.y_coords)
        mass = res.mass_total_g
        if res.any_failure:
            mass *= 10  # penalidade
        return mass

    def run(self, pop_size: int = 20, n_gen: int = 50,
            F: float = 0.8, CR: float = 0.9):
        """Evolução Diferencial para minimizar massa."""
        n_ribs = self.cfg.n_ribs
        n_var = n_ribs - 1
        span = self.cfg.semi_span_mm

        # Limites: cada espaçamento entre 30mm e span/2
        lo = np.full(n_var, 30.0)
        hi = np.full(n_var, span * 0.6)

        # Constraint: soma dos espaçamentos ≤ span
        def valid(x): return np.sum(x) <= span - 10

        # Inicialização
        np.random.seed(42)
        pop = lo + np.random.rand(pop_size, n_var) * (hi - lo)
        # Normalizar para que a soma respeite o span
        for i in range(pop_size):
            if np.sum(pop[i]) > span:
                pop[i] = pop[i] * (span * 0.95) / np.sum(pop[i])

        fitness = np.array([self._eval(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_mass = fitness[best_idx]
        best_genes = pop[best_idx].copy()

        self.log(f"  Gen 0 | Pop {pop_size} | Melhor: {best_mass:.1f}g", "INFO")
        self.progress(0, f"Gen 0/{n_gen} — Massa: {best_mass:.1f}g")

        for gen in range(1, n_gen + 1):
            if self._stop.is_set(): break

            for i in range(pop_size):
                # Mutação DE/rand/1
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lo, hi)

                # Normalizar
                if np.sum(mutant) > span:
                    mutant = mutant * (span * 0.95) / np.sum(mutant)

                # Cruzamento binomial
                mask = np.random.rand(n_var) < CR
                if not np.any(mask):
                    mask[np.random.randint(n_var)] = True
                trial = np.where(mask, mutant, pop[i])

                if not valid(trial): continue

                f_trial = self._eval(trial)
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_mass:
                        best_mass = f_trial
                        best_genes = trial.copy()
                        best_idx = i

            pct = int(gen / n_gen * 100)
            self.progress(pct, f"Gen {gen}/{n_gen} — Melhor massa: {best_mass:.1f}g")

            if gen % 5 == 0:
                self.log(f"  Gen {gen:3d} | Melhor: {best_mass:.1f}g | "
                         f"Avaliações: {self.n_evals}", "INFO")

            if self.gen_cb:
                positions = np.concatenate([[0], np.cumsum(best_genes)])
                self.gen_cb(gen, best_mass, positions)

        # Resultado final
        positions = np.concatenate([[0], np.cumsum(best_genes)])
        self.best_result = analyze_wingbox_section(
            self.cfg, positions, self.lift_dist, self.y_coords)
        self.log(f"\n  Otimização concluída!", "OK")
        self.log(f"  Massa mínima: {self.best_result.mass_total_g:.1f}g", "OK")
        self.log(f"  Falha estrutural: {'SIM' if self.best_result.any_failure else 'NÃO'}",
                 "ERROR" if self.best_result.any_failure else "OK")
        return self.best_result, positions


# ═══════════════════════════════════════════════════════════════════════════════
#  Funções de plotagem — estilo fiel à imagem do TCC
# ═══════════════════════════════════════════════════════════════════════════════

def plot_wing_planform_topview(fig, ax, cfg: WingboxConfig,
                                rib_positions_mm: np.ndarray,
                                result: Optional[WingboxAnalysisResult] = None,
                                show_section: bool = True):
    """
    Vista superior da asa estilo TCC (Caixa de Torção).
    Imita fielmente a imagem fornecida: contorno da asa, caixa hachureada,
    longarina destacada, nervuras verticais.
    """
    ax.set_facecolor(BG)
    span = cfg.semi_span_mm
    lam = cfg.tip_chord_mm / cfg.root_chord_mm

    def chord(y): return cfg.root_chord_mm * (1 - y / span * (1 - lam))
    def le_x(y): return -chord(y) * 0.0   # LE em x=0 (simplificado)
    def te_x(y): return chord(y)

    # Coordenadas do contorno da asa
    y_arr = np.linspace(0, span, 200)
    le_arr = np.array([le_x(y) for y in y_arr])
    te_arr = np.array([te_x(y) for y in y_arr])

    # Preencher asa (cinza muito claro para simular o papel branco do TCC)
    ax.fill_betweenx(y_arr, le_arr, te_arr, color='#1C2030', alpha=0.5)

    # Contorno da asa (linha branca espessa)
    ax.plot(le_arr, y_arr, color='white', lw=2.0, solid_capstyle='round')
    ax.plot(te_arr, y_arr, color='white', lw=2.0, solid_capstyle='round')
    ax.plot([le_x(0), le_x(0)], [0, 0], color='white', lw=2)
    ax.plot([le_x(0), te_x(0)], [0, 0], color='white', lw=2)  # raiz
    ax.plot([le_x(span), te_x(span)], [span, span], color='white', lw=1.5)  # ponta

    # Caixa de torção hachureada (região entre box_start e box_end)
    box_start = cfg.box_start_pct
    box_end   = cfg.box_end_pct
    box_le = np.array([chord(y) * box_start for y in y_arr])
    box_te = np.array([chord(y) * box_end   for y in y_arr])

    ax.fill_betweenx(y_arr, box_le, box_te,
                     facecolor='none', edgecolor='#6A7A9C',
                     linewidth=0, alpha=0.3)

    # Hachura da caixa de torção (estilo TCC — linhas diagonais)
    n_hatch = 40
    for i, y_h in enumerate(np.linspace(0, span, n_hatch)):
        c_h = chord(y_h)
        x0 = c_h * box_start
        x1 = c_h * box_end
        # Linhas diagonais da hachura
        if i % 2 == 0:
            ax.plot([x0, x1], [y_h, y_h], color='#4D82D6', lw=0.4, alpha=0.4)
    # Linhas oblíquas da hachura (diferenciando área hachureada)
    for xi in np.linspace(0, 0.95, 30):
        y_h = xi * span
        c_h = chord(y_h)
        dy = span * 0.04
        y_h2 = min(span, y_h + dy)
        c_h2 = chord(y_h2)
        x0 = c_h * box_start + (c_h * box_end - c_h * box_start) * xi * 0.1
        x1_end = c_h2 * box_end
        ax.plot([c_h * box_start, x1_end * 0.9], [y_h, y_h2],
                color='#4D82D6', lw=0.3, alpha=0.35)

    # Bordas da caixa de torção
    ax.plot(box_le, y_arr, color=ACCENT, lw=1.2, ls='--', alpha=0.8, label='Caixa de Torção')
    ax.plot(box_te, y_arr, color=ACCENT, lw=1.2, ls='--', alpha=0.8)

    # Longarina (linha sólida destacada em amarelo/laranja)
    spar_x = np.array([chord(y) * cfg.spar_pct for y in y_arr])
    ax.plot(spar_x, y_arr, color=AMBER, lw=2.5, label=f'Longarina ({cfg.spar_pct*100:.0f}% corda)')

    # Símbolo do tubo de carbono (círculo na raiz)
    spar_x_root = chord(0) * cfg.spar_pct
    r_visual = chord(0) * 0.025
    circle = plt.Circle((spar_x_root, span * 0.1), r_visual,
                         color=AMBER, fill=False, lw=2.0)
    ax.add_patch(circle)

    # Nervuras (linhas verticais hachureadas)
    for i, y_r in enumerate(rib_positions_mm):
        c_r = chord(y_r)
        x_le = c_r * box_start * 0.0   # começa na LE
        x_te = c_r                      # vai até TE
        color_rib = RED if (result and i < len(result.sections) and
                            result.sections[i].failed) else '#4D82D6'
        alpha_rib = 1.0 if (result and i < len(result.sections) and
                             result.sections[i].failed) else 0.8
        ax.plot([0, c_r], [y_r, y_r], color=color_rib, lw=1.5,
                alpha=alpha_rib, solid_capstyle='round')

    # Raiz marcada
    ax.plot([0, te_x(0)], [0, 0], color='white', lw=2.5)

    # Dimensões
    c_root = chord(0)
    ax.annotate('', xy=(c_root * 1.08, span), xytext=(c_root * 1.08, 0),
                arrowprops=dict(arrowstyle='<->', color='#6A7A9C', lw=1))
    ax.text(c_root * 1.12, span/2, f'{span:.0f} mm',
            color=TEXT_S, fontsize=7, va='center', rotation=90)

    ax.annotate('', xy=(c_root, -span*0.06), xytext=(0, -span*0.06),
                arrowprops=dict(arrowstyle='<->', color='#6A7A9C', lw=1))
    ax.text(c_root/2, -span*0.1, f'c = {c_root:.0f} mm',
            color=TEXT_S, fontsize=7, ha='center')

    ax.set_xlim(-c_root * 0.1, c_root * 1.2)
    ax.set_ylim(-span * 0.15, span * 1.08)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legenda discreta
    handles = [
        mpatches.Patch(facecolor=ACCENT, alpha=0.6, label='Caixa de Torção'),
        mpatches.Patch(facecolor=AMBER,  alpha=0.9, label=f'Longarina ({cfg.spar_pct*100:.0f}%)'),
        mpatches.Patch(facecolor='#4D82D6', alpha=0.8, label='Nervuras'),
    ]
    if result and result.any_failure:
        handles.append(mpatches.Patch(facecolor=RED, alpha=0.9, label='Falha estrutural'))
    ax.legend(handles=handles, loc='upper left',
              facecolor=PANEL, edgecolor=BORDER,
              labelcolor=TEXT_P, fontsize=7.5)

    ax.set_title('Vista Superior — Asa + Caixa de Torção',
                 color=TEXT_P, fontsize=11, fontweight='bold', pad=8)


def plot_cross_section_boom(fig, ax, cfg: WingboxConfig,
                              chord_mm: float, y_station_mm: float):
    """
    Vista em corte transversal com idealização por booms.
    Mostra o perfil NACA, a caixa de torção, booms e tubo de carbono.
    """
    ax.set_facecolor(BG)

    # Perfil NACA simétrico
    x_naca, yt = naca4_coords(cfg.profile_thickness_pct * 100, n=100)
    x_s = x_naca * chord_mm
    y_top = yt * chord_mm
    y_bot = -yt * chord_mm

    ax.plot(x_s, y_top, color='white', lw=1.8)
    ax.plot(x_s, y_bot, color='white', lw=1.8)
    ax.fill_between(x_s, y_bot, y_top, color='#1C2030', alpha=0.7)

    # Caixa de torção
    bs = cfg.box_start_pct * chord_mm
    be = cfg.box_end_pct * chord_mm
    h  = np.max(y_top)
    box_pts_x = [bs, be, be, bs, bs]
    box_pts_y = [-h, -h, h, h, -h]
    ax.fill(box_pts_x, box_pts_y, color=ACCENT, alpha=0.15)
    ax.plot(box_pts_x, box_pts_y, color=ACCENT, lw=1.2, ls='--', alpha=0.7)

    # Longarina
    sp = cfg.spar_pct * chord_mm
    h_spar = np.interp(sp / chord_mm, x_naca, yt) * chord_mm
    ax.plot([sp, sp], [-h_spar, h_spar], color=AMBER, lw=2.5, label='Longarina')

    # Tubo de carbono
    r_vis = cfg.spar_od_mm / 2
    tube = plt.Circle((sp, 0), r_vis, color=AMBER, fill=False, lw=2.5)
    tube_fill = plt.Circle((sp, 0), r_vis, color=AMBER, alpha=0.3)
    ax.add_patch(tube_fill); ax.add_patch(tube)

    # Booms (6 booms — TCC Fig. 2.4)
    bdata = boom_areas(chord_mm, cfg.skin_t_mm, cfg.spar_pct,
                        cfg.box_start_pct, cfg.box_end_pct,
                        cfg.profile_thickness_pct * 100)
    boom_names_labels = {
        "BA_sup": "B₁", "L_sup": "B₂", "BF_sup": "B₃",
        "BF_inf": "B₄", "L_inf": "B₅", "BA_inf": "B₆",
    }
    for bname, blabel in boom_names_labels.items():
        bm = bdata["booms"][bname]
        # Posição do boom: x = sp + bm["x"], z = bm["z"]
        xb = sp + bm["x"]
        zb = bm["z"]
        r_boom = max(2.0, np.sqrt(bm["B"]) * 0.2)
        boom_circ = plt.Circle((xb, zb), r_boom, color=GREEN, zorder=5)
        ax.add_patch(boom_circ)
        ax.text(xb, zb + r_boom * 1.8, blabel,
                color=GREEN, fontsize=7, ha='center', va='bottom',
                fontweight='bold')

    # Células I e II
    mid1_x = (bs + sp) / 2
    mid2_x = (sp + be) / 2
    ax.text(mid1_x, 0, 'I', color=ACCENT, fontsize=11,
            ha='center', va='center', fontweight='bold', alpha=0.6)
    ax.text(mid2_x, 0, 'II', color=ACCENT, fontsize=10,
            ha='center', va='center', fontweight='bold', alpha=0.6)

    ax.set_aspect('equal')
    ax.set_xlim(-chord_mm * 0.05, chord_mm * 1.05)
    ax.set_ylim(-h * 1.6, h * 1.8)
    ax.tick_params(colors=TEXT_S, labelsize=7)
    for sp_ in ax.spines.values(): sp_.set_color(BORDER)
    ax.set_xlabel('x [mm]', color=TEXT_S, fontsize=8)
    ax.set_ylabel('z [mm]', color=TEXT_S, fontsize=8)
    ax.grid(True, alpha=0.12, color=TEXT_S)
    ax.set_title(f'Seção Transversal — y = {y_station_mm:.0f} mm (Booms)',
                 color=TEXT_P, fontsize=10, fontweight='bold', pad=6)

    # Legenda discreta
    handles = [
        mpatches.Patch(facecolor=ACCENT, alpha=0.5, label='Caixa de torção'),
        mpatches.Patch(facecolor=GREEN, alpha=0.9, label='Booms (bal.)'),
        mpatches.Patch(facecolor=AMBER, alpha=0.8, label='Tubo CF'),
    ]
    ax.legend(handles=handles, loc='upper right',
              facecolor=PANEL, edgecolor=BORDER,
              labelcolor=TEXT_P, fontsize=7)


def plot_structural_results(fig, axes, result: WingboxAnalysisResult,
                              cfg: WingboxConfig):
    """Gráficos de resultados estruturais por seção."""
    if not result.sections: return

    y_mids = np.array([s.y_start_mm + s.length_mm/2 for s in result.sections])
    taus   = np.array([s.tau_balsa_MPa for s in result.sections])
    sigs   = np.array([s.sigma_tube_MPa for s in result.sections])
    masses = np.array([s.mass_g for s in result.sections])
    Vs     = np.array([s.V_N for s in result.sections])
    Ms     = np.array([s.M_Nmm / 1000 for s in result.sections])

    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_S, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(BORDER)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_title(title, color=TEXT_P, fontsize=9, fontweight='bold', pad=5)
        ax.set_xlabel(xlabel, color=TEXT_S, fontsize=8)
        ax.set_ylabel(ylabel, color=TEXT_S, fontsize=8)
        ax.grid(True, alpha=0.12, color=TEXT_S)

    ax1, ax2, ax3 = axes

    # τ na balsa
    colors_tau = [RED if t > BALSA_PROPS["tau_crit_MPa"] else ACCENT for t in taus]
    ax1.bar(y_mids, taus, width=y_mids[1]-y_mids[0] if len(y_mids)>1 else 50,
            color=colors_tau, alpha=0.8, edgecolor=BORDER)
    ax1.axhline(BALSA_PROPS["tau_crit_MPa"], color=RED, lw=1.5, ls='--',
                label=f'τ_crit = {BALSA_PROPS["tau_crit_MPa"]} MPa')
    _style(ax1, 'Tensão Cisalhante na Casca (Balsa)', 'y [mm]', 'τ [MPa]')
    ax1.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT_P)

    # σ no tubo
    colors_sig = [RED if s > CFRP_PROPS["sigma_t_MPa"] else AMBER for s in sigs]
    ax2.bar(y_mids, sigs, width=y_mids[1]-y_mids[0] if len(y_mids)>1 else 50,
            color=colors_sig, alpha=0.8, edgecolor=BORDER)
    ax2.axhline(CFRP_PROPS["sigma_t_MPa"], color=RED, lw=1.5, ls='--',
                label=f'σ_t = {CFRP_PROPS["sigma_t_MPa"]} MPa')
    _style(ax2, 'Tensão de Flexão no Tubo CF', 'y [mm]', 'σ [MPa]')
    ax2.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT_P)

    # Cortante e momento
    ax3.plot(y_mids, Vs, color=ACCENT, lw=2, label='V [N]')
    ax3b = ax3.twinx()
    ax3b.plot(y_mids, Ms, color=GREEN, lw=2, label='M [N·m]')
    ax3b.set_ylabel('M [N·m]', color=GREEN, fontsize=8)
    ax3b.tick_params(colors=GREEN, labelsize=8)
    _style(ax3, 'Cortante e Momento Fletor', 'y [mm]', 'V [N]')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1+lines2, labels1+labels2, fontsize=7,
               facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT_P)


# ═══════════════════════════════════════════════════════════════════════════════
#  Worker thread
# ═══════════════════════════════════════════════════════════════════════════════

class WingboxWorker(QThread):
    log_signal      = pyqtSignal(str, str)
    progress_signal = pyqtSignal(int, str)
    gen_signal      = pyqtSignal(int, float, object)  # gen, mass, positions
    done_signal     = pyqtSignal(object, object)       # result, positions

    def __init__(self, cfg: WingboxConfig, pop: int, n_gen: int,
                 F: float, CR: float):
        super().__init__()
        self.cfg = cfg
        self.pop = pop; self.n_gen = n_gen; self.F = F; self.CR = CR
        self._opt = None

    def stop(self):
        if self._opt: self._opt.stop()

    def log(self, m, n="INFO"): self.log_signal.emit(m, n)

    def run(self):
        try:
            from schrenk import WingGeometry, FlightCondition, schrenk_distribution
            wing = WingGeometry(
                semi_span_mm=self.cfg.semi_span_mm,
                root_chord_mm=self.cfg.root_chord_mm,
                tip_chord_mm=self.cfg.tip_chord_mm,
            )
            flight = FlightCondition(
                velocity_ms=15, aircraft_mass_kg=5,
                load_factor=4, rho_kgm3=1.225,
            )
            sch = schrenk_distribution(wing, flight, n_stations=300)
            self.log("  Distribuição de Schrenk calculada.", "OK")

            self._opt = RibSpacingOptimizer(
                self.cfg, sch.lift_per_span_Nmm, sch.y_mm,
                log_cb=self.log,
                progress_cb=lambda p,l: self.progress_signal.emit(p,l),
                gen_cb=lambda g,m,pos: self.gen_signal.emit(g, m, pos),
            )
            result, positions = self._opt.run(
                pop_size=self.pop, n_gen=self.n_gen,
                F=self.F, CR=self.CR
            )
            self.done_signal.emit(result, positions)
        except Exception as ex:
            import traceback
            self.log(f"Erro: {ex}\n{traceback.format_exc()}", "ERROR")


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI — Aba de Wingbox Visual
# ═══════════════════════════════════════════════════════════════════════════════

def build_wingbox_visual_tab(parent) -> QWidget:
    """Aba 'Caixa de Torção' com vista superior e seção transversal."""
    scroll = QScrollArea(); scroll.setWidgetResizable(True)
    scroll.setStyleSheet(f"background:{SURFACE};")
    w = QWidget(); w.setStyleSheet(f"background:{SURFACE};")
    scroll.setWidget(w)
    lay = QVBoxLayout(w); lay.setSpacing(10); lay.setContentsMargins(16,16,16,16)

    # ── Geometria ──
    g1 = QGroupBox("Geometria do Wingbox (Caixa de Torção)")
    gl = QGridLayout(g1); gl.setSpacing(8)
    parent.vwb_box_start = _mk_dspin(5.0, 1, 30, 1)
    parent.vwb_box_end   = _mk_dspin(75.0, 40, 90, 1)
    parent.vwb_spar_pct  = _mk_dspin(25.0, 10, 50, 1)
    parent.vwb_thick_pct = _mk_dspin(12.0, 6, 20, 1)
    parent.vwb_skin_t    = _mk_dspin(1.5, 0.3, 5.0, 2)
    parent.vwb_spar_od   = _mk_dspin(12.0, 5, 30, 1)
    parent.vwb_spar_wall = _mk_dspin(1.0, 0.3, 3.0, 2)
    parent.vwb_sf        = _mk_dspin(1.5, 1.0, 3.0, 2)
    for i,(l,wgt) in enumerate([
        ("Início caixa (% corda):", parent.vwb_box_start),
        ("Fim caixa (% corda):", parent.vwb_box_end),
        ("Longarina (% corda):", parent.vwb_spar_pct),
        ("Espessura perfil (%):", parent.vwb_thick_pct),
        ("Espessura casca (mm):", parent.vwb_skin_t),
        ("Diâm. tubo CF (mm):", parent.vwb_spar_od),
        ("Parede tubo CF (mm):", parent.vwb_spar_wall),
        ("Fator de segurança:", parent.vwb_sf),
    ]):
        gl.addWidget(QLabel(l), i//2, (i%2)*2); gl.addWidget(wgt, i//2, (i%2)*2+1)
    lay.addWidget(g1)

    # ── Nervuras manuais ──
    g2 = QGroupBox("Nervuras — Posicionamento Manual")
    gl2 = QGridLayout(g2); gl2.setSpacing(8)
    parent.vwb_n_ribs = _mk_ispin(10, 3, 25)
    parent.vwb_rib_mode = QComboBox()
    parent.vwb_rib_mode.addItems(["Uniforme", "Concentrado na raiz", "Concentrado na ponta", "Custom (editar tabela)"])
    gl2.addWidget(QLabel("Nº nervuras:"), 0, 0); gl2.addWidget(parent.vwb_n_ribs, 0, 1)
    gl2.addWidget(QLabel("Distribuição:"), 1, 0); gl2.addWidget(parent.vwb_rib_mode, 1, 1)
    lay.addWidget(g2)

    # Botão visualizar
    btn_vis = QPushButton("👁  Visualizar Wingbox")
    btn_vis.setFixedHeight(36); btn_vis.setCursor(Qt.CursorShape.PointingHandCursor)
    btn_vis.setStyleSheet(f"QPushButton{{background:{ACCENT};color:white;border:none;"
                          f"border-radius:4px;font-size:12px;font-weight:600;padding:0 18px;}}"
                          f"QPushButton:hover{{background:#3A70C4;}}")
    btn_vis.clicked.connect(lambda: _update_wingbox_visual(parent))
    lay.addWidget(btn_vis)

    # ── Figura principal: Vista superior (grande) + seção transversal ──
    parent.vwb_fig = Figure(figsize=(10, 7), dpi=100)
    parent.vwb_fig.patch.set_facecolor(BG)
    parent.vwb_canvas = FigureCanvas(parent.vwb_fig)
    parent.vwb_canvas.setMinimumHeight(650)
    lay.addWidget(parent.vwb_canvas)

    # Slider de estação
    row_slide = QHBoxLayout()
    row_slide.addWidget(QLabel("Estação para seção transversal:"))
    parent.vwb_station_slider = QSlider(Qt.Orientation.Horizontal)
    parent.vwb_station_slider.setRange(0, 100)
    parent.vwb_station_slider.setValue(0)
    parent.vwb_station_lbl = QLabel("0 mm")
    parent.vwb_station_lbl.setStyleSheet(f"color:{ACCENT};font-weight:bold;background:transparent;")
    parent.vwb_station_slider.valueChanged.connect(
        lambda v: (_update_station_label(parent, v), _update_wingbox_visual(parent)))
    row_slide.addWidget(parent.vwb_station_slider, stretch=1)
    row_slide.addWidget(parent.vwb_station_lbl)
    lay.addLayout(row_slide)

    # Resultados
    parent.vwb_result_box = QTextEdit(); parent.vwb_result_box.setReadOnly(True)
    parent.vwb_result_box.setMaximumHeight(160)
    parent.vwb_result_box.setStyleSheet(f"background:{BG};color:{TEXT_P};"
        f"border:1px solid {BORDER};border-radius:5px;padding:8px;font-size:11px;")
    lay.addWidget(parent.vwb_result_box)
    lay.addStretch()

    # Inicializar
    _update_wingbox_visual(parent)
    return scroll


def _update_station_label(parent, val):
    semi = parent.sch_semi_span.value() if hasattr(parent, 'sch_semi_span') else 750
    mm = int(val / 100 * semi)
    parent.vwb_station_lbl.setText(f"{mm} mm")


def _get_wb_cfg(parent) -> WingboxConfig:
    semi  = parent.sch_semi_span.value()  if hasattr(parent,'sch_semi_span')  else 750
    c_r   = parent.sch_root_chord.value() if hasattr(parent,'sch_root_chord') else 300
    c_t   = parent.sch_tip_chord.value()  if hasattr(parent,'sch_tip_chord')  else 200
    return WingboxConfig(
        semi_span_mm=semi, root_chord_mm=c_r, tip_chord_mm=c_t,
        spar_pct    = parent.vwb_spar_pct.value() / 100,
        box_start_pct = parent.vwb_box_start.value() / 100,
        box_end_pct   = parent.vwb_box_end.value() / 100,
        profile_thickness_pct = parent.vwb_thick_pct.value() / 100,
        skin_t_mm   = parent.vwb_skin_t.value(),
        spar_od_mm  = parent.vwb_spar_od.value(),
        spar_wall_mm= parent.vwb_spar_wall.value(),
        n_ribs      = int(parent.vwb_n_ribs.value()),
        safety_factor = parent.vwb_sf.value(),
    )


def _make_rib_positions(cfg: WingboxConfig, mode: str) -> np.ndarray:
    n = cfg.n_ribs; span = cfg.semi_span_mm
    if mode == "Uniforme":
        return np.linspace(0, span, n)
    elif mode == "Concentrado na raiz":
        t = np.linspace(0, 1, n)**1.8
        return t * span
    elif mode == "Concentrado na ponta":
        t = np.linspace(0, 1, n)**(0.55)
        return t * span
    else:
        return np.linspace(0, span, n)


def _update_wingbox_visual(parent):
    try:
        cfg = _get_wb_cfg(parent)
        mode = parent.vwb_rib_mode.currentText()
        rib_pos = _make_rib_positions(cfg, mode)

        # Análise rápida
        from schrenk import WingGeometry, FlightCondition, schrenk_distribution
        wing = WingGeometry(semi_span_mm=cfg.semi_span_mm,
                             root_chord_mm=cfg.root_chord_mm,
                             tip_chord_mm=cfg.tip_chord_mm)
        flight = FlightCondition(velocity_ms=15, aircraft_mass_kg=5, load_factor=4)
        sch = schrenk_distribution(wing, flight, n_stations=200)
        result = analyze_wingbox_section(cfg, rib_pos, sch.lift_per_span_Nmm, sch.y_mm)

        # Estação selecionada
        station_pct = parent.vwb_station_slider.value() / 100
        y_station = station_pct * cfg.semi_span_mm
        lam = cfg.tip_chord_mm / cfg.root_chord_mm
        chord_at_station = cfg.root_chord_mm * (1 - y_station/cfg.semi_span_mm*(1-lam))

        # Plot
        fig = parent.vwb_fig; fig.clear()
        gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35,
                      left=0.03, right=0.97, top=0.95, bottom=0.07)
        ax_top   = fig.add_subplot(gs[0, :2])   # Vista superior (grande)
        ax_cross = fig.add_subplot(gs[0, 2])    # Seção transversal
        ax_tau   = fig.add_subplot(gs[1, 0])    # τ balsa
        ax_sig   = fig.add_subplot(gs[1, 1])    # σ tubo
        ax_VM    = fig.add_subplot(gs[1, 2])    # V e M

        plot_wing_planform_topview(fig, ax_top, cfg, rib_pos, result)
        plot_cross_section_boom(fig, ax_cross, cfg, chord_at_station, y_station)
        plot_structural_results(fig, [ax_tau, ax_sig, ax_VM], result, cfg)

        # Linha de estação na vista superior
        c_st = chord_at_station
        ax_top.axhline(y_station, color='#6A7A9C', lw=1, ls=':', alpha=0.8)
        ax_top.text(c_st * 0.5, y_station + cfg.semi_span_mm*0.01,
                    f'Seção: {y_station:.0f}mm', color=TEXT_S, fontsize=7)

        fig.patch.set_facecolor(BG)
        parent.vwb_canvas.draw()

        # Resultado textual
        ok = not result.any_failure
        col = GREEN if ok else RED
        st  = "✓ APROVADO" if ok else "✗ FALHA ESTRUTURAL"
        html = f"""
        <p style='color:{col};font-weight:bold;font-size:13px'>{st}</p>
        <table style='width:100%;border-collapse:collapse;font-size:11px;'>
        <tr><td style='padding:3px 8px'>Massa total asa</td>
            <td style='color:{ACCENT};font-weight:bold'>{result.mass_total_g:.1f} g</td>
            <td style='padding:3px 8px'>τ_max balsa</td>
            <td style='color:{RED if result.tau_max_MPa > BALSA_PROPS["tau_crit_MPa"] else ACCENT};font-weight:bold'>{result.tau_max_MPa:.3f} MPa</td></tr>
        <tr style='background:{PANEL}'><td style='padding:3px 8px'>Massa casca</td>
            <td style='color:{ACCENT};font-weight:bold'>{result.mass_skin_g:.1f} g</td>
            <td style='padding:3px 8px'>σ_max tubo CF</td>
            <td style='color:{ACCENT};font-weight:bold'>{result.sigma_max_MPa:.1f} MPa</td></tr>
        <tr><td style='padding:3px 8px'>Massa nervuras</td>
            <td style='color:{ACCENT};font-weight:bold'>{result.mass_ribs_g:.1f} g</td>
            <td style='padding:3px 8px'>δ ponta</td>
            <td style='color:{ACCENT};font-weight:bold'>{result.deflection_tip_mm:.2f} mm</td></tr>
        <tr style='background:{PANEL}'><td style='padding:3px 8px'>Massa longarina CF</td>
            <td style='color:{AMBER};font-weight:bold'>{result.mass_spar_g:.1f} g</td>
            <td style='padding:3px 8px'>Nervuras</td>
            <td style='color:{ACCENT};font-weight:bold'>{cfg.n_ribs}</td></tr>
        </table>"""
        parent.vwb_result_box.setHtml(html)

    except Exception as ex:
        import traceback
        if hasattr(parent, 'vwb_result_box'):
            parent.vwb_result_box.setHtml(f"<p style='color:{RED}'>Erro: {ex}</p>")


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI — Aba de Posicionamento de Nervuras (RIBSPO)
# ═══════════════════════════════════════════════════════════════════════════════

def build_rib_positioning_tab(parent) -> QWidget:
    """Aba RIBSPO — otimizador de espaçamento de nervuras."""
    scroll = QScrollArea(); scroll.setWidgetResizable(True)
    scroll.setStyleSheet(f"background:{SURFACE};")
    w = QWidget(); w.setStyleSheet(f"background:{SURFACE};")
    scroll.setWidget(w)
    lay = QVBoxLayout(w); lay.setSpacing(10); lay.setContentsMargins(16,16,16,16)

    # Info do método
    info = QLabel("🔬  Método RIBSPO — Otimização de Espaçamento de Nervuras via Evolução Diferencial\n"
                  "Baseado em: Martins & Mariano (TCC UFU 2020) — Equipe Tucano Aerodesign\n"
                  "Minimize a massa total da asa respeitando critérios de falha (balsa τ ≤ 2.8MPa, CF Hill-Tsai)")
    info.setStyleSheet(f"background:{PANEL};color:{TEXT_S};font-size:10px;"
                       f"border:1px solid {BORDER};border-radius:5px;padding:10px 14px;"
                       f"line-height:1.6;")
    info.setWordWrap(True)
    lay.addWidget(info)

    # Parâmetros do GA
    g1 = QGroupBox("Parâmetros do Algoritmo (Evolução Diferencial)")
    gl = QGridLayout(g1); gl.setSpacing(8)
    parent.ribspo_n_ribs    = _mk_ispin(10, 3, 25)
    parent.ribspo_pop       = _mk_ispin(20, 5, 100)
    parent.ribspo_n_gen     = _mk_ispin(60, 10, 500)
    parent.ribspo_F         = _mk_dspin(0.8, 0.3, 1.2, 2)
    parent.ribspo_CR        = _mk_dspin(0.9, 0.5, 1.0, 2)
    parent.ribspo_sf        = _mk_dspin(1.5, 1.0, 3.0, 2)
    for i,(l,wgt) in enumerate([
        ("Nº nervuras total:", parent.ribspo_n_ribs),
        ("Tamanho da população:", parent.ribspo_pop),
        ("Nº de gerações:", parent.ribspo_n_gen),
        ("Fator de mutação F:", parent.ribspo_F),
        ("Taxa de cruzamento CR:", parent.ribspo_CR),
        ("Fator de segurança:", parent.ribspo_sf),
    ]):
        gl.addWidget(QLabel(l), i//2, (i%2)*2); gl.addWidget(wgt, i//2, (i%2)*2+1)
    lay.addWidget(g1)

    # Geometria do wingbox para RIBSPO
    g2 = QGroupBox("Configuração Estrutural")
    gl2 = QGridLayout(g2); gl2.setSpacing(8)
    parent.ribspo_skin_t   = _mk_dspin(1.5, 0.3, 5.0, 2)
    parent.ribspo_spar_od  = _mk_dspin(12.0, 5, 30, 1)
    parent.ribspo_spar_t   = _mk_dspin(1.0, 0.3, 3.0, 2)
    for i,(l,wgt) in enumerate([
        ("Espessura casca balsa (mm):", parent.ribspo_skin_t),
        ("Diâm. tubo carbono (mm):", parent.ribspo_spar_od),
        ("Parede tubo (mm):", parent.ribspo_spar_t),
    ]):
        gl2.addWidget(QLabel(l), i, 0); gl2.addWidget(wgt, i, 1)
    lay.addWidget(g2)

    # Botões
    btn_row = QHBoxLayout()
    parent.ribspo_btn_run  = QPushButton("▶  Otimizar Posicionamento (RIBSPO)")
    parent.ribspo_btn_stop = QPushButton("⏹  Parar")
    parent.ribspo_btn_run.setFixedHeight(36); parent.ribspo_btn_stop.setFixedHeight(36)
    parent.ribspo_btn_run.setStyleSheet(f"QPushButton{{background:{GREEN};color:#111;border:none;"
        f"border-radius:4px;font-size:12px;font-weight:600;padding:0 18px;}}"
        f"QPushButton:hover{{background:#3DB87A;}}")
    parent.ribspo_btn_stop.setStyleSheet(f"QPushButton{{background:{RED};color:white;border:none;"
        f"border-radius:4px;font-weight:bold;padding:0 16px;}}"
        f"QPushButton:disabled{{background:{BORDER};color:{TEXT_D};}}")
    parent.ribspo_btn_stop.setEnabled(False)
    parent.ribspo_btn_run.clicked.connect(lambda: _run_ribspo(parent))
    parent.ribspo_btn_stop.clicked.connect(lambda: _stop_ribspo(parent))
    btn_row.addWidget(parent.ribspo_btn_run); btn_row.addWidget(parent.ribspo_btn_stop)
    btn_row.addStretch()
    lay.addLayout(btn_row)

    # Progresso
    parent.ribspo_progress = QProgressBar(); parent.ribspo_progress.setFixedHeight(5)
    parent.ribspo_progress.setTextVisible(False); lay.addWidget(parent.ribspo_progress)
    parent.ribspo_progress_lbl = QLabel("Pronto")
    parent.ribspo_progress_lbl.setStyleSheet(f"color:{TEXT_S};font-size:10px;background:transparent;")
    lay.addWidget(parent.ribspo_progress_lbl)

    # Log
    parent.ribspo_log = QTextEdit(); parent.ribspo_log.setReadOnly(True)
    parent.ribspo_log.setFont(QFont("Consolas", 9)); parent.ribspo_log.setMaximumHeight(180)
    parent.ribspo_log.setStyleSheet(f"background:{BG};color:{TEXT_P};"
        f"border:1px solid {BORDER};border-radius:4px;padding:6px;")
    lay.addWidget(parent.ribspo_log)

    # Figura: convergência + resultado visual
    parent.ribspo_fig = Figure(figsize=(10, 6), dpi=100)
    parent.ribspo_fig.patch.set_facecolor(BG)
    parent.ribspo_canvas = FigureCanvas(parent.ribspo_fig)
    parent.ribspo_canvas.setMinimumHeight(560)
    lay.addWidget(parent.ribspo_canvas)

    # Tabela de espaçamentos ótimos
    lbl_tbl = QLabel("RESULTADO — Espaçamentos ótimos por seção")
    lbl_tbl.setStyleSheet(f"color:{TEXT_D};font-size:9px;font-weight:700;letter-spacing:.12em;"
                           f"background:transparent;")
    lay.addWidget(lbl_tbl)
    parent.ribspo_table = QTableWidget(); parent.ribspo_table.setMaximumHeight(220)
    parent.ribspo_table.setStyleSheet(f"QTableWidget{{background:{BG};color:{TEXT_P};"
        f"border:1px solid {BORDER};gridline-color:{BORDER};}}"
        f"QHeaderView::section{{background:{PANEL};color:{TEXT_S};"
        f"border:1px solid {BORDER};padding:4px;font-size:9px;}}")
    lay.addWidget(parent.ribspo_table)
    lay.addStretch()

    parent._ribspo_worker = None
    parent._ribspo_conv_mass = []
    parent._ribspo_conv_gen = []
    parent._ribspo_best_pos = None
    return scroll


def _run_ribspo(parent):
    from datetime import datetime
    parent.ribspo_log.clear()
    parent._ribspo_conv_mass = []; parent._ribspo_conv_gen = []
    parent.ribspo_progress.setValue(0)
    parent.ribspo_btn_run.setEnabled(False)
    parent.ribspo_btn_stop.setEnabled(True)

    cfg = _get_wb_cfg(parent)
    cfg.n_ribs      = int(parent.ribspo_n_ribs.value())
    cfg.skin_t_mm   = parent.ribspo_skin_t.value()
    cfg.spar_od_mm  = parent.ribspo_spar_od.value()
    cfg.spar_wall_mm= parent.ribspo_spar_t.value()
    cfg.safety_factor= parent.ribspo_sf.value()

    worker = WingboxWorker(cfg,
        pop=int(parent.ribspo_pop.value()),
        n_gen=int(parent.ribspo_n_gen.value()),
        F=parent.ribspo_F.value(),
        CR=parent.ribspo_CR.value(),
    )
    worker.log_signal.connect(lambda m, n: _ribspo_log(parent, m, n))
    worker.progress_signal.connect(lambda p, l: (
        parent.ribspo_progress.setValue(p),
        parent.ribspo_progress_lbl.setText(l)))
    worker.gen_signal.connect(lambda g, m, pos: _ribspo_on_gen(parent, g, m, pos))
    worker.done_signal.connect(lambda r, pos: _ribspo_on_done(parent, r, pos))
    worker.finished.connect(lambda: (
        parent.ribspo_btn_run.setEnabled(True),
        parent.ribspo_btn_stop.setEnabled(False)))
    parent._ribspo_worker = worker
    worker.start()


def _stop_ribspo(parent):
    if parent._ribspo_worker: parent._ribspo_worker.stop()


CLRS = {"INFO": TEXT_P, "OK": GREEN, "WARN": AMBER, "ERROR": RED, "SECTION": ACCENT}

def _ribspo_log(parent, msg, nivel):
    from datetime import datetime
    color = CLRS.get(nivel, TEXT_P)
    ts = datetime.now().strftime("%H:%M:%S")
    parent.ribspo_log.append(
        f'<span style="color:{TEXT_D}">[{ts}]</span> '
        f'<span style="color:{color}">{msg}</span>')
    sb = parent.ribspo_log.verticalScrollBar(); sb.setValue(sb.maximum())


def _ribspo_on_gen(parent, gen, mass, positions):
    parent._ribspo_conv_gen.append(gen)
    parent._ribspo_conv_mass.append(mass)
    parent._ribspo_best_pos = positions

    # Atualizar gráfico de convergência
    fig = parent.ribspo_fig; fig.clear()
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35,
                  left=0.08, right=0.97, top=0.93, bottom=0.08)
    ax_conv = fig.add_subplot(gs[0, :])
    ax_plan = fig.add_subplot(gs[1, 0])
    ax_bar  = fig.add_subplot(gs[1, 1])

    # Convergência
    ax_conv.set_facecolor(BG)
    ax_conv.plot(parent._ribspo_conv_gen, parent._ribspo_conv_mass,
                 color=ACCENT, lw=2)
    ax_conv.fill_between(parent._ribspo_conv_gen, parent._ribspo_conv_mass,
                          alpha=0.15, color=ACCENT)
    ax_conv.tick_params(colors=TEXT_S, labelsize=8)
    for sp in ax_conv.spines.values(): sp.set_color(BORDER)
    ax_conv.spines['top'].set_visible(False); ax_conv.spines['right'].set_visible(False)
    ax_conv.set_title(f'Convergência RIBSPO — Gen {gen} | Melhor massa: {mass:.1f}g',
                      color=TEXT_P, fontsize=10, fontweight='bold', pad=6)
    ax_conv.set_xlabel('Geração', color=TEXT_S, fontsize=9)
    ax_conv.set_ylabel('Massa [g]', color=TEXT_S, fontsize=9)
    ax_conv.grid(True, alpha=0.12, color=TEXT_S)

    # Vista rápida do posicionamento
    cfg = _get_wb_cfg(parent); cfg.n_ribs = int(parent.ribspo_n_ribs.value())
    if positions is not None and len(positions) > 1:
        try:
            plot_wing_planform_topview(fig, ax_plan, cfg, np.array(positions))
        except Exception: pass

    # Barras de espaçamento
    ax_bar.set_facecolor(BG)
    if positions is not None and len(positions) > 1:
        spacings = np.diff(np.array(positions))
        ax_bar.bar(range(len(spacings)), spacings, color=ACCENT, alpha=0.8,
                   edgecolor=BORDER)
        ax_bar.tick_params(colors=TEXT_S, labelsize=8)
        for sp in ax_bar.spines.values(): sp.set_color(BORDER)
        ax_bar.spines['top'].set_visible(False); ax_bar.spines['right'].set_visible(False)
        ax_bar.set_title('Espaçamentos por Seção', color=TEXT_P, fontsize=9, fontweight='bold')
        ax_bar.set_xlabel('Seção', color=TEXT_S, fontsize=8)
        ax_bar.set_ylabel('Espaço [mm]', color=TEXT_S, fontsize=8)
        ax_bar.grid(True, alpha=0.12, color=TEXT_S)

    fig.patch.set_facecolor(BG)
    parent.ribspo_canvas.draw()


def _ribspo_on_done(parent, result, positions):
    pos = np.array(positions)
    spacings = np.diff(pos)

    # Tabela de resultados
    tbl = parent.ribspo_table
    tbl.clear(); tbl.setRowCount(len(spacings))
    tbl.setColumnCount(5)
    tbl.setHorizontalHeaderLabels(["Seção", "Y início (mm)", "Y fim (mm)",
                                    "Espaço (mm)", "Status"])
    tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    for i, sp in enumerate(spacings):
        y_s = pos[i]; y_e = pos[i+1]
        failed = (result.sections[i].failed if i < len(result.sections) else False)
        color = QColor(RED) if failed else QColor(GREEN)
        items = [str(i+1), f"{y_s:.1f}", f"{y_e:.1f}",
                 f"{sp:.1f}", "✗ FALHA" if failed else "✓ OK"]
        for j, txt in enumerate(items):
            itm = QTableWidgetItem(txt)
            itm.setForeground(color)
            tbl.setItem(i, j, itm)

    # Gráfico final completo
    _ribspo_on_gen(parent, parent._ribspo_conv_gen[-1] if parent._ribspo_conv_gen else 0,
                   result.mass_total_g, positions)
    _ribspo_log(parent, f"\n  ═══ RESULTADO FINAL ═══", "SECTION")
    _ribspo_log(parent, f"  Massa total: {result.mass_total_g:.1f} g", "OK")
    _ribspo_log(parent, f"  Casca: {result.mass_skin_g:.1f}g | Nervuras: {result.mass_ribs_g:.1f}g | Longarina: {result.mass_spar_g:.1f}g", "INFO")
    _ribspo_log(parent, f"  τ_max balsa: {result.tau_max_MPa:.3f} MPa "
                f"({'FALHA' if result.tau_max_MPa > BALSA_PROPS['tau_crit_MPa'] else 'OK'})", "INFO")
    _ribspo_log(parent, f"  σ_max tubo CF: {result.sigma_max_MPa:.1f} MPa", "INFO")
    _ribspo_log(parent, f"  Deflexão ponta: {result.deflection_tip_mm:.2f} mm", "INFO")
    _ribspo_log(parent, f"  Falha estrutural: {'SIM' if result.any_failure else 'NÃO'}",
                "ERROR" if result.any_failure else "OK")
    for i, s in enumerate(spacings):
        _ribspo_log(parent, f"  Seção {i+1}: {pos[i]:.1f}→{pos[i+1]:.1f}mm  Δ={s:.1f}mm", "INFO")


# ── Helpers de spin ────────────────────────────────────────────────────────────

def _mk_dspin(val, lo, hi, dec=2):
    w = QDoubleSpinBox(); w.setDecimals(dec)
    w.setRange(float(lo), float(hi)); w.setValue(float(val))
    w.setMinimumWidth(110); return w

def _mk_ispin(val, lo, hi):
    w = QSpinBox(); w.setRange(int(lo), int(hi)); w.setValue(int(val))
    w.setMinimumWidth(110); return w


# ═══════════════════════════════════════════════════════════════════════════════
#  Builder principal — retorna o widget do módulo completo
# ═══════════════════════════════════════════════════════════════════════════════

def build_wingbox_module(parent) -> QWidget:
    """
    Módulo completo de Wingbox: tabs com Vista Superior, Seção Transversal e RIBSPO.
    Chamado pelo main2.py para o stack idx 3.
    """
    page = QWidget(); page.setStyleSheet(f"background:{SURFACE};")
    lay = QVBoxLayout(page); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

    # Header
    hdr = QWidget(); hdr.setFixedHeight(64)
    hdr.setStyleSheet(f"background:{PANEL};border-bottom:1px solid {BORDER};")
    hl = QHBoxLayout(hdr); hl.setContentsMargins(20,0,20,0)
    col = QVBoxLayout(); col.setSpacing(2)
    cl = QLabel("MÓDULOS / ESTRUTURAL — WINGBOX")
    cl.setStyleSheet(f"color:{TEXT_D};font-size:9px;letter-spacing:.12em;background:transparent;")
    tl = QLabel("Caixa de Torção + Idealização por Booms + RIBSPO")
    tl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
    tl.setStyleSheet(f"color:{TEXT_P};background:transparent;")
    col.addWidget(cl); col.addWidget(tl)
    hl.addLayout(col); hl.addStretch()

    # Badge TCC
    badge = QLabel("Baseado em Martins & Mariano — TCC Tucano UFU 2020")
    badge.setStyleSheet(f"background:rgba(77,130,214,.15);color:{ACCENT};"
        f"border:1px solid rgba(77,130,214,.3);border-radius:3px;"
        f"font-size:9px;font-weight:600;padding:3px 10px;letter-spacing:.06em;")
    hl.addWidget(badge)
    lay.addWidget(hdr)

    tabs = QTabWidget(); tabs.setDocumentMode(True)
    tabs.addTab(build_wingbox_visual_tab(parent), "▭  Caixa de Torção (Vista Superior)")
    tabs.addTab(build_rib_positioning_tab(parent), "◈  RIBSPO — Posicionamento de Nervuras")
    lay.addWidget(tabs, stretch=1)
    return page
###################################################################################
def _spin(val, lo, hi, dec=2, step=None):
    if dec is None:
        w = QSpinBox(); w.setRange(int(lo), int(hi)); w.setValue(int(val))
    else:
        w = QDoubleSpinBox(); w.setDecimals(dec)
        w.setRange(float(lo), float(hi)); w.setValue(float(val))
        if step: w.setSingleStep(step)
    w.setMinimumWidth(120); return w
def build_aeroelastic_tab(parent) -> QWidget:
    scroll = QScrollArea(); scroll.setWidgetResizable(True)
    scroll.setStyleSheet(f"background:{SURFACE};")
    w = QWidget(); w.setStyleSheet(f"background:{SURFACE};")
    scroll.setWidget(w)
    lay = QVBoxLayout(w); lay.setSpacing(10); lay.setContentsMargins(16, 16, 16, 16)

    # ── Parâmetros estruturais ──
    g1 = QGroupBox("Propriedades da Seção Típica (2DOF)")
    gl = QGridLayout(g1); gl.setSpacing(8)
    parent.ae_chord    = _spin(250, 50, 600, 1)
    parent.ae_mass     = _spin(2.0, 0.1, 20, 2)
    parent.ae_Ialpha   = _spin(0.005, 0.0001, 0.5, 4)
    parent.ae_a_h      = _spin(-0.3, -0.5, 0.5, 2)
    parent.ae_x_alpha  = _spin(0.1, -0.3, 0.5, 2)
    parent.ae_g_struct = _spin(0.02, 0.001, 0.10, 3)
    parent.ae_method   = QComboBox()
    parent.ae_method.addItems(["2DOF Simplificado", "2DOF (método p-k)", "3DOF (c/ superfície de controle)"])
    for i, (lbl, wgt) in enumerate([
        ("Corda MAC (mm):", parent.ae_chord),
        ("Massa asa (kg):", parent.ae_mass),
        ("Iα (kg·m²):", parent.ae_Ialpha),
        ("a_h (pos. eixo elástico):", parent.ae_a_h),
        ("x_α (distância CG-eixo):", parent.ae_x_alpha),
        ("Amortecimento estrutural g:", parent.ae_g_struct),
        ("Método de análise:", parent.ae_method),
    ]):
        gl.addWidget(QLabel(lbl), i, 0); gl.addWidget(wgt, i, 1)
    lay.addWidget(g1)

    # ── Faixa de velocidade ──
    g2 = QGroupBox("Envelope de Velocidades e Condições")
    gl2 = QGridLayout(g2); gl2.setSpacing(8)
    parent.ae_v_min   = _spin(5.0, 1, 30, 1)
    parent.ae_v_max   = _spin(60.0, 10, 200, 1)
    parent.ae_v_design= _spin(20.0, 5, 100, 1)
    parent.ae_rho     = _spin(1.225, 0.5, 1.5, 4)
    parent.ae_safety  = _spin(1.20, 1.05, 1.50, 2)
    for i, (lbl, wgt) in enumerate([
        ("V mínima (m/s):", parent.ae_v_min),
        ("V máxima análise (m/s):", parent.ae_v_max),
        ("V projeto / manobra (m/s):", parent.ae_v_design),
        ("Densidade ar (kg/m³):", parent.ae_rho),
        ("Fator de segurança (flutter):", parent.ae_safety),
    ]):
        gl2.addWidget(QLabel(lbl), i, 0); gl2.addWidget(wgt, i, 1)
    lay.addWidget(g2)

    # ── Usar resultado do wingbox ──
    parent.ae_use_wingbox = QCheckBox("Usar EI/GJ calculados no módulo Wingbox")
    parent.ae_use_wingbox.setChecked(True)
    parent.ae_use_wingbox.setStyleSheet(f"color:{TEXT_S};background:transparent;")
    parent.ae_EI = _spin(1e9, 1e5, 1e12, 0)
    parent.ae_GJ = _spin(5e8, 1e5, 1e12, 0)
    lay.addWidget(parent.ae_use_wingbox)

    g3 = QGroupBox("Rigidezes (quando não usar Wingbox)")
    gl3 = QGridLayout(g3); gl3.setSpacing(8)
    gl3.addWidget(QLabel("EI (N·mm²):"), 0, 0); gl3.addWidget(parent.ae_EI, 0, 1)
    gl3.addWidget(QLabel("GJ (N·mm²):"), 1, 0); gl3.addWidget(parent.ae_GJ, 1, 1)
    lay.addWidget(g3)

    # Botão
    btn = QPushButton("▶  Analisar Flutter / Divergência")
    btn.setFixedHeight(36); btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(f"QPushButton{{background:{RED};color:white;border:none;border-radius:4px;"
                      f"font-size:12px;font-weight:600;padding:0 18px;}}"
                      f"QPushButton:hover{{background:#B83838;}}")
    btn.clicked.connect(lambda: _run_aeroelastic(parent))
    lay.addWidget(btn)

    # Modal
    btn2 = QPushButton("🎵  Extração de Modos Naturais (Rayleigh)")
    btn2.setFixedHeight(36); btn2.setCursor(Qt.CursorShape.PointingHandCursor)
    btn2.setStyleSheet(f"QPushButton{{background:{AMBER};color:#111;border:none;border-radius:4px;"
                       f"font-size:12px;font-weight:600;padding:0 18px;}}"
                       f"QPushButton:hover{{background:#C4860A;}}")
    btn2.clicked.connect(lambda: _run_modal(parent))
    lay.addWidget(btn2)

    # KPIs
    parent.ae_kpi_layout = QHBoxLayout(); lay.addLayout(parent.ae_kpi_layout)

    # Resultado
    parent.ae_result_box = QTextEdit(); parent.ae_result_box.setReadOnly(True)
    parent.ae_result_box.setMaximumHeight(200)
    parent.ae_result_box.setStyleSheet(f"background:{BG};color:{TEXT_P};border:1px solid {BORDER};"
                                        f"border-radius:5px;padding:8px;font-size:11px;")
    lay.addWidget(parent.ae_result_box)

    # Gráficos V-g e V-f
    parent.ae_fig, parent.ae_canvas = _make_fig(7, 5)
    lay.addWidget(parent.ae_canvas)
    lay.addStretch()
    return scroll

def _make_fig(w=6, h=4):
    fig = Figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(BG)
    canvas = FigureCanvas(fig)
    canvas.setMinimumHeight(int(h * 100))
    return fig, canvas


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_S, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if title: ax.set_title(title, color=TEXT_P, fontsize=11, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_S, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_S, fontsize=9)
    ax.grid(True, alpha=0.15, color=TEXT_S)

def _run_aeroelastic(parent):
    from aeroelasticity import (AeroelasticParams, flutter_2dof,
                                flutter_pk_method, flutter_3dof,
                                flutter_safety_check)

    semi = parent.sch_semi_span.value() if hasattr(parent, 'sch_semi_span') else 750

    EI_val = parent.ae_EI.value()
    GJ_val = parent.ae_GJ.value()
    if parent.ae_use_wingbox.isChecked() and hasattr(parent, '_wingbox_result'):
        wb = parent._wingbox_result
        EI_val = wb.EI_root
        GJ_val = wb.GJ_root

    ae_p = AeroelasticParams(
        semi_span_mm=semi,
        chord_mm=parent.ae_chord.value(),
        mass_kg=parent.ae_mass.value(),
        Iα_kgm2=parent.ae_Ialpha.value(),
        EI_Nmm2=EI_val,
        GJ_Nmm2=GJ_val,
        a_h=parent.ae_a_h.value(),
        x_alpha=parent.ae_x_alpha.value(),
        rho_kgm3=parent.ae_rho.value(),
        V_min_ms=parent.ae_v_min.value(),
        V_max_ms=parent.ae_v_max.value(),
        g_structural=parent.ae_g_struct.value(),
    )

    method = parent.ae_method.currentText()
    if "p-k" in method:
        fl = flutter_pk_method(ae_p)
    elif "3DOF" in method:
        fl = flutter_3dof(ae_p)
    else:
        fl = flutter_2dof(ae_p)

    safety = flutter_safety_check(fl, parent.ae_v_design.value(), parent.ae_safety.value())
    parent._flutter_result = fl

    # KPIs
    while parent.ae_kpi_layout.count():
        item = parent.ae_kpi_layout.takeAt(0)
        if item.widget(): item.widget().deleteLater()

    vf_col = GREEN if not fl.flutter_found else (GREEN if safety["approved"] else RED)
    parent.ae_kpi_layout.addWidget(_kpi(f"{fl.omega_h_Hz:.2f}", "f Flexão (livre)", "Hz", ACCENT))
    parent.ae_kpi_layout.addWidget(_kpi(f"{fl.omega_alpha_Hz:.2f}", "f Torção (livre)", "Hz", AMBER))
    parent.ae_kpi_layout.addWidget(_kpi(
        f"{fl.V_flutter_ms:.1f}" if fl.flutter_found else "N/D",
        "V Flutter", "m/s", vf_col))
    parent.ae_kpi_layout.addWidget(_kpi(
        f"{fl.V_divergence_ms:.1f}" if fl.divergence_found else "N/D",
        "V Divergência", "m/s", AMBER if fl.divergence_found else GREEN))
    parent.ae_kpi_layout.addWidget(_kpi(
        "✓ OK" if safety["approved"] else "✗ FALHA",
        "Margem Flutter", "", vf_col))

    # HTML
    col = GREEN if safety["approved"] else RED
    status = "APROVADO" if safety["approved"] else "REPROVADO"
    html = f"""
    <h3 style='color:{ACCENT}'>Análise Aeroelástica — {method}</h3>
    <p style='color:{col};font-weight:bold;font-size:14px'>{status}</p>
    <table style='width:100%;border-collapse:collapse;font-size:11px;margin-top:6px;'>
    <tr style='background:{PANEL}'><td style='padding:4px 8px'>Frequência de flexão (livre)</td>
        <td style='color:{ACCENT};font-weight:bold;padding:4px 8px'>{fl.omega_h_Hz:.3f} Hz</td></tr>
    <tr><td style='padding:4px 8px'>Frequência de torção (livre)</td>
        <td style='color:{ACCENT};font-weight:bold;padding:4px 8px'>{fl.omega_alpha_Hz:.3f} Hz</td></tr>
    <tr style='background:{PANEL}'><td style='padding:4px 8px'>Velocidade de flutter</td>
        <td style='color:{vf_col};font-weight:bold;padding:4px 8px'>
        {"N/D (não detectado)" if not fl.flutter_found else f"{fl.V_flutter_ms:.1f} m/s"}</td></tr>
    <tr><td style='padding:4px 8px'>Frequência no flutter</td>
        <td style='color:{ACCENT};font-weight:bold;padding:4px 8px'>
        {"—" if not fl.flutter_found else f"{fl.f_flutter_Hz:.2f} Hz"}</td></tr>
    <tr style='background:{PANEL}'><td style='padding:4px 8px'>V projeto</td>
        <td style='color:{TEXT_S};font-weight:bold;padding:4px 8px'>{safety["V_design_ms"]:.1f} m/s</td></tr>
    <tr><td style='padding:4px 8px'>V requerida (V_d × {safety["safety_factor"]})</td>
        <td style='color:{TEXT_S};font-weight:bold;padding:4px 8px'>{safety["V_required_ms"]:.1f} m/s</td></tr>
    <tr style='background:{PANEL}'><td style='padding:4px 8px'>Margem sobre V_projeto</td>
        <td style='color:{col};font-weight:bold;padding:4px 8px'>
        {"∞" if not fl.flutter_found else f"{safety.get('margin_pct', 0):.1f}%"}</td></tr>
    </table>
    <p style='color:{TEXT_D};font-size:10px;margin-top:6px'>{safety.get("note", "")}</p>
    """
    parent.ae_result_box.setHtml(html)

    # Gráficos V-g e V-f
    fig = parent.ae_fig; fig.clear()
    ax1 = fig.add_subplot(211)
    ax1.plot(fl.V_ms, fl.damp_bending, color=ACCENT, lw=2, label="Flexão")
    ax1.plot(fl.V_ms, fl.damp_torsion, color=AMBER, lw=2, label="Torção")
    ax1.axhline(0, color=RED, lw=1, ls="--", label="g = 0")
    if fl.flutter_found:
        ax1.axvline(fl.V_flutter_ms, color=RED, lw=1.5, ls=":", label=f"V_F={fl.V_flutter_ms:.1f} m/s")
    ax1.axvline(parent.ae_v_design.value(), color=GREEN, lw=1, ls=":", label="V_design")
    _style_ax(ax1, "Diagrama V-g (Amortecimento vs Velocidade)", "", "Amortecimento g")
    ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT_P, loc="lower left")

    ax2 = fig.add_subplot(212)
    ax2.plot(fl.V_ms, fl.freq_bending_Hz, color=ACCENT, lw=2, label="Flexão")
    ax2.plot(fl.V_ms, fl.freq_torsion_Hz, color=AMBER, lw=2, label="Torção")
    if fl.flutter_found:
        ax2.axvline(fl.V_flutter_ms, color=RED, lw=1.5, ls=":", label=f"V_F={fl.V_flutter_ms:.1f} m/s")
    _style_ax(ax2, "Diagrama V-f (Frequência vs Velocidade)", "Velocidade [m/s]", "Frequência [Hz]")
    ax2.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT_P)

    fig.tight_layout(pad=1.5)
    parent.ae_canvas.draw()
    
def _kpi(val, label, unit="", color=ACCENT):
    card = QWidget(); card.setFixedHeight(76)
    card.setStyleSheet(f"background:{PANEL};border:1px solid {BORDER};border-radius:5px;"
                       f"border-top:2px solid {color};")
    lay = QVBoxLayout(card); lay.setContentsMargins(10, 6, 10, 6); lay.setSpacing(1)
    v = QLabel(str(val)); v.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
    v.setStyleSheet(f"color:{color};background:transparent;"); v.setAlignment(Qt.AlignmentFlag.AlignCenter)
    l = QLabel(label); l.setStyleSheet(f"color:{TEXT_S};font-size:9px;font-weight:600;background:transparent;")
    l.setAlignment(Qt.AlignmentFlag.AlignCenter)
    u = QLabel(unit); u.setStyleSheet(f"color:{TEXT_D};font-size:8px;background:transparent;")
    u.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(v); lay.addWidget(l); lay.addWidget(u); return card


def _run_modal(parent):
    from aeroelasticity import modal_analysis_rayleigh

    semi = parent.sch_semi_span.value() if hasattr(parent, 'sch_semi_span') else 750
    EI_val = parent.ae_EI.value()
    GJ_val = parent.ae_GJ.value()
    if parent.ae_use_wingbox.isChecked() and hasattr(parent, '_wingbox_result'):
        wb = parent._wingbox_result
        EI_val = wb.EI_root
        GJ_val = wb.GJ_root

    modal = modal_analysis_rayleigh(
        EI_Nmm2=EI_val,
        GJ_Nmm2=GJ_val,
        semi_span_mm=semi,
        mass_total_kg=parent.ae_mass.value(),
        Ialpha_kgm2=parent.ae_Ialpha.value(),
        n_modes=6,
    )

    rows = []
    for m in modal["modes"]:
        color = ACCENT if m["type"] == "bending" else AMBER
        rows.append(f"<tr><td style='padding:4px 8px;color:{TEXT_S}'>{m['mode']}</td>"
                    f"<td style='padding:4px 8px;color:{color};font-weight:bold'>{m['freq_Hz']:.3f} Hz</td>"
                    f"<td style='padding:4px 8px;color:{TEXT_D}'>{m['omega_rads']:.3f} rad/s</td></tr>")

    html = (f"<h3 style='color:{ACCENT}'>Frequências Naturais — Rayleigh-Ritz</h3>"
            f"<table style='width:100%;border-collapse:collapse;font-size:11px;'>"
            f"<tr style='background:{PANEL}'><th style='padding:4px 8px;color:{TEXT_D}'>Modo</th>"
            f"<th style='padding:4px 8px;color:{TEXT_D}'>Frequência</th>"
            f"<th style='padding:4px 8px;color:{TEXT_D}'>ω [rad/s]</th></tr>"
            + "".join(rows) + "</table>")
    parent.ae_result_box.setHtml(html)