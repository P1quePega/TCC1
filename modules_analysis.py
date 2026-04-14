"""
modules_analysis.py — Módulos adicionais para o AeroStruct Suite
Integra: Schrenk, Peso/CG, Entelagem e Sensibilidade na GUI PyQt6.

USO: Importar no main2.py e chamar os builders na classe NervuraApp.
"""

import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QDoubleSpinBox, QSpinBox, QTabWidget,
    QTextEdit, QScrollArea, QComboBox, QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# Cores do tema (importar do main2.py ou redefinir)
BG       = "#111318"
SURFACE  = "#1C2030"
PANEL    = "#21263C"
RAISED   = "#282E48"
BORDER   = "#30395A"
BORDER2  = "#3E4C70"
ACCENT   = "#4D82D6"
TEXT_P   = "#C8D4EC"
TEXT_S   = "#6A7A9C"
TEXT_D   = "#3A4562"
GREEN    = "#4EC88A"
RED      = "#D95252"
AMBER    = "#D9963A"


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers de estilo
# ═══════════════════════════════════════════════════════════════════════════════

def _spin(val, min_v, max_v, decimals=2, step=None):
    if decimals is None:
        w = QSpinBox()
        w.setRange(int(min_v), int(max_v))
        w.setValue(int(val))
    else:
        w = QDoubleSpinBox()
        w.setDecimals(decimals)
        w.setRange(float(min_v), float(max_v))
        w.setValue(float(val))
        if step:
            w.setSingleStep(step)
    w.setMinimumWidth(120)
    return w


def _kpi_card(val_str, label, unit="", color=ACCENT):
    card = QWidget()
    card.setFixedHeight(80)
    card.setStyleSheet(f"""
        background: {PANEL}; border: 1px solid {BORDER};
        border-radius: 5px; border-top: 2px solid {color};
    """)
    lay = QVBoxLayout(card)
    lay.setContentsMargins(12, 8, 12, 8)
    lay.setSpacing(2)

    v = QLabel(val_str)
    v.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
    v.setStyleSheet(f"color: {color}; background: transparent;")
    v.setAlignment(Qt.AlignmentFlag.AlignCenter)

    l = QLabel(label)
    l.setStyleSheet(f"color: {TEXT_S}; font-size: 9px; font-weight: 600; "
                    f"letter-spacing: 0.05em; background: transparent;")
    l.setAlignment(Qt.AlignmentFlag.AlignCenter)

    u = QLabel(unit)
    u.setStyleSheet(f"color: {TEXT_D}; font-size: 9px; background: transparent;")
    u.setAlignment(Qt.AlignmentFlag.AlignCenter)

    lay.addWidget(v)
    lay.addWidget(l)
    lay.addWidget(u)
    return card


def _make_fig(width=5, height=3.5):
    fig = Figure(figsize=(width, height), dpi=100)
    fig.patch.set_facecolor(BG)
    canvas = FigureCanvas(fig)
    canvas.setMinimumHeight(int(height * 100))
    return fig, canvas


def _style_axis(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_S, labelsize=9)
    ax.spines['bottom'].set_color(BORDER)
    ax.spines['left'].set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, color=TEXT_P, fontsize=11, fontweight='bold', pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_S, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_S, fontsize=9)
    ax.grid(True, alpha=0.15, color=TEXT_S)


# ═══════════════════════════════════════════════════════════════════════════════
#  Módulo: Distribuição de Sustentação (Schrenk)
# ═══════════════════════════════════════════════════════════════════════════════

def build_schrenk_tab(parent) -> QWidget:
    """Constrói a aba de distribuição de Schrenk e Discretização de Nervuras."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setStyleSheet(f"background: {SURFACE};")
    w = QWidget()
    w.setStyleSheet(f"background: {SURFACE};")
    scroll.setWidget(w)
    lay = QVBoxLayout(w)
    lay.setSpacing(10)
    lay.setContentsMargins(16, 16, 16, 16)

    # ── Inputs ──
    g1 = QGroupBox("Geometria da Asa (meia-envergadura)")
    gl1 = QGridLayout(g1)
    gl1.setSpacing(8)
    parent.sch_semi_span = _spin(750, 50, 5000, 1)
    parent.sch_root_chord = _spin(300, 50, 1000, 1)
    parent.sch_tip_chord = _spin(200, 50, 1000, 1)
    for i, (lbl, w_) in enumerate([
        ("Semi-envergadura (mm):", parent.sch_semi_span),
        ("Corda na raiz (mm):",    parent.sch_root_chord),
        ("Corda na ponta (mm):",   parent.sch_tip_chord),
    ]):
        gl1.addWidget(QLabel(lbl), i, 0)
        gl1.addWidget(w_,          i, 1)
    lay.addWidget(g1)

    # ── Nova seção g2 com os parâmetros do MATLAB ──
    g2 = QGroupBox("Condição de Voo e Discretização")
    gl2 = QGridLayout(g2)
    gl2.setSpacing(8)
    parent.sch_n_ribs = _spin(12, 2, 50, 0)
    parent.sch_velocity = _spin(44.0, 1, 100, 1)
    parent.sch_rho = _spin(1.225, 0.5, 1.5, 4)
    parent.sch_load_factor = _spin(6.0, 1, 15, 1)
    parent.sch_mass = _spin(650.0, 0.5, 1000, 1)
    for i, (lbl, w_) in enumerate([
        ("Número de Nervuras (N):", parent.sch_n_ribs),
        ("Velocidade (m/s):",       parent.sch_velocity),
        ("Densidade ar (kg/m³):",   parent.sch_rho),
        ("Fator de carga (n):",     parent.sch_load_factor),
        ("Massa de decolagem (kg):",parent.sch_mass),
    ]):
        gl2.addWidget(QLabel(lbl), i, 0)
        gl2.addWidget(w_,          i, 1)
    lay.addWidget(g2)

    # ── Novos Botões de Ação Integrados ──
    btn_row = QHBoxLayout()
    btn = QPushButton("▶  Calcular Cargas por Nervura")
    btn.setFixedHeight(36)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(f"QPushButton {{ background: {ACCENT}; color: white; border-radius: 4px; font-weight: bold; padding: 0 16px; }}")
    btn.clicked.connect(lambda: _run_schrenk(parent))
    
    parent.btn_trans_schrenk = QPushButton("⮂  Transferir Crítica para MAPDL")
    parent.btn_trans_schrenk.setFixedHeight(36)
    parent.btn_trans_schrenk.setEnabled(False)
    parent.btn_trans_schrenk.setCursor(Qt.CursorShape.PointingHandCursor)
    parent.btn_trans_schrenk.setStyleSheet(f"QPushButton {{ background: {GREEN}; color: #111; border-radius: 4px; font-weight: bold; padding: 0 16px; }} QPushButton:disabled {{ background: {BORDER}; color: {TEXT_D}; }}")
    parent.btn_trans_schrenk.clicked.connect(lambda: _transfer_to_mapdl(parent))

    parent.btn_exp_schrenk = QPushButton("↓  Exportar para CAD/SpaceClaim")
    parent.btn_exp_schrenk.setFixedHeight(36)
    parent.btn_exp_schrenk.setEnabled(False)
    parent.btn_exp_schrenk.setCursor(Qt.CursorShape.PointingHandCursor)
    parent.btn_exp_schrenk.setStyleSheet(f"QPushButton {{ background: {AMBER}; color: #111; border-radius: 4px; font-weight: bold; padding: 0 16px; }} QPushButton:disabled {{ background: {BORDER}; color: {TEXT_D}; }}")
    parent.btn_exp_schrenk.clicked.connect(lambda: _export_cad(parent))

    btn_row.addWidget(btn)
    btn_row.addWidget(parent.btn_trans_schrenk)
    btn_row.addWidget(parent.btn_exp_schrenk)
    lay.addLayout(btn_row)

    # KPIs
    parent.sch_kpi_layout = QHBoxLayout()
    lay.addLayout(parent.sch_kpi_layout)

    # Gráficos
    parent.sch_fig, parent.sch_canvas = _make_fig(7, 8)
    lay.addWidget(parent.sch_canvas)

    lay.addStretch()
    return scroll

def _run_schrenk(parent):
    from schrenk import WingGeometry, FlightCondition, discretize_rib_loads_matlab
    import numpy as np

    wing = WingGeometry(
        semi_span_mm=parent.sch_semi_span.value(),
        root_chord_mm=parent.sch_root_chord.value(),
        tip_chord_mm=parent.sch_tip_chord.value(),
    )
    flight = FlightCondition(
        velocity_ms=parent.sch_velocity.value(),
        rho_kgm3=parent.sch_rho.value(),
        load_factor=parent.sch_load_factor.value(),
        aircraft_mass_kg=parent.sch_mass.value(),
    )
    n_ribs = int(parent.sch_n_ribs.value())

    # Roda a função adaptada do MATLAB
    res = discretize_rib_loads_matlab(wing, flight, n_ribs)
    parent._schrenk_res_disc = res  # Salva para uso na exportação MDO

    idx_c = res["idx_crit"]
    forca_max = res["force_per_rib"][idx_c]
    corda_crit = res["chord_ribs"][idx_c]
    pressao_crit = res["pressure_per_rib"][idx_c]

    # Atualizar KPIs
    while parent.sch_kpi_layout.count():
        item = parent.sch_kpi_layout.takeAt(0)
        if item.widget(): item.widget().deleteLater()

    parent.sch_kpi_layout.addWidget(_kpi_card(f"{res['req_lift_half']:.1f}", "Sustentação Estrutural", "N", GREEN))
    parent.sch_kpi_layout.addWidget(_kpi_card(f"{res['scale_factor']:.3f}", "Fator de Escala", "-", TEXT_S))
    parent.sch_kpi_layout.addWidget(_kpi_card(f"Nerv. {idx_c+1}", "Nervura Crítica", f"y = {res['y_ribs'][idx_c]:.1f}mm", RED))
    parent.sch_kpi_layout.addWidget(_kpi_card(f"{forca_max:.1f}", "Carga Crítica (Baia)", "N", RED))

    # Ativa botões de integração
    parent.btn_trans_schrenk.setEnabled(True)
    parent.btn_exp_schrenk.setEnabled(True)

    # ═════════════════════════════════════════════════════════════════════════
    #  Cálculo dos Esforços Internos (Cortante, Fletor e Torsor)
    # ═════════════════════════════════════════════════════════════════════════
    y_sch = res["y_schrenk"]
    L_esc = res["L_scaled"]
    
    # 1. Esforço Cortante (V): Integração da ponta para a raiz
    V_esc = np.zeros_like(y_sch)
    for i in range(len(y_sch) - 2, -1, -1):
        dy = y_sch[i+1] - y_sch[i]
        V_esc[i] = V_esc[i+1] + L_esc[i] * dy
        
    # 2. Momento Fletor (M): Integração do Cortante
    M_esc = np.zeros_like(y_sch)
    for i in range(len(y_sch) - 2, -1, -1):
        dy = y_sch[i+1] - y_sch[i]
        M_esc[i] = M_esc[i+1] + V_esc[i] * dy

    # 3. Momento Torsor (T): Estimativa via Cm0
    # M_pitch = q * c^2 * Cm0. Usando Cm0 = -0.05 típico de perfil assimétrico.
    q_din_MPa = 0.5 * flight.rho_kgm3 * flight.velocity_ms**2 * 1e-6
    corda_y = np.interp(y_sch, res["y_ribs"], res["chord_ribs"])
    Cm0 = -0.05 
    t_dist = q_din_MPa * (corda_y**2) * Cm0 # Torque distribuído [N.mm / mm]
    
    T_esc = np.zeros_like(y_sch)
    for i in range(len(y_sch) - 2, -1, -1):
        dy = y_sch[i+1] - y_sch[i]
        T_esc[i] = T_esc[i+1] + t_dist[i] * dy

    # ═════════════════════════════════════════════════════════════════════════
    #  Plotagem dos 4 Gráficos
    # ═════════════════════════════════════════════════════════════════════════
    fig = parent.sch_fig
    fig.clear()
    
    # Gráfico 1: Sustentação (Barras + Contínuo)
    ax1 = fig.add_subplot(411)
    largura_barra = wing.semi_span_mm / n_ribs * 0.8
    ax1.bar(res["y_ribs"], res["force_per_rib"], width=largura_barra, color=ACCENT, edgecolor=BORDER, alpha=0.8, label="Força na Baia (N)")
    L_media_vao = L_esc * (wing.semi_span_mm / (n_ribs - 1))
    ax1.plot(y_sch, L_media_vao, 'r--', linewidth=1.5, label='Curva L(y) Ponderada')
    ax1.bar(res["y_ribs"][idx_c], res["force_per_rib"][idx_c], width=largura_barra, color=RED, edgecolor=BORDER, label="Nervura Crítica")
    _style_axis(ax1, f"Distribuição de Sustentação (MTOW={flight.aircraft_mass_kg}kg, n={flight.load_factor})", "", "Força [N]")
    ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT_P)

    # Gráfico 2: Esforço Cortante (V)
    ax2 = fig.add_subplot(412)
    ax2.plot(y_sch, V_esc, color=AMBER, linewidth=2)
    ax2.fill_between(y_sch, V_esc, alpha=0.15, color=AMBER)
    _style_axis(ax2, "Diagrama de Esforço Cortante (V)", "", "V [N]")

    # Gráfico 3: Momento Fletor (M)
    ax3 = fig.add_subplot(413)
    ax3.plot(y_sch, M_esc, color=RED, linewidth=2)
    ax3.fill_between(y_sch, M_esc, alpha=0.15, color=RED)
    _style_axis(ax3, "Diagrama de Momento Fletor (M)", "", "M [N·mm]")

    # Gráfico 4: Momento Torsor (T)
    ax4 = fig.add_subplot(414)
    ax4.plot(y_sch, T_esc, color=GREEN, linewidth=2)
    ax4.fill_between(y_sch, T_esc, alpha=0.15, color=GREEN)
    _style_axis(ax4, "Diagrama de Momento Torsor (T) — Estimado (Cm0 = -0.05)", "Envergadura y [mm]", "T [N·mm]")

    fig.tight_layout(pad=1.5)
    parent.sch_canvas.draw()


def _transfer_to_mapdl(parent):
    """MDO: Envia a pressão e corda exatas da nervura crítica para a aba de Otimização"""
    res = parent._schrenk_res_disc
    idx_c = res["idx_crit"]
    corda_crit = res["chord_ribs"][idx_c]
    pressao_crit = res["pressure_per_rib"][idx_c]
    
    try:
        # Preenche os campos na aba de Nervura
        parent.inp_corda.setValue(corda_crit)
        # O MAPDL costuma usar pressão com sinal negativo (atuando no perfil)
        parent.inp_pressao.setValue(-abs(pressao_crit))
        
        QMessageBox.information(parent, "Integração MDO Concluída", 
            f"Dados da Nervura Crítica ({idx_c+1}) transferidos com sucesso!\n\n"
            f"• Corda local: {corda_crit:.1f} mm\n"
            f"• Pressão distribuída: {-abs(pressao_crit):.4f} MPa")
            
        parent._switch_module(2) # Pula para a tela de Otimização (agora no idx 2)
    except Exception as e:
        QMessageBox.warning(parent, "Erro", str(e))


def _export_cad(parent):
    """Replica a Seção 8 e 9 do MATLAB: Exporta CSV do ANSYS e TXT do SpaceClaim"""
    res = parent._schrenk_res_disc
    from PyQt6.QtWidgets import QFileDialog
    
    dir_path = QFileDialog.getExistingDirectory(parent, "Selecione a pasta para exportar os arquivos de Carga")
    if not dir_path: return
    
    try:
        idx_c = res["idx_crit"]
        c = res["chord_ribs"][idx_c]
        p_val = res["pressure_per_rib"][idx_c]
        
        # Pega as coordenadas baseadas no perfil que está selecionado na aba de Otimização
        from main2 import gerar_naca4, BANCO_PERFIS
        fonte = parent.inp_fonte.currentText()
        if fonte == "NACA 4 dígitos":
            coords = gerar_naca4(parent.inp_naca.text().strip(), 150)
        elif fonte == "Banco de Dados":
            coords = BANCO_PERFIS.get(parent.inp_db.currentText(), gerar_naca4("4412", 150))
        else: # Se for DAT, pega os dados brutos (fallback para NACA pra garantir)
            coords = gerar_naca4("4412", 150)
            
        # Escalonando pra corda local
        X_aero = coords[:, 0] * c
        Z_aero = coords[:, 1] * c
        
        # Referencial ANSYS (Envergadura=X, Altura=Y, Corda=Z)
        X_ansys = np.zeros_like(X_aero)  # Secção 2D na origem
        Y_ansys = Z_aero
        Z_ansys = X_aero
        
        # Export SpaceClaim TXT
        sc_path = os.path.join(dir_path, "Perfil_Nervura_SpaceClaim.txt")
        with open(sc_path, "w") as f:
            f.write("3d=true\npolyline=false\n")
            for i in range(len(X_ansys)):
                f.write(f"{X_ansys[i]:.6f}\t{Y_ansys[i]:.6f}\t{Z_ansys[i]:.6f}\n")
                
        # Export ANSYS Loads CSV
        csv_path = os.path.join(dir_path, "Carga_Pressao_Nervura_Critica.csv")
        mat_ansys = np.column_stack((X_ansys, Y_ansys, Z_ansys, np.full_like(X_ansys, p_val)))
        np.savetxt(csv_path, mat_ansys, delimiter=",", fmt="%.6f", header="X,Y,Z,Pressure")
        
        QMessageBox.information(parent, "Exportação Concluída", 
            f"Arquivos exportados para o CAD e ANSYS com sucesso em:\n{dir_path}")
            
    except Exception as e:
        QMessageBox.warning(parent, "Erro", f"Erro na exportação: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Módulo: Peso e CG
# ═══════════════════════════════════════════════════════════════════════════════

def build_weight_cg_tab(parent) -> QWidget:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setStyleSheet(f"background: {SURFACE};")
    w = QWidget()
    w.setStyleSheet(f"background: {SURFACE};")
    scroll.setWidget(w)
    lay = QVBoxLayout(w)
    lay.setSpacing(10)
    lay.setContentsMargins(16, 16, 16, 16)

    # Nervuras
    g1 = QGroupBox("Nervuras (dados da otimização)")
    gl1 = QGridLayout(g1)
    gl1.setSpacing(8)
    parent.wc_n_ribs = _spin(8, 2, 30, decimals=None)
    parent.wc_area_casca = _spin(1157, 10, 50000, 0)
    parent.wc_area_otim = _spin(800, 10, 50000, 0)
    parent.wc_volfrac = _spin(0.30, 0.05, 1.0, 2)
    parent.wc_rib_thick = _spin(3.0, 0.5, 10, 1)
    parent.wc_rib_density = _spin(160, 10, 2000, 0)
    parent.wc_spar_pos = _spin(0.28, 0.1, 0.5, 2)
    for i, (lbl, w_) in enumerate([
        ("Nº de nervuras (meia-asa):", parent.wc_n_ribs),
        ("Área casca raiz (mm²):",     parent.wc_area_casca),
        ("Área otimizável raiz (mm²):", parent.wc_area_otim),
        ("Fração de volume final:",    parent.wc_volfrac),
        ("Espessura nervura (mm):",    parent.wc_rib_thick),
        ("Densidade nervura (kg/m³):", parent.wc_rib_density),
        ("Posição longarina (% corda):", parent.wc_spar_pos),
    ]):
        gl1.addWidget(QLabel(lbl), i, 0)
        gl1.addWidget(w_,          i, 1)
    lay.addWidget(g1)

    # Longarina
    g2 = QGroupBox("Longarina (Tubo)")
    gl2 = QGridLayout(g2)
    gl2.setSpacing(8)
    parent.wc_spar_od = _spin(12, 2, 50, 1)
    parent.wc_spar_wall = _spin(1.0, 0.2, 5, 1)
    parent.wc_spar_density = _spin(1600, 100, 3000, 0)
    for i, (lbl, w_) in enumerate([
        ("Diâmetro externo (mm):",    parent.wc_spar_od),
        ("Espessura parede (mm):",    parent.wc_spar_wall),
        ("Densidade tubo (kg/m³):",   parent.wc_spar_density),
    ]):
        gl2.addWidget(QLabel(lbl), i, 0)
        gl2.addWidget(w_,          i, 1)
    lay.addWidget(g2)

    # Entelagem + cola
    g3 = QGroupBox("Entelagem e Margem")
    gl3 = QGridLayout(g3)
    gl3.setSpacing(8)
    parent.wc_cover_gsm = _spin(35, 5, 200, 0)
    parent.wc_glue_pct = _spin(12, 0, 30, 0)
    for i, (lbl, w_) in enumerate([
        ("Gramatura entelagem (g/m²):", parent.wc_cover_gsm),
        ("Margem cola/acabamento (%):", parent.wc_glue_pct),
    ]):
        gl3.addWidget(QLabel(lbl), i, 0)
        gl3.addWidget(w_,          i, 1)
    lay.addWidget(g3)

    # Botão
    btn = QPushButton("▶  Calcular Peso e CG")
    btn.setFixedHeight(36)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(f"""
        QPushButton {{ background: {GREEN}; color: #1e1e2e; border: none;
        border-radius: 4px; font-size: 12px; font-weight: 600; padding: 0 18px; }}
        QPushButton:hover {{ background: #3DB87A; }}
    """)
    btn.clicked.connect(lambda: _run_weight_cg(parent))
    lay.addWidget(btn)

    # KPIs
    parent.wc_kpi_layout = QHBoxLayout()
    lay.addLayout(parent.wc_kpi_layout)

    # Resultado textual
    parent.wc_result_box = QTextEdit()
    parent.wc_result_box.setReadOnly(True)
    parent.wc_result_box.setMaximumHeight(300)
    parent.wc_result_box.setStyleSheet(f"""
        background: {BG}; color: {TEXT_P}; border: 1px solid {BORDER};
        border-radius: 5px; padding: 8px; font-size: 12px;
    """)
    lay.addWidget(parent.wc_result_box)

    # Gráfico (barras de massa)
    parent.wc_fig, parent.wc_canvas = _make_fig(6, 3)
    lay.addWidget(parent.wc_canvas)

    lay.addStretch()
    return scroll


def _run_weight_cg(parent):
    from weight_cg import (
        RibMass, SparConfig, CoveringConfig, GlueConfig,
        compute_weight_cg, generate_rib_masses_from_optimization,
        estimate_covering_area
    )
    from schrenk import WingGeometry

    # Pegar geometria do módulo Schrenk se disponível
    semi = parent.sch_semi_span.value() if hasattr(parent, 'sch_semi_span') else 750
    c_root = parent.sch_root_chord.value() if hasattr(parent, 'sch_root_chord') else 300
    c_tip = parent.sch_tip_chord.value() if hasattr(parent, 'sch_tip_chord') else 200

    wing = WingGeometry(semi_span_mm=semi, root_chord_mm=c_root, tip_chord_mm=c_tip)

    n_ribs = int(parent.wc_n_ribs.value())
    rib_positions = np.linspace(0, semi, n_ribs)
    lam = c_tip / c_root
    chord_at_ribs = c_root * (1 - rib_positions / semi * (1 - lam))

    ribs = generate_rib_masses_from_optimization(
        n_ribs=n_ribs,
        rib_positions_mm=rib_positions,
        chord_at_ribs_mm=chord_at_ribs,
        area_casca_root_mm2=parent.wc_area_casca.value(),
        area_otim_root_mm2=parent.wc_area_otim.value(),
        volume_fraction=parent.wc_volfrac.value(),
        thickness_mm=parent.wc_rib_thick.value(),
        density_kgm3=parent.wc_rib_density.value(),
        spar_position_pct=parent.wc_spar_pos.value(),
        root_chord_mm=c_root,
    )

    spar = SparConfig(
        outer_diameter_mm=parent.wc_spar_od.value(),
        wall_thickness_mm=parent.wc_spar_wall.value(),
        density_kgm3=parent.wc_spar_density.value(),
        x_position_pct=parent.wc_spar_pos.value(),
        semi_span_mm=semi,
    )

    covering = CoveringConfig(density_gsm=parent.wc_cover_gsm.value())
    glue = GlueConfig(margin_pct=parent.wc_glue_pct.value())

    result = compute_weight_cg(ribs, spar, covering, glue,
                                semi, c_root, c_tip, wing.mac_mm)

    parent._weight_result = result

    # KPIs
    while parent.wc_kpi_layout.count():
        item = parent.wc_kpi_layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()

    cg_color = GREEN if result.cg_ok else RED
    parent.wc_kpi_layout.addWidget(
        _kpi_card(f"{result.total_g:.1f}", "Peso Total Asa", "g", ACCENT))
    parent.wc_kpi_layout.addWidget(
        _kpi_card(f"{result.ribs_g:.1f}", "Nervuras", "g", TEXT_S))
    parent.wc_kpi_layout.addWidget(
        _kpi_card(f"{result.spar_g:.1f}", "Longarina", "g", TEXT_S))
    parent.wc_kpi_layout.addWidget(
        _kpi_card(f"{result.x_cg_pct_mac:.1f}%", "CG / MAC", "", cg_color))

    # Texto detalhado
    status = "✓ CG dentro da faixa" if result.cg_ok else "✗ CG fora da faixa recomendada"
    s_color = GREEN if result.cg_ok else RED
    html = f"""
    <h3 style='color:{ACCENT}'>Balanço de Massa — Asa Completa</h3>
    <table style='width:100%;border-collapse:collapse;font-size:12px;'>
    <tr style='background:{PANEL}'><td style='padding:5px'>Nervuras ({len(ribs)} un. × 2 meia-asas)</td>
        <td style='color:{ACCENT};font-weight:bold'>{result.ribs_g:.2f} g</td></tr>
    <tr><td style='padding:5px'>Longarina (tubo)</td>
        <td style='color:{ACCENT};font-weight:bold'>{result.spar_g:.2f} g</td></tr>
    <tr style='background:{PANEL}'><td style='padding:5px'>Entelagem</td>
        <td style='color:{ACCENT};font-weight:bold'>{result.covering_g:.2f} g</td></tr>
    <tr><td style='padding:5px'>Cola / margem ({int(parent.wc_glue_pct.value())}%)</td>
        <td style='color:{ACCENT};font-weight:bold'>{result.glue_margin_g:.2f} g</td></tr>
    <tr style='border-top:2px solid {BORDER}'><td style='padding:5px;font-weight:bold'>TOTAL</td>
        <td style='color:{ACCENT};font-weight:bold;font-size:14px'>{result.total_g:.2f} g</td></tr>
    </table>
    <br/>
    <p style='color:{s_color};font-weight:bold'>{status}</p>
    <p style='color:{TEXT_S}'>X_cg = {result.x_cg_mm:.1f} mm do LE ({result.x_cg_pct_mac:.1f}% MAC)</p>
    <p style='color:{TEXT_S}'>Faixa recomendada: {result.cg_range_min_pct}% – {result.cg_range_max_pct}% MAC</p>
    """
    parent.wc_result_box.setHtml(html)

    # Gráfico de barras
    fig = parent.wc_fig
    fig.clear()
    ax = fig.add_subplot(111)
    labels = ['Nervuras', 'Longarina', 'Entelagem', 'Cola']
    values = [result.ribs_g, result.spar_g, result.covering_g, result.glue_margin_g]
    colors = ['#4D82D6', '#4EC88A', '#D9963A', '#6A7A9C']
    bars = ax.barh(labels, values, color=colors, edgecolor=BORDER, height=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}g', va='center', fontsize=9, color=TEXT_P)
    _style_axis(ax, "Breakdown de Massa", "Massa [g]", "")
    ax.invert_yaxis()
    fig.tight_layout()
    parent.wc_canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
#  Módulo: Verificação de Entelagem
# ═══════════════════════════════════════════════════════════════════════════════

def build_covering_tab(parent) -> QWidget:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setStyleSheet(f"background: {SURFACE};")
    w = QWidget()
    w.setStyleSheet(f"background: {SURFACE};")
    scroll.setWidget(w)
    lay = QVBoxLayout(w)
    lay.setSpacing(10)
    lay.setContentsMargins(16, 16, 16, 16)

    g1 = QGroupBox("Propriedades da Entelagem")
    gl1 = QGridLayout(g1)
    gl1.setSpacing(8)
    parent.cov_E = _spin(2500, 100, 10000, 0)
    parent.cov_t = _spin(0.025, 0.005, 0.100, 3)
    parent.cov_pretension = _spin(0.01, 0.001, 0.1, 3)
    parent.cov_nu = _spin(0.35, 0.1, 0.5, 2)
    for i, (lbl, w_) in enumerate([
        ("Módulo elástico (MPa):",    parent.cov_E),
        ("Espessura (mm):",           parent.cov_t),
        ("Pré-tensão (N/mm):",        parent.cov_pretension),
        ("Poisson:",                   parent.cov_nu),
    ]):
        gl1.addWidget(QLabel(lbl), i, 0)
        gl1.addWidget(w_,          i, 1)
    lay.addWidget(g1)

    g2 = QGroupBox("Condição de Verificação")
    gl2 = QGridLayout(g2)
    gl2.setSpacing(8)
    parent.cov_spacing = _spin(80, 10, 300, 0)
    parent.cov_chord = _spin(200, 50, 500, 0)
    parent.cov_pressure = _spin(-0.05, -1.0, 0.0, 4)
    parent.cov_tolerance = _spin(0.005, 0.001, 0.020, 3)
    parent.cov_use_plate = QCheckBox("Usar modelo de placa fina (em vez de membrana)")
    parent.cov_use_plate.setStyleSheet(f"color: {TEXT_S}; background: transparent;")
    for i, (lbl, w_) in enumerate([
        ("Espaçamento entre nervuras (mm):", parent.cov_spacing),
        ("Corda local (mm):",                parent.cov_chord),
        ("Pressão no extradorso (MPa):",     parent.cov_pressure),
        ("Tolerância deformação (Δw/c):",    parent.cov_tolerance),
    ]):
        gl2.addWidget(QLabel(lbl), i, 0)
        gl2.addWidget(w_,          i, 1)
    gl2.addWidget(parent.cov_use_plate, 4, 0, 1, 2)
    lay.addWidget(g2)

    # Botões
    btn_row = QHBoxLayout()
    btn_check = QPushButton("▶  Verificar Entelagem")
    btn_check.setFixedHeight(36)
    btn_check.setCursor(Qt.CursorShape.PointingHandCursor)
    btn_check.setStyleSheet(f"""
        QPushButton {{ background: {ACCENT}; color: white; border: none;
        border-radius: 4px; font-size: 12px; font-weight: 600; padding: 0 18px; }}
        QPushButton:hover {{ background: #3A70C4; }}
    """)
    btn_check.clicked.connect(lambda: _run_covering_check(parent))

    btn_sweep = QPushButton("📊  Estudo Paramétrico")
    btn_sweep.setFixedHeight(36)
    btn_sweep.setCursor(Qt.CursorShape.PointingHandCursor)
    btn_sweep.setStyleSheet(f"""
        QPushButton {{ background: {AMBER}; color: #1e1e2e; border: none;
        border-radius: 4px; font-size: 12px; font-weight: 600; padding: 0 18px; }}
        QPushButton:hover {{ background: #C4860A; }}
    """)
    btn_sweep.clicked.connect(lambda: _run_covering_sweep(parent))

    btn_row.addWidget(btn_check)
    btn_row.addWidget(btn_sweep)
    btn_row.addStretch()
    lay.addLayout(btn_row)

    # KPIs
    parent.cov_kpi_layout = QHBoxLayout()
    lay.addLayout(parent.cov_kpi_layout)

    # Resultado
    parent.cov_result_box = QTextEdit()
    parent.cov_result_box.setReadOnly(True)
    parent.cov_result_box.setMaximumHeight(160)
    parent.cov_result_box.setStyleSheet(f"""
        background: {BG}; color: {TEXT_P}; border: 1px solid {BORDER};
        border-radius: 5px; padding: 8px;
    """)
    lay.addWidget(parent.cov_result_box)

    # Gráfico
    parent.cov_fig, parent.cov_canvas = _make_fig(6, 3.5)
    lay.addWidget(parent.cov_canvas)

    lay.addStretch()
    return scroll


def _run_covering_check(parent):
    from covering import CoveringMaterial, check_covering

    mat = CoveringMaterial(
        E_MPa=parent.cov_E.value(),
        thickness_mm=parent.cov_t.value(),
        pretension_Nmm=parent.cov_pretension.value(),
        nu=parent.cov_nu.value(),
    )
    res = check_covering(
        rib_spacing_mm=parent.cov_spacing.value(),
        chord_mm=parent.cov_chord.value(),
        pressure_MPa=parent.cov_pressure.value(),
        mat=mat,
        max_deflection_chord_ratio=parent.cov_tolerance.value(),
        use_plate_model=parent.cov_use_plate.isChecked(),
    )

    # KPIs
    while parent.cov_kpi_layout.count():
        item = parent.cov_kpi_layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()

    color = GREEN if res.approved else RED
    parent.cov_kpi_layout.addWidget(
        _kpi_card(f"{res.max_deflection_mm:.4f}", "Deflexão Máxima", "mm", color))
    parent.cov_kpi_layout.addWidget(
        _kpi_card(f"{res.max_allowed_mm:.4f}", "Limite", "mm", TEXT_S))
    parent.cov_kpi_layout.addWidget(
        _kpi_card(f"{res.deflection_chord_ratio*100:.3f}%", "Δw/c", "", color))
    parent.cov_kpi_layout.addWidget(
        _kpi_card("✓ OK" if res.approved else "✗ FALHA",
                  "Status", "", color))

    status = "APROVADO" if res.approved else "REPROVADO"
    s_color = GREEN if res.approved else RED
    html = f"""
    <p style='color:{s_color};font-weight:bold;font-size:14px'>{status}</p>
    <p style='color:{TEXT_S}'>Modelo: {res.mode} | Região: {res.critical_region}</p>
    <p style='color:{TEXT_S}'>{res.notes}</p>
    """
    parent.cov_result_box.setHtml(html)


def _run_covering_sweep(parent):
    from covering import CoveringMaterial, parametric_covering_study

    mat = CoveringMaterial(
        E_MPa=parent.cov_E.value(),
        thickness_mm=parent.cov_t.value(),
        pretension_Nmm=parent.cov_pretension.value(),
    )
    spacings = np.linspace(10, 250, 50)
    study = parametric_covering_study(
        spacings, parent.cov_chord.value(), parent.cov_pressure.value(),
        mat, parent.cov_tolerance.value()
    )

    fig = parent.cov_fig
    fig.clear()
    ax = fig.add_subplot(111)

    sp = study["spacings_mm"]
    defl = study["deflections_mm"]
    limit = study["limit_mm"]
    approved = study["approved"]

    ax.plot(sp, defl, color=ACCENT, linewidth=1.5, label='Deflexão')
    ax.axhline(limit, color=RED, linestyle='--', linewidth=1, label=f'Limite ({limit:.3f} mm)')
    ax.fill_between(sp, defl, limit, where=~approved, alpha=0.15, color=RED, label='Reprovado')
    ax.fill_between(sp, 0, defl, where=approved, alpha=0.10, color=GREEN, label='Aprovado')

    if study["max_approved_spacing_mm"] > 0:
        ax.axvline(study["max_approved_spacing_mm"], color=GREEN, linestyle=':',
                   linewidth=1.5, label=f'Máx: {study["max_approved_spacing_mm"]:.0f} mm')

    _style_axis(ax, "Deflexão da Entelagem vs Espaçamento",
                "Espaçamento entre Nervuras [mm]", "Deflexão [mm]")
    ax.legend(fontsize=8, loc='upper left', facecolor=PANEL, edgecolor=BORDER,
              labelcolor=TEXT_S)
    fig.tight_layout()
    parent.cov_canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
#  Módulo: Análise de Sensibilidade
# ═══════════════════════════════════════════════════════════════════════════════

def build_sensitivity_tab(parent) -> QWidget:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setStyleSheet(f"background: {SURFACE};")
    w = QWidget()
    w.setStyleSheet(f"background: {SURFACE};")
    scroll.setWidget(w)
    lay = QVBoxLayout(w)
    lay.setSpacing(10)
    lay.setContentsMargins(16, 16, 16, 16)

    g1 = QGroupBox("Tipo de Análise")
    gl1 = QGridLayout(g1)
    gl1.setSpacing(8)
    parent.sens_type = QComboBox()
    parent.sens_type.addItems([
        "Tornado — Estrutural (nervura)",
        "Tornado — Entelagem",
        "Varredura — Espaçamento vs Deflexão",
        "Varredura — Espessura vs Tensão",
    ])
    parent.sens_perturb = _spin(10, 1, 50, 0)
    gl1.addWidget(QLabel("Análise:"), 0, 0)
    gl1.addWidget(parent.sens_type,   0, 1)
    gl1.addWidget(QLabel("Perturbação (%):" ), 1, 0)
    gl1.addWidget(parent.sens_perturb, 1, 1)
    lay.addWidget(g1)

    btn = QPushButton("▶  Executar Análise de Sensibilidade")
    btn.setFixedHeight(36)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(f"""
        QPushButton {{ background: {AMBER}; color: #1e1e2e; border: none;
        border-radius: 4px; font-size: 12px; font-weight: 600; padding: 0 18px; }}
        QPushButton:hover {{ background: #C4860A; }}
    """)
    btn.clicked.connect(lambda: _run_sensitivity(parent))
    lay.addWidget(btn)

    parent.sens_fig, parent.sens_canvas = _make_fig(7, 5)
    lay.addWidget(parent.sens_canvas)

    parent.sens_result_box = QTextEdit()
    parent.sens_result_box.setReadOnly(True)
    parent.sens_result_box.setMaximumHeight(200)
    parent.sens_result_box.setStyleSheet(f"""
        background: {BG}; color: {TEXT_P}; border: 1px solid {BORDER};
        border-radius: 5px; padding: 8px;
    """)
    lay.addWidget(parent.sens_result_box)

    lay.addStretch()
    return scroll


def _run_sensitivity(parent):
    from sensitivity import (
        tornado_sensitivity, univariate_sweep, SensitivityVariable,
        DEFAULT_STRUCTURAL_VARS, DEFAULT_COVERING_VARS,
        structural_sensitivity_evaluator, covering_sensitivity_evaluator,
    )

    idx = parent.sens_type.currentIndex()
    perturb = parent.sens_perturb.value()
    fig = parent.sens_fig
    fig.clear()

    if idx == 0:
        # Tornado estrutural
        results = tornado_sensitivity(
            DEFAULT_STRUCTURAL_VARS, structural_sensitivity_evaluator,
            ["stress_MPa", "mass_g"], perturbation_pct=perturb)
        _plot_tornado(fig, results, "stress_MPa", "Sensibilidade — Tensão Máxima (MPa)")

    elif idx == 1:
        # Tornado entelagem
        results = tornado_sensitivity(
            DEFAULT_COVERING_VARS, covering_sensitivity_evaluator,
            ["deflection_mm"], perturbation_pct=perturb)
        _plot_tornado(fig, results, "deflection_mm", "Sensibilidade — Deflexão Entelagem (mm)")

    elif idx == 2:
        # Varredura espaçamento
        var = SensitivityVariable("rib_spacing", "Espaçamento", "mm", 80, 20, 200, 20)
        base = {v.name: v.base_value for v in DEFAULT_COVERING_VARS}
        res = univariate_sweep(var, covering_sensitivity_evaluator, base,
                               ["deflection_mm", "approved"])
        ax = fig.add_subplot(111)
        ax.plot(res.variable_values, res.responses["deflection_mm"],
                color=ACCENT, linewidth=1.5)
        _style_axis(ax, "Deflexão vs Espaçamento entre Nervuras",
                    "Espaçamento [mm]", "Deflexão [mm]")

    elif idx == 3:
        # Varredura espessura
        var = SensitivityVariable("thickness", "Espessura", "mm", 3.0, 0.5, 8.0, 20)
        base = {v.name: v.base_value for v in DEFAULT_STRUCTURAL_VARS}
        res = univariate_sweep(var, structural_sensitivity_evaluator, base,
                               ["stress_MPa", "mass_g"])
        ax1 = fig.add_subplot(211)
        ax1.plot(res.variable_values, res.responses["stress_MPa"],
                 color=ACCENT, linewidth=1.5)
        _style_axis(ax1, "Tensão vs Espessura", "", "σ [MPa]")

        ax2 = fig.add_subplot(212)
        ax2.plot(res.variable_values, res.responses["mass_g"],
                 color=GREEN, linewidth=1.5)
        _style_axis(ax2, "Massa vs Espessura", "Espessura [mm]", "Massa [g]")

    fig.tight_layout(pad=2.0)
    parent.sens_canvas.draw()

    # Texto
    if idx in [0, 1]:
        _show_tornado_text(parent, results)


def _plot_tornado(fig, results, response_key, title):
    ax = fig.add_subplot(111)
    labels = []
    low_vals = []
    high_vals = []
    base_val = 0

    for name, data in results.items():
        labels.append(data["variable"])
        base_val = data[f"{response_key}_base"]
        low_vals.append(data[f"{response_key}_minus"] - base_val)
        high_vals.append(data[f"{response_key}_plus"] - base_val)

    # Ordenar por impacto total
    impacts = [abs(h) + abs(l) for h, l in zip(high_vals, low_vals)]
    order = np.argsort(impacts)
    labels = [labels[i] for i in order]
    low_vals = [low_vals[i] for i in order]
    high_vals = [high_vals[i] for i in order]

    y_pos = range(len(labels))
    ax.barh(y_pos, high_vals, color=ACCENT, alpha=0.8, label='+10%')
    ax.barh(y_pos, low_vals, color=RED, alpha=0.8, label='-10%')
    ax.axvline(0, color=TEXT_S, linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    _style_axis(ax, title, f"Δ em relação ao base ({base_val:.4f})", "")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT_S)


def _show_tornado_text(parent, results):
    lines = []
    for name, data in results.items():
        for key in data:
            if key.endswith("_elasticity"):
                rname = key.replace("_elasticity", "")
                lines.append(f"{data['variable']} → {rname}: elasticidade = {data[key]:.3f}")
    parent.sens_result_box.setPlainText(
        "Elasticidade (ΔR/R) / (ΔX/X):\n" + "\n".join(lines)
    )