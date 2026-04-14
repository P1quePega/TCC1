"""
main2.py — T.O.C.A. v0.2 (Mamutes do Cerrado / MMTS)
Todos os módulos integrados: Schrenk, Peso/CG, Entelagem, Sensibilidade,
Wingbox, Aeroelasticidade, MDO/GA, Otimização de Nervura, ANSYS.
"""

import sys, os, glob, shutil, json, time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from shapely.geometry import Polygon

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QTabWidget, QTextEdit, QProgressBar, QGroupBox, QDoubleSpinBox,
    QSpinBox, QMessageBox, QScrollArea, QComboBox, QStackedWidget,
    QFrame, QSizePolicy, QSpacerItem, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap

from modules_analysis import (
    build_schrenk_tab, build_weight_cg_tab,
    build_covering_tab, build_sensitivity_tab,
)
from module_wingbox_visual import build_wingbox_visual_tab, build_aeroelastic_tab
from modules_mdo import build_mdo_module
from module_wingbox_visual import build_wingbox_module

# ═══════════════════════════════════════════════════════════════════════════════
#  Paleta — Midnight Slate
# ═══════════════════════════════════════════════════════════════════════════════
BG      = "#111318"; SURFACE = "#1C2030"; PANEL   = "#21263C"
RAISED  = "#282E48"; BORDER  = "#30395A"; BORDER2 = "#3E4C70"
ACCENT  = "#4D82D6"; ACCENT_H = "#6196E8"
TEXT_P  = "#C8D4EC"; TEXT_S  = "#6A7A9C"; TEXT_D  = "#3A4562"
GREEN   = "#4EC88A"; RED     = "#D95252"; AMBER   = "#D9963A"

# ═══════════════════════════════════════════════════════════════════════════════
#  NACA 4-dígitos
# ═══════════════════════════════════════════════════════════════════════════════

def gerar_naca4(code: str, n_points: int = 150) -> np.ndarray:
    m = int(code[0]) / 100.0; p = int(code[1]) / 10.0; t = int(code[2:4]) / 100.0
    beta = np.linspace(0, np.pi, n_points); x = 0.5 * (1 - np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
    yc = np.zeros_like(x); dyc = np.zeros_like(x)
    if m > 0 and p > 0:
        fr = x <= p; rr = x > p
        yc[fr] = (m/p**2)*(2*p*x[fr]-x[fr]**2); yc[rr] = (m/(1-p)**2)*(1-2*p+2*p*x[rr]-x[rr]**2)
        dyc[fr] = (2*m/p**2)*(p-x[fr]); dyc[rr] = (2*m/(1-p)**2)*(p-x[rr])
    theta = np.arctan(dyc)
    xu = x-yt*np.sin(theta); yu = yc+yt*np.cos(theta)
    xl = x+yt*np.sin(theta); yl = yc-yt*np.cos(theta)
    return np.vstack([np.column_stack([xu[::-1], yu[::-1]]),
                      np.column_stack([xl[1:], yl[1:]])])

BANCO_PERFIS = {}
BANCO_AERONAVES = {}

def carregar_bancos_de_dados():
    global BANCO_PERFIS, BANCO_AERONAVES
    caminho_aero = "aeronaves.json"
    if os.path.exists(caminho_aero):
        try:
            with open(caminho_aero, 'r', encoding='utf-8') as f:
                BANCO_AERONAVES = json.load(f)
        except Exception as e:
            print(f"Erro ao ler aeronaves.json: {e}")
    else:
        BANCO_AERONAVES = {"Aeronave Padrão": {
            "semi_span": 750.0, "root_chord": 300.0, "tip_chord": 200.0, "n_ribs": 12,
            "velocity": 15.0, "mass": 5.0, "load_factor": 4.0, "airfoil": "NACA 4412 (Gerado)"
        }}
        with open(caminho_aero, 'w', encoding='utf-8') as f:
            json.dump(BANCO_AERONAVES, f, indent=4)

    pasta_perfis = "banco_perfis"
    os.makedirs(pasta_perfis, exist_ok=True)
    for arq in glob.glob(os.path.join(pasta_perfis, "*.dat")):
        nome = os.path.splitext(os.path.basename(arq))[0]
        try:
            try: coords = np.loadtxt(arq, skiprows=0)
            except Exception: coords = np.loadtxt(arq, skiprows=1)
            BANCO_PERFIS[nome] = coords
        except Exception as e:
            print(f"Aviso: {arq}: {e}")
    BANCO_PERFIS["NACA 4412 (Gerado)"] = gerar_naca4("4412", 150)

carregar_bancos_de_dados()

# ═══════════════════════════════════════════════════════════════════════════════
#  Workers
# ═══════════════════════════════════════════════════════════════════════════════

class AeroWorker(QThread):
    log_signal      = pyqtSignal(str, str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, params):
        super().__init__(); self.params = params; self._stopped = False

    def stop(self): self._stopped = True
    def log(self, msg, nivel="INFO"): self.log_signal.emit(msg, nivel)

    def run(self):
        try:
            from schrenk import WingGeometry, FlightCondition, schrenk_distribution, discretize_rib_loads_matlab
            p = self.params
            self.log("="*60); self.log("CÁLCULO DE CARGAS (Schrenk + Discretização)"); self.log("="*60)
            self.progress_signal.emit(10, "Configurando geometria...")
            wing = WingGeometry(semi_span_mm=p['semi_span'],
                                root_chord_mm=p['root_chord'],
                                tip_chord_mm=p['tip_chord'])
            flight = FlightCondition(velocity_ms=p['velocidade'],
                                     rho_kgm3=p.get('rho', 1.225),
                                     load_factor=p['fator_carga'],
                                     aircraft_mass_kg=p['massa'])
            self.progress_signal.emit(30, "Calculando Schrenk...")
            result = schrenk_distribution(wing, flight, n_stations=500)
            self.log(f"  Sustentação: {result.total_lift_N:.2f} N | Cortante raiz: {result.max_shear_N:.2f} N", "OK")
            if self._stopped: return
            n_ribs = int(p.get('n_ribs', 12))
            self.progress_signal.emit(60, "Discretizando nervuras...")
            rib_data = discretize_rib_loads_matlab(wing, flight, n_ribs)
            idx_c = rib_data['idx_crit']
            self.log(f"  Nervura crítica: #{idx_c+1} | Y={rib_data['y_ribs'][idx_c]:.1f}mm | "
                     f"F={rib_data['force_per_rib'][idx_c]:.3f}N | "
                     f"P={rib_data['pressure_per_rib'][idx_c]:.6f}MPa", "WARN")
            pressao = float(-abs(rib_data['pressure_per_rib'][idx_c]))
            self.progress_signal.emit(100, "Concluído!")
            self.finished_signal.emit({
                "pressao_extraida": pressao, "solver_usado": "Schrenk",
                "rib_data": rib_data, "idx_crit": idx_c, "n_ribs": n_ribs,
                "schrenk_result": result, "wing": wing, "flight": flight,
            })
        except Exception as e:
            import traceback
            self.log(f"Erro: {e}", "ERROR"); self.log(traceback.format_exc(), "ERROR")


class PipelineWorker(QThread):
    log_signal      = pyqtSignal(str, str)
    progress_signal = pyqtSignal(int, str)
    image_signal    = pyqtSignal(str, str)
    result_signal   = pyqtSignal(dict)
    error_signal    = pyqtSignal(str)

    def __init__(self, params):
        super().__init__(); self.params = params; self._stopped = False

    def stop(self): self._stopped = True
    def log(self, msg, nivel="INFO"): self.log_signal.emit(msg, nivel)

    def _capture_png(self, mapdl, img_dir, name, titulo=""):
        try: plt.close('all')
        except Exception: pass
        pngs = sorted(glob.glob(os.path.join(mapdl.directory, f"{mapdl.jobname}*.png")),
                      key=os.path.getmtime)
        if pngs:
            dest = os.path.join(img_dir, f"{name}.png")
            shutil.copy2(pngs[-1], dest)
            self.image_signal.emit(dest, titulo or name)
            return dest
        return None

    def _safe_plot(self, mapdl, cmd, img_dir, name, titulo):
        try:
            mapdl.run("/SHOW, PNG"); mapdl.run(cmd); mapdl.run("/SHOW, CLOSE")
        except Exception as ex:
            self.log(f"  Aviso plot '{name}': {ex}", "WARN")
        finally:
            try: plt.close('all')
            except Exception: pass
        return self._capture_png(mapdl, img_dir, name, titulo)

    def run(self):
        try: self._run_pipeline()
        except Exception as e:
            import traceback
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")

    def _run_pipeline(self):
        p = self.params
        plt.switch_backend('Agg'); plt.show = lambda *a, **kw: None
        self.progress_signal.emit(2, "Iniciando pipeline...")
        self.log("="*60); self.log("PIPELINE OTIMIZAÇÃO TOPOLÓGICA"); self.log("="*60)

        img_dir = os.path.join(os.getcwd(), f"{p['nome_projeto']}_resultados")
        os.makedirs(img_dir, exist_ok=True)
        self.progress_signal.emit(10, "Processando geometria...")

        # Obter coordenadas do perfil
        if p.get('caminho_dat') and os.path.exists(p['caminho_dat']):
            try: coords = np.loadtxt(p['caminho_dat'], skiprows=0)
            except Exception: coords = np.loadtxt(p['caminho_dat'], skiprows=1)
        elif p.get('banco_perfil') and p['banco_perfil'] in BANCO_PERFIS:
            coords = BANCO_PERFIS[p['banco_perfil']].copy()
        else:
            coords = gerar_naca4(p.get('naca_code', '4412'))

        # Escalar e filtrar
        cs = coords * p['corda_mm']
        cf = [cs[0]]
        for pt in cs[1:]:
            if np.linalg.norm(pt - cf[-1]) > 1e-4: cf.append(pt)
        if np.linalg.norm(cf[0] - cf[-1]) <= 1e-4: cf.pop()
        cf = np.array(cf)

        poly_ext = Polygon(cf)
        if not poly_ext.is_valid: poly_ext = poly_ext.buffer(0)
        poly_int = poly_ext.buffer(-p['espessura_casca'], join_style=1)
        ci = np.array(poly_int.exterior.coords)
        if np.linalg.norm(ci[0] - ci[-1]) <= 1e-4: ci = ci[:-1]
        area_ext = poly_ext.area; area_casca = area_ext - poly_int.area
        self.log(f"  Área externa: {area_ext:.2f} mm² | Casca: {area_casca:.2f} mm²")
        self.progress_signal.emit(20, "Geometria OK")
        if self._stopped: return

        self.progress_signal.emit(25, "Iniciando MAPDL...")
        try:
            from ansys.mapdl.core import launch_mapdl
        except ImportError:
            raise ImportError("PyMAPDL não encontrado. pip install ansys-mapdl-core")

        mapdl = launch_mapdl(jobname=p['nome_projeto'], override=True,
                             port=p.get('mapdl_port', 50056), cleanup_on_exit=False)
        self.log(f"  MAPDL {mapdl.version} conectado", "OK")
        self.progress_signal.emit(30, "MAPDL OK")

        try:
            mapdl.clear(); mapdl.prep7()
            mapdl.mp('EX', 1, p['modulo_elasticidade'])
            mapdl.mp('PRXY', 1, p['poisson'])
            mapdl.mp('DENS', 1, p['densidade'])
            mapdl.et(1, 'PLANE183')

            # Perfil externo
            for i, pt in enumerate(cf): mapdl.k(1+i, pt[0], pt[1], 0)
            n_out = len(cf)
            for i in range(n_out-1): mapdl.l(1+i, 2+i)
            mapdl.l(n_out, 1); mapdl.lsel('S','LINE','',1,n_out)
            mapdl.run("CM,COMP_LINES_OUTER,LINE"); mapdl.al('ALL')
            mapdl.asel('S','AREA','',1); mapdl.run("CM,COMP_OUTER,AREA")

            # Perfil interno
            mapdl.allsel(); ki = n_out+1; li = n_out+1
            for i, pt in enumerate(ci): mapdl.k(ki+i, pt[0], pt[1], 0)
            n_in = len(ci)
            for i in range(n_in-1): mapdl.l(ki+i, ki+i+1)
            mapdl.l(ki+n_in-1, ki); mapdl.lsel('S','LINE','',li,li+n_in-1)
            mapdl.al('ALL'); mapdl.asel('S','AREA','',2); mapdl.run("CM,COMP_INNER,AREA")

            mapdl.allsel()
            mapdl.asba('COMP_OUTER','COMP_INNER',keep2="KEEP")
            mapdl.run("CM,COMP_BLUE,AREA"); mapdl.cmsel('S','COMP_INNER')
            mapdl.run("CM,COMP_YELLOW,AREA")
            self.progress_signal.emit(42, "Geometria MAPDL OK")

            mapdl.allsel(); mapdl.run("/VIEW,1,0,0,1"); mapdl.run("/ANG,1,0")
            self._safe_plot(mapdl,"APLOT",img_dir,"01_geometria","Geometria")

            mapdl.allsel(); mapdl.esize(p['tamanho_elemento']); mapdl.amesh('ALL')
            n_el = mapdl.mesh.n_elem; n_no = mapdl.mesh.n_node
            self.log(f"  Malha: {n_el} elementos, {n_no} nós", "OK")
            self._safe_plot(mapdl,"EPLOT",img_dir,"02_malha","Malha")
            self.progress_signal.emit(55, "Malha OK")

            # BCs
            mapdl.cmsel('S','COMP_LINES_OUTER'); mapdl.sfl('ALL','PRES',p['pressao_aerodinamica'])
            c = p['corda_mm']
            mapdl.allsel()
            mapdl.nsel('S','LOC','X',p['x_long_ini_pct']*c,p['x_long_fim_pct']*c)
            mapdl.nsel('R','LOC','Y',-0.05*c,0.05*c)
            n_eng = int(mapdl.get("NCOUNT","NODE",0,"COUNT"))
            self.log(f"  {n_eng} nós engastados", "OK")
            mapdl.d('ALL','UX',0); mapdl.d('ALL','UY',0); mapdl.allsel()
            self._safe_plot(mapdl,"EPLOT",img_dir,"03_bc","Condições de Contorno")
            self.progress_signal.emit(60, "BCs OK")
            if self._stopped: return

            # TopOpt SIMP
            self.log("\n[4/5] OTIMIZAÇÃO TOPOLÓGICA (SIMP)", "SECTION")
            mapdl.cmsel('S','COMP_YELLOW','AREA'); mapdl.run("ESLA,S")
            n_des = int(mapdl.get("ECOUNT","ELEM",0,"COUNT"))
            mapdl.run("CM,TOPO_DESIGN,ELEM"); mapdl.allsel()
            self.log(f"  {n_des} elementos no domínio de design", "OK")
            mapdl.run("FINISH"); mapdl.run("/SOLU"); mapdl.run("ANTYPE,0")
            mapdl.run("TOVAR,MATDEN,DENSITY,0.001,1.0")
            mapdl.run("TOCOMP,TOPO_DESIGN,MATDEN")
            mapdl.run("TOVAR,VOLUME,OBJ")
            mapdl.run(f"TOVAR,SEQV,CON,0,{p['tensao_max']}")
            mapdl.run(f"TOFREQ,{p['max_iter']},{p['convergencia']},SIMP")
            self.progress_signal.emit(62, "SOLVE estrutural base...")
            mapdl.run("SOLVE"); self.log("  SOLVE base OK", "OK")
            self.progress_signal.emit(65, "TOEXE em execução (aguarde)...")
            self.log("  Executando TOEXE — pode levar vários minutos...", "WARN")
            mapdl.run("TOEXE"); self.log("  TOEXE concluído!", "OK")
            self.progress_signal.emit(80, "TopOpt OK")

            # Pós-processamento
            self.log("\n[5/5] PÓS-PROCESSAMENTO", "SECTION")
            mapdl.run("FINISH"); mapdl.run("/POST1"); mapdl.run("SET,LAST")
            stress_max = stress_mean = disp_max = 0.0
            try:
                mapdl.allsel()
                d = mapdl.post_processing.nodal_displacement("NORM")
                if d is not None and d.size > 0: disp_max = float(np.nanmax(d))
                s = mapdl.post_processing.nodal_eqv_stress()
                if s is not None and s.size > 0:
                    stress_max = float(np.nanmax(s))
                    atv = s[s > max(stress_max*0.01, 1e-6)]
                    if atv.size > 0: stress_mean = float(np.nanmean(atv))
            except Exception: pass

            self.log(f"  σ_max = {stress_max:.4f} MPa | δ_max = {disp_max:.6f} mm", "OK")
            ok = stress_max <= p['tensao_max']
            self.log(f"  {'APROVADO' if ok else 'REPROVADO'} — σ_adm = {p['tensao_max']} MPa",
                     "OK" if ok else "ERROR")

            self._safe_plot(mapdl, "PLNSOL,S,EQV,0,1", img_dir, "04_stress",
                            f"Von Mises (máx: {stress_max:.3f} MPa)")
            self._safe_plot(mapdl, "PLNSOL,U,SUM,0,1", img_dir, "05_deslocamento",
                            f"Deslocamento (máx: {disp_max:.4f} mm)")
            for cmd in ["PLETAB,TDENS","PLNSOL,TOPO,DENS","PLNSOL,NMISC,1"]:
                try:
                    if "PLETAB" in cmd: mapdl.run("ETABLE,TDENS,TOPO,DENS",mute=True)
                    self._safe_plot(mapdl, cmd, img_dir, "06_topologia",
                                    "Nervura Otimizada — Densidades"); break
                except Exception: continue

            mapdl.run("FINISH"); mapdl.run("/PREP7"); mapdl.allsel()
            mapdl.save(p['nome_projeto'], "DB")

            self.progress_signal.emit(95, "Finalizando...")
            self.result_signal.emit({
                "params": p, "n_elementos": n_el, "n_nos": n_no,
                "stress_max": stress_max, "stress_mean": stress_mean, "disp_max": disp_max,
                "area_externa": area_ext, "area_casca": area_casca,
                "img_dir": img_dir, "mapdl_workdir": mapdl.directory,
                "timestamp": datetime.now().isoformat(),
            })
            self.progress_signal.emit(100, "Pipeline concluído!")
            self.log("\nPipeline finalizado com sucesso!", "OK")
        finally:
            try: mapdl.exit()
            except Exception: pass
            self.log("  Sessão MAPDL encerrada")


# ═══════════════════════════════════════════════════════════════════════════════
#  Aplicação Principal
# ═══════════════════════════════════════════════════════════════════════════════

class NervuraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None; self.results = None
        self._apply_theme(); self._setup_ui()

    # ── Tema ─────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        self.setStyleSheet(f"""
        QMainWindow,QWidget{{background:{SURFACE};color:{TEXT_P};
            font-family:'Segoe UI',sans-serif;font-size:12px;}}
        QFrame[frameShape="4"],QFrame[frameShape="5"]{{color:{BORDER};max-height:1px;}}
        QGroupBox{{background:{PANEL};border:1px solid {BORDER};border-radius:6px;
            margin-top:10px;padding:10px 8px 8px;font-weight:600;font-size:11px;
            color:{TEXT_S};letter-spacing:.04em;}}
        QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 4px;color:{TEXT_S};}}
        QLineEdit,QDoubleSpinBox,QSpinBox,QComboBox{{background:{RAISED};border:1px solid {BORDER};
            border-radius:4px;color:{TEXT_P};padding:4px 8px;min-height:26px;
            selection-background-color:{ACCENT};}}
        QLineEdit:focus,QDoubleSpinBox:focus,QSpinBox:focus,QComboBox:focus{{border:1px solid {ACCENT};}}
        QComboBox::drop-down{{border:none;background:{RAISED};border-left:1px solid {BORDER};
            width:24px;border-radius:0 4px 4px 0;}}
        QComboBox QAbstractItemView{{background:{RAISED};color:{TEXT_P};border:1px solid {BORDER2};
            selection-background-color:{ACCENT};outline:none;}}
        QDoubleSpinBox::up-button,QSpinBox::up-button,
        QDoubleSpinBox::down-button,QSpinBox::down-button{{background:{RAISED};border:none;width:18px;}}
        QTabWidget::pane{{background:{SURFACE};border:1px solid {BORDER};
            border-radius:0 0 6px 6px;top:-1px;}}
        QTabBar::tab{{background:{PANEL};color:{TEXT_S};padding:7px 18px;border:1px solid {BORDER};
            border-bottom:none;border-radius:4px 4px 0 0;margin-right:2px;font-size:11px;}}
        QTabBar::tab:selected{{background:{SURFACE};color:{TEXT_P};
            border-bottom:2px solid {ACCENT};font-weight:600;}}
        QTabBar::tab:hover:!selected{{background:{RAISED};color:{TEXT_P};}}
        QScrollArea{{border:none;background:transparent;}}
        QScrollBar:vertical{{background:{PANEL};width:8px;border-radius:4px;}}
        QScrollBar::handle:vertical{{background:{BORDER2};border-radius:4px;min-height:20px;}}
        QScrollBar::handle:vertical:hover{{background:{ACCENT};}}
        QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0px;}}
        QProgressBar{{background:{PANEL};border:1px solid {BORDER};border-radius:3px;height:6px;}}
        QProgressBar::chunk{{background:{ACCENT};border-radius:3px;}}
        QLabel{{color:{TEXT_P};background:transparent;}}
        QCheckBox{{color:{TEXT_S};background:transparent;spacing:6px;}}
        QCheckBox::indicator{{width:14px;height:14px;border:1px solid {BORDER};
            border-radius:3px;background:{RAISED};}}
        QCheckBox::indicator:checked{{background:{ACCENT};border-color:{ACCENT};}}
        QListWidget{{background:{RAISED};color:{TEXT_P};border:1px solid {BORDER};border-radius:4px;}}
        QTableWidget{{background:{BG};color:{TEXT_P};gridline-color:{BORDER};border:none;}}
        QHeaderView::section{{background:{PANEL};color:{TEXT_S};border:1px solid {BORDER};
            padding:4px;font-size:9px;font-weight:600;letter-spacing:.08em;}}
        QToolTip{{background:{RAISED};color:{TEXT_P};border:1px solid {BORDER2};
            padding:4px 8px;border-radius:4px;}}
        """)

    # ── Setup UI ──────────────────────────────────────────────────────────────

    def _setup_ui(self):
        self.setWindowTitle("T.O.C.A. v0.2 — Mamutes do Cerrado / MMTS Aerodesign")
        self.setMinimumSize(1360, 880)
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        self.root_stack = QStackedWidget(); root.addWidget(self.root_stack)

        # Página 0: Home
        self._build_home_page()

        # Página 1: Workspace
        workspace = QWidget(); workspace.setStyleSheet(f"background:{BG};")
        ws_lay = QHBoxLayout(workspace); ws_lay.setContentsMargins(0,0,0,0); ws_lay.setSpacing(0)
        self.stack = QStackedWidget(); self.stack.setStyleSheet(f"background:{SURFACE};")
        ws_lay.addWidget(self._build_sidebar())
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color:{BORDER};max-width:1px;"); ws_lay.addWidget(sep)
        ws_lay.addWidget(self.stack, stretch=1)
        self.root_stack.addWidget(workspace)

        # Módulos (ordem = índice no stack)
        self._build_module_analysis()       # 0 — Cargas & Análise
        self._build_module_aero()           # 1 — Análise Aerodinâmica
        self._build_module_nervura()        # 2 — Otimização de Nervura (ANSYS)
        self._build_module_wingbox_page()   # 3 — Wingbox & Longarinas
        self._build_module_aeroelastic()    # 4 — Aeroelasticidade
        self._build_module_mdo_page()       # 5 — MDO / GA
        self._build_module_wing()           # 6 — Posicionamento (em dev.)

    def _enter_workspace(self, idx=0):
        self.root_stack.setCurrentIndex(1); self._switch_module(idx)

    def _go_home(self): self.root_stack.setCurrentIndex(0)

    # ── Home ─────────────────────────────────────────────────────────────────

    def _build_home_page(self):
        page = QWidget(); page.setStyleSheet(f"background:{BG};")
        root = QVBoxLayout(page); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{BG};border:none;")
        content = QWidget(); content.setStyleSheet(f"background:{BG};")
        scroll.setWidget(content)
        lay = QVBoxLayout(content); lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(0); lay.setContentsMargins(40,50,40,40)

        logo_lbl = QLabel(); logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = QPixmap("logo_equipe.png")
        if not pix.isNull():
            logo_lbl.setPixmap(pix.scaledToHeight(130, Qt.TransformationMode.SmoothTransformation))
        else:
            logo_lbl.setText("MAMUTES DO CERRADO")
            logo_lbl.setStyleSheet(f"color:{TEXT_S};font-size:16px;font-weight:bold;background:transparent;")
        lay.addWidget(logo_lbl); lay.addSpacing(18)

        title = QLabel("T.O.C.A."); title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 42, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{TEXT_P};letter-spacing:.22em;background:transparent;")
        lay.addWidget(title)

        sub = QLabel("Toolkit de Otimização Computacional de Aeronaves")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(f"color:{TEXT_S};font-size:12px;letter-spacing:.06em;background:transparent;")
        lay.addWidget(sub); lay.addSpacing(6)

        ver = QLabel("v0.2  ·  Mamutes do Cerrado  ·  MMTS Aerodesign 2026")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver.setStyleSheet(f"color:{TEXT_D};font-size:9px;letter-spacing:.08em;background:transparent;")
        lay.addWidget(ver); lay.addSpacing(36)

        btn_start = QPushButton("▶   Iniciar Projeto")
        btn_start.setFixedSize(260,44); btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_start.setFont(QFont("Segoe UI",13,QFont.Weight.Bold))
        btn_start.setStyleSheet(f"QPushButton{{background:{ACCENT};color:white;border:none;"
            f"border-radius:6px;letter-spacing:.04em;}}"
            f"QPushButton:hover{{background:{ACCENT_H};}}")
        btn_start.clicked.connect(lambda: self._enter_workspace(0))
        lay.addWidget(btn_start, alignment=Qt.AlignmentFlag.AlignCenter); lay.addSpacing(36)

        sep_line = QFrame(); sep_line.setFrameShape(QFrame.Shape.HLine)
        sep_line.setFixedWidth(560); sep_line.setStyleSheet(f"color:{BORDER};")
        lay.addWidget(sep_line, alignment=Qt.AlignmentFlag.AlignCenter); lay.addSpacing(24)

        lbl_atalhos = QLabel("ACESSO RÁPIDO")
        lbl_atalhos.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_atalhos.setStyleSheet(f"color:{TEXT_D};font-size:9px;font-weight:700;"
                                   f"letter-spacing:.18em;background:transparent;")
        lay.addWidget(lbl_atalhos); lay.addSpacing(14)

        grid_c = QWidget(); grid_c.setFixedWidth(700)
        grid_c.setStyleSheet("background:transparent;")
        from PyQt6.QtWidgets import QGridLayout as QGL
        grid = QGL(grid_c); grid.setSpacing(12)

        shortcuts = [
            ("≈", "Cargas &\nSustentação",    "Schrenk + Discretização de Nervuras",     0, f"border-top:2px solid {ACCENT}"),
            ("⬡", "Nervura\nANSYS",           "TopOpt SIMP + PyMAPDL",                   2, f"border-top:2px solid {GREEN}"),
            ("▭", "Wingbox &\nLongarinas",     "Caixão torção + CLPT + Tapering",         3, f"border-top:2px solid {AMBER}"),
            ("∿", "Flutter &\nDivergência",    "Cooper 2DOF/3DOF + Diagramas V-g V-f",   4, f"border-top:2px solid {RED}"),
            ("⊕", "MDO —\nAlg. Genético",     "NSGA-II multi-obj + Batch Processing",   5, f"border-top:2px solid #7C5CBF"),
            ("🔬", "Materiais",                "Balsa, Divinycell, CFRP, Sanduíche+CLPT",5, f"border-top:2px solid {TEXT_S}"),
        ]
        for i, (icon, name, desc, idx, border) in enumerate(shortcuts):
            card = QPushButton(); card.setFixedSize(210, 108)
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.setStyleSheet(f"QPushButton{{background:{PANEL};border:1px solid {BORDER};"
                               f"{border};border-radius:6px;text-align:center;}}"
                               f"QPushButton:hover{{border-color:{ACCENT};background:{RAISED};}}")
            cl = QVBoxLayout(card); cl.setContentsMargins(10,8,10,8); cl.setSpacing(3)
            il = QLabel(icon); il.setAlignment(Qt.AlignmentFlag.AlignCenter)
            il.setStyleSheet(f"font-size:18px;background:transparent;color:{TEXT_P};")
            nl = QLabel(name); nl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            nl.setStyleSheet(f"color:{TEXT_P};font-size:11px;font-weight:600;background:transparent;")
            dl = QLabel(desc); dl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dl.setStyleSheet(f"color:{TEXT_D};font-size:8px;background:transparent;")
            dl.setWordWrap(True)
            cl.addWidget(il); cl.addWidget(nl); cl.addWidget(dl)
            card.clicked.connect(lambda _, i=idx: self._enter_workspace(i))
            grid.addWidget(card, i // 3, i % 3)

        lay.addWidget(grid_c, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addStretch()
        root.addWidget(scroll, stretch=1)
        self.root_stack.addWidget(page)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget(); sidebar.setFixedWidth(218)
        sidebar.setStyleSheet(f"background:{BG};")
        lay = QVBoxLayout(sidebar); lay.setContentsMargins(0,24,0,16); lay.setSpacing(0)

        logo_lbl = QLabel(); logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = QPixmap("logo_equipe.png")
        if not pix.isNull():
            logo_lbl.setPixmap(pix.scaledToWidth(140, Qt.TransformationMode.SmoothTransformation))
        else:
            logo_lbl.setText("MAMUTES DO CERRADO")
            logo_lbl.setStyleSheet(f"color:{TEXT_D};font-size:9px;font-weight:800;background:transparent;")
        lay.addWidget(logo_lbl); lay.addSpacing(10)

        app_name = QLabel("T.O.C.A."); app_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        app_name.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        app_name.setStyleSheet(f"color:{TEXT_P};letter-spacing:.18em;background:transparent;")
        lay.addWidget(app_name)

        acr = QLabel("Toolkit de Otimização\nComputacional de Aeronaves")
        acr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        acr.setStyleSheet(f"color:{TEXT_D};font-size:8px;letter-spacing:.06em;"
                           f"line-height:1.5;background:transparent;")
        lay.addWidget(acr); lay.addSpacing(8)

        bw = QWidget(); bl = QHBoxLayout(bw); bl.setContentsMargins(0,0,0,0)
        badge = QLabel("v0.2  |  MMTS 2026")
        badge.setStyleSheet(f"background:{RAISED};color:{ACCENT};border:1px solid {BORDER};"
                            f"border-radius:3px;padding:2px 8px;font-size:9px;font-weight:700;")
        bl.addWidget(badge); lay.addWidget(bw); lay.addSpacing(14)

        # Preset global
        preset_lbl = QLabel("PRESET GLOBAL")
        preset_lbl.setStyleSheet(f"color:{TEXT_D};font-size:9px;font-weight:700;"
                                  f"letter-spacing:.18em;padding:0 16px 6px;background:transparent;")
        lay.addWidget(preset_lbl)
        cc = QWidget(); cc.setStyleSheet("background:transparent;")
        cc_l = QVBoxLayout(cc); cc_l.setContentsMargins(16,0,16,8); cc_l.setSpacing(0)
        self.combo_aeronaves = QComboBox()
        self.combo_aeronaves.addItem("— Selecionar Aeronave —")
        self.combo_aeronaves.addItems(list(BANCO_AERONAVES.keys()))
        self.combo_aeronaves.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo_aeronaves.activated.connect(self._apply_aircraft_preset)
        cc_l.addWidget(self.combo_aeronaves); lay.addWidget(cc)

        # Módulos
        sec_lbl = QLabel("MÓDULOS DE ANÁLISE")
        sec_lbl.setStyleSheet(f"color:{TEXT_D};font-size:9px;font-weight:700;"
                               f"letter-spacing:.16em;padding:0 16px 6px;background:transparent;")
        lay.addWidget(sec_lbl)

        self._nav_buttons = []
        modules = [
            ("  ≈  Cargas & Sustentação",        0, True),
            ("  ≀  Análise Aerodinâmica",         1, True),
            ("  ⬡  Otimização de Nervura",        2, True),
            ("  ▭  Wingbox & RIBSPO",             3, True),
            ("  ∿  Aeroelasticidade & Flutter",   4, True),
            ("  ⊕  MDO — Alg. Genético",         5, True),
            ("  ◈  Wingbox Analítico (CLPT)",     6, True),
        ]
        for label, idx, enabled in modules:
            btn = self._nav_button(label, idx)
            if not enabled: btn.setEnabled(False); btn.setToolTip("Em desenvolvimento")
            self._nav_buttons.append(btn); lay.addWidget(btn)

        lay.addStretch()

        btn_home = QPushButton("  ◉  Início")
        btn_home.setFixedHeight(34); btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_home.setStyleSheet(f"QPushButton{{text-align:left;padding:0 12px;border:none;"
            f"border-left:2px solid transparent;border-radius:0;background:transparent;"
            f"color:{TEXT_D};font-size:11px;}}"
            f"QPushButton:hover{{background:{PANEL};color:{TEXT_P};border-left:2px solid {BORDER2};}}")
        btn_home.clicked.connect(self._go_home); lay.addWidget(btn_home)
        lay.addSpacing(6)

        self.status_lbl = QLabel("● MAPDL inativo")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_lbl.setStyleSheet(f"color:{TEXT_D};font-size:10px;font-weight:bold;"
                                       f"background:transparent;margin-bottom:5px;")
        lay.addWidget(self.status_lbl)

        footer = QLabel("PyMAPDL · pymoo · Shapely\nSIMP · NSGA-II · Cooper\n© 2026 MMTS")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet(f"color:{TEXT_D};font-size:9px;line-height:1.7;padding:8px 14px;"
                              f"border-top:1px solid {BORDER};background:transparent;")
        lay.addWidget(footer)
        return sidebar

    def _nav_button(self, text, idx):
        btn = QPushButton(text); btn.setCheckable(True); btn.setChecked(idx == 0)
        btn.setFixedHeight(36); btn.setCursor(Qt.CursorShape.PointingHandCursor)
        active = (f"QPushButton{{text-align:left;padding:0 10px;border:none;"
                  f"border-left:2px solid {ACCENT};border-radius:0;background:{PANEL};"
                  f"color:{TEXT_P};font-size:11px;font-weight:600;}}")
        inactive = (f"QPushButton{{text-align:left;padding:0 10px;border:none;"
                    f"border-left:2px solid transparent;border-radius:0;background:transparent;"
                    f"color:{TEXT_S};font-size:11px;}}"
                    f"QPushButton:hover:!disabled{{background:{PANEL};color:{TEXT_P};"
                    f"border-left:2px solid {BORDER2};}}"
                    f"QPushButton:disabled{{color:{TEXT_D};}}")
        btn._active_style = active; btn._inactive_style = inactive
        btn.setStyleSheet(active if idx == 0 else inactive)
        btn.clicked.connect(lambda _, i=idx: self._switch_module(i))
        return btn

    def _switch_module(self, idx):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == idx)
            if btn.isEnabled():
                btn.setStyleSheet(btn._active_style if i == idx else btn._inactive_style)

    # ── Cabeçalho de módulo ───────────────────────────────────────────────────

    def _module_header(self, crumb, title):
        w = QWidget(); w.setFixedHeight(64)
        w.setStyleSheet(f"background:{PANEL};border-bottom:1px solid {BORDER};")
        lay = QHBoxLayout(w); lay.setContentsMargins(20,0,20,0)
        col = QVBoxLayout(); col.setSpacing(2)
        cl = QLabel(crumb); cl.setStyleSheet(f"color:{TEXT_D};font-size:9px;letter-spacing:.12em;background:transparent;")
        tl = QLabel(title); tl.setFont(QFont("Segoe UI",13,QFont.Weight.Bold))
        tl.setStyleSheet(f"color:{TEXT_P};background:transparent;")
        col.addWidget(cl); col.addWidget(tl)
        lay.addLayout(col); lay.addStretch()
        return w

    # ── Botão de ação ─────────────────────────────────────────────────────────

    def _action_btn(self, text, color, hover):
        btn = QPushButton(text); btn.setFixedHeight(34); btn.setMinimumWidth(160)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"QPushButton{{background:{color};color:white;border:none;"
            f"border-radius:4px;font-size:12px;font-weight:600;padding:0 18px;}}"
            f"QPushButton:hover:!disabled{{background:{hover};}}"
            f"QPushButton:disabled{{background:{BORDER};color:{TEXT_D};}}")
        return btn

    # ══════════════════════════════════════════════════════════════════════════
    #  Módulo 0 — Cargas, Peso/CG, Entelagem, Sensibilidade
    # ══════════════════════════════════════════════════════════════════════════

    def _build_module_analysis(self):
        page = QWidget(); page.setStyleSheet(f"background:{SURFACE};")
        lay = QVBoxLayout(page); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self._module_header("MÓDULOS / CARGAS E ANÁLISE",
                                          "Sustentação, Peso, Entelagem e Sensibilidade"))
        tabs = QTabWidget(); tabs.setDocumentMode(True)
        tabs.addTab(build_schrenk_tab(self),     "📐  Schrenk")
        tabs.addTab(build_weight_cg_tab(self),   "⚖  Peso & CG")
        tabs.addTab(build_covering_tab(self),    "🎯  Entelagem")
        tabs.addTab(build_sensitivity_tab(self), "📊  Sensibilidade")
        lay.addWidget(tabs, stretch=1)
        self.stack.addWidget(page)

    # ══════════════════════════════════════════════════════════════════════════
    #  Módulo 1 — Análise Aerodinâmica
    # ══════════════════════════════════════════════════════════════════════════

    def _build_module_aero(self):
        page = QWidget(); page.setStyleSheet(f"background:{SURFACE};")
        lay = QVBoxLayout(page); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self._module_header("MÓDULOS / AERODINÂMICA",
                                          "Pressão na Nervura Crítica (Schrenk)"))
        self.tabs_aero = QTabWidget(); self.tabs_aero.setDocumentMode(True)
        lay.addWidget(self.tabs_aero, stretch=1)
        self._build_tab_aero_inputs()
        self._build_tab_aero_log()
        lay.addWidget(self._build_aero_action_bar())
        self.stack.addWidget(page)

    def _build_tab_aero_inputs(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{SURFACE};")
        w = QWidget(); w.setStyleSheet(f"background:{SURFACE};")
        scroll.setWidget(w); lay = QVBoxLayout(w)
        lay.setSpacing(10); lay.setContentsMargins(16,16,16,16)

        g1 = QGroupBox("Condições de Voo"); gl = QGridLayout(g1); gl.setSpacing(8)
        self.aero_vel  = self._mk_spin(15.69, 1, 300, 2)
        self.aero_dens = self._mk_spin(1.225, 0, 2.0, 3)
        self.aero_carga= self._mk_spin(2.5, 1, 10, 1)
        for i,(l,w_) in enumerate([("Velocidade (m/s):",self.aero_vel),
                                    ("Densidade ar (kg/m³):",self.aero_dens),
                                    ("Fator de carga (g):",self.aero_carga)]):
            gl.addWidget(QLabel(l),i,0); gl.addWidget(w_,i,1)
        lay.addWidget(g1); lay.addStretch()
        self.tabs_aero.addTab(scroll,"⚙  Parâmetros")

    def _build_tab_aero_log(self):
        w = QWidget(); w.setStyleSheet(f"background:{SURFACE};")
        lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12)
        self.log_box_aero = QTextEdit(); self.log_box_aero.setReadOnly(True)
        self.log_box_aero.setFont(QFont("Consolas",10))
        self.log_box_aero.setStyleSheet(f"background:{BG};color:{TEXT_P};"
            f"border:1px solid {BORDER};border-radius:5px;padding:8px;")
        lay.addWidget(self.log_box_aero)
        self.tabs_aero.addTab(w,"📋  Log")

    def _build_aero_action_bar(self):
        bar = QWidget(); bar.setFixedHeight(58)
        bar.setStyleSheet(f"background:{PANEL};border-top:1px solid {BORDER};")
        lay = QHBoxLayout(bar); lay.setContentsMargins(16,0,16,0); lay.setSpacing(8)
        self.btn_run_aero = QPushButton("▶  Calcular Cargas Aerodinâmicas")
        self.btn_run_aero.setFixedHeight(34)
        self.btn_run_aero.setStyleSheet(f"QPushButton{{background:{ACCENT};color:white;border:none;"
            f"border-radius:4px;font-size:12px;font-weight:bold;padding:0 18px;}}")
        self.btn_run_aero.clicked.connect(self._run_aero)
        self.btn_transfer = QPushButton("⮂  Transferir Pressão → Nervura")
        self.btn_transfer.setFixedHeight(34); self.btn_transfer.setEnabled(False)
        self.btn_transfer.setStyleSheet(f"QPushButton{{background:#2E7D52;color:white;border:none;"
            f"border-radius:4px;font-weight:bold;padding:0 18px;}}"
            f"QPushButton:disabled{{background:{BORDER};color:{TEXT_D};}}")
        self.btn_transfer.clicked.connect(self._transfer_pressure)
        lay.addWidget(self.btn_run_aero); lay.addWidget(self.btn_transfer)
        lay.addSpacing(16)
        pc = QVBoxLayout(); pc.setSpacing(3)
        self.prog_lbl_aero = QLabel("Pronto")
        self.prog_lbl_aero.setStyleSheet(f"color:{TEXT_S};font-size:10px;background:transparent;")
        self.prog_aero = QProgressBar(); self.prog_aero.setFixedHeight(5); self.prog_aero.setTextVisible(False)
        pc.addWidget(self.prog_lbl_aero); pc.addWidget(self.prog_aero)
        lay.addLayout(pc, stretch=1)
        return bar

    # ══════════════════════════════════════════════════════════════════════════
    #  Módulo 2 — Otimização de Nervura (ANSYS)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_module_nervura(self):
        page = QWidget(); page.setStyleSheet(f"background:{SURFACE};")
        lay = QVBoxLayout(page); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self._module_header("MÓDULOS / ESTRUTURAL",
                                          "Otimização Topológica de Nervura (ANSYS MAPDL)"))
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        lay.addWidget(self.tabs, stretch=1)
        self._build_tab_inputs()
        self._build_tab_log()
        self._build_tab_images()
        self._build_tab_results()
        lay.addWidget(self._build_action_bar())
        self.stack.addWidget(page)

    # ── Abas da nervura ───────────────────────────────────────────────────────

    def _build_tab_inputs(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{SURFACE};")
        w = QWidget(); w.setStyleSheet(f"background:{SURFACE};")
        scroll.setWidget(w); lay = QVBoxLayout(w)
        lay.setSpacing(10); lay.setContentsMargins(16,16,16,16)

        # Projeto
        g = QGroupBox("Projeto"); gl = QGridLayout(g); gl.setSpacing(8)
        self.inp_nome = QLineEdit("Nervura_TopOpt")
        self.inp_port = self._mk_spin(50056, 1024, 65535, decimals=None)
        gl.addWidget(QLabel("Nome:"),0,0); gl.addWidget(self.inp_nome,0,1)
        gl.addWidget(QLabel("Porta MAPDL:"),1,0); gl.addWidget(self.inp_port,1,1)
        lay.addWidget(g)

        # Perfil + Material da Nervura
        g2 = QGroupBox("Perfil Aerodinâmico e Material da Nervura")
        gl2 = QGridLayout(g2); gl2.setSpacing(8)
        self.inp_fonte = QComboBox()
        self.inp_fonte.addItems(["Banco de Dados","NACA 4 dígitos","Arquivo .dat"])
        self.inp_fonte.currentIndexChanged.connect(self._on_fonte_changed)
        self.inp_dat = QLineEdit(); self.inp_dat.setPlaceholderText("Caminho .dat")
        self.btn_browse = QPushButton("Procurar…")
        self.btn_browse.setStyleSheet(f"QPushButton{{background:{RAISED};color:{TEXT_P};"
            f"border:1px solid {BORDER};border-radius:4px;padding:4px;}}")
        self.btn_browse.clicked.connect(self._browse_dat)
        self.inp_naca = QLineEdit("4412")
        self.inp_db = QComboBox(); self.inp_db.addItems(list(BANCO_PERFIS.keys()))
        self.lbl_dinamico = QLabel("Perfil:")
        self.inp_corda = self._mk_spin(200,1,5000,1)
        self.inp_esp   = self._mk_spin(3.0,0.1,50,2)

        # Material da nervura
        from materials import get_all_materials
        self.inp_rib_mat = QComboBox()
        self.inp_rib_mat.addItems(list(get_all_materials().keys()))
        idx_balsa = self.inp_rib_mat.findText("Balsa C-grain")
        if idx_balsa >= 0: self.inp_rib_mat.setCurrentIndex(idx_balsa)
        self.inp_rib_mat.currentIndexChanged.connect(self._on_rib_mat_changed)

        for i,(l,wgt) in enumerate([
            ("Fonte do perfil:",self.inp_fonte),
            (self.lbl_dinamico,self.inp_db),
            ("Corda (mm):",self.inp_corda),
            ("Esp. casca (mm):",self.inp_esp),
            ("Material nervura:",self.inp_rib_mat),
        ]):
            if isinstance(l,str): gl2.addWidget(QLabel(l),i,0)
            else: gl2.addWidget(l,i,0)
            gl2.addWidget(wgt,i,1)
        gl2.addWidget(self.inp_naca,1,1); gl2.addWidget(self.inp_dat,1,1)
        gl2.addWidget(self.btn_browse,1,2)
        self.inp_dat.setVisible(False); self.btn_browse.setVisible(False)
        self.inp_naca.setVisible(False)

        # Preview do perfil
        self.fig = Figure(figsize=(4,3),dpi=100); self.fig.patch.set_facecolor(BG)
        self.canvas = FigureCanvas(self.fig); self.canvas.setMinimumHeight(180)
        self.ax = self.fig.add_subplot(111); self.ax.set_facecolor(BG)
        row_w = QWidget(); row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0,0,0,0); row_l.addWidget(QWidget())
        lay.addWidget(g2)
        lay.addWidget(self.canvas)

        for sig in [self.inp_corda.valueChanged, self.inp_esp.valueChanged,
                    self.inp_naca.textChanged, self.inp_db.currentIndexChanged,
                    self.inp_dat.textChanged, self.inp_fonte.currentIndexChanged]:
            sig.connect(self._update_preview)
        self._update_preview()

        # Geometria da malha
        g3 = QGroupBox("Malha & Engaste"); gl3 = QGridLayout(g3); gl3.setSpacing(8)
        self.inp_long_ini  = self._mk_spin(0.25,0.05,0.50,2)
        self.inp_long_fim  = self._mk_spin(0.30,0.10,0.60,2)
        self.inp_elem_size = self._mk_spin(2.0,0.1,20,1)
        for i,(l,wgt) in enumerate([("Engaste início (% corda):",self.inp_long_ini),
                                     ("Engaste fim (% corda):",self.inp_long_fim),
                                     ("Tamanho elemento (mm):",self.inp_elem_size)]):
            gl3.addWidget(QLabel(l),i,0); gl3.addWidget(wgt,i,1)
        lay.addWidget(g3)

        # Cargas e material
        g4 = QGroupBox("Cargas e Propriedades (MAPDL)")
        gl4 = QGridLayout(g4); gl4.setSpacing(8)
        self.inp_pressao = self._mk_spin(-0.05,-100,100,4)
        self.inp_ex      = self._mk_spin(3500,1,1e7,0)
        self.inp_prxy    = self._mk_spin(0.3,0,0.5,2)
        self.inp_dens    = self._mk_spin(1.2e-4,0,1,7)
        for i,(l,wgt) in enumerate([("Pressão aerodinámica (MPa):",self.inp_pressao),
                                     ("Módulo E (MPa):",self.inp_ex),
                                     ("Poisson:",self.inp_prxy),
                                     ("Densidade (t/mm³):",self.inp_dens)]):
            gl4.addWidget(QLabel(l),i,0); gl4.addWidget(wgt,i,1)
        lay.addWidget(g4)

        # TopOpt
        g5 = QGroupBox("Parâmetros de Otimização Topológica")
        gl5 = QGridLayout(g5); gl5.setSpacing(8)
        self.inp_tensao_max   = self._mk_spin(5.0,0.1,1000,1)
        self.inp_max_iter     = self._mk_spin(50,5,500,None)
        self.inp_convergencia = self._mk_spin(0.001,0.0001,0.1,4)
        for i,(l,wgt) in enumerate([("Tensão máx. (MPa):",self.inp_tensao_max),
                                     ("Máx. iterações:",self.inp_max_iter),
                                     ("Convergência:",self.inp_convergencia)]):
            gl5.addWidget(QLabel(l),i,0); gl5.addWidget(wgt,i,1)
        lay.addWidget(g5)
        lay.addStretch()
        self.tabs.addTab(scroll,"⚙  Parâmetros")

    def _on_rib_mat_changed(self):
        """Auto-preenche E, nu, densidade ao trocar material da nervura."""
        from materials import get_all_materials
        mat = get_all_materials().get(self.inp_rib_mat.currentText())
        if not mat: return
        self.inp_ex.setValue(mat.E_MPa)
        self.inp_prxy.setValue(mat.nu)
        # Converter kg/m³ → t/mm³
        self.inp_dens.setValue(mat.density_kgm3 * 1e-12 * 1e6)

    def _build_tab_log(self):
        w = QWidget(); w.setStyleSheet(f"background:{SURFACE};")
        lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(8)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas",10))
        self.log_box.setStyleSheet(f"background:{BG};color:{TEXT_P};border:1px solid {BORDER};"
            f"border-radius:5px;padding:8px;selection-background-color:{ACCENT};")
        btn_clear = QPushButton("Limpar log"); btn_clear.setFixedWidth(110)
        btn_clear.setStyleSheet(f"QPushButton{{background:{RAISED};color:{TEXT_S};"
            f"border:1px solid {BORDER};border-radius:4px;padding:5px 12px;font-size:11px;}}")
        btn_clear.clicked.connect(self.log_box.clear)
        lay.addWidget(self.log_box); lay.addWidget(btn_clear)
        self.tabs.addTab(w,"📋  Log")

    def _build_tab_images(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{SURFACE};")
        w = QWidget(); w.setStyleSheet(f"background:{SURFACE};")
        scroll.setWidget(w)
        self.img_layout = QVBoxLayout(w); self.img_layout.setSpacing(12)
        self.img_layout.setContentsMargins(12,12,12,12)
        ph = QLabel("As imagens ANSYS aparecerão aqui.")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet(f"color:{TEXT_D};font-size:12px;padding:60px;background:transparent;")
        ph.setObjectName("img_placeholder")
        self.img_layout.addWidget(ph); self.img_layout.addStretch()
        self.tabs.addTab(scroll,"🖼  ANSYS Imagens")

    def _build_tab_results(self):
        w = QWidget(); w.setStyleSheet(f"background:{SURFACE};")
        lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12)
        self.results_box = QTextEdit(); self.results_box.setReadOnly(True)
        self.results_box.setFont(QFont("Segoe UI",11))
        self.results_box.setPlaceholderText("Resultados após pipeline.")
        self.results_box.setStyleSheet(f"background:{BG};color:{TEXT_P};"
            f"border:1px solid {BORDER};border-radius:5px;padding:8px;")
        lay.addWidget(self.results_box)
        self.tabs.addTab(w,"📊  Resultados")

    def _build_action_bar(self):
        bar = QWidget(); bar.setFixedHeight(58)
        bar.setStyleSheet(f"background:{PANEL};border-top:1px solid {BORDER};")
        lay = QHBoxLayout(bar); lay.setContentsMargins(16,0,16,0); lay.setSpacing(8)
        self.btn_run    = self._action_btn("▶  Executar Pipeline",    ACCENT,   "#3A70C4")
        self.btn_stop   = self._action_btn("⏹  Parar",               RED,      "#B83838")
        self.btn_report = self._action_btn("↓  Exportar Relatório",   "#2E7D52","#246040")
        self.btn_stop.setEnabled(False); self.btn_report.setEnabled(False)
        lay.addWidget(self.btn_run); lay.addWidget(self.btn_stop); lay.addWidget(self.btn_report)
        lay.addSpacing(16)
        pc = QVBoxLayout(); pc.setSpacing(3)
        self.progress_lbl = QLabel("Pronto")
        self.progress_lbl.setStyleSheet(f"color:{TEXT_S};font-size:10px;background:transparent;")
        self.progress = QProgressBar(); self.progress.setFixedHeight(5); self.progress.setTextVisible(False)
        pc.addWidget(self.progress_lbl); pc.addWidget(self.progress)
        lay.addLayout(pc, stretch=1)
        self.btn_run.clicked.connect(self._run)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_report.clicked.connect(self._export_report)
        return bar

    # ══════════════════════════════════════════════════════════════════════════
    #  Módulo 3 — Wingbox & Longarinas
    # ══════════════════════════════════════════════════════════════════════════

    def _build_module_wingbox_page(self):
        """Módulo 3 — Wingbox Visual (Vista Superior + RIBSPO)."""
        self.stack.addWidget(build_wingbox_module(self))

    # ══════════════════════════════════════════════════════════════════════════
    #  Módulo 4 — Aeroelasticidade
    # ══════════════════════════════════════════════════════════════════════════

    def _build_module_aeroelastic(self):
        page = QWidget(); page.setStyleSheet(f"background:{SURFACE};")
        lay = QVBoxLayout(page); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self._module_header("MÓDULOS / DINÂMICA",
                                          "Aeroelasticidade — Flutter, Divergência e Modos Naturais"))
        tabs = QTabWidget(); tabs.setDocumentMode(True)
        tabs.addTab(build_aeroelastic_tab(self), "∿  Análise V-g / V-f (Cooper)")
        lay.addWidget(tabs, stretch=1)
        self.stack.addWidget(page)

    # ══════════════════════════════════════════════════════════════════════════
    #  Módulo 5 — MDO / GA
    # ══════════════════════════════════════════════════════════════════════════

    def _build_module_mdo_page(self):
        self.stack.addWidget(build_mdo_module(self))

    # ══════════════════════════════════════════════════════════════════════════
    #  Módulo 6 — Posicionamento (em dev.)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_module_wing(self):
        """Módulo 6 — Wingbox Analítico com CLPT e longarinas."""
        page = QWidget(); page.setStyleSheet(f"background:{SURFACE};")
        lay = QVBoxLayout(page); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self._module_header(
            "MÓDULOS / ESTRUTURAL",
            "Wingbox Analítico — CLPT, Longarinas e Tapering"))
        tabs = QTabWidget(); tabs.setDocumentMode(True)
        from module_wingbox_visual import build_wingbox_visual_tab
        tabs.addTab(build_wingbox_visual_tab(self), "▭  GJ / EI / Longarina")
        lay.addWidget(tabs, stretch=1)
        self.stack.addWidget(page)

    # ══════════════════════════════════════════════════════════════════════════
    #  Helpers de UI
    # ══════════════════════════════════════════════════════════════════════════

    def _mk_spin(self, val, lo, hi, decimals=2, step=None):
        if decimals is None:
            w = QSpinBox(); w.setRange(int(lo),int(hi)); w.setValue(int(val))
        else:
            w = QDoubleSpinBox(); w.setDecimals(decimals)
            w.setRange(float(lo),float(hi)); w.setValue(float(val))
            if step: w.setSingleStep(step)
        w.setMinimumWidth(140); return w

    def _browse_dat(self):
        path, _ = QFileDialog.getOpenFileName(self,"Selecionar perfil .dat","","DAT (*.dat);;All (*)")
        if path: self.inp_dat.setText(path)

    def _on_fonte_changed(self, idx):
        fonte = self.inp_fonte.currentText()
        is_dat  = fonte == "Arquivo .dat"
        is_naca = fonte == "NACA 4 dígitos"
        is_db   = fonte == "Banco de Dados"
        self.inp_dat.setVisible(is_dat); self.btn_browse.setVisible(is_dat)
        self.inp_naca.setVisible(is_naca); self.inp_db.setVisible(is_db)
        self.lbl_dinamico.setText("Caminho .dat:" if is_dat else
                                  "Código NACA:" if is_naca else "Selecione:")

    def _update_preview(self, *args):
        try:
            corda = self.inp_corda.value(); esp = self.inp_esp.value()
            fonte = self.inp_fonte.currentText(); coords = None
            if fonte == "NACA 4 dígitos":
                code = self.inp_naca.text().strip()
                if len(code) == 4 and code.isdigit(): coords = gerar_naca4(code, 150)
            elif fonte == "Banco de Dados":
                nome = self.inp_db.currentText()
                if nome in BANCO_PERFIS: coords = BANCO_PERFIS[nome]
            else:
                path = self.inp_dat.text().strip()
                if os.path.exists(path):
                    try: coords = np.loadtxt(path, skiprows=0)
                    except Exception: coords = np.loadtxt(path, skiprows=1)
            self.ax.clear()
            if coords is not None and len(coords) > 0:
                cs = coords * corda
                poly_e = Polygon(cs)
                if not poly_e.is_valid: poly_e = poly_e.buffer(0)
                poly_i = poly_e.buffer(-esp, join_style=1)
                if not poly_e.is_empty:
                    xe, ye = poly_e.exterior.xy
                    self.ax.plot(xe, ye, color=ACCENT, lw=1.5, label='Externa')
                if not poly_i.is_empty:
                    xi, yi = poly_i.exterior.xy
                    self.ax.plot(xi, yi, color=AMBER, lw=1.5, ls='--', label='Casca')
            self.ax.set_facecolor(BG); self.ax.axis('equal')
            self.ax.grid(True, color=BORDER, ls=':', lw=0.5)
            self.ax.tick_params(colors=TEXT_S, labelsize=8)
            for sp in self.ax.spines.values(): sp.set_color(BORDER)
            self.ax.legend(loc='upper right', fontsize=8, facecolor=PANEL,
                           edgecolor=BORDER, labelcolor=TEXT_P)
            self.fig.tight_layout(); self.canvas.draw()
        except Exception: pass

    # ── Preset de aeronave ────────────────────────────────────────────────────

    def _apply_aircraft_preset(self):
        idx = self.combo_aeronaves.currentIndex()
        if idx == 0: return
        nome = self.combo_aeronaves.currentText()
        preset = BANCO_AERONAVES.get(nome)
        if not preset: return
        try:
            if hasattr(self,'sch_semi_span'): self.sch_semi_span.setValue(preset['semi_span'])
            if hasattr(self,'sch_root_chord'): self.sch_root_chord.setValue(preset['root_chord'])
            if hasattr(self,'sch_tip_chord'): self.sch_tip_chord.setValue(preset['tip_chord'])
            if hasattr(self,'sch_n_ribs'): self.sch_n_ribs.setValue(preset['n_ribs'])
            if hasattr(self,'sch_velocity'): self.sch_velocity.setValue(preset['velocity'])
            if hasattr(self,'sch_mass'): self.sch_mass.setValue(preset['mass'])
            if hasattr(self,'sch_load_factor'): self.sch_load_factor.setValue(preset['load_factor'])
            if hasattr(self,'inp_corda'): self.inp_corda.setValue(preset['root_chord'])
            if hasattr(self,'aero_vel'): self.aero_vel.setValue(preset['velocity'])
            if hasattr(self,'aero_carga'): self.aero_carga.setValue(preset['load_factor'])
            if hasattr(self,'wc_n_ribs'): self.wc_n_ribs.setValue(preset['n_ribs'])
            if hasattr(self,'_update_preview'): self._update_preview()
            if hasattr(self,'status_lbl'):
                self.status_lbl.setText(f"● Preset '{nome}' carregado")
                self.status_lbl.setStyleSheet(f"color:{GREEN};font-size:10px;font-weight:bold;"
                                               f"background:transparent;margin-bottom:5px;")
        except Exception as e:
            QMessageBox.warning(self,"Erro ao carregar preset", str(e))

    # ── Coleta de parâmetros (nervura) ────────────────────────────────────────

    def _collect_params(self):
        sv = lambda w: w.value() if hasattr(w,'value') else w.text()
        fonte = self.inp_fonte.currentText()
        fp = "dat" if fonte=="Arquivo .dat" else "naca" if fonte=="NACA 4 dígitos" else "banco"
        return {
            "nome_projeto":         self.inp_nome.text().strip() or "Nervura",
            "mapdl_port":           int(sv(self.inp_port)),
            "fonte_perfil":         fp,
            "caminho_dat":          self.inp_dat.text().strip() if fp=="dat" else "",
            "naca_code":            self.inp_naca.text().strip() if fp=="naca" else "",
            "banco_perfil":         self.inp_db.currentText() if fp=="banco" else "",
            "corda_mm":             sv(self.inp_corda),
            "espessura_casca":      sv(self.inp_esp),
            "x_long_ini_pct":       sv(self.inp_long_ini),
            "x_long_fim_pct":       sv(self.inp_long_fim),
            "tamanho_elemento":     sv(self.inp_elem_size),
            "pressao_aerodinamica": sv(self.inp_pressao),
            "modulo_elasticidade":  sv(self.inp_ex),
            "poisson":              sv(self.inp_prxy),
            "densidade":            sv(self.inp_dens),
            "tensao_max":           sv(self.inp_tensao_max),
            "max_iter":             int(sv(self.inp_max_iter)),
            "convergencia":         sv(self.inp_convergencia),
        }

    # ── Execução pipeline ─────────────────────────────────────────────────────

    COLORS = {"INFO":TEXT_P,"OK":GREEN,"WARN":AMBER,"ERROR":RED,"SECTION":ACCENT}

    def _append_log(self, msg, nivel="INFO"):
        color = self.COLORS.get(nivel, TEXT_P)
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f'<span style="color:{TEXT_D}">[{ts}]</span> '
                            f'<span style="color:{color}">{msg}</span>')
        sb = self.log_box.verticalScrollBar(); sb.setValue(sb.maximum())

    def _run(self):
        p = self._collect_params()
        if p['fonte_perfil']=='dat' and not p['caminho_dat']:
            QMessageBox.warning(self,"Input inválido","Selecione o arquivo .dat."); return
        if p['fonte_perfil']=='naca':
            code = p.get('naca_code','')
            if len(code)!=4 or not code.isdigit():
                QMessageBox.warning(self,"Input inválido","Código NACA deve ter 4 dígitos."); return
        self._clear_images(); self.tabs.setCurrentIndex(1)
        self.log_box.clear(); self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True); self.btn_report.setEnabled(False)
        self.results = None
        self.status_lbl.setText("● MAPDL conectando...")
        self.status_lbl.setStyleSheet(f"color:{AMBER};font-size:10px;background:transparent;")
        self.worker = PipelineWorker(p)
        self.worker.log_signal.connect(self._append_log)
        self.worker.progress_signal.connect(lambda v,l: (self.progress.setValue(v),
                                                         self.progress_lbl.setText(l)))
        self.worker.image_signal.connect(self._add_ansys_image)
        self.worker.result_signal.connect(self._on_result)
        self.worker.error_signal.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _stop(self):
        if self.worker: self.worker.stop(); self._append_log("Parada solicitada.", "WARN")

    def _clear_images(self):
        while self.img_layout.count():
            item = self.img_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        ph = QLabel("Aguardando pipeline...")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet(f"color:{TEXT_D};font-size:12px;padding:60px;background:transparent;")
        ph.setObjectName("img_placeholder")
        self.img_layout.addWidget(ph); self.img_layout.addStretch()

    def _add_ansys_image(self, filepath, titulo):
        ph = self.findChild(QLabel, "img_placeholder")
        if ph: ph.setParent(None); ph.deleteLater()
        card = QGroupBox(titulo)
        card.setStyleSheet(f"QGroupBox{{background:{PANEL};border:1px solid {BORDER};"
            f"border-radius:6px;margin-top:10px;padding:12px;font-weight:600;color:{ACCENT};}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:10px;color:{ACCENT};}}")
        cl = QVBoxLayout(card); il = QLabel()
        il.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if os.path.exists(filepath):
            pix = QPixmap(filepath)
            if not pix.isNull():
                il.setPixmap(pix.scaledToWidth(min(820,pix.width()),Qt.TransformationMode.SmoothTransformation))
            else: il.setText(f"Erro ao carregar: {filepath}")
        else: il.setText(f"Não encontrado: {filepath}")
        cl.addWidget(il)
        cnt = self.img_layout.count()
        self.img_layout.insertWidget(max(0,cnt-1), card)
        self.tabs.setCurrentIndex(2)

    def _on_result(self, results):
        self.results = results; self._show_results(results); self.tabs.setCurrentIndex(3)

    def _on_error(self, msg):
        self._append_log(f"ERRO: {msg}", "ERROR")
        self.status_lbl.setText("● MAPDL erro")
        self.status_lbl.setStyleSheet(f"color:{RED};font-size:10px;background:transparent;")
        QMessageBox.critical(self,"Erro no Pipeline", msg[:500])

    def _on_finished(self):
        self.btn_run.setEnabled(True); self.btn_stop.setEnabled(False)
        if self.results:
            self.btn_report.setEnabled(True)
            self.status_lbl.setText("● MAPDL concluído")
            self.status_lbl.setStyleSheet(f"color:{GREEN};font-size:10px;background:transparent;")
        else:
            self.status_lbl.setText("● MAPDL inativo")
            self.status_lbl.setStyleSheet(f"color:{TEXT_D};font-size:10px;background:transparent;")

    def _show_results(self, r):
        p = r['params']; ok = r['stress_max'] <= p['tensao_max']
        sc = GREEN if ok else RED; st = "APROVADO" if ok else "REPROVADO"
        self.results_box.setHtml(f"""
        <div style='font-family:Segoe UI,sans-serif;padding:4px;'>
        <h2 style='color:{ACCENT};font-size:15px'>{p['nome_projeto']}</h2>
        <p style='color:{sc};font-size:14px;font-weight:bold'>{st}</p>
        <table style='width:100%;border-collapse:collapse;font-size:12px;'>
        <tr><td style='padding:6px 10px;border-bottom:1px solid {BORDER}'>Elementos</td>
            <td style='color:{ACCENT};font-weight:bold'>{r['n_elementos']}</td></tr>
        <tr style='background:{PANEL}'><td style='padding:6px 10px;border-bottom:1px solid {BORDER}'>Nós</td>
            <td style='color:{ACCENT};font-weight:bold'>{r['n_nos']}</td></tr>
        <tr><td style='padding:6px 10px;border-bottom:1px solid {BORDER}'>σ Von Mises máx.</td>
            <td style='color:{sc};font-weight:bold'>{r['stress_max']:.4f} MPa</td></tr>
        <tr style='background:{PANEL}'><td style='padding:6px 10px;border-bottom:1px solid {BORDER}'>σ admissível</td>
            <td style='color:{ACCENT};font-weight:bold'>{p['tensao_max']} MPa</td></tr>
        <tr><td style='padding:6px 10px;border-bottom:1px solid {BORDER}'>Deslocamento máx.</td>
            <td style='color:{ACCENT};font-weight:bold'>{r['disp_max']:.6f} mm</td></tr>
        <tr style='background:{PANEL}'><td style='padding:6px 10px'>Área casca</td>
            <td style='color:{ACCENT};font-weight:bold'>{r['area_casca']:.2f} mm²</td></tr>
        </table>
        <p style='color:{TEXT_D};font-size:10px;margin-top:8px'>{r['timestamp']}</p>
        </div>""")

    def _export_report(self):
        if not self.results: return
        path, _ = QFileDialog.getSaveFileName(self,"Salvar relatório",
            f"{self.results['params']['nome_projeto']}_relatorio.html","HTML (*.html)")
        if not path: return
        from report import gerar_relatorio_html
        gerar_relatorio_html(self.results, path)
        QMessageBox.information(self,"Relatório exportado",f"Salvo em:\n{path}")
        import webbrowser; webbrowser.open(path)

    # ── Análise aerodinâmica ──────────────────────────────────────────────────

    def _run_aero(self):
        semi  = self.sch_semi_span.value()  if hasattr(self,'sch_semi_span')  else 750
        c_r   = self.sch_root_chord.value() if hasattr(self,'sch_root_chord') else 300
        c_t   = self.sch_tip_chord.value()  if hasattr(self,'sch_tip_chord')  else 200
        massa = self.sch_mass.value()        if hasattr(self,'sch_mass')       else 5.0
        n_r   = int(self.wc_n_ribs.value()) if hasattr(self,'wc_n_ribs')      else 12
        params = {"semi_span":semi,"root_chord":c_r,"tip_chord":c_t,
                  "velocidade":self.aero_vel.value(),"rho":self.aero_dens.value(),
                  "fator_carga":self.aero_carga.value(),"massa":massa,"n_ribs":n_r}
        self.tabs_aero.setCurrentIndex(1); self.log_box_aero.clear()
        self.btn_run_aero.setEnabled(False); self.btn_transfer.setEnabled(False)
        self.worker_aero = AeroWorker(params)
        self.worker_aero.log_signal.connect(self._on_aero_log)
        self.worker_aero.progress_signal.connect(lambda v,l: (self.prog_aero.setValue(v),
                                                               self.prog_lbl_aero.setText(l)))
        self.worker_aero.finished_signal.connect(self._on_aero_finished)
        self.worker_aero.start()

    def _on_aero_log(self, msg, nivel):
        color = self.COLORS.get(nivel, TEXT_P)
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box_aero.append(f'<span style="color:{TEXT_D}">[{ts}]</span> '
                                  f'<span style="color:{color}">{msg}</span>')

    def _on_aero_finished(self, results):
        self.btn_run_aero.setEnabled(True); self.btn_transfer.setEnabled(True)
        self._ultima_pressao = results["pressao_extraida"]
        self._aero_results = results

    def _transfer_pressure(self):
        try:
            if hasattr(self,'inp_pressao'):
                self.inp_pressao.setValue(self._ultima_pressao)
                QMessageBox.information(self,"Transferência OK",
                    f"Pressão {self._ultima_pressao:.6f} MPa → módulo estrutural.")
                self._enter_workspace(2)
        except Exception as e:
            QMessageBox.warning(self,"Erro",str(e))


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = NervuraApp()
    win.show()
    sys.exit(app.exec())