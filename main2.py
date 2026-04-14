"""
main2.py — T.O.C.A. v0.3 (Mamutes do Cerrado / MMTS)
Toolkit de Otimização Computacional de Aeronaves

M�dulos (sidebar):
  0. Cargas & Sustentação   — Schrenk, Peso/CG, Entelagem, Sensibilidade
  1. Análise Aerodinâmica   — Pressão nervura crítica
  2. Posicionamento Nervuras — RIBSPO (SEPARADO do wingbox)
  3. Wingbox                — Caixa de Torção (idealização única por booms)
  4. Dim. Longarinas        — NOVO (Megson Cap.16-20 + Cooper Cap.6)
  5. Aeroelasticidade       — Flutter 2DOF/3DOF (aeroelasticidade_solvers.py)
  6. Otimização Nervura     — ANSYS MAPDL TopOpt (material variável)
  7. MDO (ÚLTIMA)           — Pipeline completo NSGA-II

Refs: Megson 6th Ed., Cooper 2008, Schrenk 1940.
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

from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap

from modules_analysis import (
    build_schrenk_tab, build_weight_cg_tab,
    build_covering_tab, build_sensitivity_tab,
)
from module_wingbox_visual import (
    build_wingbox_visual_tab, build_rib_positioning_tab,
    build_aeroelastic_tab, build_wingbox_module,
)
from modules_mdo import build_mdo_module

# Paleta
BG="#111318"; SURFACE="#1C2030"; PANEL="#21263C"
RAISED="#282E48"; BORDER="#30395A"; BORDER2="#3E4C70"
ACCENT="#4D82D6"; ACCENT_H="#6196E8"
TEXT_P="#C8D4EC"; TEXT_S="#6A7A9C"; TEXT_D="#3A4562"
GREEN="#4EC88A"; RED="#D95252"; AMBER="#D9963A"
PURPLE="#7C5CBF"; TEAL="#2BB5B5"

def gerar_naca4(code, n_points=150):
    m=int(code[0])/100.0; p=int(code[1])/10.0; t=int(code[2:4])/100.0
    beta=np.linspace(0,np.pi,n_points); x=0.5*(1-np.cos(beta))
    yt=5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
    yc=np.zeros_like(x); dyc=np.zeros_like(x)
    if m>0 and p>0:
        fr=x<=p; rr=x>p
        yc[fr]=(m/p**2)*(2*p*x[fr]-x[fr]**2); yc[rr]=(m/(1-p)**2)*(1-2*p+2*p*x[rr]-x[rr]**2)
        dyc[fr]=(2*m/p**2)*(p-x[fr]); dyc[rr]=(2*m/(1-p)**2)*(p-x[rr])
    theta=np.arctan(dyc)
    xu=x-yt*np.sin(theta); yu=yc+yt*np.cos(theta)
    xl=x+yt*np.sin(theta); yl=yc-yt*np.cos(theta)
    return np.vstack([np.column_stack([xu[::-1],yu[::-1]]),np.column_stack([xl[1:],yl[1:]])])

BANCO_PERFIS={}; BANCO_AERONAVES={}

def carregar_bancos_de_dados():
    global BANCO_PERFIS, BANCO_AERONAVES
    if os.path.exists("aeronaves.json"):
        try:
            with open("aeronaves.json",'r',encoding='utf-8') as f: BANCO_AERONAVES=json.load(f)
        except: pass
    if not BANCO_AERONAVES:
        BANCO_AERONAVES={"Aeronave Padrão":{"semi_span":750.0,"root_chord":300.0,"tip_chord":200.0,"n_ribs":12,"velocity":15.0,"mass":5.0,"load_factor":4.0,"airfoil":"NACA 4412 (Gerado)"}}
        with open("aeronaves.json",'w',encoding='utf-8') as f: json.dump(BANCO_AERONAVES,f,indent=4)
    os.makedirs("banco_perfis",exist_ok=True)
    for arq in glob.glob(os.path.join("banco_perfis","*.dat")):
        nome=os.path.splitext(os.path.basename(arq))[0]
        try:
            try: coords=np.loadtxt(arq,skiprows=0)
            except: coords=np.loadtxt(arq,skiprows=1)
            BANCO_PERFIS[nome]=coords
        except: pass
    BANCO_PERFIS["NACA 4412 (Gerado)"]=gerar_naca4("4412",150)

carregar_bancos_de_dados()

# Workers
class AeroWorker(QThread):
    log_signal=pyqtSignal(str,str); progress_signal=pyqtSignal(int,str); finished_signal=pyqtSignal(dict)
    def __init__(s,params): super().__init__(); s.params=params; s._stopped=False
    def stop(s): s._stopped=True
    def log(s,msg,n="INFO"): s.log_signal.emit(msg,n)
    def run(s):
        try:
            from schrenk import WingGeometry,FlightCondition,schrenk_distribution,discretize_rib_loads_matlab
            p=s.params; s.log("="*60); s.log("CÁLCULO DE CARGAS (Schrenk)"); s.log("="*60)
            s.progress_signal.emit(10,"Geometria...")
            wing=WingGeometry(semi_span_mm=p['semi_span'],root_chord_mm=p['root_chord'],tip_chord_mm=p['tip_chord'])
            flight=FlightCondition(velocity_ms=p['velocidade'],rho_kgm3=p.get('rho',1.225),load_factor=p['fator_carga'],aircraft_mass_kg=p['massa'])
            s.progress_signal.emit(30,"Schrenk..."); result=schrenk_distribution(wing,flight,n_stations=500)
            s.log(f"  L={result.total_lift_N:.2f}N | V_raiz={result.max_shear_N:.2f}N","OK")
            n_ribs=int(p.get('n_ribs',12)); s.progress_signal.emit(60,"Discretizando...")
            rib_data=discretize_rib_loads_matlab(wing,flight,n_ribs); idx_c=rib_data['idx_crit']
            s.log(f"  Nerv. crit #{idx_c+1} P={rib_data['pressure_per_rib'][idx_c]:.6f}MPa","WARN")
            pressao=float(-abs(rib_data['pressure_per_rib'][idx_c]))
            s.progress_signal.emit(100,"OK!")
            s.finished_signal.emit({"pressao_extraida":pressao,"rib_data":rib_data,"idx_crit":idx_c,"n_ribs":n_ribs,"schrenk_result":result,"wing":wing,"flight":flight})
        except Exception as e:
            import traceback; s.log(f"Erro: {e}","ERROR"); s.log(traceback.format_exc(),"ERROR")

class PipelineWorker(QThread):
    log_signal=pyqtSignal(str,str); progress_signal=pyqtSignal(int,str)
    image_signal=pyqtSignal(str,str); result_signal=pyqtSignal(dict); error_signal=pyqtSignal(str)
    def __init__(s,params): super().__init__(); s.params=params; s._stopped=False
    def stop(s): s._stopped=True
    def log(s,msg,n="INFO"): s.log_signal.emit(msg,n)
    def run(s):
        try: s._pipeline()
        except Exception as e:
            import traceback; s.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")
    def _pipeline(s):
        p=s.params; plt.switch_backend('Agg'); plt.show=lambda *a,**kw:None
        s.progress_signal.emit(2,"Iniciando..."); s.log("="*60); s.log("PIPELINE TOPOPT"); s.log("="*60)
        img_dir=os.path.join(os.getcwd(),f"{p['nome_projeto']}_resultados"); os.makedirs(img_dir,exist_ok=True)
        if p.get('caminho_dat') and os.path.exists(p['caminho_dat']):
            try: coords=np.loadtxt(p['caminho_dat'],skiprows=0)
            except: coords=np.loadtxt(p['caminho_dat'],skiprows=1)
        elif p.get('banco_perfil') and p['banco_perfil'] in BANCO_PERFIS: coords=BANCO_PERFIS[p['banco_perfil']].copy()
        else: coords=gerar_naca4(p.get('naca_code','4412'))
        cs=coords*p['corda_mm']; cf=[cs[0]]
        for pt in cs[1:]:
            if np.linalg.norm(pt-cf[-1])>1e-4: cf.append(pt)
        if np.linalg.norm(cf[0]-cf[-1])<=1e-4: cf.pop()
        cf=np.array(cf)
        poly_ext=Polygon(cf)
        if not poly_ext.is_valid: poly_ext=poly_ext.buffer(0)
        poly_int=poly_ext.buffer(-p['espessura_casca'],join_style=1)
        ci=np.array(poly_int.exterior.coords)
        if np.linalg.norm(ci[0]-ci[-1])<=1e-4: ci=ci[:-1]
        area_casca=poly_ext.area-poly_int.area
        s.progress_signal.emit(25,"MAPDL...")
        try: from ansys.mapdl.core import launch_mapdl
        except ImportError: raise ImportError("PyMAPDL não encontrado")
        mapdl=launch_mapdl(jobname=p['nome_projeto'],override=True,port=p.get('mapdl_port',50056),cleanup_on_exit=False)
        s.log(f"  MAPDL {mapdl.version}","OK")
        try:
            mapdl.clear(); mapdl.prep7()
            mapdl.mp('EX',1,p['modulo_elasticidade']); mapdl.mp('PRXY',1,p['poisson']); mapdl.mp('DENS',1,p['densidade'])
            mapdl.et(1,'PLANE183')
            for i,pt in enumerate(cf): mapdl.k(1+i,pt[0],pt[1],0)
            n_out=len(cf)
            for i in range(n_out-1): mapdl.l(1+i,2+i)
            mapdl.l(n_out,1); mapdl.lsel('S','LINE','',1,n_out); mapdl.run("CM,COMP_LINES_OUTER,LINE"); mapdl.al('ALL')
            mapdl.asel('S','AREA','',1); mapdl.run("CM,COMP_OUTER,AREA")
            mapdl.allsel(); ki=n_out+1
            for i,pt in enumerate(ci): mapdl.k(ki+i,pt[0],pt[1],0)
            n_in=len(ci)
            for i in range(n_in-1): mapdl.l(ki+i,ki+i+1)
            mapdl.l(ki+n_in-1,ki); mapdl.lsel('S','LINE','',ki,ki+n_in-1); mapdl.al('ALL')
            mapdl.asel('S','AREA','',2); mapdl.run("CM,COMP_INNER,AREA")
            mapdl.allsel(); mapdl.asba('COMP_OUTER','COMP_INNER',keep2="KEEP")
            mapdl.run("CM,COMP_BLUE,AREA"); mapdl.cmsel('S','COMP_INNER'); mapdl.run("CM,COMP_YELLOW,AREA")
            mapdl.allsel(); mapdl.esize(p['tamanho_elemento']); mapdl.amesh('ALL')
            n_el=mapdl.mesh.n_elem; n_no=mapdl.mesh.n_node; s.log(f"  Malha: {n_el}el, {n_no}nós","OK")
            mapdl.cmsel('S','COMP_LINES_OUTER'); mapdl.sfl('ALL','PRES',p['pressao_aerodinamica'])
            c=p['corda_mm']; mapdl.allsel()
            mapdl.nsel('S','LOC','X',p['x_long_ini_pct']*c,p['x_long_fim_pct']*c)
            mapdl.nsel('R','LOC','Y',-0.05*c,0.05*c); mapdl.d('ALL','UX',0); mapdl.d('ALL','UY',0); mapdl.allsel()
            mapdl.cmsel('S','COMP_YELLOW','AREA'); mapdl.run("ESLA,S")
            mapdl.run("CM,TOPO_DESIGN,ELEM"); mapdl.allsel()
            mapdl.run("FINISH"); mapdl.run("/SOLU"); mapdl.run("ANTYPE,0")
            mapdl.run("TOVAR,MATDEN,DENSITY,0.001,1.0"); mapdl.run("TOCOMP,TOPO_DESIGN,MATDEN")
            mapdl.run("TOVAR,VOLUME,OBJ"); mapdl.run(f"TOVAR,SEQV,CON,0,{p['tensao_max']}")
            mapdl.run(f"TOFREQ,{p['max_iter']},{p['convergencia']},SIMP")
            s.progress_signal.emit(62,"SOLVE..."); mapdl.run("SOLVE")
            s.progress_signal.emit(65,"TOEXE..."); mapdl.run("TOEXE"); s.log("  TOEXE OK","OK")
            mapdl.run("FINISH"); mapdl.run("/POST1"); mapdl.run("SET,LAST")
            stress_max=stress_mean=disp_max=0.0
            try:
                mapdl.allsel(); d=mapdl.post_processing.nodal_displacement("NORM")
                if d is not None and d.size>0: disp_max=float(np.nanmax(d))
                sv=mapdl.post_processing.nodal_eqv_stress()
                if sv is not None and sv.size>0: stress_max=float(np.nanmax(sv)); stress_mean=float(np.nanmean(sv))
            except: pass
            s.result_signal.emit({"params":p,"n_elementos":n_el,"n_nos":n_no,"stress_max":stress_max,"stress_mean":stress_mean,"disp_max":disp_max,"area_casca":area_casca,"img_dir":img_dir,"timestamp":datetime.now().isoformat()})
            s.progress_signal.emit(100,"Concluído!")
        finally:
            try: mapdl.exit()
            except: pass

# ═══════════════════════════════════════════════════════════════════════════════
class NervuraApp(QMainWindow):
    def __init__(self):
        super().__init__(); self.worker=None; self.results=None
        self._apply_theme(); self._setup_ui()

    def _apply_theme(self):
        self.setStyleSheet(f"""
        QMainWindow,QWidget{{background:{SURFACE};color:{TEXT_P};font-family:'Segoe UI',sans-serif;font-size:12px;}}
        QGroupBox{{background:{PANEL};border:1px solid {BORDER};border-radius:6px;margin-top:10px;padding:10px 8px 8px;font-weight:600;font-size:11px;color:{TEXT_S};}}
        QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 4px;color:{TEXT_S};}}
        QLineEdit,QDoubleSpinBox,QSpinBox,QComboBox{{background:{RAISED};border:1px solid {BORDER};border-radius:4px;color:{TEXT_P};padding:4px 8px;min-height:26px;}}
        QLineEdit:focus,QDoubleSpinBox:focus,QSpinBox:focus,QComboBox:focus{{border:1px solid {ACCENT};}}
        QComboBox::drop-down{{border:none;background:{RAISED};border-left:1px solid {BORDER};width:24px;}}
        QComboBox QAbstractItemView{{background:{RAISED};color:{TEXT_P};border:1px solid {BORDER2};selection-background-color:{ACCENT};}}
        QDoubleSpinBox::up-button,QSpinBox::up-button,QDoubleSpinBox::down-button,QSpinBox::down-button{{background:{RAISED};border:none;width:18px;}}
        QTabWidget::pane{{background:{SURFACE};border:1px solid {BORDER};border-radius:0 0 6px 6px;top:-1px;}}
        QTabBar::tab{{background:{PANEL};color:{TEXT_S};padding:7px 18px;border:1px solid {BORDER};border-bottom:none;border-radius:4px 4px 0 0;margin-right:2px;font-size:11px;}}
        QTabBar::tab:selected{{background:{SURFACE};color:{TEXT_P};border-bottom:2px solid {ACCENT};font-weight:600;}}
        QScrollArea{{border:none;background:transparent;}}
        QScrollBar:vertical{{background:{PANEL};width:8px;border-radius:4px;}}
        QScrollBar::handle:vertical{{background:{BORDER2};border-radius:4px;min-height:20px;}}
        QProgressBar{{background:{PANEL};border:1px solid {BORDER};border-radius:3px;height:6px;}}
        QProgressBar::chunk{{background:{ACCENT};border-radius:3px;}}
        QLabel{{color:{TEXT_P};background:transparent;}}
        QCheckBox{{color:{TEXT_S};background:transparent;}}
        QCheckBox::indicator{{width:14px;height:14px;border:1px solid {BORDER};border-radius:3px;background:{RAISED};}}
        QCheckBox::indicator:checked{{background:{ACCENT};border-color:{ACCENT};}}
        QTableWidget{{background:{BG};color:{TEXT_P};gridline-color:{BORDER};border:none;}}
        QHeaderView::section{{background:{PANEL};color:{TEXT_S};border:1px solid {BORDER};padding:4px;font-size:9px;}}
        """)

    def _setup_ui(self):
        self.setWindowTitle("T.O.C.A. v0.3 — Mamutes do Cerrado / MMTS Aerodesign")
        self.setMinimumSize(1400,900)
        central=QWidget(); self.setCentralWidget(central)
        root=QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        self.root_stack=QStackedWidget(); root.addWidget(self.root_stack)
        self._build_home_page()
        workspace=QWidget(); workspace.setStyleSheet(f"background:{BG};")
        ws=QHBoxLayout(workspace); ws.setContentsMargins(0,0,0,0); ws.setSpacing(0)
        self.stack=QStackedWidget(); self.stack.setStyleSheet(f"background:{SURFACE};")
        ws.addWidget(self._build_sidebar())
        sep=QFrame(); sep.setFrameShape(QFrame.Shape.VLine); sep.setStyleSheet(f"color:{BORDER};max-width:1px;")
        ws.addWidget(sep); ws.addWidget(self.stack,stretch=1)
        self.root_stack.addWidget(workspace)
        # 8 módulos na NOVA ORDEM
        self._build_module_analysis()       # 0
        self._build_module_aero()           # 1
        self._build_module_ribspo()         # 2 SEPARADO
        self._build_module_wingbox_page()   # 3
        self._build_module_spar_sizing()    # 4 NOVO
        self._build_module_aeroelastic()    # 5
        self._build_module_nervura()        # 6
        self._build_module_mdo_page()       # 7 ÚLTIMA

    def _enter_workspace(self,idx=0): self.root_stack.setCurrentIndex(1); self._switch_module(idx)
    def _go_home(self): self.root_stack.setCurrentIndex(0)

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
        sec=QLabel("MÓDULOS DE ANÁLISE"); sec.setStyleSheet(f"color:{TEXT_D};font-size:9px;font-weight:700;letter-spacing:.16em;padding:0 16px 6px;background:transparent;")
        lay.addWidget(sec)
        self._nav_buttons=[]
        modules=[
            ("  ≈  Cargas & Sustentação",0),("  ≀  Análise Aerodinâmica",1),
            ("  ◈  Posicionamento Nervuras",2),("  ▭  Wingbox (Caixa Torção)",3),
            ("  ⫾  Dim. Longarinas",4),("  ∿  Aeroelasticidade",5),
            ("  ⬡  Nervura ANSYS",6),("  ⊕  MDO — Pipeline",7),
        ]
        for label,idx in modules:
            btn=self._nav_button(label,idx); self._nav_buttons.append(btn); lay.addWidget(btn)
        lay.addStretch()
        btn_home=QPushButton("  ◉  Início"); btn_home.setFixedHeight(34); btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_home.setStyleSheet(f"QPushButton{{text-align:left;padding:0 12px;border:none;border-left:2px solid transparent;background:transparent;color:{TEXT_D};font-size:11px;}}QPushButton:hover{{background:{PANEL};color:{TEXT_P};}}")
        btn_home.clicked.connect(self._go_home); lay.addWidget(btn_home)
        self.status_lbl=QLabel("● MAPDL inativo"); self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_lbl.setStyleSheet(f"color:{TEXT_D};font-size:10px;font-weight:bold;background:transparent;")
        lay.addWidget(self.status_lbl)
        footer=QLabel("Megson · Cooper · Schrenk\n© 2026 MMTS"); footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet(f"color:{TEXT_D};font-size:9px;padding:8px 14px;border-top:1px solid {BORDER};background:transparent;")
        lay.addWidget(footer); return sidebar

    def _nav_button(self,text,idx):
        btn=QPushButton(text); btn.setCheckable(True); btn.setChecked(idx==0); btn.setFixedHeight(36)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"QPushButton{{text-align:left;padding:0 10px;border:none;border-left:2px solid {'"+ACCENT+"' if idx==0 else 'transparent'};background:{'"+PANEL+"' if idx==0 else 'transparent'};color:{'"+TEXT_P+"' if idx==0 else '"+TEXT_S+"'};font-size:11px;{'font-weight:600;' if idx==0 else ''}}}QPushButton:hover{{background:{PANEL};color:{TEXT_P};border-left:2px solid {BORDER2};}}")
        btn.clicked.connect(lambda:self._switch_module(idx)); return btn

    def _switch_module(self,idx):
        self.stack.setCurrentIndex(idx)
        for i,b in enumerate(self._nav_buttons):
            a=(i==idx)
            b.setChecked(a)
            b.setStyleSheet(f"QPushButton{{text-align:left;padding:0 10px;border:none;border-left:2px solid {ACCENT if a else 'transparent'};background:{PANEL if a else 'transparent'};color:{TEXT_P if a else TEXT_S};font-size:11px;{'font-weight:600;' if a else ''}}}QPushButton:hover{{background:{PANEL};color:{TEXT_P};border-left:2px solid {BORDER2};}}")

    def _module_header(self,crumb,title):
        w=QWidget(); w.setFixedHeight(64); w.setStyleSheet(f"background:{PANEL};border-bottom:1px solid {BORDER};")
        lay=QHBoxLayout(w); lay.setContentsMargins(20,0,20,0)
        col=QVBoxLayout(); col.setSpacing(2)
        cl=QLabel(crumb); cl.setStyleSheet(f"color:{TEXT_D};font-size:9px;letter-spacing:.12em;background:transparent;")
        tl=QLabel(title); tl.setFont(QFont("Segoe UI",13,QFont.Weight.Bold)); tl.setStyleSheet(f"color:{TEXT_P};background:transparent;")
        col.addWidget(cl); col.addWidget(tl); lay.addLayout(col); lay.addStretch(); return w

    # Módulo 0
    def _build_module_analysis(self):
        page=QWidget(); page.setStyleSheet(f"background:{SURFACE};")
        lay=QVBoxLayout(page); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self._module_header("MÓDULOS / CARGAS","Sustentação, Peso, Entelagem e Sensibilidade"))
        tabs=QTabWidget(); tabs.setDocumentMode(True)
        tabs.addTab(build_schrenk_tab(self),"📐  Schrenk"); tabs.addTab(build_weight_cg_tab(self),"⚖  Peso & CG")
        tabs.addTab(build_covering_tab(self),"🎯  Entelagem"); tabs.addTab(build_sensitivity_tab(self),"📊  Sensibilidade")
        lay.addWidget(tabs,stretch=1); self.stack.addWidget(page)

    # Módulo 1
    def _build_module_aero(self):
        page=QWidget(); page.setStyleSheet(f"background:{SURFACE};")
        lay=QVBoxLayout(page); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self._module_header("MÓDULOS / AERODINÂMICA","Pressão Nervura Crítica (Schrenk)"))
        self.tabs_aero=QTabWidget(); self.tabs_aero.setDocumentMode(True)
        lay.addWidget(self.tabs_aero,stretch=1)
        # Aba params
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setStyleSheet(f"background:{SURFACE};")
        w=QWidget(); w.setStyleSheet(f"background:{SURFACE};"); scroll.setWidget(w); vl=QVBoxLayout(w); vl.setSpacing(10); vl.setContentsMargins(16,16,16,16)
        g1=QGroupBox("Condições de Voo"); gl=QGridLayout(g1); gl.setSpacing(8)
        self.aero_vel=self._mk_spin(15.69,1,300,2); self.aero_dens=self._mk_spin(1.225,0,2,3); self.aero_carga=self._mk_spin(2.5,1,10,1)
        for i,(l,wgt) in enumerate([("Velocidade (m/s):",self.aero_vel),("Densidade (kg/m³):",self.aero_dens),("Fator carga:",self.aero_carga)]):
            gl.addWidget(QLabel(l),i,0); gl.addWidget(wgt,i,1)
        vl.addWidget(g1); vl.addStretch(); self.tabs_aero.addTab(scroll,"⚙  Parâmetros")
        # Aba log
        wl=QWidget(); wl.setStyleSheet(f"background:{SURFACE};"); ll=QVBoxLayout(wl); ll.setContentsMargins(12,12,12,12)
        self.log_box_aero=QTextEdit(); self.log_box_aero.setReadOnly(True); self.log_box_aero.setFont(QFont("Consolas",10))
        self.log_box_aero.setStyleSheet(f"background:{BG};color:{TEXT_P};border:1px solid {BORDER};border-radius:5px;padding:8px;")
        ll.addWidget(self.log_box_aero); self.tabs_aero.addTab(wl,"📋  Log")
        # Action bar
        bar=QWidget(); bar.setFixedHeight(58); bar.setStyleSheet(f"background:{PANEL};border-top:1px solid {BORDER};")
        bl=QHBoxLayout(bar); bl.setContentsMargins(16,0,16,0); bl.setSpacing(8)
        self.btn_run_aero=QPushButton("▶  Calcular Cargas"); self.btn_run_aero.setFixedHeight(34)
        self.btn_run_aero.setStyleSheet(f"QPushButton{{background:{ACCENT};color:white;border:none;border-radius:4px;font-weight:bold;padding:0 18px;}}")
        self.btn_run_aero.clicked.connect(self._run_aero)
        self.btn_transfer=QPushButton("⮂  Transferir → Nervura"); self.btn_transfer.setFixedHeight(34); self.btn_transfer.setEnabled(False)
        self.btn_transfer.setStyleSheet(f"QPushButton{{background:#2E7D52;color:white;border:none;border-radius:4px;font-weight:bold;padding:0 18px;}}QPushButton:disabled{{background:{BORDER};color:{TEXT_D};}}")
        self.btn_transfer.clicked.connect(self._transfer_pressure)
        bl.addWidget(self.btn_run_aero); bl.addWidget(self.btn_transfer); bl.addSpacing(16)
        pc=QVBoxLayout(); self.prog_lbl_aero=QLabel("Pronto"); self.prog_lbl_aero.setStyleSheet(f"color:{TEXT_S};font-size:10px;background:transparent;")
        self.prog_aero=QProgressBar(); self.prog_aero.setFixedHeight(5); self.prog_aero.setTextVisible(False)
        pc.addWidget(self.prog_lbl_aero); pc.addWidget(self.prog_aero); bl.addLayout(pc,stretch=1)
        lay.addWidget(bar); self.stack.addWidget(page)

    # Módulo 2 — RIBSPO SEPARADO
    def _build_module_ribspo(self):
        page=QWidget(); page.setStyleSheet(f"background:{SURFACE};")
        lay=QVBoxLayout(page); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self._module_header("MÓDULOS / POSICIONAMENTO","RIBSPO — Otimização Espaçamento Nervuras (Separado)"))
        tabs=QTabWidget(); tabs.setDocumentMode(True)
        tabs.addTab(build_rib_positioning_tab(self),"◈  RIBSPO — Evolução Diferencial")
        lay.addWidget(tabs,stretch=1); self.stack.addWidget(page)

    # Módulo 3 — Wingbox
    def _build_module_wingbox_page(self): self.stack.addWidget(build_wingbox_module(self))

    # Módulo 4 — Dim. Longarinas NOVO
    def _build_module_spar_sizing(self):
        from spar_sizing import PROFILE_TYPES, SPAR_PRESETS, size_spar, spar_tapering_analysis
        from materials import get_all_materials
        page=QWidget(); page.setStyleSheet(f"background:{SURFACE};")
        lay=QVBoxLayout(page); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self._module_header("MÓDULOS / LONGARINAS","Dimensionamento Preliminar (Megson Cap.16-20, Cooper Cap.6)"))
        tabs=QTabWidget(); tabs.setDocumentMode(True)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setStyleSheet(f"background:{SURFACE};")
        w=QWidget(); w.setStyleSheet(f"background:{SURFACE};"); scroll.setWidget(w)
        slay=QVBoxLayout(w); slay.setSpacing(10); slay.setContentsMargins(16,16,16,16)
        g1=QGroupBox("Seção da Longarina"); gl=QGridLayout(g1); gl.setSpacing(8)
        self.spar_profile=QComboBox(); self.spar_profile.addItems(PROFILE_TYPES)
        self.spar_preset=QComboBox(); self.spar_preset.addItem("— Personalizado —"); self.spar_preset.addItems(list(SPAR_PRESETS.keys()))
        self.spar_preset.activated.connect(self._apply_spar_preset)
        self.spar_h=self._mk_spin(12,1,100,1); self.spar_w=self._mk_spin(12,1,100,1); self.spar_wall=self._mk_spin(1,0.1,10,2)
        self.spar_ft=self._mk_spin(1.5,0.1,10,2); self.spar_wt=self._mk_spin(1,0.1,10,2); self.spar_length=self._mk_spin(1500,100,5000,0)
        self.spar_mat=QComboBox(); self.spar_mat.addItems(list(get_all_materials().keys()))
        idx_cf=self.spar_mat.findText("CFRP UD (0°)")
        if idx_cf>=0: self.spar_mat.setCurrentIndex(idx_cf)
        for i,(l,wgt) in enumerate([("Preset:",self.spar_preset),("Tipo:",self.spar_profile),("Material:",self.spar_mat),
            ("Altura (mm):",self.spar_h),("Largura (mm):",self.spar_w),("Parede (mm):",self.spar_wall),
            ("Flange (mm):",self.spar_ft),("Web (mm):",self.spar_wt),("Comprimento (mm):",self.spar_length)]):
            gl.addWidget(QLabel(l),i,0); gl.addWidget(wgt,i,1)
        slay.addWidget(g1)
        g2=QGroupBox("Cargas (Megson Cap.16)"); gl2=QGridLayout(g2); gl2.setSpacing(8)
        self.spar_M=self._mk_spin(0,-1e8,1e8,1); self.spar_V=self._mk_spin(0,-1e5,1e5,1)
        self.spar_T=self._mk_spin(0,-1e8,1e8,1); self.spar_sf=self._mk_spin(1.5,1,3,2)
        for i,(l,wgt) in enumerate([("M fletor (N·mm):",self.spar_M),("V cortante (N):",self.spar_V),("T torque (N·mm):",self.spar_T),("Fator seg.:",self.spar_sf)]):
            gl2.addWidget(QLabel(l),i,0); gl2.addWidget(wgt,i,1)
        slay.addWidget(g2)
        btn=QPushButton("▶  Dimensionar"); btn.setFixedHeight(36); btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"QPushButton{{background:{TEAL};color:#111;border:none;border-radius:4px;font-size:12px;font-weight:600;padding:0 18px;}}")
        btn.clicked.connect(self._run_spar_sizing); slay.addWidget(btn)
        self.spar_result_box=QTextEdit(); self.spar_result_box.setReadOnly(True); self.spar_result_box.setMaximumHeight(250)
        self.spar_result_box.setStyleSheet(f"background:{BG};color:{TEXT_P};border:1px solid {BORDER};border-radius:5px;padding:8px;font-size:11px;")
        slay.addWidget(self.spar_result_box)
        self.spar_fig=Figure(figsize=(10,5),dpi=100); self.spar_fig.patch.set_facecolor(BG)
        self.spar_canvas=FigureCanvas(self.spar_fig); self.spar_canvas.setMinimumHeight(400)
        slay.addWidget(self.spar_canvas); slay.addStretch()
        tabs.addTab(scroll,"⫾  Dimensionamento (Megson)")
        lay.addWidget(tabs,stretch=1); self.stack.addWidget(page)

    def _apply_spar_preset(self):
        from spar_sizing import SPAR_PRESETS
        p=SPAR_PRESETS.get(self.spar_preset.currentText())
        if not p: return
        idx=self.spar_profile.findText(p.get("profile_type","Tubular"))
        if idx>=0: self.spar_profile.setCurrentIndex(idx)
        for k,w in [("height_mm",self.spar_h),("width_mm",self.spar_w),("wall_mm",self.spar_wall),("flange_t_mm",self.spar_ft),("web_t_mm",self.spar_wt)]:
            if k in p: w.setValue(p[k])

    def _run_spar_sizing(self):
        from spar_sizing import size_spar, spar_tapering_analysis
        r=size_spar(self.spar_profile.currentText(),self.spar_mat.currentText(),self.spar_h.value(),self.spar_w.value(),
            self.spar_wall.value(),self.spar_ft.value(),self.spar_wt.value(),self.spar_length.value(),
            self.spar_M.value(),self.spar_V.value(),self.spar_T.value(),self.spar_sf.value())
        col=GREEN if r.approved else RED; st="✓ APROVADO" if r.approved else "✗ REPROVADO"
        self.spar_result_box.setHtml(f"""
        <h3 style='color:{ACCENT}'>Longarina — {r.profile_type} ({r.material_name})</h3>
        <p style='color:{col};font-weight:bold;font-size:14px'>{st} | Modo: {r.critical_mode}</p>
        <table style='width:100%;border-collapse:collapse;font-size:11px;'>
        <tr style='background:{PANEL}'><td style='padding:3px 8px'>A={r.area_mm2:.2f}mm²</td><td>Ixx={r.Ixx_mm4:.1f}mm⁴</td><td>J={r.J_mm4:.1f}mm⁴</td><td>Massa={r.mass_total_g:.2f}g</td></tr>
        <tr><td style='padding:3px 8px'>σ_flex={r.sigma_bending_MPa:.3f}MPa</td><td>MS={r.MS_bending:.2f}</td><td>τ_cis={r.tau_shear_MPa:.3f}MPa</td><td>MS={r.MS_shear:.2f}</td></tr>
        <tr style='background:{PANEL}'><td style='padding:3px 8px'>τ_tor={r.tau_torsion_MPa:.3f}MPa</td><td>MS={r.MS_torsion:.2f}</td><td>σ_VM={r.sigma_von_mises_MPa:.3f}MPa</td><td style='color:{col}'>MS={r.MS_von_mises:.2f}</td></tr>
        <tr><td style='padding:3px 8px'>EI={r.EI_Nmm2:.2e} N·mm²</td><td colspan='3'>GJ={r.GJ_Nmm2:.2e} N·mm²</td></tr>
        </table>""")
        # Tapering
        semi=self.sch_semi_span.value() if hasattr(self,'sch_semi_span') else 750
        taper=0.67
        if hasattr(self,'sch_tip_chord') and hasattr(self,'sch_root_chord'):
            rc=self.sch_root_chord.value(); tc=self.sch_tip_chord.value()
            taper=tc/rc if rc>0 else 0.67
        tap=spar_tapering_analysis(self.spar_profile.currentText(),self.spar_mat.currentText(),
            self.spar_h.value(),self.spar_w.value(),taper,self.spar_wall.value(),self.spar_ft.value(),self.spar_wt.value(),semi,30)
        fig=self.spar_fig; fig.clear(); axes=fig.subplots(2,2); y=tap["y_mm"]
        def _s(ax,t,xl,yl):
            ax.set_facecolor(BG); ax.tick_params(colors=TEXT_S,labelsize=8)
            for s in ax.spines.values(): s.set_color(BORDER)
            ax.set_title(t,color=TEXT_P,fontsize=9,fontweight='bold',pad=5)
            ax.set_xlabel(xl,color=TEXT_S,fontsize=8); ax.set_ylabel(yl,color=TEXT_S,fontsize=8)
            ax.grid(True,alpha=0.12,color=TEXT_S)
        axes[0,0].plot(y,tap["EI_Nmm2"],color=ACCENT,lw=2); _s(axes[0,0],"EI","y [mm]","EI [N·mm²]")
        axes[0,1].plot(y,tap["GJ_Nmm2"],color=AMBER,lw=2); _s(axes[0,1],"GJ","y [mm]","GJ [N·mm²]")
        axes[1,0].plot(y,tap["Ixx_mm4"],color=ACCENT,lw=2,label="Ixx"); axes[1,0].plot(y,tap["J_mm4"],color=AMBER,lw=2,label="J")
        _s(axes[1,0],"Ixx e J","y [mm]","mm⁴"); axes[1,0].legend(fontsize=7,facecolor=PANEL,edgecolor=BORDER,labelcolor=TEXT_P)
        axes[1,1].plot(y,tap["mass_cumul_g"],color=GREEN,lw=2); _s(axes[1,1],f"Massa ({tap['total_mass_g']:.1f}g)","y [mm]","g")
        fig.tight_layout(pad=1.5); self.spar_canvas.draw()
        self._spar_result=r; self._spar_tapering=tap

    # Módulo 5 — Aeroelasticidade
    def _build_module_aeroelastic(self):
        page=QWidget(); page.setStyleSheet(f"background:{SURFACE};")
        lay=QVBoxLayout(page); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self._module_header("MÓDULOS / DINÂMICA","Aeroelasticidade — Flutter e Divergência (Cooper 2008)"))
        tabs=QTabWidget(); tabs.setDocumentMode(True)
        tabs.addTab(build_aeroelastic_tab(self),"∿  V-g / V-f (Cooper)")
        lay.addWidget(tabs,stretch=1); self.stack.addWidget(page)

    # Módulo 6 — Nervura ANSYS
    def _build_module_nervura(self):
        page=QWidget(); page.setStyleSheet(f"background:{SURFACE};")
        lay=QVBoxLayout(page); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        lay.addWidget(self._module_header("MÓDULOS / ESTRUTURAL","Otimização Topológica de Nervura (ANSYS MAPDL)"))
        self.tabs=QTabWidget(); self.tabs.setDocumentMode(True); lay.addWidget(self.tabs,stretch=1)
        # Parâmetros
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setStyleSheet(f"background:{SURFACE};")
        w=QWidget(); w.setStyleSheet(f"background:{SURFACE};"); scroll.setWidget(w); vl=QVBoxLayout(w); vl.setSpacing(10); vl.setContentsMargins(16,16,16,16)
        g=QGroupBox("Projeto"); gl=QGridLayout(g); gl.setSpacing(8)
        self.inp_nome=QLineEdit("Nervura_TopOpt"); self.inp_port=self._mk_spin(50056,1024,65535,decimals=None)
        gl.addWidget(QLabel("Nome:"),0,0); gl.addWidget(self.inp_nome,0,1)
        gl.addWidget(QLabel("Porta MAPDL:"),1,0); gl.addWidget(self.inp_port,1,1); vl.addWidget(g)
        g2=QGroupBox("Perfil e Material"); gl2=QGridLayout(g2); gl2.setSpacing(8)
        self.inp_fonte=QComboBox(); self.inp_fonte.addItems(["Banco de Dados","NACA 4 dígitos","Arquivo .dat"])
        self.inp_fonte.currentIndexChanged.connect(self._on_fonte_changed)
        self.inp_dat=QLineEdit(); self.inp_naca=QLineEdit("4412"); self.inp_db=QComboBox(); self.inp_db.addItems(list(BANCO_PERFIS.keys()))
        self.lbl_dinamico=QLabel("Perfil:"); self.inp_corda=self._mk_spin(200,1,5000,1); self.inp_esp=self._mk_spin(3,0.1,50,2)
        self.btn_browse=QPushButton("..."); self.btn_browse.clicked.connect(self._browse_dat)
        from materials import get_all_materials
        self.inp_rib_mat=QComboBox(); self.inp_rib_mat.addItems(list(get_all_materials().keys()))
        idx_b=self.inp_rib_mat.findText("Balsa C-grain")
        if idx_b>=0: self.inp_rib_mat.setCurrentIndex(idx_b)
        self.inp_rib_mat.currentIndexChanged.connect(self._on_rib_mat_changed)
        for i,(l,wgt) in enumerate([("Fonte:",self.inp_fonte),("Perfil:",self.inp_db),("Corda (mm):",self.inp_corda),("Esp. casca (mm):",self.inp_esp),("Material:",self.inp_rib_mat)]):
            gl2.addWidget(QLabel(l),i,0); gl2.addWidget(wgt,i,1)
        self.inp_dat.setVisible(False); self.btn_browse.setVisible(False); self.inp_naca.setVisible(False); vl.addWidget(g2)
        self.fig=Figure(figsize=(4,2.5),dpi=100); self.fig.patch.set_facecolor(BG)
        self.canvas=FigureCanvas(self.fig); self.canvas.setMinimumHeight(150); self.ax=self.fig.add_subplot(111); vl.addWidget(self.canvas)
        g3=QGroupBox("Malha"); gl3=QGridLayout(g3); gl3.setSpacing(8)
        self.inp_long_ini=self._mk_spin(0.25,0.05,0.5,2); self.inp_long_fim=self._mk_spin(0.30,0.1,0.6,2); self.inp_elem_size=self._mk_spin(2,0.1,20,1)
        for i,(l,wgt) in enumerate([("Engaste ini (%):",self.inp_long_ini),("Engaste fim (%):",self.inp_long_fim),("Elem. (mm):",self.inp_elem_size)]):
            gl3.addWidget(QLabel(l),i,0); gl3.addWidget(wgt,i,1)
        vl.addWidget(g3)
        g4=QGroupBox("Cargas MAPDL"); gl4=QGridLayout(g4); gl4.setSpacing(8)
        self.inp_pressao=self._mk_spin(-0.05,-100,100,4); self.inp_ex=self._mk_spin(3500,1,1e7,0)
        self.inp_prxy=self._mk_spin(0.3,0,0.5,2); self.inp_dens=self._mk_spin(1.2e-4,0,1,7)
        for i,(l,wgt) in enumerate([("Pressão (MPa):",self.inp_pressao),("E (MPa):",self.inp_ex),("ν:",self.inp_prxy),("ρ (t/mm³):",self.inp_dens)]):
            gl4.addWidget(QLabel(l),i,0); gl4.addWidget(wgt,i,1)
        vl.addWidget(g4)
        g5=QGroupBox("TopOpt"); gl5=QGridLayout(g5); gl5.setSpacing(8)
        self.inp_tensao_max=self._mk_spin(5,0.1,1000,1); self.inp_max_iter=self._mk_spin(50,5,500,None); self.inp_convergencia=self._mk_spin(0.001,0.0001,0.1,4)
        for i,(l,wgt) in enumerate([("σ máx (MPa):",self.inp_tensao_max),("Iterações:",self.inp_max_iter),("Conv.:",self.inp_convergencia)]):
            gl5.addWidget(QLabel(l),i,0); gl5.addWidget(wgt,i,1)
        vl.addWidget(g5); vl.addStretch(); self.tabs.addTab(scroll,"⚙  Parâmetros")
        # Log
        wl=QWidget(); wl.setStyleSheet(f"background:{SURFACE};"); ll=QVBoxLayout(wl); ll.setContentsMargins(12,12,12,12)
        self.log_box=QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setFont(QFont("Consolas",10))
        self.log_box.setStyleSheet(f"background:{BG};color:{TEXT_P};border:1px solid {BORDER};border-radius:5px;padding:8px;")
        ll.addWidget(self.log_box); self.tabs.addTab(wl,"📋  Log")
        # Images
        sw=QScrollArea(); sw.setWidgetResizable(True); sw.setStyleSheet(f"background:{SURFACE};")
        iw=QWidget(); iw.setStyleSheet(f"background:{SURFACE};"); sw.setWidget(iw)
        self.img_layout=QVBoxLayout(iw); self.img_layout.setSpacing(12); self.img_layout.setContentsMargins(12,12,12,12)
        self.img_layout.addStretch(); self.tabs.addTab(sw,"🖼  ANSYS")
        # Results
        rw=QWidget(); rw.setStyleSheet(f"background:{SURFACE};"); rl=QVBoxLayout(rw); rl.setContentsMargins(12,12,12,12)
        self.results_box=QTextEdit(); self.results_box.setReadOnly(True)
        self.results_box.setStyleSheet(f"background:{BG};color:{TEXT_P};border:1px solid {BORDER};border-radius:5px;padding:8px;")
        rl.addWidget(self.results_box); self.tabs.addTab(rw,"📊  Resultados")
        # Action bar
        bar=QWidget(); bar.setFixedHeight(58); bar.setStyleSheet(f"background:{PANEL};border-top:1px solid {BORDER};")
        bl=QHBoxLayout(bar); bl.setContentsMargins(16,0,16,0); bl.setSpacing(8)
        self.btn_run=QPushButton("▶  Pipeline ANSYS"); self.btn_run.setFixedHeight(34)
        self.btn_run.setStyleSheet(f"QPushButton{{background:{GREEN};color:#111;border:none;border-radius:4px;font-weight:bold;padding:0 18px;}}")
        self.btn_run.clicked.connect(self._run)
        self.btn_stop=QPushButton("⏹  Parar"); self.btn_stop.setFixedHeight(34); self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(f"QPushButton{{background:{RED};color:white;border:none;border-radius:4px;font-weight:bold;padding:0 16px;}}QPushButton:disabled{{background:{BORDER};}}")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_report=QPushButton("📄  Relatório"); self.btn_report.setFixedHeight(34); self.btn_report.setEnabled(False)
        self.btn_report.clicked.connect(self._export_report)
        bl.addWidget(self.btn_run); bl.addWidget(self.btn_stop); bl.addWidget(self.btn_report)
        self.progress_lbl=QLabel("Pronto"); self.progress_lbl.setStyleSheet(f"color:{TEXT_S};font-size:10px;background:transparent;")
        self.progress=QProgressBar(); self.progress.setFixedHeight(5); self.progress.setTextVisible(False)
        pc=QVBoxLayout(); pc.addWidget(self.progress_lbl); pc.addWidget(self.progress); bl.addLayout(pc,stretch=1)
        lay.addWidget(bar); self.stack.addWidget(page)
        for sig in [self.inp_corda.valueChanged,self.inp_esp.valueChanged,self.inp_naca.textChanged,self.inp_db.currentIndexChanged,self.inp_fonte.currentIndexChanged]:
            sig.connect(self._update_preview)
        self._update_preview()

    def _on_rib_mat_changed(self):
        from materials import get_all_materials
        mat=get_all_materials().get(self.inp_rib_mat.currentText())
        if not mat: return
        self.inp_ex.setValue(mat.E_MPa); self.inp_prxy.setValue(mat.nu); self.inp_dens.setValue(mat.density_kgm3*1e-12*1e6)

    # Módulo 7 — MDO
    def _build_module_mdo_page(self): self.stack.addWidget(build_mdo_module(self))

    # Helpers
    def _mk_spin(self,val,lo,hi,decimals=2,step=None):
        if decimals is None: w=QSpinBox(); w.setRange(int(lo),int(hi)); w.setValue(int(val))
        else: w=QDoubleSpinBox(); w.setDecimals(decimals); w.setRange(float(lo),float(hi)); w.setValue(float(val))
        w.setMinimumWidth(140); return w

    def _browse_dat(self):
        path,_=QFileDialog.getOpenFileName(self,"Selecionar .dat","","DAT (*.dat);;All (*)")
        if path: self.inp_dat.setText(path)

    def _on_fonte_changed(self,idx):
        f=self.inp_fonte.currentText()
        self.inp_dat.setVisible(f=="Arquivo .dat"); self.btn_browse.setVisible(f=="Arquivo .dat")
        self.inp_naca.setVisible(f=="NACA 4 dígitos"); self.inp_db.setVisible(f=="Banco de Dados")

    def _update_preview(self,*a):
        try:
            corda=self.inp_corda.value(); esp=self.inp_esp.value(); fonte=self.inp_fonte.currentText(); coords=None
            if fonte=="NACA 4 dígitos":
                code=self.inp_naca.text().strip()
                if len(code)==4 and code.isdigit(): coords=gerar_naca4(code)
            elif fonte=="Banco de Dados":
                n=self.inp_db.currentText()
                if n in BANCO_PERFIS: coords=BANCO_PERFIS[n]
            self.ax.clear()
            if coords is not None:
                cs=coords*corda; poly_e=Polygon(cs)
                if not poly_e.is_valid: poly_e=poly_e.buffer(0)
                poly_i=poly_e.buffer(-esp,join_style=1)
                if not poly_e.is_empty: xe,ye=poly_e.exterior.xy; self.ax.plot(xe,ye,color=ACCENT,lw=1.5)
                if not poly_i.is_empty: xi,yi=poly_i.exterior.xy; self.ax.plot(xi,yi,color=AMBER,lw=1.5,ls='--')
            self.ax.set_facecolor(BG); self.ax.axis('equal'); self.ax.grid(True,color=BORDER,ls=':',lw=0.5)
            self.ax.tick_params(colors=TEXT_S,labelsize=8)
            for sp in self.ax.spines.values(): sp.set_color(BORDER)
            self.fig.tight_layout(); self.canvas.draw()
        except: pass

    def _apply_aircraft_preset(self):
        idx=self.combo_aeronaves.currentIndex()
        if idx==0: return
        p=BANCO_AERONAVES.get(self.combo_aeronaves.currentText())
        if not p: return
        for attr,key in [('sch_semi_span','semi_span'),('sch_root_chord','root_chord'),('sch_tip_chord','tip_chord'),('sch_mass','mass'),('aero_vel','velocity'),('aero_carga','load_factor'),('inp_corda','root_chord')]:
            if hasattr(self,attr) and key in p: getattr(self,attr).setValue(p[key])
        if hasattr(self,'_update_preview'): self._update_preview()

    def _collect_params(self):
        sv=lambda w:w.value() if hasattr(w,'value') else w.text()
        f=self.inp_fonte.currentText(); fp="dat" if f=="Arquivo .dat" else "naca" if f=="NACA 4 dígitos" else "banco"
        return {"nome_projeto":self.inp_nome.text().strip() or "Nervura","mapdl_port":int(sv(self.inp_port)),"fonte_perfil":fp,
            "caminho_dat":self.inp_dat.text().strip() if fp=="dat" else "","naca_code":self.inp_naca.text().strip() if fp=="naca" else "",
            "banco_perfil":self.inp_db.currentText() if fp=="banco" else "","corda_mm":sv(self.inp_corda),"espessura_casca":sv(self.inp_esp),
            "x_long_ini_pct":sv(self.inp_long_ini),"x_long_fim_pct":sv(self.inp_long_fim),"tamanho_elemento":sv(self.inp_elem_size),
            "pressao_aerodinamica":sv(self.inp_pressao),"modulo_elasticidade":sv(self.inp_ex),"poisson":sv(self.inp_prxy),
            "densidade":sv(self.inp_dens),"tensao_max":sv(self.inp_tensao_max),"max_iter":int(sv(self.inp_max_iter)),"convergencia":sv(self.inp_convergencia)}

    COLORS={"INFO":TEXT_P,"OK":GREEN,"WARN":AMBER,"ERROR":RED,"SECTION":ACCENT}
    def _append_log(self,msg,nivel="INFO"):
        c=self.COLORS.get(nivel,TEXT_P); ts=datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f'<span style="color:{TEXT_D}">[{ts}]</span> <span style="color:{c}">{msg}</span>')

    def _run(self):
        p=self._collect_params(); self.tabs.setCurrentIndex(1); self.log_box.clear()
        self.btn_run.setEnabled(False); self.btn_stop.setEnabled(True); self.results=None
        self.worker=PipelineWorker(p)
        self.worker.log_signal.connect(self._append_log)
        self.worker.progress_signal.connect(lambda v,l:(self.progress.setValue(v),self.progress_lbl.setText(l)))
        self.worker.image_signal.connect(self._add_ansys_image)
        self.worker.result_signal.connect(self._on_result)
        self.worker.error_signal.connect(lambda m:QMessageBox.critical(self,"Erro",m[:500]))
        self.worker.finished.connect(lambda:(self.btn_run.setEnabled(True),self.btn_stop.setEnabled(False)))
        self.worker.start()

    def _stop(self):
        if self.worker: self.worker.stop()

    def _add_ansys_image(self,fp,titulo):
        card=QGroupBox(titulo); card.setStyleSheet(f"QGroupBox{{background:{PANEL};border:1px solid {BORDER};border-radius:6px;margin-top:10px;padding:12px;color:{ACCENT};}}")
        cl=QVBoxLayout(card); il=QLabel(); il.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if os.path.exists(fp):
            pix=QPixmap(fp)
            if not pix.isNull(): il.setPixmap(pix.scaledToWidth(min(820,pix.width()),Qt.TransformationMode.SmoothTransformation))
        cl.addWidget(il); cnt=self.img_layout.count(); self.img_layout.insertWidget(max(0,cnt-1),card)

    def _on_result(self,r):
        self.results=r; ok=r['stress_max']<=r['params']['tensao_max']
        self.results_box.setHtml(f"<p style='color:{GREEN if ok else RED};font-weight:bold'>{'APROVADO' if ok else 'REPROVADO'}</p><p>σ={r['stress_max']:.4f}MPa | δ={r['disp_max']:.6f}mm</p>")
        self.tabs.setCurrentIndex(3); self.btn_report.setEnabled(True)

    def _export_report(self):
        if not self.results: return
        path,_=QFileDialog.getSaveFileName(self,"Salvar","","HTML (*.html)")
        if not path: return
        from report import gerar_relatorio_html; gerar_relatorio_html(self.results,path)
        QMessageBox.information(self,"OK",f"Salvo: {path}")

    def _run_aero(self):
        semi=self.sch_semi_span.value() if hasattr(self,'sch_semi_span') else 750
        c_r=self.sch_root_chord.value() if hasattr(self,'sch_root_chord') else 300
        c_t=self.sch_tip_chord.value() if hasattr(self,'sch_tip_chord') else 200
        massa=self.sch_mass.value() if hasattr(self,'sch_mass') else 5
        n_r=int(self.wc_n_ribs.value()) if hasattr(self,'wc_n_ribs') else 12
        params={"semi_span":semi,"root_chord":c_r,"tip_chord":c_t,"velocidade":self.aero_vel.value(),"rho":self.aero_dens.value(),"fator_carga":self.aero_carga.value(),"massa":massa,"n_ribs":n_r}
        self.tabs_aero.setCurrentIndex(1); self.log_box_aero.clear(); self.btn_run_aero.setEnabled(False); self.btn_transfer.setEnabled(False)
        self.worker_aero=AeroWorker(params)
        self.worker_aero.log_signal.connect(lambda m,n:self.log_box_aero.append(f'<span style="color:{self.COLORS.get(n,TEXT_P)}">{m}</span>'))
        self.worker_aero.progress_signal.connect(lambda v,l:(self.prog_aero.setValue(v),self.prog_lbl_aero.setText(l)))
        self.worker_aero.finished_signal.connect(self._on_aero_finished)
        self.worker_aero.start()

    def _on_aero_finished(self,r):
        self.btn_run_aero.setEnabled(True); self.btn_transfer.setEnabled(True)
        self._ultima_pressao=r["pressao_extraida"]; self._aero_results=r

    def _transfer_pressure(self):
        if hasattr(self,'inp_pressao') and hasattr(self,'_ultima_pressao'):
            self.inp_pressao.setValue(self._ultima_pressao)
            QMessageBox.information(self,"OK",f"Pressão {self._ultima_pressao:.6f} MPa transferida")
            self._enter_workspace(6)

if __name__=="__main__":
    app=QApplication(sys.argv); app.setStyle("Fusion"); win=NervuraApp(); win.show(); sys.exit(app.exec())