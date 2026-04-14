"""
wingbox.py — Dimensionamento Global da Asa (Wingbox)
Análise de Caixão de Torção, Biblioteca de Longarinas com Tapering,
e integração com CLPT para laminados de fibra de carbono.

Referências:
- Megson, T.H.G. "Aircraft Structures for Engineering Students", 5th Ed.
- Bruhn, E.F. "Analysis and Design of Flight Vehicle Structures"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from materials import StructuralMaterial, clpt_ABD_matrices, clpt_effective_properties


# ─── Perfis de longarina ──────────────────────────────────────────────────────

@dataclass
class SparProfile:
    """Seção transversal de uma longarina."""
    profile_type: str = "Tubular"   # "Tubular", "I", "C", "Box", "Solid"

    # Parâmetros comuns
    height_mm: float = 20.0         # Altura total da seção [mm]
    width_mm: float = 15.0          # Largura total [mm]

    # Tubular
    wall_mm: float = 1.0            # Espessura da parede [mm]

    # Perfil I/C/Box
    flange_t_mm: float = 1.5        # Espessura da mesa [mm]
    web_t_mm: float = 1.0           # Espessura da alma [mm]
    flange_w_mm: float = 0.0        # Largura da mesa (0 = usar width_mm)

    # Material e comprimento
    material: Optional[StructuralMaterial] = None
    length_mm: float = 1500.0       # Comprimento (envergadura total) [mm]

    @property
    def area_mm2(self) -> float:
        """Área da seção transversal [mm²]."""
        t = self.profile_type
        if t == "Tubular":
            r_o = self.height_mm / 2
            r_i = max(0, r_o - self.wall_mm)
            return np.pi * (r_o**2 - r_i**2)
        elif t == "Solid":
            return self.height_mm * self.width_mm
        elif t in ("I", "C", "Box"):
            hw = self.height_mm - 2 * self.flange_t_mm
            fw = self.flange_w_mm if self.flange_w_mm > 0 else self.width_mm
            if t == "I":
                return 2 * (fw * self.flange_t_mm) + hw * self.web_t_mm
            elif t == "C":
                return 2 * (fw * self.flange_t_mm) + hw * self.web_t_mm
            else:  # Box
                return 2 * (fw * self.flange_t_mm + hw * self.web_t_mm)
        return 1.0

    @property
    def Ixx_mm4(self) -> float:
        """Momento de inércia em relação ao eixo X (flexão) [mm⁴]."""
        t = self.profile_type
        h = self.height_mm
        w = self.width_mm
        if t == "Tubular":
            r_o = h / 2
            r_i = max(0, r_o - self.wall_mm)
            return np.pi / 4 * (r_o**4 - r_i**4)
        elif t == "Solid":
            return w * h**3 / 12
        elif t in ("I", "C", "Box"):
            fw = self.flange_w_mm if self.flange_w_mm > 0 else w
            hw = h - 2 * self.flange_t_mm
            I_flanges = 2 * (fw * self.flange_t_mm**3 / 12 +
                             fw * self.flange_t_mm * ((h - self.flange_t_mm)/2)**2)
            I_web = self.web_t_mm * hw**3 / 12
            if t == "Box":
                I_web *= 2
            return I_flanges + I_web
        return 1.0

    @property
    def J_mm4(self) -> float:
        """Constante torcional J [mm⁴]."""
        t = self.profile_type
        if t == "Tubular":
            r_o = self.height_mm / 2
            r_i = max(0, r_o - self.wall_mm)
            return np.pi / 2 * (r_o**4 - r_i**4)
        elif t == "Solid":
            h, b = self.height_mm, self.width_mm
            # Fórmula de Saint-Venant para seção retangular
            k = (1/3) * (1 - 0.63 * min(h, b) / max(h, b))
            return k * min(h, b)**3 * max(h, b)
        elif t == "I":
            # Seção aberta — J ≈ Σ(l_i × t_i³)/3
            fw = self.flange_w_mm if self.flange_w_mm > 0 else self.width_mm
            hw = self.height_mm - 2 * self.flange_t_mm
            return (2 * fw * self.flange_t_mm**3 + hw * self.web_t_mm**3) / 3
        elif t == "Box":
            # Seção fechada: J = 4A²/(∮ds/t)
            h = self.height_mm - self.flange_t_mm
            b = self.width_mm - self.web_t_mm
            A_enc = h * b
            perimeter = 2 * (h / self.flange_t_mm + b / self.web_t_mm)
            return 4 * A_enc**2 / perimeter if perimeter > 0 else 1.0
        return 1.0

    @property
    def mass_g(self) -> float:
        if self.material is None:
            return 0.0
        vol_mm3 = self.area_mm2 * self.length_mm
        return self.material.density_kgm3 * vol_mm3 * 1e-6 * 1000  # g

    def taper(self, taper_ratio: float) -> "SparProfile":
        """
        Retorna uma cópia do perfil com dimensões escaladas pelo fator de afinamento.
        taper_ratio: relação ponta/raiz (0<λ≤1)
        """
        import copy
        tip = copy.deepcopy(self)
        tip.height_mm *= taper_ratio
        tip.width_mm *= taper_ratio
        tip.wall_mm *= taper_ratio
        tip.flange_t_mm *= taper_ratio
        tip.web_t_mm *= taper_ratio
        return tip


# ─── Caixão de torção ──────────────────────────────────────────────────────────

@dataclass
class TorsionBoxSection:
    """Seção do caixão de torção em uma estação spanwise."""
    y_mm: float = 0.0              # Posição spanwise [mm]
    chord_mm: float = 300.0        # Corda local [mm]
    spar_front_pct: float = 0.15   # Posição longarina frontal (% corda)
    spar_rear_pct: float = 0.65    # Posição longarina traseira (% corda)
    height_pct: float = 0.12       # Altura do perfil (% corda — média)

    # Espessuras das paredes do caixão
    skin_top_mm: float = 1.0       # Pele superior [mm]
    skin_bot_mm: float = 1.0       # Pele inferior [mm]
    spar_front_mm: float = 1.5     # Alma longarina frontal [mm]
    spar_rear_mm: float = 1.2      # Alma longarina traseira [mm]

    # Materiais
    skin_material: Optional[StructuralMaterial] = None
    spar_material: Optional[StructuralMaterial] = None

    @property
    def box_width_mm(self) -> float:
        return (self.spar_rear_pct - self.spar_front_pct) * self.chord_mm

    @property
    def box_height_mm(self) -> float:
        return self.height_pct * self.chord_mm

    @property
    def enclosed_area_mm2(self) -> float:
        """Área fechada do caixão A_enc [mm²]."""
        return self.box_width_mm * self.box_height_mm

    @property
    def shear_flow_J_mm4(self) -> float:
        """
        Constante torcional J do caixão fechado (Bredt-Batho) [mm⁴].
        J = 4·A²·t_eff/perímetro
        """
        A = self.enclosed_area_mm2
        b = self.box_width_mm
        h = self.box_height_mm
        # Espessura efetiva média
        perimeter = (b / ((self.skin_top_mm + self.skin_bot_mm) / 2) +
                     h / ((self.spar_front_mm + self.spar_rear_mm) / 2))
        return 4 * A**2 / perimeter if perimeter > 0 else 0.0

    @property
    def Ixx_mm4(self) -> float:
        """Momento de inércia efetivo do caixão (peles + almas) [mm⁴]."""
        h = self.box_height_mm
        b = self.box_width_mm
        # Contribuição das peles (mesas)
        I_skins = 2 * (b * ((self.skin_top_mm + self.skin_bot_mm) / 2) * (h / 2)**2)
        # Contribuição das almas (vigas laterais)
        I_webs = 2 * (((self.spar_front_mm + self.spar_rear_mm) / 2) * h**3 / 12)
        return I_skins + I_webs

    @property
    def shear_center_pct(self) -> float:
        """Posição do centro de cisalhamento (% corda), aproximado."""
        # Para caixão simétrico: centro de cisalhamento ≈ centroide do caixão
        return (self.spar_front_pct + self.spar_rear_pct) / 2


@dataclass
class WingboxResult:
    """Resultado da análise do wingbox."""
    y_mm: np.ndarray = field(default_factory=lambda: np.array([]))
    J_mm4: np.ndarray = field(default_factory=lambda: np.array([]))
    Ixx_mm4: np.ndarray = field(default_factory=lambda: np.array([]))
    GJ_Nmm2: np.ndarray = field(default_factory=lambda: np.array([]))
    EI_Nmm2: np.ndarray = field(default_factory=lambda: np.array([]))
    shear_center_pct: np.ndarray = field(default_factory=lambda: np.array([]))

    # Valores na raiz
    GJ_root: float = 0.0
    EI_root: float = 0.0
    J_root: float = 0.0
    Ixx_root: float = 0.0

    # Frequências estimadas (Rayleigh)
    freq_bending_Hz: float = 0.0
    freq_torsion_Hz: float = 0.0

    # Spar mass
    spar_mass_g: float = 0.0
    total_wingbox_mass_g: float = 0.0


def analyze_wingbox(
    semi_span_mm: float,
    root_chord_mm: float,
    tip_chord_mm: float,
    spar_root: SparProfile,
    spar_tip: Optional[SparProfile] = None,
    n_stations: int = 50,
    wing_mass_kg: float = 2.0,
) -> WingboxResult:
    """
    Analisa o wingbox ao longo da envergadura.

    Assume longarina simples com tapering linear entre raiz e ponta.
    Calcula GJ, EI, e frequências naturais via método de Rayleigh.
    """
    y = np.linspace(0, semi_span_mm, n_stations)
    eta = y / semi_span_mm  # 0 → 1

    # Tapering linear das dimensões
    if spar_tip is None:
        # Taper proporcional à corda
        taper = tip_chord_mm / root_chord_mm
        spar_tip = spar_root.taper(taper)

    # Interpolação linear de J e Ixx ao longo da envergadura
    J_root = spar_root.J_mm4
    J_tip = spar_tip.J_mm4
    I_root = spar_root.Ixx_mm4
    I_tip = spar_tip.Ixx_mm4

    J_arr = J_root + eta * (J_tip - J_root)
    I_arr = I_root + eta * (I_tip - I_root)

    # Módulos efetivos
    if spar_root.material is not None:
        mat = spar_root.material
        E = mat.E_MPa
        G = mat.G_effective
    else:
        E = 135000.0  # CFRP default
        G = 4200.0

    GJ_arr = G * J_arr    # Rigidez torcional [N·mm²]
    EI_arr = E * I_arr    # Rigidez à flexão [N·mm²]

    # Centro de cisalhamento (constante para longarina única)
    shear_ctr = np.full(n_stations, 0.25)  # ~25% corda (simplificado)

    # Massa da longarina
    area_root = spar_root.area_mm2
    area_tip = spar_tip.area_mm2
    rho = spar_root.material.density_kgm3 if spar_root.material else 1600.0
    area_arr = area_root + eta * (area_tip - area_root)
    spar_mass_g = float(np.trapezoid(area_arr * rho * 1e-6, y)) * 1000  # g

    # Frequência natural de flexão (método de Rayleigh — asa em balanço)
    # f_b = (λ_1²/(2π·L²)) × sqrt(EI_root / (m/L))
    L = semi_span_mm
    m_per_L = wing_mass_kg * 1000 / L  # g/mm → simplificado
    if EI_arr[0] > 0 and m_per_L > 0:
        freq_b = (3.516 / (2 * np.pi * L**2)) * np.sqrt(EI_arr[0] / (m_per_L * 1e-6))
    else:
        freq_b = 0.0

    # Frequência natural de torção (barra engastada)
    # f_t = (π/(2L)) × sqrt(GJ_root / (m_theta/L))
    # m_theta: massa por unidade de comprimento × raio de giração²
    r_gyr = (root_chord_mm + tip_chord_mm) / 4 / np.sqrt(6)  # raio de giração
    I_mass_root = m_per_L * 1e-6 * r_gyr**2  # [kg·mm²/mm]
    if GJ_arr[0] > 0 and I_mass_root > 0:
        freq_t = (np.pi / (2 * L)) * np.sqrt(GJ_arr[0] / I_mass_root) / (2 * np.pi)
    else:
        freq_t = 0.0

    result = WingboxResult(
        y_mm=y,
        J_mm4=J_arr,
        Ixx_mm4=I_arr,
        GJ_Nmm2=GJ_arr,
        EI_Nmm2=EI_arr,
        shear_center_pct=shear_ctr,
        GJ_root=float(GJ_arr[0]),
        EI_root=float(EI_arr[0]),
        J_root=float(J_root),
        Ixx_root=float(I_root),
        freq_bending_Hz=freq_b,
        freq_torsion_Hz=freq_t,
        spar_mass_g=spar_mass_g,
        total_wingbox_mass_g=spar_mass_g,
    )
    return result


# ─── Análise de falha do wingbox ──────────────────────────────────────────────

def wingbox_stress_check(
    result: WingboxResult,
    spar_root: SparProfile,
    moment_root_Nmm: float,
    shear_root_N: float,
    torque_root_Nmm: float,
) -> dict:
    """
    Verifica tensões na raiz do wingbox.
    Retorna dict com tensões calculadas e margem de segurança.
    """
    mat = spar_root.material
    if mat is None:
        return {}

    h = spar_root.height_mm
    I = spar_root.Ixx_mm4
    J = spar_root.J_mm4

    # Tensão de flexão (fibra mais afastada)
    sigma_b = moment_root_Nmm * (h / 2) / I if I > 0 else 0.0

    # Tensão de cisalhamento por cortante (VQ/Ib)
    # Simplificado: tau_shear = 1.5 * V/A para seção tubular
    A = spar_root.area_mm2
    tau_shear = 1.5 * shear_root_N / A if A > 0 else 0.0

    # Tensão de cisalhamento por torção (Bredth-Batho para tubo)
    # tau_t = T*r/J
    r_out = h / 2
    tau_torsion = torque_root_Nmm * r_out / J if J > 0 else 0.0

    # Tensão combinada (Von Mises para barra)
    tau_total = np.sqrt(tau_shear**2 + tau_torsion**2)
    sigma_vm = np.sqrt(sigma_b**2 + 3 * tau_total**2)

    sigma_adm = mat.sigma_t_MPa
    tau_adm = mat.tau_MPa
    MS_bending = sigma_adm / sigma_b - 1 if sigma_b > 0 else float("inf")
    MS_shear = tau_adm / tau_total - 1 if tau_total > 0 else float("inf")
    MS_vm = sigma_adm / sigma_vm - 1 if sigma_vm > 0 else float("inf")

    return {
        "sigma_bending_MPa": sigma_b,
        "tau_shear_MPa": tau_shear,
        "tau_torsion_MPa": tau_torsion,
        "tau_total_MPa": tau_total,
        "sigma_vm_MPa": sigma_vm,
        "MS_bending": MS_bending,
        "MS_shear": MS_shear,
        "MS_vm": MS_vm,
        "approved": min(MS_bending, MS_shear, MS_vm) >= 0,
    }


# ─── Biblioteca de seções de longarina pré-definidas ─────────────────────────

def spar_library() -> Dict[str, dict]:
    """Retorna perfis pré-configurados para longarinas de aeromodelo."""
    return {
        "Tubo CF ø12×1mm":    {"profile_type": "Tubular",  "height_mm": 12, "width_mm": 12, "wall_mm": 1.0,  "flange_t_mm": 0, "web_t_mm": 0},
        "Tubo CF ø16×1.5mm":  {"profile_type": "Tubular",  "height_mm": 16, "width_mm": 16, "wall_mm": 1.5,  "flange_t_mm": 0, "web_t_mm": 0},
        "Tubo CF ø20×2mm":    {"profile_type": "Tubular",  "height_mm": 20, "width_mm": 20, "wall_mm": 2.0,  "flange_t_mm": 0, "web_t_mm": 0},
        "Perfil I 20×15×1mm": {"profile_type": "I",         "height_mm": 20, "width_mm": 15, "wall_mm": 0,    "flange_t_mm": 1.5, "web_t_mm": 1.0},
        "Perfil I 30×20×2mm": {"profile_type": "I",         "height_mm": 30, "width_mm": 20, "wall_mm": 0,    "flange_t_mm": 2.0, "web_t_mm": 1.5},
        "Perfil C 20×10×1mm": {"profile_type": "C",         "height_mm": 20, "width_mm": 10, "wall_mm": 0,    "flange_t_mm": 1.5, "web_t_mm": 1.0},
        "Caixão 20×15×1mm":   {"profile_type": "Box",       "height_mm": 20, "width_mm": 15, "wall_mm": 0,    "flange_t_mm": 1.0, "web_t_mm": 1.0},
        "Sólido 15×5mm":      {"profile_type": "Solid",     "height_mm": 15, "width_mm": 5,  "wall_mm": 0,    "flange_t_mm": 0, "web_t_mm": 0},
    }