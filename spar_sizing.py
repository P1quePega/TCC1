"""
spar_sizing.py — Dimensionamento Preliminar de Longarinas
Cálculos baseados em:
  - Megson, T.H.G. "Aircraft Structures for Engineering Students", 6th Ed.
    Cap. 20 (Wing Spars), Cap. 16 (Bending of Open/Closed Sections)
  - Cooper, J.E. "Introduction to Aircraft Aeroelasticity and Loads", Wiley, 2008.
    Cap. 6 (Static Aeroelasticity), Cap. 11 (Flutter)

Módulo para cálculo preliminar de seções transversais de longarinas,
incluindo: área, Ixx, J, tensões, margens de segurança e massa.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from materials import StructuralMaterial, get_all_materials


# ─── Seções transversais de longarina ────────────────────────────────────────

PROFILE_TYPES = ["Tubular", "I-beam", "C-channel", "Box", "Sólido Retangular"]


@dataclass
class SparSectionResult:
    """Resultado do dimensionamento de uma seção de longarina."""
    profile_type: str = "Tubular"
    material_name: str = ""

    # Dimensões
    height_mm: float = 0.0
    width_mm: float = 0.0
    wall_mm: float = 0.0
    flange_t_mm: float = 0.0
    web_t_mm: float = 0.0

    # Propriedades da seção
    area_mm2: float = 0.0
    Ixx_mm4: float = 0.0
    Iyy_mm4: float = 0.0
    J_mm4: float = 0.0
    Zx_mm3: float = 0.0          # Módulo de seção (Ixx / y_max)
    rg_mm: float = 0.0           # Raio de giração

    # Rigidezes
    EI_Nmm2: float = 0.0
    GJ_Nmm2: float = 0.0

    # Cargas aplicadas
    M_applied_Nmm: float = 0.0
    V_applied_N: float = 0.0
    T_applied_Nmm: float = 0.0

    # Tensões calculadas (Megson Cap. 16)
    sigma_bending_MPa: float = 0.0     # σ = M·y/I (Megson Eq. 16.18)
    tau_shear_MPa: float = 0.0         # τ = VQ/Ib (Megson Eq. 16.23)
    tau_torsion_MPa: float = 0.0       # τ = T·r/J (tubular) ou Bredt (fechada)
    sigma_von_mises_MPa: float = 0.0   # σ_vm = √(σ² + 3τ²)

    # Tensões admissíveis
    sigma_adm_MPa: float = 0.0
    tau_adm_MPa: float = 0.0

    # Margens de segurança (Megson Cap. 13)
    MS_bending: float = 0.0
    MS_shear: float = 0.0
    MS_torsion: float = 0.0
    MS_von_mises: float = 0.0

    # Massa
    mass_per_length_g_mm: float = 0.0  # g/mm
    mass_total_g: float = 0.0
    length_mm: float = 0.0

    # Status
    approved: bool = True
    critical_mode: str = ""
    notes: str = ""


def compute_section_properties(
    profile_type: str,
    height_mm: float,
    width_mm: float,
    wall_mm: float = 1.0,
    flange_t_mm: float = 1.5,
    web_t_mm: float = 1.0,
) -> Dict[str, float]:
    """
    Calcula propriedades geométricas da seção transversal.

    Referência: Megson Cap. 16.2-16.4 para seções abertas e fechadas.
    Para seções tubulares: Megson Eq. 17.11 (Bredt-Batho para J).
    Para seções I: Megson Eq. 18.5 (seção aberta, J = Σ(bt³/3)).
    """
    props = {"area": 0, "Ixx": 0, "Iyy": 0, "J": 0, "Zx": 0, "rg": 0}

    h = height_mm
    w = width_mm

    if profile_type == "Tubular":
        r_o = h / 2
        r_i = max(0, r_o - wall_mm)
        A = np.pi * (r_o**2 - r_i**2)
        Ixx = np.pi / 4 * (r_o**4 - r_i**4)
        J = np.pi / 2 * (r_o**4 - r_i**4)  # Megson Eq. 17.1
        Iyy = Ixx
        Zx = Ixx / r_o if r_o > 0 else 0

    elif profile_type == "Sólido Retangular":
        A = h * w
        Ixx = w * h**3 / 12
        Iyy = h * w**3 / 12
        # Saint-Venant para seção retangular (Megson Eq. 17.5)
        a, b = max(h, w), min(h, w)
        k = (1/3) * (1 - 0.63 * b / a * (1 - b**4 / (12 * a**4)))
        J = k * a * b**3
        Zx = Ixx / (h / 2) if h > 0 else 0

    elif profile_type == "I-beam":
        hw = h - 2 * flange_t_mm
        fw = w
        A = 2 * fw * flange_t_mm + hw * web_t_mm
        # Ixx: contribuição das mesas (Steiner) + alma
        I_flanges = 2 * (fw * flange_t_mm**3 / 12 +
                         fw * flange_t_mm * ((h - flange_t_mm) / 2)**2)
        I_web = web_t_mm * hw**3 / 12
        Ixx = I_flanges + I_web
        Iyy = 2 * flange_t_mm * fw**3 / 12 + hw * web_t_mm**3 / 12
        # J seção aberta: Megson Eq. 18.5 — J = Σ(b_i · t_i³) / 3
        J = (2 * fw * flange_t_mm**3 + hw * web_t_mm**3) / 3
        Zx = Ixx / (h / 2) if h > 0 else 0

    elif profile_type == "C-channel":
        hw = h - 2 * flange_t_mm
        fw = w
        A = 2 * fw * flange_t_mm + hw * web_t_mm
        I_flanges = 2 * (fw * flange_t_mm**3 / 12 +
                         fw * flange_t_mm * ((h - flange_t_mm) / 2)**2)
        I_web = web_t_mm * hw**3 / 12
        Ixx = I_flanges + I_web
        Iyy = 2 * flange_t_mm * fw**3 / 12 + hw * web_t_mm**3 / 12
        J = (2 * fw * flange_t_mm**3 + hw * web_t_mm**3) / 3
        Zx = Ixx / (h / 2) if h > 0 else 0

    elif profile_type == "Box":
        hw = h - 2 * flange_t_mm
        bw = w - 2 * web_t_mm
        A = 2 * (w * flange_t_mm + hw * web_t_mm)
        I_flanges = 2 * (w * flange_t_mm**3 / 12 +
                         w * flange_t_mm * ((h - flange_t_mm) / 2)**2)
        I_webs = 2 * web_t_mm * hw**3 / 12
        Ixx = I_flanges + I_webs
        Iyy = 2 * (flange_t_mm * w**3 / 12 + hw * web_t_mm**3 / 12 +
                    hw * web_t_mm * ((w - web_t_mm) / 2)**2)
        # J seção fechada: Bredt-Batho (Megson Eq. 17.11)
        A_enc = hw * bw
        perim_over_t = 2 * (bw / flange_t_mm + hw / web_t_mm)
        J = 4 * A_enc**2 / perim_over_t if perim_over_t > 0 else 0
        Zx = Ixx / (h / 2) if h > 0 else 0

    rg = np.sqrt(Ixx / A) if A > 0 else 0

    return {
        "area": A, "Ixx": Ixx, "Iyy": Iyy, "J": J, "Zx": Zx, "rg": rg
    }


def size_spar(
    profile_type: str,
    material_name: str,
    height_mm: float,
    width_mm: float,
    wall_mm: float = 1.0,
    flange_t_mm: float = 1.5,
    web_t_mm: float = 1.0,
    length_mm: float = 1500.0,
    M_Nmm: float = 0.0,
    V_N: float = 0.0,
    T_Nmm: float = 0.0,
    safety_factor: float = 1.5,
) -> SparSectionResult:
    """
    Dimensiona uma longarina e verifica tensões.

    Referências:
    - Tensão de flexão: σ = M·y_max / Ixx  (Megson Eq. 16.18)
    - Tensão cisalhante: τ = V·Q / (I·b)   (Megson Eq. 16.23)
      Para seção tubular: τ ≈ 2V / A
      Para perfil I: τ ≈ V / (h_w · t_w)
    - Tensão de torção:
      Seção fechada: τ = T / (2·A_enc·t)    (Bredt-Batho, Megson Eq. 17.11)
      Seção aberta: τ = T·t_max / J          (Megson Eq. 18.3)
    - Von Mises: σ_vm = √(σ² + 3·τ²)        (Megson Eq. 14.31)
    - Margem de segurança: MS = σ_adm / (SF·σ_calc) - 1  (Megson Cap. 13)
    """
    result = SparSectionResult()
    result.profile_type = profile_type
    result.material_name = material_name
    result.height_mm = height_mm
    result.width_mm = width_mm
    result.wall_mm = wall_mm
    result.flange_t_mm = flange_t_mm
    result.web_t_mm = web_t_mm
    result.length_mm = length_mm
    result.M_applied_Nmm = M_Nmm
    result.V_applied_N = V_N
    result.T_applied_Nmm = T_Nmm

    # Material
    mats = get_all_materials()
    mat = mats.get(material_name)
    if mat is None:
        result.notes = f"Material '{material_name}' não encontrado."
        result.approved = False
        return result

    E = mat.E_MPa
    G = mat.G_effective
    result.sigma_adm_MPa = mat.sigma_t_MPa
    result.tau_adm_MPa = mat.tau_MPa

    # Propriedades da seção
    props = compute_section_properties(
        profile_type, height_mm, width_mm, wall_mm, flange_t_mm, web_t_mm
    )
    result.area_mm2 = props["area"]
    result.Ixx_mm4 = props["Ixx"]
    result.Iyy_mm4 = props["Iyy"]
    result.J_mm4 = props["J"]
    result.Zx_mm3 = props["Zx"]
    result.rg_mm = props["rg"]

    result.EI_Nmm2 = E * props["Ixx"]
    result.GJ_Nmm2 = G * props["J"]

    # Massa
    rho = mat.density_kgm3
    vol_mm3 = props["area"] * length_mm
    result.mass_total_g = rho * vol_mm3 * 1e-6  # kg → g (×1000, ÷1e9 mm³→m³)
    result.mass_per_length_g_mm = result.mass_total_g / length_mm if length_mm > 0 else 0

    # ── Tensão de flexão (Megson Eq. 16.18) ──
    y_max = height_mm / 2
    I = props["Ixx"]
    result.sigma_bending_MPa = abs(M_Nmm) * y_max / I if I > 0 else 0

    # ── Tensão cisalhante por cortante (Megson Eq. 16.23) ──
    if profile_type == "Tubular":
        # Para tubo de paredes finas: τ_max ≈ 2V / A
        result.tau_shear_MPa = 2 * abs(V_N) / props["area"] if props["area"] > 0 else 0
    elif profile_type in ("I-beam", "C-channel"):
        # τ_max na alma: V / (h_w · t_w)
        hw = height_mm - 2 * flange_t_mm
        result.tau_shear_MPa = abs(V_N) / (hw * web_t_mm) if hw * web_t_mm > 0 else 0
    elif profile_type == "Box":
        hw = height_mm - 2 * flange_t_mm
        result.tau_shear_MPa = abs(V_N) / (2 * hw * web_t_mm) if hw * web_t_mm > 0 else 0
    else:
        # Sólido: τ = 3V / (2A) para seção retangular
        result.tau_shear_MPa = 1.5 * abs(V_N) / props["area"] if props["area"] > 0 else 0

    # ── Tensão de torção ──
    if profile_type in ("Tubular", "Box"):
        # Seção fechada: Bredt-Batho τ = T / (2·A_enc·t_min)
        if profile_type == "Tubular":
            r_o = height_mm / 2
            r_i = max(0, r_o - wall_mm)
            A_enc = np.pi * ((r_o + r_i) / 2)**2
            t_min = wall_mm
        else:
            hw = height_mm - 2 * flange_t_mm
            bw = width_mm - 2 * web_t_mm
            A_enc = hw * bw
            t_min = min(flange_t_mm, web_t_mm)
        result.tau_torsion_MPa = abs(T_Nmm) / (2 * A_enc * t_min) if A_enc * t_min > 0 else 0
    else:
        # Seção aberta: τ = T·t_max / J  (Megson Eq. 18.3)
        t_max = max(flange_t_mm, web_t_mm, wall_mm)
        result.tau_torsion_MPa = abs(T_Nmm) * t_max / props["J"] if props["J"] > 0 else 0

    # ── Von Mises (Megson Eq. 14.31) ──
    tau_total = np.sqrt(result.tau_shear_MPa**2 + result.tau_torsion_MPa**2)
    result.sigma_von_mises_MPa = np.sqrt(
        result.sigma_bending_MPa**2 + 3 * tau_total**2
    )

    # ── Margens de segurança (Megson Cap. 13) ──
    sf = safety_factor
    sig_adm = mat.sigma_t_MPa
    tau_adm = mat.tau_MPa

    result.MS_bending = sig_adm / (sf * result.sigma_bending_MPa) - 1 if result.sigma_bending_MPa > 1e-10 else 999.0
    result.MS_shear = tau_adm / (sf * result.tau_shear_MPa) - 1 if result.tau_shear_MPa > 1e-10 else 999.0
    result.MS_torsion = tau_adm / (sf * result.tau_torsion_MPa) - 1 if result.tau_torsion_MPa > 1e-10 else 999.0
    result.MS_von_mises = sig_adm / (sf * result.sigma_von_mises_MPa) - 1 if result.sigma_von_mises_MPa > 1e-10 else 999.0

    # Status
    min_ms = min(result.MS_bending, result.MS_shear, result.MS_torsion, result.MS_von_mises)
    result.approved = min_ms >= 0

    if result.MS_von_mises < 0:
        result.critical_mode = "Von Mises"
    elif result.MS_bending < 0:
        result.critical_mode = "Flexão"
    elif result.MS_shear < 0:
        result.critical_mode = "Cisalhamento"
    elif result.MS_torsion < 0:
        result.critical_mode = "Torção"
    else:
        result.critical_mode = "N/A (aprovado)"

    ms_strs = [f"MS_flexão={result.MS_bending:.2f}",
               f"MS_cisalh={result.MS_shear:.2f}",
               f"MS_torção={result.MS_torsion:.2f}",
               f"MS_VM={result.MS_von_mises:.2f}"]
    result.notes = " | ".join(ms_strs)

    return result


def spar_tapering_analysis(
    profile_type: str,
    material_name: str,
    root_height_mm: float,
    root_width_mm: float,
    taper_ratio: float,
    wall_mm: float = 1.0,
    flange_t_mm: float = 1.5,
    web_t_mm: float = 1.0,
    semi_span_mm: float = 750.0,
    n_stations: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Analisa a longarina ao longo da envergadura com tapering linear.

    Retorna distribuições spanwise de EI, GJ, Ixx, J, área, massa cumulativa.
    Referência: Megson Cap. 20 — distribuição spanwise de propriedades.
    """
    mats = get_all_materials()
    mat = mats.get(material_name)
    E = mat.E_MPa if mat else 135000
    G = mat.G_effective if mat else 4200
    rho = mat.density_kgm3 if mat else 1600

    y = np.linspace(0, semi_span_mm, n_stations)
    eta = y / semi_span_mm

    # Tapering linear
    h_arr = root_height_mm * (1 - eta * (1 - taper_ratio))
    w_arr = root_width_mm * (1 - eta * (1 - taper_ratio))

    Ixx_arr = np.zeros(n_stations)
    J_arr = np.zeros(n_stations)
    A_arr = np.zeros(n_stations)

    for i in range(n_stations):
        p = compute_section_properties(
            profile_type, h_arr[i], w_arr[i],
            wall_mm * (1 - eta[i] * (1 - taper_ratio)),
            flange_t_mm * (1 - eta[i] * (1 - taper_ratio)),
            web_t_mm * (1 - eta[i] * (1 - taper_ratio)),
        )
        Ixx_arr[i] = p["Ixx"]
        J_arr[i] = p["J"]
        A_arr[i] = p["area"]

    EI_arr = E * Ixx_arr
    GJ_arr = G * J_arr

    # Massa acumulada
    mass_cumul = np.zeros(n_stations)
    for i in range(1, n_stations):
        dy = y[i] - y[i - 1]
        A_mid = (A_arr[i] + A_arr[i - 1]) / 2
        mass_cumul[i] = mass_cumul[i - 1] + A_mid * dy * rho * 1e-6  # g

    return {
        "y_mm": y, "eta": eta,
        "height_mm": h_arr, "width_mm": w_arr,
        "Ixx_mm4": Ixx_arr, "J_mm4": J_arr, "area_mm2": A_arr,
        "EI_Nmm2": EI_arr, "GJ_Nmm2": GJ_arr,
        "mass_cumul_g": mass_cumul,
        "EI_root": float(EI_arr[0]), "GJ_root": float(GJ_arr[0]),
        "total_mass_g": float(mass_cumul[-1]),
    }


def compare_profiles(
    profiles: List[Dict],
    material_name: str,
    length_mm: float,
    M_Nmm: float = 0.0,
    V_N: float = 0.0,
    T_Nmm: float = 0.0,
) -> List[SparSectionResult]:
    """
    Compara múltiplos perfis de longarina sob as mesmas cargas.
    profiles: lista de dicts com keys: profile_type, height_mm, width_mm, wall_mm, etc.
    """
    results = []
    for p in profiles:
        r = size_spar(
            profile_type=p.get("profile_type", "Tubular"),
            material_name=material_name,
            height_mm=p.get("height_mm", 12),
            width_mm=p.get("width_mm", 12),
            wall_mm=p.get("wall_mm", 1),
            flange_t_mm=p.get("flange_t_mm", 1.5),
            web_t_mm=p.get("web_t_mm", 1),
            length_mm=length_mm,
            M_Nmm=M_Nmm, V_N=V_N, T_Nmm=T_Nmm,
        )
        results.append(r)
    results.sort(key=lambda x: x.mass_total_g)
    return results


# ─── Biblioteca pré-definida (para dropdown na GUI) ──────────────────────────

SPAR_PRESETS = {
    "Tubo CF ø10×0.8mm":  {"profile_type": "Tubular", "height_mm": 10, "width_mm": 10, "wall_mm": 0.8},
    "Tubo CF ø12×1.0mm":  {"profile_type": "Tubular", "height_mm": 12, "width_mm": 12, "wall_mm": 1.0},
    "Tubo CF ø16×1.5mm":  {"profile_type": "Tubular", "height_mm": 16, "width_mm": 16, "wall_mm": 1.5},
    "Tubo CF ø20×2.0mm":  {"profile_type": "Tubular", "height_mm": 20, "width_mm": 20, "wall_mm": 2.0},
    "Perfil I 20×15×1":   {"profile_type": "I-beam", "height_mm": 20, "width_mm": 15, "flange_t_mm": 1.5, "web_t_mm": 1.0},
    "Perfil I 30×20×2":   {"profile_type": "I-beam", "height_mm": 30, "width_mm": 20, "flange_t_mm": 2.0, "web_t_mm": 1.5},
    "Perfil C 20×10×1":   {"profile_type": "C-channel", "height_mm": 20, "width_mm": 10, "flange_t_mm": 1.5, "web_t_mm": 1.0},
    "Caixão 20×15×1":     {"profile_type": "Box", "height_mm": 20, "width_mm": 15, "flange_t_mm": 1.0, "web_t_mm": 1.0},
    "Sólido 15×5 (balsa)":{"profile_type": "Sólido Retangular", "height_mm": 15, "width_mm": 5, "wall_mm": 0},
    "Sólido 20×8 (balsa)":{"profile_type": "Sólido Retangular", "height_mm": 20, "width_mm": 8, "wall_mm": 0},
}