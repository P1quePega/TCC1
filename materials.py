"""
materials.py — Biblioteca de Materiais Estruturais
Inclui: Balsa, Divinycell (H60/H80/H100), Painel Sanduíche Carbono-Divinycell,
Fibra de Carbono (Unidirecional e Tecido), GFRP, Alumínio, Fita de Carbono.

Propriedades editáveis via dicionário — compatível com a GUI do T.O.C.A.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import numpy as np
import json
import os


# ─── Dataclass de material ───────────────────────────────────────────────────

@dataclass
class StructuralMaterial:
    """Material estrutural com propriedades mecânicas completas."""
    name: str
    category: str                       # "foam", "wood", "composite", "metal"
    description: str = ""

    # Propriedades elásticas
    E_MPa: float = 1000.0               # Módulo de Young (isotrópico ou Ex) [MPa]
    E2_MPa: float = 0.0                 # Módulo transversal (0 = isotrópico) [MPa]
    G_MPa: float = 0.0                  # Módulo de cisalhamento [MPa]
    nu: float = 0.3                     # Coeficiente de Poisson

    # Resistências
    sigma_t_MPa: float = 10.0           # Resistência à tração [MPa]
    sigma_c_MPa: float = 10.0           # Resistência à compressão [MPa]
    tau_MPa: float = 5.0                # Resistência ao cisalhamento [MPa]

    # Propriedades físicas
    density_kgm3: float = 200.0         # Densidade [kg/m³]
    cost_per_kg: float = 10.0           # Custo aproximado [R$/kg]

    # Flags de uso
    is_core: bool = False               # Pode ser usado como núcleo de sanduíche
    is_skin: bool = True                # Pode ser usado como pele
    is_anisotropic: bool = False        # Requer análise laminada (CLPT)

    # Espessura típica de ply para compósitos [mm]
    ply_thickness_mm: float = 0.0

    # Para compósitos: propriedades de ply unidirecional
    E1_MPa: float = 0.0   # Módulo longitudinal [MPa]
    E2_ply_MPa: float = 0.0  # Módulo transversal [MPa]
    G12_MPa: float = 0.0  # Módulo de cisalhamento [MPa]
    nu12: float = 0.0     # Poisson longitudinal

    @property
    def G_effective(self) -> float:
        """Módulo de cisalhamento efetivo."""
        if self.G_MPa > 0:
            return self.G_MPa
        return self.E_MPa / (2 * (1 + self.nu))

    @property
    def E2_effective(self) -> float:
        return self.E2_MPa if self.E2_MPa > 0 else self.E_MPa

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StructuralMaterial":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─── Banco de materiais padrão ───────────────────────────────────────────────

DEFAULT_MATERIALS: Dict[str, StructuralMaterial] = {

    # ── Madeira ──────────────────────────────────────────────────────────────

    "Balsa C-grain": StructuralMaterial(
        name="Balsa C-grain",
        category="wood",
        description="Madeira de balsa de grão transversal — nervuras de aeromodelo",
        E_MPa=2300.0,
        E2_MPa=120.0,
        G_MPa=80.0,
        nu=0.23,
        sigma_t_MPa=7.5,
        sigma_c_MPa=5.0,
        tau_MPa=1.5,
        density_kgm3=160.0,
        cost_per_kg=120.0,
        is_core=False,
        is_skin=True,
        is_anisotropic=True,
        ply_thickness_mm=0.0,
    ),

    "Balsa A-grain": StructuralMaterial(
        name="Balsa A-grain",
        category="wood",
        description="Balsa de grão longitudinal — longarinas e reforços",
        E_MPa=3500.0,
        E2_MPa=80.0,
        G_MPa=100.0,
        nu=0.20,
        sigma_t_MPa=14.0,
        sigma_c_MPa=10.0,
        tau_MPa=2.0,
        density_kgm3=120.0,
        cost_per_kg=130.0,
        is_core=False,
        is_skin=True,
        is_anisotropic=True,
    ),

    # ── Espumas (Divinycell) ──────────────────────────────────────────────────

    "Divinycell H60": StructuralMaterial(
        name="Divinycell H60",
        category="foam",
        description="Espuma PVC H60 — núcleo de painel sanduíche leve",
        E_MPa=60.0,
        G_MPa=22.0,
        nu=0.32,
        sigma_t_MPa=1.7,
        sigma_c_MPa=0.9,
        tau_MPa=0.8,
        density_kgm3=60.0,
        cost_per_kg=180.0,
        is_core=True,
        is_skin=False,
    ),

    "Divinycell H80": StructuralMaterial(
        name="Divinycell H80",
        category="foam",
        description="Espuma PVC H80 — núcleo de painel sanduíche estrutural",
        E_MPa=85.0,
        G_MPa=31.0,
        nu=0.32,
        sigma_t_MPa=2.6,
        sigma_c_MPa=1.4,
        tau_MPa=1.15,
        density_kgm3=80.0,
        cost_per_kg=200.0,
        is_core=True,
        is_skin=False,
    ),

    "Divinycell H100": StructuralMaterial(
        name="Divinycell H100",
        category="foam",
        description="Espuma PVC H100 — alta resistência estrutural",
        E_MPa=130.0,
        G_MPa=40.0,
        nu=0.32,
        sigma_t_MPa=3.5,
        sigma_c_MPa=2.2,
        tau_MPa=1.6,
        density_kgm3=100.0,
        cost_per_kg=250.0,
        is_core=True,
        is_skin=False,
    ),

    # ── Compósitos ────────────────────────────────────────────────────────────

    "CFRP UD (0°)": StructuralMaterial(
        name="CFRP UD (0°)",
        category="composite",
        description="Fibra de carbono unidirecional — pré-impregnada T300/epoxy",
        E_MPa=135000.0,
        E2_MPa=8000.0,
        G_MPa=4200.0,
        nu=0.27,
        sigma_t_MPa=1500.0,
        sigma_c_MPa=900.0,
        tau_MPa=70.0,
        density_kgm3=1600.0,
        cost_per_kg=350.0,
        is_core=False,
        is_skin=True,
        is_anisotropic=True,
        ply_thickness_mm=0.125,
        E1_MPa=135000.0,
        E2_ply_MPa=8000.0,
        G12_MPa=4200.0,
        nu12=0.27,
    ),

    "CFRP Tecido (0/90)": StructuralMaterial(
        name="CFRP Tecido (0/90)",
        category="composite",
        description="Tecido de fibra de carbono plain weave — HexPly M21",
        E_MPa=65000.0,
        E2_MPa=65000.0,
        G_MPa=5000.0,
        nu=0.05,
        sigma_t_MPa=820.0,
        sigma_c_MPa=700.0,
        tau_MPa=90.0,
        density_kgm3=1580.0,
        cost_per_kg=400.0,
        is_core=False,
        is_skin=True,
        is_anisotropic=False,
        ply_thickness_mm=0.200,
        E1_MPa=65000.0,
        E2_ply_MPa=65000.0,
        G12_MPa=5000.0,
        nu12=0.05,
    ),

    "GFRP Tecido (E-glass)": StructuralMaterial(
        name="GFRP Tecido (E-glass)",
        category="composite",
        description="Fibra de vidro E-glass / epoxy — solução econômica",
        E_MPa=18000.0,
        E2_MPa=18000.0,
        G_MPa=3500.0,
        nu=0.13,
        sigma_t_MPa=300.0,
        sigma_c_MPa=250.0,
        tau_MPa=40.0,
        density_kgm3=1800.0,
        cost_per_kg=80.0,
        is_core=False,
        is_skin=True,
        is_anisotropic=False,
        ply_thickness_mm=0.25,
    ),

    # ── Painéis Sanduíche ──────────────────────────────────────────────────────

    "Sanduíche CFRP/H80 (0.5mm+10mm+0.5mm)": StructuralMaterial(
        name="Sanduíche CFRP/H80 (0.5mm+10mm+0.5mm)",
        category="composite",
        description="Painel sanduíche CFRP 2×0.5mm + Divinycell H80 10mm — nervura leve",
        # Propriedades efetivas calculadas via regra das misturas
        E_MPa=8400.0,          # Rigidez à flexão efetiva
        G_MPa=28.0,            # Cisalhamento do núcleo (dominante)
        nu=0.3,
        sigma_t_MPa=180.0,     # Limitado pela pele de CFRP
        sigma_c_MPa=150.0,
        tau_MPa=1.15,          # Limitado pelo núcleo H80
        density_kgm3=120.0,    # Média ponderada
        cost_per_kg=480.0,
        is_core=False,
        is_skin=True,
        is_anisotropic=False,
        ply_thickness_mm=11.0,  # Espessura total do painel
    ),

    "Sanduíche CFRP/H60 (0.3mm+6mm+0.3mm)": StructuralMaterial(
        name="Sanduíche CFRP/H60 (0.3mm+6mm+0.3mm)",
        category="composite",
        description="Painel sanduíche leve — nervuras intermediárias",
        E_MPa=6200.0,
        G_MPa=20.0,
        nu=0.3,
        sigma_t_MPa=120.0,
        sigma_c_MPa=100.0,
        tau_MPa=0.8,
        density_kgm3=90.0,
        cost_per_kg=420.0,
        is_core=False,
        is_skin=True,
        is_anisotropic=False,
        ply_thickness_mm=6.6,
    ),

    # ── Metais ────────────────────────────────────────────────────────────────

    "Alumínio 6061-T6": StructuralMaterial(
        name="Alumínio 6061-T6",
        category="metal",
        description="Liga de alumínio aeronáutica — referência de comparação",
        E_MPa=68900.0,
        G_MPa=26000.0,
        nu=0.33,
        sigma_t_MPa=276.0,
        sigma_c_MPa=276.0,
        tau_MPa=160.0,
        density_kgm3=2700.0,
        cost_per_kg=25.0,
        is_core=False,
        is_skin=True,
        is_anisotropic=False,
    ),
}


# ─── Funções de gerenciamento ─────────────────────────────────────────────────

def save_custom_materials(materials: Dict[str, StructuralMaterial],
                           path: str = "custom_materials.json") -> None:
    data = {k: v.to_dict() for k, v in materials.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_custom_materials(path: str = "custom_materials.json") -> Dict[str, StructuralMaterial]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: StructuralMaterial.from_dict(v) for k, v in data.items()}


def get_all_materials() -> Dict[str, StructuralMaterial]:
    """Retorna o banco padrão + materiais customizados do usuário."""
    all_mats = dict(DEFAULT_MATERIALS)
    custom = load_custom_materials()
    all_mats.update(custom)
    return all_mats


def material_names_by_category(category: Optional[str] = None) -> list:
    mats = get_all_materials()
    if category:
        return [n for n, m in mats.items() if m.category == category]
    return list(mats.keys())


# ─── CLPT — Teoria Clássica de Laminados ─────────────────────────────────────

def ply_Q_matrix(E1: float, E2: float, G12: float, nu12: float) -> np.ndarray:
    """Matriz de rigidez reduzida Q de um ply unidirecional [MPa]."""
    nu21 = nu12 * E2 / E1
    denom = 1 - nu12 * nu21
    Q = np.array([
        [E1 / denom,         nu12 * E2 / denom, 0],
        [nu12 * E2 / denom,  E2 / denom,         0],
        [0,                  0,                   G12]
    ])
    return Q


def ply_Qbar_matrix(Q: np.ndarray, theta_deg: float) -> np.ndarray:
    """Matriz Q rotacionada para ângulo θ [°]."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    c2, s2, cs = c**2, s**2, c * s

    T = np.array([
        [c2,    s2,    2*cs],
        [s2,    c2,   -2*cs],
        [-cs,   cs,   c2-s2]
    ])
    Tinv = np.array([
        [c2,    s2,   -2*cs],
        [s2,    c2,    2*cs],
        [cs,   -cs,   c2-s2]
    ])
    return Tinv @ Q @ T


def clpt_ABD_matrices(plies: list) -> tuple:
    """
    Calcula as matrizes A, B, D do laminado (CLPT).

    plies: lista de dicts com keys: theta_deg, thickness_mm, material (StructuralMaterial)
    Retorna (A, B, D) — matrizes 3×3 em N/mm e N·mm.
    """
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))

    # Calcular z de cada ply (z=0 no plano médio)
    total_t = sum(p["thickness_mm"] for p in plies)
    z = -total_t / 2
    for ply in plies:
        mat = ply["material"]
        t = ply["thickness_mm"]
        theta = ply["theta_deg"]

        E1 = mat.E1_MPa if mat.E1_MPa > 0 else mat.E_MPa
        E2 = mat.E2_ply_MPa if mat.E2_ply_MPa > 0 else mat.E2_effective
        G12 = mat.G12_MPa if mat.G12_MPa > 0 else mat.G_effective
        nu12 = mat.nu12 if mat.nu12 > 0 else mat.nu

        Q = ply_Q_matrix(E1, E2, G12, nu12)
        Qbar = ply_Qbar_matrix(Q, theta)

        z0 = z
        z1 = z + t
        z = z1

        A += Qbar * (z1 - z0)
        B += 0.5 * Qbar * (z1**2 - z0**2)
        D += (1/3) * Qbar * (z1**3 - z0**3)

    return A, B, D


def clpt_optimize_angles(
    mat: StructuralMaterial,
    n_plies: int,
    ply_thickness_mm: float,
    target: str = "torsion",       # "torsion" | "bending" | "balanced"
) -> list:
    """
    Otimiza ângulos de ply para maximizar rigidez torcional ou à flexão.
    Retorna sequência de ângulos [°].
    """
    if target == "torsion":
        # ±45° maximiza G_xy (rigidez torcional)
        base = [45, -45]
    elif target == "bending":
        # 0° maximiza Ex (rigidez à flexão)
        base = [0, 0]
    else:
        # Balanceado: 0/±45/90
        base = [0, 45, -45, 90]

    # Repetir para n_plies (simétrico)
    angles = []
    for i in range(n_plies):
        angles.append(base[i % len(base)])

    # Garantir simetria
    n = len(angles)
    if n % 2 == 0:
        angles = angles[:n//2] + angles[n//2:][::-1]
    return angles


def clpt_effective_properties(A: np.ndarray, B: np.ndarray,
                               D: np.ndarray, total_t: float) -> dict:
    """Extrai propriedades efetivas do laminado a partir das matrizes ABD."""
    # Compliance [A*] = [A]^-1
    try:
        Astar = np.linalg.inv(A)
        Ex = 1 / (Astar[0, 0] * total_t)
        Ey = 1 / (Astar[1, 1] * total_t)
        Gxy = 1 / (Astar[2, 2] * total_t)
        nuxy = -Astar[0, 1] / Astar[0, 0]

        # Rigidez à flexão
        Dstar = np.linalg.inv(D)
        Dx = 1 / Dstar[0, 0]
        Dy = 1 / Dstar[1, 1]
        Dxy = 1 / Dstar[2, 2]
    except np.linalg.LinAlgError:
        Ex = Ey = Gxy = nuxy = Dx = Dy = Dxy = 0.0

    return {
        "Ex_MPa": Ex, "Ey_MPa": Ey, "Gxy_MPa": Gxy, "nuxy": nuxy,
        "Dx_Nmm": Dx, "Dy_Nmm": Dy, "Dxy_Nmm": Dxy,
    }


# ─── Helpers para nervuras ────────────────────────────────────────────────────

def rib_effective_properties(mat: StructuralMaterial,
                              thickness_mm: float,
                              use_clpt: bool = False,
                              ply_sequence: Optional[list] = None) -> dict:
    """
    Retorna propriedades efetivas de uma nervura para uso no cálculo de massa/rigidez.
    """
    if use_clpt and mat.is_anisotropic and ply_sequence:
        plies = [{"theta_deg": a, "thickness_mm": mat.ply_thickness_mm,
                  "material": mat} for a in ply_sequence]
        A, B, D = clpt_ABD_matrices(plies)
        t_total = sum(p["thickness_mm"] for p in plies)
        props = clpt_effective_properties(A, B, D, t_total)
        return {
            "E_MPa": props["Ex_MPa"],
            "G_MPa": props["Gxy_MPa"],
            "nu": props["nuxy"],
            "thickness_mm": t_total,
            "density_kgm3": mat.density_kgm3,
            "sigma_t_MPa": mat.sigma_t_MPa,
            "sigma_c_MPa": mat.sigma_c_MPa,
            "tau_MPa": mat.tau_MPa,
        }
    else:
        return {
            "E_MPa": mat.E_MPa,
            "G_MPa": mat.G_effective,
            "nu": mat.nu,
            "thickness_mm": thickness_mm,
            "density_kgm3": mat.density_kgm3,
            "sigma_t_MPa": mat.sigma_t_MPa,
            "sigma_c_MPa": mat.sigma_c_MPa,
            "tau_MPa": mat.tau_MPa,
        }