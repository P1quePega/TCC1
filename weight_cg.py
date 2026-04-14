"""
weight_cg.py — Estimador de Peso e Centro de Gravidade (CG) da Asa
Soma massas: nervuras otimizadas + tubo longarina + entelagem + cola.
Calcula X_cg em relação ao bordo de ataque da MAC.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RibMass:
    """Massa de uma nervura individual."""
    y_mm: float                 # Posição spanwise [mm]
    chord_mm: float             # Corda local [mm]
    area_casca_mm2: float       # Área da casca [mm²]
    area_otim_mm2: float        # Área efetiva após otimização [mm²]
    thickness_mm: float         # Espessura da nervura [mm]
    density_kgm3: float         # Densidade do material [kg/m³]
    spar_position_pct: float    # Posição da longarina (% corda)

    @property
    def volume_mm3(self) -> float:
        return (self.area_casca_mm2 + self.area_otim_mm2) * self.thickness_mm

    @property
    def mass_g(self) -> float:
        vol_m3 = self.volume_mm3 * 1e-9  # mm³ → m³
        return self.density_kgm3 * vol_m3 * 1000  # kg → g

    @property
    def x_cg_mm(self) -> float:
        """CG aproximado da nervura: centroide da corda ponderado pela longarina."""
        return self.spar_position_pct * self.chord_mm


@dataclass
class SparConfig:
    """Configuração do tubo longarina."""
    material: str = "Carbono"
    outer_diameter_mm: float = 12.0
    wall_thickness_mm: float = 1.0
    density_kgm3: float = 1600.0       # Fibra de carbono
    x_position_pct: float = 0.28       # Posição (%corda na raiz)
    semi_span_mm: float = 750.0

    @property
    def inner_diameter_mm(self) -> float:
        return self.outer_diameter_mm - 2 * self.wall_thickness_mm

    @property
    def cross_section_mm2(self) -> float:
        r_out = self.outer_diameter_mm / 2
        r_in = self.inner_diameter_mm / 2
        return np.pi * (r_out**2 - r_in**2)

    @property
    def mass_g(self) -> float:
        length_mm = self.semi_span_mm * 2
        vol_m3 = self.cross_section_mm2 * length_mm * 1e-9
        return self.density_kgm3 * vol_m3 * 1000  # g

    @property
    def linear_mass_gmm(self) -> float:
        """Massa linear [g/mm]."""
        return self.mass_g / (self.semi_span_mm * 2)


@dataclass
class CoveringConfig:
    """Configuração da entelagem (Monokote/Oracover)."""
    material: str = "Monokote"
    density_gsm: float = 35.0        # Gramatura [g/m²]
    # Se não souber a gramatura, calcular via espessura e densidade
    thickness_mm: float = 0.025
    density_kgm3: float = 1400.0     # Polímero (PET/PE)

    @property
    def gsm_from_thickness(self) -> float:
        """Gramatura calculada a partir de espessura e densidade [g/m²]."""
        return self.density_kgm3 * self.thickness_mm * 1e-3 * 1e6

    def mass_for_area_g(self, area_mm2: float) -> float:
        """Massa da entelagem para uma dada área superficial [g]."""
        area_m2 = area_mm2 * 1e-6
        return self.density_gsm * area_m2


@dataclass
class GlueConfig:
    """Margem percentual para cola, acabamento, fixações."""
    margin_pct: float = 12.0  # % adicionada sobre massa total


@dataclass
class WeightBreakdown:
    """Resultado do balanço de massa."""
    ribs_g: float = 0.0
    spar_g: float = 0.0
    covering_g: float = 0.0
    glue_margin_g: float = 0.0
    total_g: float = 0.0
    # Detalhamento por nervura
    rib_details: List[dict] = field(default_factory=list)
    # CG
    x_cg_mm: float = 0.0              # CG da asa [mm] a partir do bordo de ataque
    x_cg_pct_mac: float = 0.0         # CG como % da MAC (objetivo: 20-30%)
    mac_mm: float = 0.0
    # Flags
    cg_ok: bool = True
    cg_range_min_pct: float = 18.0
    cg_range_max_pct: float = 32.0


def estimate_covering_area(wing_semi_span_mm: float,
                            root_chord_mm: float,
                            tip_chord_mm: float,
                            airfoil_perimeter_ratio: float = 2.08) -> float:
    """
    Estima área superficial total da asa para entelagem [mm²].
    airfoil_perimeter_ratio: perímetro / corda (≈ 2.05-2.10 para NACA 4 dígitos)
    Fator 2 porque são extradorso + intradorso.
    """
    # Perímetro médio
    avg_perimeter = (root_chord_mm + tip_chord_mm) / 2 * airfoil_perimeter_ratio
    # Área = perímetro médio × envergadura (meia-asa × 2)
    return avg_perimeter * wing_semi_span_mm * 2


def compute_weight_cg(
    rib_masses: List[RibMass],
    spar: SparConfig,
    covering: CoveringConfig,
    glue: GlueConfig,
    wing_semi_span_mm: float,
    root_chord_mm: float,
    tip_chord_mm: float,
    mac_mm: Optional[float] = None,
) -> WeightBreakdown:
    """
    Calcula peso total e CG da asa.

    Retorna WeightBreakdown com massas parciais, total e posição do CG.
    """
    result = WeightBreakdown()

    # MAC
    lam = tip_chord_mm / root_chord_mm
    if mac_mm is None:
        mac_mm = (2/3) * root_chord_mm * (1 + lam + lam**2) / (1 + lam)
    result.mac_mm = mac_mm

    # 1. Nervuras
    total_rib_mass = 0.0
    moment_x_ribs = 0.0  # Somatório (massa × x_cg)
    for rib in rib_masses:
        m = rib.mass_g
        total_rib_mass += m
        moment_x_ribs += m * rib.x_cg_mm
        result.rib_details.append({
            "y_mm": rib.y_mm,
            "chord_mm": rib.chord_mm,
            "mass_g": m,
            "x_cg_mm": rib.x_cg_mm,
        })
    result.ribs_g = total_rib_mass

    # 2. Longarina
    result.spar_g = spar.mass_g
    x_cg_spar = spar.x_position_pct * root_chord_mm  # Simplificado

    # 3. Entelagem
    covering_area = estimate_covering_area(
        wing_semi_span_mm, root_chord_mm, tip_chord_mm
    )
    result.covering_g = covering.mass_for_area_g(covering_area)
    x_cg_covering = 0.40 * root_chord_mm  # CG da entelagem ≈ 40% corda

    # 4. Subtotal sem margem
    subtotal = result.ribs_g + result.spar_g + result.covering_g

    # 5. Margem de cola
    result.glue_margin_g = subtotal * (glue.margin_pct / 100)

    # 6. Total
    result.total_g = subtotal + result.glue_margin_g

    # 7. CG da asa (momentos)
    # Σ(m_i × x_i) / Σ(m_i)
    total_moment = (
        moment_x_ribs
        + result.spar_g * x_cg_spar
        + result.covering_g * x_cg_covering
        + result.glue_margin_g * (moment_x_ribs / max(total_rib_mass, 1e-6))
    )

    result.x_cg_mm = total_moment / max(result.total_g, 1e-6)
    result.x_cg_pct_mac = (result.x_cg_mm / mac_mm) * 100

    # 8. Verificação
    result.cg_ok = (result.cg_range_min_pct <= result.x_cg_pct_mac <= result.cg_range_max_pct)

    return result


def generate_rib_masses_from_optimization(
    n_ribs: int,
    rib_positions_mm: np.ndarray,
    chord_at_ribs_mm: np.ndarray,
    area_casca_root_mm2: float,
    area_otim_root_mm2: float,
    volume_fraction: float,
    thickness_mm: float,
    density_kgm3: float,
    spar_position_pct: float,
    root_chord_mm: float,
) -> List[RibMass]:
    """
    Gera lista de RibMass para todas as nervuras, escalando a área
    proporcionalmente à corda local (nervura da raiz como referência).
    """
    ribs = []
    for i in range(n_ribs):
        scale = (chord_at_ribs_mm[i] / root_chord_mm)**2
        ribs.append(RibMass(
            y_mm=float(rib_positions_mm[i]),
            chord_mm=float(chord_at_ribs_mm[i]),
            area_casca_mm2=area_casca_root_mm2 * scale,
            area_otim_mm2=area_otim_root_mm2 * volume_fraction * scale,
            thickness_mm=thickness_mm,
            density_kgm3=density_kgm3,
            spar_position_pct=spar_position_pct,
        ))
    return ribs