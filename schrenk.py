"""
schrenk.py — Distribuição de Sustentação de Schrenk e Carga Crítica
Calcula a distribuição de sustentação ao longo da envergadura usando
a aproximação de Schrenk (média entre distribuição do planform e elíptica).

Referência: Schrenk, O. "A Simple Approximation Method for Obtaining the
Spanwise Lift Distribution", NACA TM-948, 1940.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class WingGeometry:
    """Geometria da asa (meia-envergadura)."""
    semi_span_mm: float = 750.0       # Meia-envergadura [mm]
    root_chord_mm: float = 300.0      # Corda na raiz [mm]
    tip_chord_mm: float = 200.0       # Corda na ponta [mm]
    sweep_deg: float = 0.0            # Enflechamento [°]
    dihedral_deg: float = 0.0         # Diedro [°]
    twist_deg: float = 0.0            # Torção geométrica (ponta) [°]

    @property
    def taper_ratio(self) -> float:
        return self.tip_chord_mm / self.root_chord_mm

    @property
    def wing_area_mm2(self) -> float:
        """Área da asa completa [mm²]."""
        return 2 * self.semi_span_mm * (self.root_chord_mm + self.tip_chord_mm) / 2

    @property
    def aspect_ratio(self) -> float:
        span = 2 * self.semi_span_mm
        return span**2 / self.wing_area_mm2

    @property
    def mac_mm(self) -> float:
        """Corda aerodinâmica média (MAC) [mm]."""
        lam = self.taper_ratio
        return (2/3) * self.root_chord_mm * (1 + lam + lam**2) / (1 + lam)

    @property
    def mac_y_mm(self) -> float:
        """Posição spanwise do MAC [mm]."""
        lam = self.taper_ratio
        return (self.semi_span_mm / 3) * (1 + 2*lam) / (1 + lam)


@dataclass
class FlightCondition:
    """Condição de voo."""
    velocity_ms: float = 15.0         # Velocidade [m/s]
    rho_kgm3: float = 1.225           # Densidade do ar [kg/m³]
    load_factor: float = 4.0          # Fator de carga (n)
    aircraft_mass_kg: float = 5.0     # Massa da aeronave [kg]
    CL_wing: float = 0.8              # CL da asa
    CL_distribution: str = "schrenk"  # "schrenk", "elliptic", "planform"

    @property
    def q_Pa(self) -> float:
        """Pressão dinâmica [Pa]."""
        return 0.5 * self.rho_kgm3 * self.velocity_ms**2

    @property
    def total_lift_N(self) -> float:
        """Sustentação total requerida [N]."""
        return self.load_factor * self.aircraft_mass_kg * 9.81


@dataclass
class SchrenkResult:
    """Resultados da distribuição de Schrenk."""
    y_mm: np.ndarray = field(default_factory=lambda: np.array([]))
    chord_mm: np.ndarray = field(default_factory=lambda: np.array([]))
    cl_local: np.ndarray = field(default_factory=lambda: np.array([]))
    lift_per_span_Nmm: np.ndarray = field(default_factory=lambda: np.array([]))
    shear_N: np.ndarray = field(default_factory=lambda: np.array([]))
    moment_Nmm: np.ndarray = field(default_factory=lambda: np.array([]))
    # Valores críticos
    max_shear_N: float = 0.0
    max_moment_Nmm: float = 0.0
    root_lift_Nmm: float = 0.0
    total_lift_N: float = 0.0
    # Pressão para o ANSYS (na nervura crítica)
    pressure_at_station_MPa: float = 0.0


def chord_distribution(wing: WingGeometry, y: np.ndarray) -> np.ndarray:
    """Distribuição de corda ao longo da envergadura (asa trapezoidal)."""
    lam = wing.taper_ratio
    eta = y / wing.semi_span_mm  # Posição normalizada (0 = raiz, 1 = ponta)
    return wing.root_chord_mm * (1 - eta * (1 - lam))


def elliptic_distribution(wing: WingGeometry, y: np.ndarray) -> np.ndarray:
    """Distribuição elíptica de sustentação (corda equivalente)."""
    b2 = wing.semi_span_mm
    return (4 * wing.wing_area_mm2 / (np.pi * 2 * b2)) * np.sqrt(1 - (y / b2)**2)


def schrenk_distribution(wing: WingGeometry, flight: FlightCondition,
                          n_stations: int = 200) -> SchrenkResult:
    """
    Calcula a distribuição de Schrenk completa.

    Retorna distribuição de sustentação, cortante e momento fletor
    ao longo da meia-envergadura.
    """
    y = np.linspace(0, wing.semi_span_mm * 0.999, n_stations)

    # 1. Distribuição do planform (corda real)
    c_planform = chord_distribution(wing, y)

    # 2. Distribuição elíptica
    c_elliptic = elliptic_distribution(wing, y)

    # 3. Schrenk: média aritmética
    c_schrenk = 0.5 * (c_planform + c_elliptic)

    # 4. Normalizar para que a integral dê a área da asa / 2
    integral_schrenk = np.trapz(c_schrenk, y)
    half_area = wing.wing_area_mm2 / 2
    c_schrenk_norm = c_schrenk * (half_area / integral_schrenk)

    # 5. Distribuição de sustentação [N/mm]
    L_total = flight.total_lift_N
    L_half = L_total / 2  # Uma meia-asa carrega metade

    # L'(y) = q * c_l(y) * c(y), mas por Schrenk:
    # L'(y) proporcional a c_schrenk_norm(y)
    integral_norm = np.trapz(c_schrenk_norm, y)
    lift_dist = c_schrenk_norm * (L_half / integral_norm)  # [N/mm]

    # 6. cl local
    q_Pa = flight.q_Pa
    q_MPa = q_Pa * 1e-6  # Pa → MPa
    cl_local = lift_dist / (q_MPa * c_planform)  # adimensional

    # 7. Cortante (integração da ponta para a raiz)
    shear = np.zeros_like(y)
    for i in range(len(y) - 2, -1, -1):
        dy = y[i+1] - y[i]
        shear[i] = shear[i+1] + lift_dist[i] * dy

    # 8. Momento fletor (integração da ponta para a raiz)
    moment = np.zeros_like(y)
    for i in range(len(y) - 2, -1, -1):
        dy = y[i+1] - y[i]
        moment[i] = moment[i+1] + shear[i] * dy

    result = SchrenkResult(
        y_mm=y,
        chord_mm=c_planform,
        cl_local=cl_local,
        lift_per_span_Nmm=lift_dist,
        shear_N=shear,
        moment_Nmm=moment,
        max_shear_N=float(np.max(np.abs(shear))),
        max_moment_Nmm=float(np.max(np.abs(moment))),
        root_lift_Nmm=float(lift_dist[0]),
        total_lift_N=float(np.trapz(lift_dist, y) * 2),
    )

    return result


def pressure_at_station(result: SchrenkResult, wing: WingGeometry,
                         flight: FlightCondition,
                         y_station_mm: float) -> float:
    """
    Calcula a pressão aerodinâmica equivalente [MPa] em uma estação
    spanwise específica, para usar como BC no ANSYS.
    """
    # Interpolar lift/span naquela posição
    lift_at_y = np.interp(y_station_mm, result.y_mm, result.lift_per_span_Nmm)
    chord_at_y = np.interp(y_station_mm, result.y_mm, result.chord_mm)

    # Pressão = força por unidade de envergadura / corda
    # [N/mm] / [mm] = [N/mm²] = [MPa]
    pressure_MPa = lift_at_y / chord_at_y

    return float(pressure_MPa)


def critical_rib_loads(result: SchrenkResult, wing: WingGeometry,
                        rib_positions_mm: np.ndarray) -> dict:
    """
    Calcula cargas em cada nervura: cortante, momento, pressão local.
    Retorna dicionário com arrays para cada nervura.
    """
    n_ribs = len(rib_positions_mm)
    data = {
        "y_mm": rib_positions_mm,
        "shear_N": np.zeros(n_ribs),
        "moment_Nmm": np.zeros(n_ribs),
        "lift_Nmm": np.zeros(n_ribs),
        "chord_mm": np.zeros(n_ribs),
        "pressure_MPa": np.zeros(n_ribs),
    }

    for i, y_rib in enumerate(rib_positions_mm):
        data["shear_N"][i] = np.interp(y_rib, result.y_mm, result.shear_N)
        data["moment_Nmm"][i] = np.interp(y_rib, result.y_mm, result.moment_Nmm)
        data["lift_Nmm"][i] = np.interp(y_rib, result.y_mm, result.lift_per_span_Nmm)
        data["chord_mm"][i] = np.interp(y_rib, result.y_mm, result.chord_mm)
        data["pressure_MPa"][i] = data["lift_Nmm"][i] / data["chord_mm"][i]
def discretize_rib_loads_matlab(wing: WingGeometry, flight: FlightCondition, n_ribs: int):
    """
    Replicação exata da lógica do 'Loads_Per_Rib.m'.
    Escalona a sustentação para MTOW*n e discretiza a força por baia tributária.
    """
    result = schrenk_distribution(wing, flight, n_stations=500)
    
    # 1. Escalonamento Estrutural (MTOW, n)
    # L_req para uma semi-asa
    req_lift_half = (flight.aircraft_mass_kg * 9.81 * flight.load_factor) / 2
    vlm_lift = np.trapz(result.lift_per_span_Nmm, result.y_mm)
    
    scale_factor = req_lift_half / vlm_lift if vlm_lift > 0 else 1.0
    L_scaled = result.lift_per_span_Nmm * scale_factor
    
    # 2. Geometria das Nervuras
    y_ribs = np.linspace(0, wing.semi_span_mm, n_ribs)
    chord_ribs = np.interp(y_ribs, result.y_mm, result.chord_mm)
    
    force_per_rib = np.zeros(n_ribs)
    pressure_per_rib = np.zeros(n_ribs)
    
    # 3. Discretização por Baia Tributária
    for i in range(n_ribs):
        if i == 0:
            y_in = 0
            y_out = (y_ribs[i] + y_ribs[i+1]) / 2
        elif i == n_ribs - 1:
            y_in = (y_ribs[i-1] + y_ribs[i]) / 2
            y_out = y_ribs[i]
        else:
            y_in = (y_ribs[i-1] + y_ribs[i]) / 2
            y_out = (y_ribs[i] + y_ribs[i+1]) / 2
            
        # Integração da força na baia (N)
        idx = (result.y_mm >= y_in) & (result.y_mm <= y_out)
        if np.any(idx):
            force = np.trapz(L_scaled[idx], result.y_mm[idx])
        else:
            force = 0
            
        force_per_rib[i] = force
        
        # Pressão simplificada atuando na área da nervura (para envio ao MAPDL)
        # Área eq = Corda * Vão da baia
        bay_width = y_out - y_in
        if bay_width > 0:
            # Em MPa (N/mm²)
            pressure_per_rib[i] = force / (chord_ribs[i] * bay_width)
            
    # 4. Identificando a Nervura Crítica
    idx_crit = np.argmax(force_per_rib)
    
    return {
        "y_ribs": y_ribs,
        "chord_ribs": chord_ribs,
        "force_per_rib": force_per_rib,
        "pressure_per_rib": pressure_per_rib,
        "idx_crit": idx_crit,
        "y_schrenk": result.y_mm,
        "L_scaled": L_scaled,
        "scale_factor": scale_factor,
        "req_lift_half": req_lift_half
    }

    return data