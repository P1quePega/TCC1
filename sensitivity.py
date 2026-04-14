"""
sensitivity.py — Análise de Sensibilidade e Estudos Paramétricos
Avalia como os resultados (tensão, massa, deflexão) variam com mudanças
nos parâmetros de entrada (espessura, pressão, material, etc.).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict


@dataclass
class SensitivityVariable:
    """Uma variável para análise de sensibilidade."""
    name: str
    label: str
    unit: str
    base_value: float
    min_value: float
    max_value: float
    n_points: int = 10

    @property
    def values(self) -> np.ndarray:
        return np.linspace(self.min_value, self.max_value, self.n_points)


@dataclass
class SensitivityResult:
    """Resultado de uma análise de sensibilidade univariada."""
    variable_name: str
    variable_label: str
    variable_unit: str
    variable_values: np.ndarray = field(default_factory=lambda: np.array([]))
    base_value: float = 0.0
    # Respostas
    responses: Dict[str, np.ndarray] = field(default_factory=dict)
    response_labels: Dict[str, str] = field(default_factory=dict)
    response_units: Dict[str, str] = field(default_factory=dict)
    # Gradientes normalizados (elasticidade)
    elasticities: Dict[str, float] = field(default_factory=dict)


def tornado_sensitivity(
    variables: List[SensitivityVariable],
    evaluate_fn: Callable,
    response_names: List[str],
    response_labels: Optional[Dict[str, str]] = None,
    perturbation_pct: float = 10.0,
) -> Dict[str, dict]:
    """
    Análise tornado: varia cada parâmetro ±perturbation_pct e mede o impacto.

    evaluate_fn: função que recebe dict de parâmetros e retorna dict de respostas.
    """
    results = {}
    base_params = {v.name: v.base_value for v in variables}
    base_response = evaluate_fn(base_params)

    for var in variables:
        delta = var.base_value * (perturbation_pct / 100)
        if delta < 1e-12:
            delta = 1e-6

        # Perturbação +
        params_plus = base_params.copy()
        params_plus[var.name] = var.base_value + delta
        resp_plus = evaluate_fn(params_plus)

        # Perturbação -
        params_minus = base_params.copy()
        params_minus[var.name] = var.base_value - delta
        resp_minus = evaluate_fn(params_minus)

        var_result = {"variable": var.label, "unit": var.unit, "base": var.base_value}

        for rname in response_names:
            r_base = base_response.get(rname, 0)
            r_plus = resp_plus.get(rname, 0)
            r_minus = resp_minus.get(rname, 0)

            # Elasticidade: (ΔR/R) / (ΔX/X)
            if abs(r_base) > 1e-12 and abs(var.base_value) > 1e-12:
                elasticity = ((r_plus - r_minus) / (2 * r_base)) / (delta / var.base_value)
            else:
                elasticity = 0.0

            var_result[f"{rname}_minus"] = r_minus
            var_result[f"{rname}_base"] = r_base
            var_result[f"{rname}_plus"] = r_plus
            var_result[f"{rname}_elasticity"] = elasticity

        results[var.name] = var_result

    return results


def univariate_sweep(
    variable: SensitivityVariable,
    evaluate_fn: Callable,
    base_params: dict,
    response_names: List[str],
    response_labels: Optional[Dict[str, str]] = None,
    response_units: Optional[Dict[str, str]] = None,
) -> SensitivityResult:
    """
    Varredura univariada: varia um parâmetro, mantém os outros fixos.
    """
    result = SensitivityResult(
        variable_name=variable.name,
        variable_label=variable.label,
        variable_unit=variable.unit,
        variable_values=variable.values,
        base_value=variable.base_value,
    )

    for rname in response_names:
        result.responses[rname] = np.zeros(variable.n_points)
        result.response_labels[rname] = (response_labels or {}).get(rname, rname)
        result.response_units[rname] = (response_units or {}).get(rname, "")

    for i, val in enumerate(variable.values):
        params = base_params.copy()
        params[variable.name] = val
        resp = evaluate_fn(params)
        for rname in response_names:
            result.responses[rname][i] = resp.get(rname, 0)

    # Calcular elasticidades no ponto base
    idx_base = np.argmin(np.abs(variable.values - variable.base_value))
    for rname in response_names:
        r_vals = result.responses[rname]
        r_base = r_vals[idx_base]
        if abs(r_base) > 1e-12 and len(r_vals) >= 3:
            dr = np.gradient(r_vals, variable.values)
            dx_ratio = variable.values[idx_base] / r_base if abs(r_base) > 1e-12 else 0
            result.elasticities[rname] = float(dr[idx_base] * dx_ratio)
        else:
            result.elasticities[rname] = 0.0

    return result


def covering_sensitivity_evaluator(params: dict) -> dict:
    """
    Função de avaliação para análise de sensibilidade da entelagem.
    Usada com tornado_sensitivity e univariate_sweep.
    """
    from covering import CoveringMaterial, check_covering

    mat = CoveringMaterial(
        E_MPa=params.get("covering_E", 2500),
        thickness_mm=params.get("covering_t", 0.025),
        pretension_Nmm=params.get("covering_pretension", 0.01),
    )
    res = check_covering(
        rib_spacing_mm=params.get("rib_spacing", 80),
        chord_mm=params.get("chord", 200),
        pressure_MPa=params.get("pressure", -0.05),
        mat=mat,
    )
    return {
        "deflection_mm": res.max_deflection_mm,
        "deflection_ratio": res.deflection_chord_ratio,
        "approved": 1.0 if res.approved else 0.0,
    }


def structural_sensitivity_evaluator(params: dict) -> dict:
    """
    Função de avaliação simplificada (analítica) para sensibilidade
    estrutural da nervura. Usa beam bending como proxy rápido.
    """
    # Modelo simplificado: nervura como viga em balanço
    c = params.get("chord", 200)           # corda [mm]
    t = params.get("thickness", 3.0)       # espessura nervura [mm]
    E = params.get("E", 3500)              # módulo [MPa]
    p = abs(params.get("pressure", 0.05))  # pressão [MPa]
    t_skin = params.get("skin_t", 3.0)     # casca [mm]
    vf = params.get("vol_fraction", 0.3)   # fração de volume

    # Área da seção = corda × espessura × volfrac
    A = c * t * vf
    # Momento de inércia ≈ (c × t³)/12
    I = c * t**3 / 12

    # Carga distribuída
    w = p * c  # [N/mm] (carga por comprimento)

    # Tensão de flexão σ = M·c / I, onde M = w·L²/2, L ≈ metade da altura do perfil
    h_perfil = 0.12 * c  # ~12% da corda para perfis típicos
    M = w * (h_perfil/2)**2 / 2
    sigma_max = M * (t/2) / I if I > 0 else 0

    # Deflexão
    delta_max = w * (h_perfil/2)**4 / (8 * E * I) if E * I > 0 else 0

    # Massa
    rho = params.get("density_kgm3", 160)  # kg/m³
    mass_g = A * t_skin * rho * 1e-6  # Simplificado

    return {
        "stress_MPa": sigma_max,
        "deflection_mm": delta_max,
        "mass_g": mass_g,
    }


# ── Variáveis padrão para análises rápidas ────────────────────────────────────

DEFAULT_STRUCTURAL_VARS = [
    SensitivityVariable("pressure", "Pressão", "MPa", -0.05, -0.10, -0.01, 10),
    SensitivityVariable("thickness", "Espessura nervura", "mm", 3.0, 1.0, 6.0, 10),
    SensitivityVariable("skin_t", "Espessura casca", "mm", 3.0, 1.0, 6.0, 10),
    SensitivityVariable("E", "Módulo de elasticidade", "MPa", 3500, 1000, 10000, 10),
    SensitivityVariable("vol_fraction", "Fração de volume", "-", 0.3, 0.1, 0.6, 10),
    SensitivityVariable("chord", "Corda", "mm", 200, 100, 400, 10),
]

DEFAULT_COVERING_VARS = [
    SensitivityVariable("rib_spacing", "Espaçamento nervuras", "mm", 80, 30, 200, 15),
    SensitivityVariable("covering_t", "Espessura Monokote", "mm", 0.025, 0.010, 0.060, 10),
    SensitivityVariable("covering_E", "Módulo Monokote", "MPa", 2500, 500, 5000, 10),
    SensitivityVariable("pressure", "Pressão", "MPa", -0.05, -0.10, -0.01, 10),
    SensitivityVariable("covering_pretension", "Pré-tensão", "N/mm", 0.01, 0.001, 0.05, 10),
]