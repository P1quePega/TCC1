"""
covering.py — Verificação de Deflexão da Entelagem (Monokote/Oracover)
Modelo de membrana retangular sob pressão uniforme para verificar se o
Monokote afunda entre nervuras, degradando o perfil aerodinâmico.

Modelo: Placa fina simplesmente apoiada (Navier/Levy) ou membrana tensionada.
Para filmes poliméricos finos como Monokote, o modelo de membrana com
pré-tensão é mais representativo.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CoveringMaterial:
    """Propriedades do material de entelagem."""
    name: str = "Monokote"
    E_MPa: float = 2500.0          # Módulo de elasticidade do polímero [MPa]
    nu: float = 0.35               # Poisson
    thickness_mm: float = 0.025    # Espessura [mm] (~25 μm)
    density_kgm3: float = 1400.0   # Densidade
    pretension_Nmm: float = 0.01   # Pré-tensão por largura unitária [N/mm]


@dataclass
class CoveringCheckResult:
    """Resultado da verificação de entelagem."""
    # Inputs
    rib_spacing_mm: float = 0.0
    chord_mm: float = 0.0
    pressure_MPa: float = 0.0
    # Resultados
    max_deflection_mm: float = 0.0
    deflection_chord_ratio: float = 0.0  # Δw/c
    # Limites
    max_allowed_mm: float = 0.0
    max_allowed_ratio: float = 0.0
    # Status
    approved: bool = True
    mode: str = "membrane"  # "membrane" ou "plate"
    # Detalhamento
    critical_region: str = ""  # "extradorso" ou "intradorso"
    notes: str = ""


def membrane_deflection(span_mm: float, pressure_MPa: float,
                         mat: CoveringMaterial) -> float:
    """
    Deflexão máxima de uma membrana retangular sob pressão uniforme.

    Modelo simplificado (1D): membrana uniformemente carregada, apoiada
    nas duas extremidades (nervuras), com pré-tensão T₀.

    Para uma faixa unitária de largura:
        w_max = p × L² / (8 × T₀/t + 8 × E × t × (w_max/L)²)

    Solução iterativa (não-linear) para deflexões grandes.
    Se pré-tensão alta: w_max ≈ p × L² / (8 × T₀)
    """
    p = abs(pressure_MPa)  # Magnitude
    L = span_mm
    t = mat.thickness_mm
    E = mat.E_MPa
    T0 = mat.pretension_Nmm  # Pré-tensão [N/mm de largura]

    if p < 1e-10 or L < 1e-6:
        return 0.0

    # Solução linear (pré-tensão dominante)
    # Para uma faixa: w = p*L²/(8*T0) quando T0 >> E*t*(w/L)²
    w_linear = p * L**2 / (8 * max(T0, 1e-10))

    # Solução não-linear iterativa (Newton)
    w = w_linear
    for _ in range(50):
        # Equilíbrio: p*L/2 = (T0 + E*t*(w/L)²*L) * (2*w/L)
        # Simplificado: p = 8*w/L² * (T0 + E*t*w²/L²)
        f = 8*w/L**2 * (T0 + E*t*(w/L)**2) - p
        df = 8/L**2 * (T0 + E*t*(w/L)**2) + 8*w/L**2 * (2*E*t*w/L**2)
        if abs(df) < 1e-30:
            break
        w_new = w - f / df
        if w_new < 0:
            w_new = w / 2
        if abs(w_new - w) < 1e-8:
            break
        w = w_new

    return max(w, 0.0)


def plate_deflection(a_mm: float, b_mm: float,
                      pressure_MPa: float, mat: CoveringMaterial) -> float:
    """
    Deflexão máxima de placa fina retangular simplesmente apoiada
    sob pressão uniforme (solução de Navier).

    a = dimensão curta (espaçamento entre nervuras)
    b = dimensão longa (corda, aproximada como comprimento do painel)

    w_max = α × p × a⁴ / D

    onde D = E×t³ / (12×(1-ν²)) e α depende da razão a/b.
    """
    p = abs(pressure_MPa)
    t = mat.thickness_mm
    E = mat.E_MPa
    nu = mat.nu

    if p < 1e-10 or a_mm < 1e-6:
        return 0.0

    D = E * t**3 / (12 * (1 - nu**2))

    # Coeficiente α para placa simplesmente apoiada (Timoshenko)
    ratio = min(a_mm, b_mm) / max(a_mm, b_mm)
    # Tabela de Timoshenko para placa SS-SS-SS-SS
    # a/b:   1.0   0.8   0.6   0.5   0.4   0.2   0.0
    # α:   0.0444 0.0384 0.0264 0.0202 0.0140 0.0042 0.0
    ratios = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    alphas = [0.0, 0.0042, 0.0140, 0.0202, 0.0264, 0.0384, 0.0444]
    alpha = np.interp(ratio, ratios, alphas)

    a = min(a_mm, b_mm)
    w_max = alpha * p * a**4 / D

    return max(w_max, 0.0)


def check_covering(
    rib_spacing_mm: float,
    chord_mm: float,
    pressure_MPa: float,
    mat: CoveringMaterial,
    max_deflection_chord_ratio: float = 0.005,
    use_plate_model: bool = False,
) -> CoveringCheckResult:
    """
    Verifica se a deflexão da entelagem é aceitável.

    Parâmetros:
        rib_spacing_mm: Distância entre nervuras [mm]
        chord_mm: Corda local [mm]
        pressure_MPa: Pressão aerodinâmica [MPa] (negativa = sucção)
        mat: Material da entelagem
        max_deflection_chord_ratio: Deflexão máx permitida como fração da corda
        use_plate_model: Usar modelo de placa fina (vs membrana)

    Retorna:
        CoveringCheckResult com status e valores.
    """
    result = CoveringCheckResult(
        rib_spacing_mm=rib_spacing_mm,
        chord_mm=chord_mm,
        pressure_MPa=pressure_MPa,
    )

    # Calcular deflexão
    if use_plate_model:
        w_max = plate_deflection(rib_spacing_mm, chord_mm * 0.6, pressure_MPa, mat)
        result.mode = "plate"
    else:
        w_max = membrane_deflection(rib_spacing_mm, pressure_MPa, mat)
        result.mode = "membrane"

    result.max_deflection_mm = w_max
    result.deflection_chord_ratio = w_max / chord_mm if chord_mm > 0 else 0

    # Limites
    result.max_allowed_ratio = max_deflection_chord_ratio
    result.max_allowed_mm = max_deflection_chord_ratio * chord_mm

    # Verificação
    result.approved = (w_max <= result.max_allowed_mm)

    # Região crítica
    if pressure_MPa < 0:
        result.critical_region = "extradorso (sucção)"
    else:
        result.critical_region = "intradorso (compressão)"

    # Notas
    if not result.approved:
        excess = (w_max / result.max_allowed_mm - 1) * 100
        result.notes = (
            f"Deflexão excede o limite em {excess:.0f}%. "
            f"Soluções: reduzir espaçamento entre nervuras, "
            f"usar filme mais espesso, ou adicionar stringers."
        )
    else:
        margin = (1 - w_max / result.max_allowed_mm) * 100
        result.notes = f"Margem de segurança: {margin:.0f}%"

    return result


def parametric_covering_study(
    spacings_mm: np.ndarray,
    chord_mm: float,
    pressure_MPa: float,
    mat: CoveringMaterial,
    max_ratio: float = 0.005,
) -> dict:
    """
    Estudo paramétrico: varia espaçamento e retorna deflexões.
    Útil para definir distância máxima entre nervuras.
    """
    deflections = np.zeros_like(spacings_mm)
    approved = np.zeros_like(spacings_mm, dtype=bool)

    for i, sp in enumerate(spacings_mm):
        res = check_covering(sp, chord_mm, pressure_MPa, mat, max_ratio)
        deflections[i] = res.max_deflection_mm
        approved[i] = res.approved

    # Encontrar espaçamento máximo aprovado
    approved_spacings = spacings_mm[approved]
    max_spacing = float(np.max(approved_spacings)) if len(approved_spacings) > 0 else 0.0

    return {
        "spacings_mm": spacings_mm,
        "deflections_mm": deflections,
        "approved": approved,
        "max_approved_spacing_mm": max_spacing,
        "limit_mm": max_ratio * chord_mm,
    }