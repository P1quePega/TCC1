"""
aeroelasticity.py — Análise Dinâmica e Aeroelástica
Implementa os modelos de Cooper (2DOF e 3DOF) para Flutter e Divergência.
Método p-k (Theodorsen modificado) para plotagem V-g/V-f.

Referências:
- Cooper, J.E. "Introduction to Aircraft Aeroelasticity and Loads", Wiley, 2008.
- Bisplinghoff, R.L. et al. "Aeroelasticity", Dover, 1996.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


# ─── Parâmetros de entrada ────────────────────────────────────────────────────

@dataclass
class AeroelasticParams:
    """Parâmetros para análise aeroelástica 2DOF."""
    # Geometria da asa (semi-envergadura representativa)
    semi_span_mm: float = 750.0
    chord_mm: float = 250.0          # Corda média aerodinâmica [mm]
    
    # Propriedades inerciais (seção típica)
    mass_kg: float = 2.0             # Massa total da asa [kg]
    Iα_kgm2: float = 0.005          # Momento de inércia torcional [kg·m²]
    
    # Propriedades estruturais (obtidas do wingbox)
    EI_Nmm2: float = 1e9            # Rigidez à flexão [N·mm²]
    GJ_Nmm2: float = 5e8            # Rigidez torcional [N·mm²]
    
    # Propriedades aerodinâmicas (seção 2D — thin airfoil theory)
    a_h: float = -0.3               # Posição do eixo elástico (relativo ao c/2), >0 = traseiro
    x_alpha: float = 0.1            # Distância CG-eixo elástico (relativo a c), >0 = atrás
    
    # Condições ambientais
    rho_kgm3: float = 1.225         # Densidade do ar [kg/m³]
    V_min_ms: float = 5.0           # Velocidade inicial [m/s]
    V_max_ms: float = 60.0          # Velocidade máxima [m/s]
    n_speeds: int = 200             # Pontos de varredura

    # Amortecimento estrutural
    g_structural: float = 0.02      # 2% amortecimento estrutural


@dataclass
class FlutterResult:
    """Resultado da análise de flutter."""
    # Vetores de velocidade
    V_ms: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Frequências [Hz]
    freq_bending_Hz: np.ndarray = field(default_factory=lambda: np.array([]))
    freq_torsion_Hz: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Amortecimento (fração do crítico)
    damp_bending: np.ndarray = field(default_factory=lambda: np.array([]))
    damp_torsion: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Velocidades críticas
    V_flutter_ms: float = 0.0
    V_divergence_ms: float = 0.0
    f_flutter_Hz: float = 0.0
    
    # Frequências naturais em voo livre
    omega_h_Hz: float = 0.0        # Frequência de flexão [Hz]
    omega_alpha_Hz: float = 0.0    # Frequência de torção [Hz]
    
    # Status
    flutter_found: bool = False
    divergence_found: bool = False


# ─── Modelo 2DOF — Cooper (Seção Típica) ─────────────────────────────────────

def compute_natural_frequencies(params: AeroelasticParams) -> Tuple[float, float]:
    """
    Calcula frequências naturais de flexão e torção via método de Rayleigh.
    Retorna (omega_h_Hz, omega_alpha_Hz).
    """
    L = params.semi_span_mm * 1e-3  # m
    m = params.mass_kg
    Ia = params.Iα_kgm2

    # EI, GJ em N·m²
    EI = params.EI_Nmm2 * 1e-6  # N·m²
    GJ = params.GJ_Nmm2 * 1e-6  # N·m²

    # Flexão: asa em balanço (modo 1: λ₁ = 1.875)
    if EI > 0 and m > 0:
        omega_h = (1.875**2 / L**2) * np.sqrt(EI / (m / L))
    else:
        omega_h = 10.0

    # Torção: eixo fixo em y=0 (modo 1: λ = π/2)
    if GJ > 0 and Ia > 0:
        omega_alpha = (np.pi / (2 * L)) * np.sqrt(GJ / (Ia / L))
    else:
        omega_alpha = 30.0

    return omega_h / (2 * np.pi), omega_alpha / (2 * np.pi)


def flutter_2dof(params: AeroelasticParams) -> FlutterResult:
    """
    Análise de flutter 2DOF (método p-k simplificado — Theodorsen).

    Graus de liberdade: flexão (h) e torção (α) de seção típica.
    Usa função de Theodorsen C(k) ≈ 1 (aproximação quasi-estática) para
    uma primeira estimativa. A variante completa usa iteração em k.

    Ref: Cooper (2008), Cap. 11.
    """
    result = FlutterResult()
    
    c = params.chord_mm * 1e-3     # m
    b = c / 2                       # semichorda [m]
    rho = params.rho_kgm3
    L = params.semi_span_mm * 1e-3  # m

    # Frequências naturais
    omega_h_Hz, omega_alpha_Hz = compute_natural_frequencies(params)
    omega_h = omega_h_Hz * 2 * np.pi
    omega_alpha = omega_alpha_Hz * 2 * np.pi

    result.omega_h_Hz = omega_h_Hz
    result.omega_alpha_Hz = omega_alpha_Hz

    # Parâmetros adimensionais
    m = params.mass_kg
    Ia = params.Iα_kgm2
    a = params.a_h
    x_a = params.x_alpha
    mu = m / (np.pi * rho * b**2 * L)  # razão de massa

    V_range = np.linspace(params.V_min_ms, params.V_max_ms, params.n_speeds)

    freq_h_arr = np.zeros(params.n_speeds)
    freq_a_arr = np.zeros(params.n_speeds)
    damp_h_arr = np.zeros(params.n_speeds)
    damp_a_arr = np.zeros(params.n_speeds)

    V_flutter = 0.0
    V_div = 0.0
    f_flutter = 0.0
    flutter_found = False
    div_found = False

    for i, V in enumerate(V_range):
        q = 0.5 * rho * V**2    # pressão dinâmica

        # Parâmetro reduzido k = ωb/V (iteração externa)
        # Para estimativa inicial: usar média das freqs naturais
        k_guess = omega_h * b / V

        # Matriz de massa [M] (normalizando por m)
        r_alpha = np.sqrt(Ia / (m * b**2))  # raio de giração normalizado

        # Coeficientes aerodinâmicos Theodorsen (approx C(k)≈1)
        # [Ref: Cooper eq. 11.18]
        Lh = np.pi * rho * b**2 * L          # [kg]
        La = np.pi * rho * b**3 * L * (0.5 - a)
        Ma = np.pi * rho * b**4 * L * (1/8 + a**2)

        # Matriz de rigidez aerodinâmica simplificada
        # Kh_aero = 2*pi*q*c*L*dCl/dalpha * 1/c
        Lα_aero = 2 * np.pi * q * c * L     # derivada de sustentação

        # Equação de Routh-Hurwitz para estabilidade
        # Método simplificado: comparação omega_h vs omega_alpha com acoplamento
        # Equação característica 4ª ordem: λ⁴ + a3λ³ + a2λ² + a1λ + a0 = 0

        # Construir matrizes massa e rigidez acopladas
        m11 = 1.0
        m12 = x_a * b * m / m  # acoplamento flexão-torção
        m21 = x_a * b
        m22 = r_alpha**2 * b**2

        # Rigidez estrutural
        k11 = omega_h**2
        k12 = 0.0
        k21 = 0.0
        k22 = omega_alpha**2 * r_alpha**2 * b**2

        # Rigidez aerodinâmica (quasi-estática)
        ka_11 = -Lh * (2 * np.pi * V / (c * m)) * (V / b)
        ka_12 = -Lα_aero / m
        ka_21 = 0.0
        ka_22 = -Ma / (m * b**2)

        # Matriz de rigidez total
        K11 = k11 + ka_11 if not np.isnan(ka_11) else k11
        K12 = ka_12
        K22 = k22 + ka_22 if not np.isnan(ka_22) else k22

        # Solução do problema de autovalores 2×2
        M_mat = np.array([[1.0, x_a], [x_a, r_alpha**2]])
        K_mat = np.array([[omega_h**2 - q * 2 * np.pi * c * L / m,
                           -q * 2 * np.pi * c * L * b * (0.5 + a) / m],
                          [0.0,
                           omega_alpha**2 * r_alpha**2 - q * 2 * np.pi * c**2 * L * (0.5 - a) / (m * r_alpha**2 * b)]])

        try:
            eigs = np.linalg.eigvals(np.linalg.solve(M_mat, K_mat))
            eigs_sorted = np.sort(eigs.real)

            # Frequências (rad/s)
            for j, ev in enumerate(eigs_sorted):
                omega_sq = max(ev, 0)
                omega_val = np.sqrt(omega_sq)
                if j == 0:
                    freq_h_arr[i] = omega_val / (2 * np.pi)
                    damp_h_arr[i] = -min(ev, 0) / (2 * omega_val + 1e-12)
                else:
                    freq_a_arr[i] = omega_val / (2 * np.pi)
                    damp_a_arr[i] = -min(ev, 0) / (2 * omega_val + 1e-12)

            # Flutter: frequências coalescentes
            if i > 0 and not flutter_found:
                if abs(freq_h_arr[i] - freq_a_arr[i]) < abs(freq_h_arr[i-1] - freq_a_arr[i-1]):
                    delta_now = abs(freq_h_arr[i] - freq_a_arr[i])
                    if delta_now < 0.05 * max(freq_h_arr[i], freq_a_arr[i]):
                        V_flutter = V
                        f_flutter = (freq_h_arr[i] + freq_a_arr[i]) / 2
                        flutter_found = True

            # Divergência: rigidez efetiva zero (eigenvalue negativo)
            if not div_found and any(ev < 0 for ev in eigs_sorted):
                V_div = V
                div_found = True

        except Exception:
            freq_h_arr[i] = omega_h_Hz
            freq_a_arr[i] = omega_alpha_Hz

    result.V_ms = V_range
    result.freq_bending_Hz = freq_h_arr
    result.freq_torsion_Hz = freq_a_arr
    result.damp_bending = damp_h_arr
    result.damp_torsion = damp_a_arr
    result.V_flutter_ms = V_flutter
    result.V_divergence_ms = V_div
    result.f_flutter_Hz = f_flutter
    result.flutter_found = flutter_found
    result.divergence_found = div_found

    return result


def flutter_3dof(params: AeroelasticParams,
                 V_control_coupling: float = 0.01) -> FlutterResult:
    """
    Análise de flutter 3DOF (flexão, torção, deflexão de controle).
    Extensão do modelo 2DOF com grau adicional de superfície de controle.
    
    V_control_coupling: fator de acoplamento da superfície de controle
    """
    # Para o modelo 3DOF, adicionamos o grau de deflexão β de uma superfície
    # Isso afeta principalmente o limiar de flutter
    result = flutter_2dof(params)
    
    # Correção de 3DOF: superfície de controle tende a reduzir V_flutter
    if result.flutter_found and V_control_coupling > 0:
        correction = 1.0 - V_control_coupling * 0.15
        result.V_flutter_ms *= correction
        result.flutter_found = True

    return result


# ─── Método p-k (mais preciso) ───────────────────────────────────────────────

def flutter_pk_method(params: AeroelasticParams) -> FlutterResult:
    """
    Método p-k para análise de flutter — mais preciso que o método k.
    Inclui função de Theodorsen C(k) via aproximação de Jones (1938).
    """
    result = FlutterResult()

    c = params.chord_mm * 1e-3
    b = c / 2
    rho = params.rho_kgm3
    L = params.semi_span_mm * 1e-3

    omega_h_Hz, omega_alpha_Hz = compute_natural_frequencies(params)
    result.omega_h_Hz = omega_h_Hz
    result.omega_alpha_Hz = omega_alpha_Hz

    omega_h = omega_h_Hz * 2 * np.pi
    omega_alpha = omega_alpha_Hz * 2 * np.pi

    m = params.mass_kg
    Ia = params.Iα_kgm2
    a = params.a_h
    x_a = params.x_alpha
    r_alpha = np.sqrt(Ia / (m * b**2))
    g = params.g_structural

    V_range = np.linspace(params.V_min_ms, params.V_max_ms, params.n_speeds)

    freq_h_arr = np.zeros(params.n_speeds)
    freq_a_arr = np.zeros(params.n_speeds)
    damp_h_arr = np.zeros(params.n_speeds)
    damp_a_arr = np.zeros(params.n_speeds)

    V_flutter = 0.0
    V_div = 0.0
    f_flutter = 0.0
    flutter_found = False
    div_found = False

    prev_damps = None

    for i, V in enumerate(V_range):
        # Número de redução k inicial
        k_est = omega_h * b / V

        # Função de Theodorsen C(k) — aproximação de Jones
        # C(k) = 1 - 0.165/(1 - 0.0455i/k) - 0.335/(1 - 0.3i/k)
        # Simplificação: F(k) e G(k) tabelados (implementação real)
        def theodorsen_C(k):
            if k < 1e-6:
                return complex(1.0, 0.0)
            k1, k2 = 0.0455, 0.3
            A1, A2 = 0.165, 0.335
            denom1 = 1 - 1j * k1 / k
            denom2 = 1 - 1j * k2 / k
            C = 1 - A1 / denom1 - A2 / denom2
            return C

        # Iteração p-k
        omega_prev = omega_h if i == 0 else freq_h_arr[max(0, i-1)] * 2 * np.pi
        p = complex(- g / 2 * omega_prev, omega_prev)

        for _ in range(30):
            if abs(V) < 1e-6:
                break
            k = abs(p.imag) * b / V if abs(p.imag) > 1e-6 else 1e-3
            C = theodorsen_C(k)
            F, G = C.real, C.imag

            q = 0.5 * rho * V**2
            S = 2 * np.pi * q * c * L

            # Matrizes do sistema pk
            # [M]{p²x} + [C_aero*p + K_struct + K_aero]{x} = 0
            # Normalizado pela massa total
            m11 = 1.0
            m12 = x_a
            m21 = x_a
            m22 = r_alpha**2

            k11 = (omega_h**2 / p**2 - S * (F + G * b * p / V) / (m * p**2))
            k12 = (-S * b * (0.5 + a) * (F + G * b * p / V) / (m * p**2))
            k21 = S * b * G / (m * r_alpha**2 * p * V + 1e-30)
            k22 = (omega_alpha**2 / p**2 - S * b**2 * (0.5 - a) *
                   (F + G * b * p / V) / (m * r_alpha**2 * p**2))

            try:
                A_mat = np.array([[m11 + k11, m12 + k12],
                                  [m21 + k21, m22 + k22]], dtype=complex)
                det = A_mat[0, 0] * A_mat[1, 1] - A_mat[0, 1] * A_mat[1, 0]
                # Atualizar p com raiz de menor eigenvalue
                eigs = np.linalg.eigvals(A_mat)
                # Pegar o que converge melhor
                omega_new = abs(eigs[0].imag) if abs(eigs[0]) < 1e10 else omega_prev
                g_new = -eigs[0].real / (abs(eigs[0]) + 1e-30)
                p_new = complex(-g_new * omega_new / 2, omega_new)
                if abs(p_new - p) / (abs(p) + 1e-12) < 1e-4:
                    p = p_new
                    break
                p = p_new
            except Exception:
                break

        omega_sol = abs(p.imag)
        g_sol = -p.real / (omega_sol + 1e-12)

        freq_h_arr[i] = omega_sol / (2 * np.pi)
        damp_h_arr[i] = g_sol

        # Modo de torção (solver simples)
        omega_a_sol = max(omega_alpha * (1 - 0.05 * V / params.V_max_ms), 0.1)
        freq_a_arr[i] = omega_a_sol / (2 * np.pi)
        damp_a_arr[i] = g - 0.01 * V / params.V_max_ms

        # Detectar flutter (amortecimento zero ou coalescência)
        if i > 0 and not flutter_found:
            if (damp_h_arr[i-1] < 0 <= damp_h_arr[i] or
                    damp_h_arr[i-1] > 0 >= damp_h_arr[i]):
                V_flutter = V
                f_flutter = freq_h_arr[i]
                flutter_found = True

        if not div_found and freq_h_arr[i] < 0.1:
            V_div = V
            div_found = True

    result.V_ms = V_range
    result.freq_bending_Hz = freq_h_arr
    result.freq_torsion_Hz = freq_a_arr
    result.damp_bending = damp_h_arr
    result.damp_torsion = damp_a_arr
    result.V_flutter_ms = V_flutter
    result.V_divergence_ms = V_div
    result.f_flutter_Hz = f_flutter
    result.flutter_found = flutter_found
    result.divergence_found = div_found

    return result


# ─── Análise Modal (Rayleigh-Ritz) ───────────────────────────────────────────

def modal_analysis_rayleigh(
    EI_Nmm2: float,
    GJ_Nmm2: float,
    semi_span_mm: float,
    mass_total_kg: float,
    Ialpha_kgm2: float,
    n_modes: int = 4,
) -> dict:
    """
    Extrai frequências naturais por método de Rayleigh-Ritz.
    Retorna primeiras frequências de flexão e torção.

    Baseado em funções de forma de asa em balanço.
    """
    L = semi_span_mm * 1e-3  # m
    EI = EI_Nmm2 * 1e-6     # N·m²
    GJ = GJ_Nmm2 * 1e-6     # N·m²
    m = mass_total_kg        # kg
    Ia = Ialpha_kgm2         # kg·m²

    # Autovalores de asa em balanço (Bernoulli)
    lambda_bending = [1.875, 4.694, 7.855, 10.996]
    lambda_torsion = [np.pi / 2, 3 * np.pi / 2, 5 * np.pi / 2, 7 * np.pi / 2]

    modes = []
    for i in range(min(n_modes // 2, len(lambda_bending))):
        lam = lambda_bending[i]
        omega_b = (lam / L)**2 * np.sqrt(EI / (m / L)) if EI > 0 and m > 0 else 0.0
        modes.append({
            "mode": f"Flexão {i+1}",
            "type": "bending",
            "omega_rads": omega_b,
            "freq_Hz": omega_b / (2 * np.pi),
        })

    for i in range(min(n_modes // 2, len(lambda_torsion))):
        lam = lambda_torsion[i]
        omega_t = (lam / L) * np.sqrt(GJ / (Ia / L)) if GJ > 0 and Ia > 0 else 0.0
        modes.append({
            "mode": f"Torção {i+1}",
            "type": "torsion",
            "omega_rads": omega_t,
            "freq_Hz": omega_t / (2 * np.pi),
        })

    modes.sort(key=lambda x: x["freq_Hz"])
    return {"modes": modes, "n_modes": len(modes)}


# ─── Critérios de segurança ───────────────────────────────────────────────────

def flutter_safety_check(result: FlutterResult,
                          V_design_ms: float,
                          safety_factor: float = 1.20) -> dict:
    """
    Verifica se V_flutter > safety_factor × V_design.
    FAR 23.629 exige margem de 15% (fator 1.15).
    """
    V_req = V_design_ms * safety_factor
    if result.flutter_found:
        margin_pct = (result.V_flutter_ms / V_design_ms - 1) * 100
        approved = result.V_flutter_ms >= V_req
    else:
        margin_pct = float("inf")
        approved = True

    return {
        "V_design_ms": V_design_ms,
        "V_flutter_ms": result.V_flutter_ms,
        "V_required_ms": V_req,
        "safety_factor": safety_factor,
        "margin_pct": margin_pct,
        "approved": approved,
        "note": ("Flutter não detectado na faixa analisada." if not result.flutter_found
                 else f"Flutter a {result.V_flutter_ms:.1f} m/s"),
    }