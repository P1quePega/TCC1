"""
mdo_optimizer.py — Otimização MDO Multi-Objetivo com Algoritmo Genético (NSGA-II)
Usa pymoo para variar: n_ribs, espaçamento, material, espessura
Objetivos: Minimizar massa total + tensão máxima
Constraints: σ ≤ σ_adm, δ_entelagem ≤ limite, V_flutter ≥ V_design × 1.15

Pipeline completo:
  1. Schrenk → Cargas por nervura
  2. Verificação de entelagem
  3. (Opcional) ANSYS → Tensão real
  4. Aeroelasticidade → Flutter check
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Any
import threading
import time


# ─── Variáveis de design ──────────────────────────────────────────────────────

@dataclass
class MDOVariable:
    """Uma variável de design do MDO."""
    name: str
    label: str
    unit: str
    lower: float
    upper: float
    is_integer: bool = False
    current: float = 0.0

    @property
    def bounds(self):
        return (self.lower, self.upper)


@dataclass
class MDOConstraint:
    """Uma constraint do MDO."""
    name: str
    label: str
    limit: float
    operator: str = "<="   # "<=" ou ">="
    enabled: bool = True
    current_value: float = 0.0

    def satisfied(self, value: float) -> bool:
        if self.operator == "<=":
            return value <= self.limit
        return value >= self.limit


@dataclass
class MDOConfig:
    """Configuração completa do MDO."""
    # Algoritmo
    pop_size: int = 40
    n_gen: int = 50
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    mutation_eta: float = 20.0
    seed: int = 42

    # Objetivos
    objective_mass: bool = True
    objective_stress: bool = True
    objective_cost: bool = False

    # Pipeline
    run_ansys: bool = False
    run_flutter: bool = True
    run_covering: bool = True

    # Batch mode
    batch_mode: bool = False
    batch_profiles: List[str] = field(default_factory=list)

    # Variáveis de design (defaults)
    variables: List[MDOVariable] = field(default_factory=lambda: [
        MDOVariable("n_ribs",      "Nº Nervuras",          "",    4,    20,  True,  10),
        MDOVariable("rib_spacing", "Esp. entre nervuras",  "mm",  40,   180, False, 80),
        MDOVariable("rib_thick",   "Esp. nervura",         "mm",  1.0,  6.0, False, 3.0),
        MDOVariable("spar_od",     "Diâm. longarina",      "mm",  8.0,  25.0,False, 12.0),
        MDOVariable("spar_wall",   "Parede longarina",     "mm",  0.5,  3.0, False, 1.0),
    ])

    # Constraints
    constraints: List[MDOConstraint] = field(default_factory=lambda: [
        MDOConstraint("stress_max",   "Tensão Von Mises",  5.0,  "<=", True),
        MDOConstraint("covering_def", "Deflexão Entelagem",0.005,"<=", True),
        MDOConstraint("v_flutter",    "V Flutter (m/s)",   15.0, ">=", True),
        MDOConstraint("total_mass",   "Massa total (g)",   2000, "<=", False),
    ])


@dataclass
class MDOIndividual:
    """Um indivíduo na população do GA."""
    genes: np.ndarray                # Variáveis de design
    objectives: np.ndarray = field(default_factory=lambda: np.zeros(2))
    constraint_violations: np.ndarray = field(default_factory=lambda: np.zeros(4))
    feasible: bool = False
    details: dict = field(default_factory=dict)
    generation: int = 0


@dataclass
class MDOGeneration:
    """Estado de uma geração do GA."""
    gen_number: int = 0
    population: List[MDOIndividual] = field(default_factory=list)
    pareto_front: List[MDOIndividual] = field(default_factory=list)
    best_mass: float = float("inf")
    best_stress: float = float("inf")
    min_constraint_viol: float = float("inf")
    timestamp: float = 0.0


@dataclass
class MDOResult:
    """Resultado completo da otimização MDO."""
    generations: List[MDOGeneration] = field(default_factory=list)
    pareto_front: List[MDOIndividual] = field(default_factory=list)
    best_individual: Optional[MDOIndividual] = None
    total_evaluations: int = 0
    elapsed_time_s: float = 0.0
    converged: bool = False
    batch_results: List[dict] = field(default_factory=list)


# ─── Função de avaliação ──────────────────────────────────────────────────────

class MDOEvaluator:
    """
    Avaliador do MDO: recebe genes e retorna objetivos + violações de constraints.
    Integra Schrenk, Entelagem, Wingbox e Aeroelasticidade.
    """

    def __init__(self, base_params: dict, config: MDOConfig,
                 material_name: str = "Balsa C-grain",
                 log_callback: Optional[Callable] = None):
        self.params = base_params
        self.config = config
        self.material_name = material_name
        self.log = log_callback or (lambda msg, nivel="INFO": None)
        self._lock = threading.Lock()

    def _decode_genes(self, genes: np.ndarray) -> dict:
        """Converte vetor de genes em parâmetros físicos."""
        vars_ = self.config.variables
        decoded = {}
        for i, v in enumerate(vars_):
            val = genes[i] if i < len(genes) else v.current
            if v.is_integer:
                val = int(round(val))
            decoded[v.name] = val
        return decoded

    def evaluate(self, genes: np.ndarray) -> MDOIndividual:
        """Avalia um indivíduo completo."""
        ind = MDOIndividual(genes=genes.copy())
        p = self._decode_genes(genes)
        base = self.params

        try:
            # 1. Schrenk — cargas por nervura
            from schrenk import WingGeometry, FlightCondition, discretize_rib_loads_matlab
            wing = WingGeometry(
                semi_span_mm=base.get("semi_span", 750),
                root_chord_mm=base.get("root_chord", 300),
                tip_chord_mm=base.get("tip_chord", 200),
            )
            flight = FlightCondition(
                velocity_ms=base.get("velocity", 15),
                aircraft_mass_kg=base.get("mass", 5),
                load_factor=base.get("load_factor", 4),
            )
            n_ribs = int(p.get("n_ribs", 10))
            rib_data = discretize_rib_loads_matlab(wing, flight, n_ribs)
            idx_c = rib_data["idx_crit"]
            pressao_crit = rib_data["pressure_per_rib"][idx_c]
            corda_crit = rib_data["chord_ribs"][idx_c]

            # 2. Estimativa de tensão (analítica — sem ANSYS)
            from materials import get_all_materials
            mat = get_all_materials().get(self.material_name)
            E = mat.E_MPa if mat else 3500
            t = p.get("rib_thick", 3.0)
            h_perf = 0.12 * corda_crit
            w = abs(pressao_crit) * corda_crit
            I = corda_crit * t**3 / 12
            M = w * (h_perf / 2)**2 / 2
            sigma_max = M * (t / 2) / I if I > 1e-10 else 999.0

            # 3. Massa da asa (simplificada)
            from weight_cg import (SparConfig, CoveringConfig, GlueConfig,
                                   generate_rib_masses_from_optimization, compute_weight_cg)
            rib_pos = np.linspace(0, wing.semi_span_mm, n_ribs)
            lam = wing.tip_chord_mm / wing.root_chord_mm
            chord_ribs = wing.root_chord_mm * (1 - rib_pos / wing.semi_span_mm * (1 - lam))
            dens = mat.density_kgm3 if mat else 160

            ribs = generate_rib_masses_from_optimization(
                n_ribs=n_ribs,
                rib_positions_mm=rib_pos,
                chord_at_ribs_mm=chord_ribs,
                area_casca_root_mm2=base.get("area_casca_root", 1200),
                area_otim_root_mm2=base.get("area_otim_root", 800),
                volume_fraction=0.5,
                thickness_mm=t,
                density_kgm3=dens,
                spar_position_pct=0.28,
                root_chord_mm=wing.root_chord_mm,
            )
            spar = SparConfig(
                outer_diameter_mm=p.get("spar_od", 12),
                wall_thickness_mm=p.get("spar_wall", 1),
                density_kgm3=1600,
                semi_span_mm=wing.semi_span_mm,
            )
            wc = compute_weight_cg(ribs, spar, CoveringConfig(), GlueConfig(),
                                   wing.semi_span_mm, wing.root_chord_mm, wing.tip_chord_mm)
            total_mass_g = wc.total_g

            # 4. Deflexão de entelagem
            covering_def = 0.0
            if self.config.run_covering:
                from covering import CoveringMaterial, check_covering
                cov_mat = CoveringMaterial()
                spacing = p.get("rib_spacing", wing.semi_span_mm / n_ribs)
                cov_res = check_covering(spacing, corda_crit, pressao_crit, cov_mat)
                covering_def = cov_res.deflection_chord_ratio

            # 5. Flutter (rápido, sem ANSYS)
            v_flutter = float("inf")
            if self.config.run_flutter:
                from aeroelasticity import AeroelasticParams, flutter_2dof
                from wingbox import SparProfile, analyze_wingbox
                from materials import get_all_materials
                cf_mat = get_all_materials().get("CFRP UD (0°)")
                spar_p = SparProfile(
                    profile_type="Tubular",
                    height_mm=p.get("spar_od", 12),
                    wall_mm=p.get("spar_wall", 1),
                    material=cf_mat,
                    length_mm=wing.semi_span_mm * 2,
                )
                wb = analyze_wingbox(wing.semi_span_mm, wing.root_chord_mm,
                                     wing.tip_chord_mm, spar_p,
                                     wing_mass_kg=flight.aircraft_mass_kg)
                ae_p = AeroelasticParams(
                    semi_span_mm=wing.semi_span_mm,
                    chord_mm=(wing.root_chord_mm + wing.tip_chord_mm) / 2,
                    mass_kg=flight.aircraft_mass_kg / 2,
                    EI_Nmm2=wb.EI_root,
                    GJ_Nmm2=wb.GJ_root,
                )
                fl = flutter_2dof(ae_p)
                v_flutter = fl.V_flutter_ms if fl.flutter_found else 999.0

            # Objetivos
            ind.objectives = np.array([total_mass_g, sigma_max])

            # Violações de constraints (g(x) ≤ 0 = satisfeito)
            consts = self.config.constraints
            viol = np.zeros(len(consts))
            for j, con in enumerate(consts):
                if not con.enabled:
                    continue
                if con.name == "stress_max":
                    viol[j] = max(0, sigma_max - con.limit)
                elif con.name == "covering_def":
                    viol[j] = max(0, covering_def - con.limit)
                elif con.name == "v_flutter":
                    viol[j] = max(0, con.limit - v_flutter)
                elif con.name == "total_mass":
                    viol[j] = max(0, total_mass_g - con.limit)

            ind.constraint_violations = viol
            ind.feasible = np.all(viol <= 0)
            ind.details = {
                "n_ribs": n_ribs,
                "mass_g": total_mass_g,
                "sigma_max": sigma_max,
                "covering_def": covering_def,
                "v_flutter": v_flutter,
                "pressao_crit": pressao_crit,
                "material": self.material_name,
            }

        except Exception as ex:
            ind.objectives = np.array([1e9, 1e9])
            ind.constraint_violations = np.ones(len(self.config.constraints)) * 1e6
            ind.feasible = False
            ind.details = {"error": str(ex)}

        return ind


# ─── Motor NSGA-II (sem dependência de pymoo como fallback) ──────────────────

def _nsga2_dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Retorna True se a domina b (Pareto)."""
    return np.all(a <= b) and np.any(a < b)


def _fast_nondominated_sort(population: List[MDOIndividual]) -> List[List[int]]:
    """Ordenação por não-dominância (NSGA-II fast-sort)."""
    n = len(population)
    S = [[] for _ in range(n)]
    n_dominated = np.zeros(n, dtype=int)
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _nsga2_dominates(population[i].objectives, population[j].objectives):
                S[i].append(j)
            elif _nsga2_dominates(population[j].objectives, population[i].objectives):
                n_dominated[i] += 1
        if n_dominated[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _crowding_distance(front_indices: list, population: List[MDOIndividual]) -> np.ndarray:
    """Calcula distância de crowding para indivíduos de uma frente."""
    n = len(front_indices)
    if n <= 2:
        return np.full(n, float("inf"))

    n_obj = len(population[0].objectives)
    dist = np.zeros(n)

    for m in range(n_obj):
        vals = [population[i].objectives[m] for i in front_indices]
        sorted_idx = np.argsort(vals)
        dist[sorted_idx[0]] = dist[sorted_idx[-1]] = float("inf")
        val_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
        if val_range < 1e-12:
            continue
        for k in range(1, n - 1):
            dist[sorted_idx[k]] += (vals[sorted_idx[k+1]] - vals[sorted_idx[k-1]]) / val_range

    return dist


def _sbx_crossover(p1: np.ndarray, p2: np.ndarray,
                    bounds: list, eta_c: float = 20.0) -> tuple:
    """Simulated Binary Crossover (SBX)."""
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if np.random.rand() > 0.5:
            continue
        lo, hi = bounds[i]
        if abs(p2[i] - p1[i]) < 1e-10:
            continue
        beta = 1 + 2 * min(p1[i] - lo, hi - p1[i]) / (abs(p2[i] - p1[i]) + 1e-12)
        alpha = 2 - beta**(-(eta_c + 1))
        u = np.random.rand()
        if u <= 1.0 / alpha:
            betaq = (u * alpha) ** (1 / (eta_c + 1))
        else:
            betaq = (1 / (2 - u * alpha)) ** (1 / (eta_c + 1))
        c1[i] = 0.5 * ((1 + betaq) * p1[i] + (1 - betaq) * p2[i])
        c2[i] = 0.5 * ((1 - betaq) * p1[i] + (1 + betaq) * p2[i])
        c1[i] = np.clip(c1[i], lo, hi)
        c2[i] = np.clip(c2[i], lo, hi)
    return c1, c2


def _polynomial_mutation(x: np.ndarray, bounds: list,
                          prob: float = 0.1, eta_m: float = 20.0) -> np.ndarray:
    """Polynomial Mutation."""
    x_mut = x.copy()
    for i in range(len(x)):
        if np.random.rand() > prob:
            continue
        lo, hi = bounds[i]
        delta = min(x[i] - lo, hi - x[i]) / (hi - lo + 1e-12)
        u = np.random.rand()
        if u < 0.5:
            delta_q = (2*u + (1-2*u)*(1-delta)**(eta_m+1))**(1/(eta_m+1)) - 1
        else:
            delta_q = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta)**(eta_m+1))**(1/(eta_m+1))
        x_mut[i] = np.clip(x[i] + delta_q * (hi - lo), lo, hi)
    return x_mut


class NSGAIIOptimizer:
    """
    NSGA-II implementado localmente (fallback quando pymoo não está disponível).
    Suporte a pymoo quando instalado.
    """

    def __init__(self, evaluator: MDOEvaluator, config: MDOConfig,
                 callback: Optional[Callable] = None):
        self.evaluator = evaluator
        self.config = config
        self.callback = callback  # callback(MDOGeneration) chamado a cada geração
        self._stop_flag = threading.Event()
        self.result = MDOResult()

    def stop(self):
        self._stop_flag.set()

    def _initial_population(self) -> List[MDOIndividual]:
        pop = []
        for _ in range(self.config.pop_size):
            genes = np.array([
                np.random.uniform(v.lower, v.upper)
                for v in self.config.variables
            ])
            ind = self.evaluator.evaluate(genes)
            pop.append(ind)
        return pop

    def run(self) -> MDOResult:
        """Executa o NSGA-II com pymoo (se disponível) ou implementação local."""
        try:
            return self._run_pymoo()
        except ImportError:
            return self._run_local()

    def _run_local(self) -> MDOResult:
        """Implementação local do NSGA-II."""
        t0 = time.time()
        bounds = [(v.lower, v.upper) for v in self.config.variables]
        pop = self._initial_population()
        self.result.total_evaluations = len(pop)

        for gen in range(self.config.n_gen):
            if self._stop_flag.is_set():
                break

            # Torneio + cruzamento + mutação
            offspring = []
            while len(offspring) < self.config.pop_size:
                idxs = np.random.choice(len(pop), 4, replace=False)
                p1 = pop[idxs[0]] if pop[idxs[0]].objectives.sum() < pop[idxs[1]].objectives.sum() else pop[idxs[1]]
                p2 = pop[idxs[2]] if pop[idxs[2]].objectives.sum() < pop[idxs[3]].objectives.sum() else pop[idxs[3]]

                if np.random.rand() < self.config.crossover_prob:
                    c1g, c2g = _sbx_crossover(p1.genes, p2.genes, bounds, self.config.mutation_eta)
                else:
                    c1g, c2g = p1.genes.copy(), p2.genes.copy()

                c1g = _polynomial_mutation(c1g, bounds, self.config.mutation_prob)
                c2g = _polynomial_mutation(c2g, bounds, self.config.mutation_prob)

                offspring.append(self.evaluator.evaluate(c1g))
                offspring.append(self.evaluator.evaluate(c2g))

            self.result.total_evaluations += len(offspring)

            # Combinação + seleção por frentes de Pareto
            combined = pop + offspring
            fronts = _fast_nondominated_sort(combined)
            next_pop = []
            for front in fronts:
                if len(next_pop) + len(front) <= self.config.pop_size:
                    next_pop.extend([combined[i] for i in front])
                else:
                    crowding = _crowding_distance(front, combined)
                    sorted_f = sorted(zip(crowding, front), key=lambda x: -x[0])
                    remaining = self.config.pop_size - len(next_pop)
                    next_pop.extend([combined[i] for _, i in sorted_f[:remaining]])
                    break

            pop = next_pop

            # Registrar geração
            pareto_inds = [combined[i] for i in fronts[0]] if fronts else []
            for ind in pareto_inds:
                ind.generation = gen

            feasible = [p for p in pop if p.feasible]
            best_mass = min((p.objectives[0] for p in feasible), default=float("inf"))
            best_stress = min((p.objectives[1] for p in feasible), default=float("inf"))

            gen_obj = MDOGeneration(
                gen_number=gen,
                population=list(pop),
                pareto_front=pareto_inds,
                best_mass=best_mass,
                best_stress=best_stress,
                timestamp=time.time() - t0,
            )
            self.result.generations.append(gen_obj)

            if self.callback:
                self.callback(gen_obj)

        # Resultado final
        all_inds = [ind for gen in self.result.generations for ind in gen.pareto_front]
        feasible = [i for i in all_inds if i.feasible]
        if feasible:
            self.result.pareto_front = feasible
            self.result.best_individual = min(feasible,
                key=lambda x: x.objectives[0] + x.objectives[1] * 0.5)
        self.result.elapsed_time_s = time.time() - t0
        self.result.converged = not self._stop_flag.is_set()
        return self.result

    def _run_pymoo(self) -> MDOResult:
        """Roda via pymoo quando instalado."""
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.optimize import minimize
        from pymoo.core.callback import Callback

        n_var = len(self.config.variables)
        n_obj = 2
        n_con = sum(1 for c in self.config.constraints if c.enabled)

        xl = np.array([v.lower for v in self.config.variables])
        xu = np.array([v.upper for v in self.config.variables])

        evaluator = self.evaluator
        config = self.config
        result_obj = self.result
        stop_flag = self._stop_flag  # 1. MOVIDO PARA CIMA
        

        class WingProblem(Problem):
            def __init__(self):
                super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_con,
                                 xl=xl, xu=xu)

            def _evaluate(self, X, out, *args, **kwargs):
                objs = np.zeros((len(X), n_obj))
                cvs = np.zeros((len(X), n_con))
                for i, genes in enumerate(X):
                    if stop_flag.is_set():  # 2. CORRIGIDO: usando a variável do escopo externo
                        break
                    ind = evaluator.evaluate(genes)
                    objs[i] = ind.objectives
                    active = [c for c in config.constraints if c.enabled]
                    for j, con in enumerate(active):
                        cvs[i, j] = ind.constraint_violations[j] if j < len(ind.constraint_violations) else 0
                out["F"] = objs
                out["G"] = cvs

        stop_flag = self._stop_flag
        gui_callback = self.callback

        class GACallback(Callback):
            def notify(self, algorithm):
                if stop_flag.is_set():
                    algorithm.termination.force_termination = True
                    return
                gen = algorithm.n_gen
                X = algorithm.pop.get("X")
                F = algorithm.pop.get("F")
                feasible_mask = algorithm.pop.get("feasible").flatten()
                
                gen_obj = MDOGeneration(gen_number=gen)
                if F is not None and len(F) > 0:
                    gen_obj.best_mass = float(np.min(F[:, 0]))
                    gen_obj.best_stress = float(np.min(F[:, 1]))

                result_obj.generations.append(gen_obj)
                if self.callback:
                    self.callback(gen_obj)

        t0 = time.time()
        algorithm = NSGA2(
            pop_size=config.pop_size,
            crossover=SBX(prob=config.crossover_prob, eta=config.mutation_eta),
            mutation=PM(prob=config.mutation_prob, eta=config.mutation_eta),
        )

        res = minimize(
            WingProblem(),
            algorithm,
            ("n_gen", config.n_gen),
            callback=GACallback(),
            seed=config.seed,
            verbose=False,
        )

        self.result.elapsed_time_s = time.time() - t0
        self.result.converged = True

        if res.X is not None:
            for genes in res.X:
                ind = evaluator.evaluate(genes)
                self.result.pareto_front.append(ind)
            feasible = [i for i in self.result.pareto_front if i.feasible]
            if feasible:
                self.result.best_individual = min(feasible, key=lambda x: x.objectives[0])

        return self.result


# ─── Batch Processing ─────────────────────────────────────────────────────────

class BatchProcessor:
    """
    Modo Batch: varre perfis aerodinâmicos e retorna relatório comparativo.
    """

    def __init__(self, base_params: dict, config: MDOConfig,
                 profiles: List[str],
                 log_callback: Optional[Callable] = None,
                 progress_callback: Optional[Callable] = None):
        self.base_params = base_params
        self.config = config
        self.profiles = profiles
        self.log = log_callback or (lambda msg, nivel="INFO": None)
        self.progress = progress_callback or (lambda pct, label: None)
        self._stop_flag = threading.Event()

    def stop(self):
        self._stop_flag.set()

    def run(self) -> List[dict]:
        results = []
        n = len(self.profiles)

        for idx, profile_name in enumerate(self.profiles):
            if self._stop_flag.is_set():
                break

            self.progress(int(idx / n * 100), f"Avaliando: {profile_name}")
            self.log(f"\n{'='*50}", "SECTION")
            self.log(f"PERFIL: {profile_name}", "SECTION")
            self.log(f"{'='*50}", "SECTION")

            try:
                # Rodar GA rápido (poucas gerações para batch)
                quick_config = MDOConfig(
                    pop_size=max(10, self.config.pop_size // 2),
                    n_gen=max(10, self.config.n_gen // 4),
                )
                params = dict(self.base_params)
                params["airfoil"] = profile_name

                evaluator = MDOEvaluator(params, quick_config,
                                         log_callback=self.log)
                optimizer = NSGAIIOptimizer(evaluator, quick_config)
                mdo_result = optimizer._run_local()

                best = mdo_result.best_individual
                result_entry = {
                    "profile": profile_name,
                    "best_mass_g": best.objectives[0] if best else float("inf"),
                    "best_stress_MPa": best.objectives[1] if best else float("inf"),
                    "feasible": best.feasible if best else False,
                    "n_ribs": best.details.get("n_ribs", 0) if best else 0,
                    "details": best.details if best else {},
                }
                results.append(result_entry)
                self.log(f"  → Massa mínima: {result_entry['best_mass_g']:.1f} g", "OK")
                self.log(f"  → Tensão:  {result_entry['best_stress_MPa']:.2f} MPa", "OK")
                self.log(f"  → Factível: {result_entry['feasible']}", "OK")

            except Exception as ex:
                self.log(f"Erro no perfil {profile_name}: {ex}", "ERROR")
                results.append({
                    "profile": profile_name,
                    "best_mass_g": float("inf"),
                    "best_stress_MPa": float("inf"),
                    "feasible": False,
                    "error": str(ex),
                })

        # Ordenar por massa (menor primeiro)
        results.sort(key=lambda r: r["best_mass_g"])
        self.progress(100, "Batch concluído!")
        return results