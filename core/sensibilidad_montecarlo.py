"""
Módulo de Incertidumbre, Barrido de Parámetros y Simulación Monte Carlo.
Evalúa la propagación de incertidumbre en los parámetros adoptados (sin datos de campo).
Incluye:
- Barrido 1D de parámetros (e, dt_imp, m_t, u, m_c)
- Simulación Monte Carlo (500 a 2000 corridas con semilla fija y percentiles P5, P50, P95)
- Mapa bidimensional de apertura (f vs u) con isolínea de apertura N_tope = 0
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
from .parameters import SimulatorConfig, RejillaParams, TroncoParams, ImpactoParams
from .rejilla import calcular_estatica_rejilla, calcular_f_apertura
from .impacto import calcular_impacto_2x2
from .tronco import simular_tronco_rk4


@dataclass
class MonteCarloSummary:
    """Resumen estadístico de la simulación Monte Carlo."""
    n_corridas: int
    semilla: int
    # Percentiles P5, P50, P95 y media para N_tope
    N_tope_p5: float
    N_tope_p50: float
    N_tope_p95: float
    N_tope_media: float
    N_tope_std: float
    # Percentiles para Fuerza de impacto promedio
    F_prom_p5: float
    F_prom_p50: float
    F_prom_p95: float
    F_prom_media: float
    # Percentiles para Deflexión angular máxima
    dtheta_p5: float
    dtheta_p50: float
    dtheta_p95: float
    dtheta_media: float
    # Probabilidad de apertura
    prob_apertura_pct: float
    # Datos crudos para histogramas
    N_topes: np.ndarray
    F_proms: np.ndarray
    dthetas_deg: np.ndarray


def barrido_sensibilidad(
    config: SimulatorConfig,
    n_puntos: int = 25
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Realiza un barrido unidimensional de parámetros clave alrededor de sus valores adoptados.
    """
    rej_p = config.rejilla
    tro_p = config.tronco
    imp_p = config.impacto
    f_base = config.flujo.f
    u_base = rej_p.u_corriente

    # 1. Barrido de e entre 0.0 y 0.50
    e_vals = np.linspace(0.0, 0.50, n_puntos)
    F_prom_vs_e = []
    dtheta_vs_e = []
    for e in e_vals:
        res_imp = calcular_impacto_2x2(tro_p, rej_p, imp_p, v_impacto=1.899, e_override=e)
        F_prom_vs_e.append(res_imp.F_prom_art)
        dtheta_vs_e.append(res_imp.dtheta_max_deg)

    # 2. Barrido de dt_imp entre 0.03 y 0.15 s
    dt_vals = np.linspace(0.03, 0.15, n_puntos)
    F_prom_vs_dt = []
    for dt_i in dt_vals:
        imp_temp = ImpactoParams(y_imp=imp_p.y_imp, dt_imp=dt_i, e=imp_p.e)
        res_imp = calcular_impacto_2x2(tro_p, rej_p, imp_temp, v_impacto=1.899)
        F_prom_vs_dt.append(res_imp.F_prom_art)

    # 3. Barrido de masa de contrapeso m_c entre 100 y 400 kg
    mc_vals = np.linspace(100.0, 400.0, n_puntos)
    N_tope_vs_mc = []
    f_ap_vs_mc = []
    for mc in mc_vals:
        rej_temp = RejillaParams(
            rho_s=rej_p.rho_s,
            theta0_deg=rej_p.theta0_deg,
            H_eje=rej_p.H_eje,
            h_agua=rej_p.h_agua,
            b_canal=rej_p.b_canal,
            r_c=rej_p.r_c,
            m_brazo=rej_p.m_brazo,
            m_c=mc,
            C_D_r=rej_p.C_D_r,
            u_corriente=u_base
        )
        res_rej = calcular_estatica_rejilla(rej_temp, f=f_base, u_corriente=u_base)
        N_tope_vs_mc.append(res_rej.N_tope)
        f_ap_vs_mc.append(res_rej.f_ap)

    # 4. Barrido de velocidad de corriente u entre 0.5 y 2.5 m/s
    u_vals = np.linspace(0.5, 2.5, n_puntos)
    N_tope_vs_u = []
    for u in u_vals:
        res_rej = calcular_estatica_rejilla(rej_p, f=f_base, u_corriente=u)
        N_tope_vs_u.append(res_rej.N_tope)

    return {
        "barrido_e": {
            "valores": e_vals,
            "F_prom": np.array(F_prom_vs_e),
            "dtheta_deg": np.array(dtheta_vs_e)
        },
        "barrido_dt": {
            "valores": dt_vals,
            "F_prom": np.array(F_prom_vs_dt)
        },
        "barrido_mc": {
            "valores": mc_vals,
            "N_tope": np.array(N_tope_vs_mc),
            "f_ap": np.array(f_ap_vs_mc)
        },
        "barrido_u": {
            "valores": u_vals,
            "N_tope": np.array(N_tope_vs_u)
        }
    }


def simulacion_montecarlo(
    config: SimulatorConfig,
    n_corridas: int = 1000,
    seed: int = 42
) -> MonteCarloSummary:
    """
    Ejecuta un análisis de incertidumbre Monte Carlo con semilla fijada para reproducibilidad.
    """
    rng = np.random.default_rng(seed)
    rej_p = config.rejilla
    tro_p = config.tronco
    imp_p = config.impacto

    # Distribuciones de incertidumbre en los parámetros adoptados:
    e_samples = rng.uniform(0.00, 0.40, n_corridas)
    dt_samples = rng.uniform(0.04, 0.12, n_corridas)
    mc_samples = rng.uniform(200.0, 300.0, n_corridas)
    u_samples = rng.uniform(0.80, 1.40, n_corridas)
    f_samples = rng.uniform(0.15, 0.45, n_corridas)
    m_t_samples = rng.uniform(45.0, 70.0, n_corridas)

    N_topes = np.zeros(n_corridas)
    F_proms = np.zeros(n_corridas)
    dthetas = np.zeros(n_corridas)
    abiertas = 0

    for i in range(n_corridas):
        # Crear instancias locales
        r_temp = RejillaParams(
            rho_s=rej_p.rho_s,
            theta0_deg=rej_p.theta0_deg,
            H_eje=rej_p.H_eje,
            h_agua=rej_p.h_agua,
            b_canal=rej_p.b_canal,
            r_c=rej_p.r_c,
            m_brazo=rej_p.m_brazo,
            m_c=mc_samples[i],
            C_D_r=rej_p.C_D_r,
            u_corriente=u_samples[i]
        )
        t_temp = TroncoParams(
            rho_mad=tro_p.rho_mad,
            D_t=tro_p.D_t,
            L_t=tro_p.L_t,
            C_D_t=tro_p.C_D_t,
            A_sub_ratio=tro_p.A_sub_ratio,
            v0=tro_p.v0,
            u_corriente=u_samples[i]
        )
        # Ajustar masa del tronco
        v_t_est = m_t_samples[i] / tro_p.rho_mad
        
        i_temp = ImpactoParams(
            y_imp=imp_p.y_imp,
            dt_imp=dt_samples[i],
            e=e_samples[i]
        )

        res_rej = calcular_estatica_rejilla(r_temp, f=f_samples[i], u_corriente=u_samples[i])
        N_topes[i] = res_rej.N_tope
        if res_rej.N_tope <= 0.0:
            abiertas += 1

        # Estimación rápida de impacto a v_aprox = 1.90 m/s
        res_imp = calcular_impacto_2x2(t_temp, r_temp, i_temp, v_impacto=1.899)
        F_proms[i] = res_imp.F_prom_art
        dthetas[i] = res_imp.dtheta_max_deg

    prob_ap = (abiertas / n_corridas) * 100.0

    return MonteCarloSummary(
        n_corridas=n_corridas,
        semilla=seed,
        N_tope_p5=float(np.percentile(N_topes, 5)),
        N_tope_p50=float(np.percentile(N_topes, 50)),
        N_tope_p95=float(np.percentile(N_topes, 95)),
        N_tope_media=float(np.mean(N_topes)),
        N_tope_std=float(np.std(N_topes)),
        F_prom_p5=float(np.percentile(F_proms, 5)),
        F_prom_p50=float(np.percentile(F_proms, 50)),
        F_prom_p95=float(np.percentile(F_proms, 95)),
        F_prom_media=float(np.mean(F_proms)),
        dtheta_p5=float(np.percentile(dthetas, 5)),
        dtheta_p50=float(np.percentile(dthetas, 50)),
        dtheta_p95=float(np.percentile(dthetas, 95)),
        dtheta_media=float(np.mean(dthetas)),
        prob_apertura_pct=prob_ap,
        N_topes=N_topes,
        F_proms=F_proms,
        dthetas_deg=dthetas
    )


def mapa_apertura(
    params: RejillaParams,
    f_res: int = 60,
    u_res: int = 60
) -> Dict[str, Any]:
    """
    Calcula una malla bidimensional (f, u) para delimitar la frontera de apertura pasiva N_tope = 0.
    """
    f_vec = np.linspace(0.0, 0.85, f_res)
    u_vec = np.linspace(0.4, 2.5, u_res)
    F_grid, U_grid = np.meshgrid(f_vec, u_vec)

    N_tope_grid = np.zeros_like(F_grid)
    for i in range(u_res):
        u_val = u_vec[i]
        for j in range(f_res):
            f_val = f_vec[j]
            res = calcular_estatica_rejilla(params, f=f_val, u_corriente=u_val)
            N_tope_grid[i, j] = res.N_tope

    # Frontera crítica f_ap(u)
    f_ap_curve = []
    for u_val in u_vec:
        f_ap_curve.append(calcular_f_apertura(params, u_corriente=u_val))

    return {
        "f_vec": f_vec,
        "u_vec": u_vec,
        "F_grid": F_grid,
        "U_grid": U_grid,
        "N_tope_grid": N_tope_grid,
        "f_ap_curve": np.array(f_ap_curve)
    }
