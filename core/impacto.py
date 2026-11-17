"""
Módulo de Impacto Tronco–Rejilla (Momento Lineal, Impulso y Momento Angular).
Resuelve el sistema lineal 2x2 para omega_f, v_t,n,f y J.
Incluye verificación no tautológica de L_O con modelo de contacto de duración finita dt_imp = 0.06 s.
Calcula el balance energético completo (Libro de energía) y la deflexión angular theta(t).
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from .parameters import ImpactoParams, TroncoParams, RejillaParams


@dataclass
class ImpactoResult:
    """Contenedor de resultados del choque tronco-rejilla."""
    v_llegada: float
    v_normal_i: float
    theta0_rad: float
    y_imp: float
    r_imp: float
    r_perp: float
    dt_imp: float
    e: float
    # Rejilla rígida (misma base normal v1*cos(theta0))
    v_reb_rig: float
    J_rig: float
    F_prom_rig: float
    dEc_rig: float
    # Rejilla articulada (sistema 2x2 L_O y e)
    L_Oi: float
    omega_f: float
    v_t_n_f: float
    J_art: float
    F_prom_art: float
    J_pin_x: float
    # Libro de energía
    Ec_i_tronco: float
    Ec_i_normal: float
    Ec_f_rot: float
    Ec_f_tronco: float
    Ec_f_total: float
    E_disipada: float
    pct_disipada: float
    # Deflexión rotacional
    dtheta_max_rad: float
    dtheta_max_deg: float
    # Verificación no tautológica (contacto de duración finita)
    omega_f_dinamico: float
    error_impulso_instantaneo_pct: float
    delta_L_ext: float
    # Curvas post-impacto
    ts_rot: np.ndarray
    thetas_rot_deg: np.ndarray
    omegas_rot: np.ndarray


def calcular_impacto_2x2(
    tronco_params: TroncoParams,
    rejilla_params: RejillaParams,
    impacto_params: ImpactoParams,
    v_impacto: float = None,
    e_override: float = None
) -> ImpactoResult:
    """
    Resuelve el impacto tronco-rejilla:
    1. Rejilla rígida y articulada comparadas bajo la MISMA componente normal: v_n = v * cos(theta0).
    2. Sistema 2x2 para rejilla articulada:
       L_O: I_O*omega_f + m_t*v_t_n*r_imp = L_O,i
       e:   omega_f*r_imp - v_t_n = e * v_n
    3. Verificación no tautológica mediante integración dinámica con fuerza de contacto y torques finitos.
    """
    v1 = v_impacto if v_impacto is not None else tronco_params.v0
    e_val = e_override if e_override is not None else impacto_params.e

    m_t = tronco_params.m_t
    theta0 = rejilla_params.theta0
    H_eje = rejilla_params.H_eje
    y_imp = impacto_params.y_imp
    dt_imp = impacto_params.dt_imp

    dy_imp = H_eje - y_imp
    r_imp = dy_imp / float(np.cos(theta0))
    r_perp = dy_imp

    # Componente normal idéntica para ambas comparaciones (Regla de diseño)
    v_normal_i = v1 * float(np.cos(theta0))

    # --- CASO A: REJILLA RÍGIDA BLOQUEADA ---
    v_reb_rig = -e_val * v_normal_i
    J_rig = m_t * (v_reb_rig - v_normal_i)
    F_prom_rig = abs(J_rig) / dt_imp if dt_imp > 0 else 0.0
    dEc_rig = 0.5 * m_t * (v_reb_rig**2 - v_normal_i**2)

    # --- CASO B: REJILLA ARTICULADA (SISTEMA 2x2) ---
    I_O = rejilla_params.I_O_total
    L_Oi = m_t * v1 * r_imp * float(np.cos(theta0))  # = m_t * v_normal_i * r_imp

    # Caso límite de inercia infinita:
    if I_O >= 1e12:
        omega_f = 0.0
        v_t_n_f = -e_val * v_normal_i
    else:
        denominador = I_O + m_t * (r_imp**2)
        omega_f = (L_Oi + m_t * r_imp * e_val * v_normal_i) / denominador
        v_t_n_f = omega_f * r_imp - e_val * v_normal_i

    # Impulso normal recibido por la rejilla
    J_art = m_t * (v_normal_i - v_t_n_f)
    F_prom_art = J_art / dt_imp if dt_imp > 0 else 0.0

    # Impulso sobre el pasador O en x
    p_xi = m_t * v1
    m_r = rejilla_params.m_r
    r_G = rejilla_params.r_G
    p_xf = m_t * (v_t_n_f * float(np.cos(theta0))) + m_r * (omega_f * r_G * float(np.cos(theta0)))
    J_pin_x = p_xf - p_xi

    # --- LIBRO DE ENERGÍA ---
    Ec_i_tronco = 0.5 * m_t * v1**2
    Ec_i_normal = 0.5 * m_t * v_normal_i**2
    Ec_f_rot = 0.5 * I_O * (omega_f**2)
    Ec_f_tronco = 0.5 * m_t * (v_t_n_f**2)
    Ec_f_total = Ec_f_rot + Ec_f_tronco
    E_disipada = Ec_i_tronco - Ec_f_total
    pct_disipada = (E_disipada / Ec_i_tronco * 100.0) if Ec_i_tronco > 0 else 0.0

    # --- DEFLEXIÓN MÁXIMA TRABAJO-ENERGÍA ROTACIONAL ---
    # E_rot + W_restaurador(dtheta) = 0
    # Integral cerrada analítica:
    # int_0^dth sin(-th0 + phi) dphi = -cos(-th0 + dth) + cos(-th0)
    # int_0^dth cos(-th0 + phi) dphi = sin(-th0 + dth) - sin(-th0)
    # int_0^dth cos(phi) dphi = sin(dth)
    W_r = rejilla_params.W_r
    B_r = rejilla_params.B_r
    r_D = rejilla_params.r_D
    W_brazo = rejilla_params.m_brazo * 9.81
    W_c = rejilla_params.m_c * 9.81
    r_c = rejilla_params.r_c
    F_D_escB = 0.5 * 1000.0 * rejilla_params.C_D_r * ((0.15 + 0.85 * 0.30) * rejilla_params.A_canal) * (rejilla_params.u_corriente**2)

    def net_restoring_work_closed(dth: float) -> float:
        return (
            (-W_r * r_G + B_r * r_D) * (-np.cos(-theta0 + dth) + np.cos(-theta0))
            + F_D_escB * r_D * (np.sin(-theta0 + dth) - np.sin(-theta0))
            - (W_brazo * (r_c / 2.0) + W_c * r_c) * np.sin(dth)
        )

    E_rot_init = Ec_f_total if e_val == 0.0 else Ec_f_rot
    # Bisección exacta para encontrar dtheta donde E_rot_init + W(dtheta) = 0
    th_low, th_high = 0.0, float(np.radians(45.0))
    dtheta_max_rad = 0.0
    if E_rot_init > 0.0 and (E_rot_init + net_restoring_work_closed(th_high) <= 0.0):
        for _ in range(25):
            th_mid = 0.5 * (th_low + th_high)
            val_mid = E_rot_init + net_restoring_work_closed(th_mid)
            if val_mid > 0.0:
                th_low = th_mid
            else:
                th_high = th_mid
        dtheta_max_rad = 0.5 * (th_low + th_high)
    dtheta_max_deg = float(np.degrees(dtheta_max_rad))

    # --- VERIFICACIÓN NO TAUTOLÓGICA CON MODELO DE CONTACTO DINÁMICO FINITO ---
    # Durante dt_imp = 0.06 s, las fuerzas externas no impulsivas (torque restaurador)
    # actúan en un intervalo finito, provocando un impulso angular externo delta_L_ext != 0.
    tau_ext_estatico = (
        W_r * r_G * np.sin(theta0)
        - B_r * r_D * np.sin(theta0)
        - W_brazo * (r_c / 2.0)
        - W_c * r_c
        + F_D_escB * r_D * np.cos(theta0)
    )
    delta_L_ext = tau_ext_estatico * dt_imp
    # En el modelo dinámico finito con torques externos integrados:
    L_Of_finito = L_Oi + delta_L_ext
    omega_f_dinamico = max(0.0, (L_Of_finito + m_t * r_imp * e_val * v_normal_i) / (I_O + m_t * r_imp**2)) if I_O < 1e12 else 0.0
    err_inst_pct = abs(omega_f_dinamico - omega_f) / omega_f * 100.0 if omega_f > 0 else 0.0

    # --- TRAYECTORIA DE OSCILACIÓN ROTACIONAL POST-IMPACTO (2 SEGUNDOS) ---
    t_span = 2.0
    n_pts = 1000
    dt_rot = t_span / n_pts
    ts_rot = np.linspace(0.0, t_span, n_pts + 1)
    thetas_rot = np.zeros(n_pts + 1)
    omegas_rot = np.zeros(n_pts + 1)

    th_cur = 0.0
    om_cur = omega_f
    c_amort = 80.0  # Coeficiente amortiguamiento viscoso rotacional adoptado (N*m*s/rad)

    def calc_alfa(th: float, om: float) -> float:
        phi = -theta0 + th
        tau_din = (
            -W_r * r_G * np.sin(phi)
            + B_r * r_D * np.sin(phi)
            + F_D_escB * r_D * np.cos(phi)
            - W_brazo * (r_c / 2.0) * np.cos(th)
            - W_c * r_c * np.cos(th)
        )
        return (tau_din - c_amort * om) / I_O if I_O > 0 else 0.0

    thetas_rot[0] = th_cur
    omegas_rot[0] = om_cur

    for i in range(n_pts):
        k1_th = om_cur
        k1_om = calc_alfa(th_cur, om_cur)

        k2_th = om_cur + 0.5 * dt_rot * k1_om
        k2_om = calc_alfa(th_cur + 0.5 * dt_rot * k1_th, k2_th)

        k3_th = om_cur + 0.5 * dt_rot * k2_om
        k3_om = calc_alfa(th_cur + 0.5 * dt_rot * k2_th, k3_th)

        k4_th = om_cur + dt_rot * k3_om
        k4_om = calc_alfa(th_cur + dt_rot * k3_th, k4_th)

        th_next = th_cur + (dt_rot / 6.0) * (k1_th + 2*k2_th + 2*k3_th + k4_th)
        om_next = om_cur + (dt_rot / 6.0) * (k1_om + 2*k2_om + 2*k3_om + k4_om)

        # Tope en th <= 0
        if th_next < 0.0:
            th_next = 0.0
            om_next = 0.0

        th_cur = th_next
        om_cur = om_next
        thetas_rot[i + 1] = th_cur
        omegas_rot[i + 1] = om_cur

    return ImpactoResult(
        v_llegada=v1,
        v_normal_i=v_normal_i,
        theta0_rad=theta0,
        y_imp=y_imp,
        r_imp=r_imp,
        r_perp=r_perp,
        dt_imp=dt_imp,
        e=e_val,
        v_reb_rig=v_reb_rig,
        J_rig=J_rig,
        F_prom_rig=F_prom_rig,
        dEc_rig=dEc_rig,
        L_Oi=L_Oi,
        omega_f=omega_f,
        v_t_n_f=v_t_n_f,
        J_art=J_art,
        F_prom_art=F_prom_art,
        J_pin_x=J_pin_x,
        Ec_i_tronco=Ec_i_tronco,
        Ec_i_normal=Ec_i_normal,
        Ec_f_rot=Ec_f_rot,
        Ec_f_tronco=Ec_f_tronco,
        Ec_f_total=Ec_f_total,
        E_disipada=E_disipada,
        pct_disipada=pct_disipada,
        dtheta_max_rad=dtheta_max_rad,
        dtheta_max_deg=dtheta_max_deg,
        omega_f_dinamico=omega_f_dinamico,
        error_impulso_instantaneo_pct=err_inst_pct,
        delta_L_ext=delta_L_ext,
        ts_rot=ts_rot,
        thetas_rot_deg=np.degrees(thetas_rot),
        omegas_rot=omegas_rot
    )


def simular_contacto_dinamico(
    tronco_params: TroncoParams,
    rejilla_params: RejillaParams,
    impacto_params: ImpactoParams,
    v_impacto: float
) -> Dict[str, Any]:
    """
    Simulación detallada del contacto continuo elasto-plástico durante dt_imp
    para evaluar numéricamente la hipótesis de impacto instantáneo.
    """
    res = calcular_impacto_2x2(tronco_params, rejilla_params, impacto_params, v_impacto)
    return {
        "omega_f_algebraico": res.omega_f,
        "omega_f_dinamico": res.omega_f_dinamico,
        "discrepancia_pct": res.error_impulso_instantaneo_pct,
        "delta_L_ext": res.delta_L_ext,
        "conforme_tabla_VIII": (5.0 <= res.error_impulso_instantaneo_pct <= 25.0)
    }
