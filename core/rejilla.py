"""
Módulo de la Rejilla Articulada Contrapesada (cuerpo rígido, estática y rotación).
Calcula masa, inercia, torques, N_tope, reacciones en articulación y fracción de apertura pasiva f_ap.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np
from .parameters import RejillaParams


@dataclass
class RejillaResult:
    """Contenedor de resultados del análisis estático y dinámico de la rejilla."""
    f: float
    u_corriente: float
    L_r: float
    m_r: float
    W_r: float
    B_r: float
    r_G: float
    r_D: float
    r_c: float
    m_brazo: float
    m_c: float
    W_brazo: float
    W_c: float
    I_rej_O: float
    I_brazo_O: float
    I_c_O: float
    I_O_total: float
    A_p: float
    F_D: float
    tau_W_r: float
    tau_B_r: float
    tau_brazo: float
    tau_cw: float
    tau_D: float
    tau_cierre_neto: float
    N_tope: float
    Ox: float
    Oy: float
    O_mag: float
    f_ap: float
    estado: str               # "CERRADA", "CERCA DE APERTURA", "ABIERTA"
    color_semaforo: str       # "green", "yellow", "red"
    porcentaje_tope_f0: float


def calcular_f_apertura(params: RejillaParams, u_corriente: float = None) -> float:
    """
    Calcula algebraicamente la fracción de obstrucción crítica f_ap donde N_tope = 0.
    """
    u_w = u_corriente if u_corriente is not None else params.u_corriente
    if u_w <= 0:
        return 1.0

    W_r = params.W_r
    B_r = params.B_r
    r_G = params.r_G
    r_D = params.r_D
    theta0 = params.theta0
    r_c = params.r_c
    W_brazo = params.m_brazo * 9.81
    W_c = params.m_c * 9.81
    A_canal = params.A_canal
    rho_w = 1000.0
    C_D_r = params.C_D_r

    tau_W_r = W_r * r_G * np.sin(theta0)
    tau_B_r = -B_r * r_D * np.sin(theta0)
    tau_brazo = -W_brazo * (r_c / 2.0)
    tau_cw = -W_c * r_c

    # tau_D_crit = -(tau_W_r + tau_B_r + tau_brazo + tau_cw)
    tau_D_crit = -(tau_W_r + tau_B_r + tau_brazo + tau_cw)
    denominador = 0.5 * rho_w * C_D_r * (u_w**2) * r_D * np.cos(theta0)
    if denominador <= 0:
        return 1.0

    A_p_crit = tau_D_crit / denominador
    # A_p = (0.15 + 0.85 * f) * A_canal
    f_ap = (A_p_crit / A_canal - 0.15) / 0.85
    return float(np.clip(f_ap, 0.0, 1.0))


def calcular_estatica_rejilla(
    params: RejillaParams,
    f: float,
    u_corriente: float = None
) -> RejillaResult:
    """
    Evalúa el equilibrio estático de momentos y fuerzas sobre la rejilla articulada.
    """
    u_w = u_corriente if u_corriente is not None else params.u_corriente
    L_r = params.L_r
    m_r = params.m_r
    W_r = params.W_r
    B_r = params.B_r
    r_G = params.r_G
    r_D = params.r_D
    theta0 = params.theta0
    r_c = params.r_c
    m_brazo = params.m_brazo
    m_c = params.m_c
    W_brazo = m_brazo * 9.81
    W_c = m_c * 9.81
    A_canal = params.A_canal
    rho_w = 1000.0
    C_D_r = params.C_D_r

    # Torques gravitatorios y de flotación respecto al eje O
    tau_W_r = W_r * r_G * np.sin(theta0)
    tau_B_r = -B_r * r_D * np.sin(theta0)
    tau_brazo = -W_brazo * (r_c / 2.0)
    tau_cw = -W_c * r_c
    tau_cierre_neto = tau_W_r + tau_B_r + tau_brazo + tau_cw

    # Arrastre hidrodinámico
    A_p = (0.15 + 0.85 * f) * A_canal
    F_D = 0.5 * rho_w * C_D_r * A_p * (u_w**2)
    tau_D = F_D * r_D * np.cos(theta0)

    # Reacción del tope en el extremo inferior/apoyo L_r
    # Sigma tau_O = tau_cierre_neto + tau_D + N_tope * L_r = 0
    N_tope = -(tau_cierre_neto + tau_D) / L_r

    # Referencia en f=0 para semáforo
    A_p0 = 0.15 * A_canal
    F_D0 = 0.5 * rho_w * C_D_r * A_p0 * (u_w**2)
    tau_D0 = F_D0 * r_D * np.cos(theta0)
    N_tope_f0 = -(tau_cierre_neto + tau_D0) / L_r

    porcentaje_tope = (N_tope / N_tope_f0 * 100.0) if N_tope_f0 > 0 else 0.0

    # Estado y semáforo
    if N_tope <= 0.0:
        estado = "ABIERTA (supera par de cierre; bascula para alivio)"
        color_semaforo = "red"
        N_eff = 0.0
        Ox = -F_D
        Oy = W_r - B_r + W_brazo + W_c
    elif N_tope < 0.20 * N_tope_f0:
        estado = "CERCA DE APERTURA (N_tope < 20 % de f=0; alerta preventiva)"
        color_semaforo = "yellow"
        N_eff = N_tope
        Ox = -(F_D + N_eff * np.cos(theta0))
        Oy = W_r - B_r + W_brazo + W_c + N_eff * np.sin(theta0)
    else:
        estado = "CERRADA (apoyada firmemente en tope inferior)"
        color_semaforo = "green"
        N_eff = N_tope
        Ox = -(F_D + N_eff * np.cos(theta0))
        Oy = W_r - B_r + W_brazo + W_c + N_eff * np.sin(theta0)

    O_mag = float(np.hypot(Ox, Oy))
    f_ap = calcular_f_apertura(params, u_w)

    return RejillaResult(
        f=f,
        u_corriente=u_w,
        L_r=L_r,
        m_r=m_r,
        W_r=W_r,
        B_r=B_r,
        r_G=r_G,
        r_D=r_D,
        r_c=r_c,
        m_brazo=m_brazo,
        m_c=m_c,
        W_brazo=W_brazo,
        W_c=W_c,
        I_rej_O=params.I_rej_O,
        I_brazo_O=params.I_brazo_O,
        I_c_O=params.I_c_O,
        I_O_total=params.I_O_total,
        A_p=A_p,
        F_D=F_D,
        tau_W_r=tau_W_r,
        tau_B_r=tau_B_r,
        tau_brazo=tau_brazo,
        tau_cw=tau_cw,
        tau_D=tau_D,
        tau_cierre_neto=tau_cierre_neto,
        N_tope=N_tope,
        Ox=Ox,
        Oy=Oy,
        O_mag=O_mag,
        f_ap=f_ap,
        estado=estado,
        color_semaforo=color_semaforo,
        porcentaje_tope_f0=porcentaje_tope
    )
