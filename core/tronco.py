"""
Módulo de Dinámica Traslacional del Tronco (partícula).
Resuelve m * dv/dt = 0.5 * rho_w * C_D_t * A_sub_t * (u - v) * |u - v| con RK4.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from .parameters import TroncoParams


@dataclass
class TroncoResult:
    """Contenedor de resultados del movimiento del tronco flotante."""
    m_t: float
    V_t: float
    W_t: float
    A_sub_t: float
    u_corriente: float
    v0: float
    vf_rk4: float
    t_rk4: float
    W_trapecio: float
    Ec0: float
    Ecf: float
    dEc: float
    err_teorema: float
    a_mrua: float
    t_mrua: float
    F_mrua: float
    W_mrua: float
    dif_t_mrua_pct: float
    ts: np.ndarray
    xs: np.ndarray
    vs: np.ndarray
    Fs: np.ndarray


def simular_tronco_rk4(
    params: TroncoParams,
    u_corriente: float = None,
    d: float = 6.00,
    dt: float = 0.001
) -> TroncoResult:
    """
    Integra numéricamente con RK4 la ecuación diferencial del movimiento del tronco:
    m_t * dv/dt = 0.5 * rho_w * C_D_t * A_sub_t * (u - v)*|u - v|
    """
    u_w = u_corriente if u_corriente is not None else params.u_corriente
    m_t = params.m_t
    A_sub_t = params.A_sub_t
    C_D_t = params.C_D_t
    rho_w = 1000.0
    v0 = params.v0

    def fuerza_arrastre(vel: float) -> float:
        rel = u_w - vel
        return 0.5 * rho_w * C_D_t * A_sub_t * rel * abs(rel)

    t, x, v = 0.0, 0.0, v0
    ts, xs, vs, Fs = [t], [x], [v], []
    Fs.append(fuerza_arrastre(v))

    # Límite cuando v0 == u_w: la fuerza es exactamente 0
    if abs(u_w - v0) < 1e-12:
        t_lim = d / v0 if v0 > 0 else 0.0
        ts = np.linspace(0.0, t_lim, 1001)
        xs = v0 * ts
        vs = np.full_like(ts, v0)
        Fs = np.zeros_like(ts)
        t_rk4 = t_lim
        vf_rk4 = v0
    else:
        while x < d:
            k1_x = v
            k1_v = fuerza_arrastre(v) / m_t

            v2_k = v + 0.5 * dt * k1_v
            k2_x = v2_k
            k2_v = fuerza_arrastre(v2_k) / m_t

            v3_k = v + 0.5 * dt * k2_v
            k3_x = v3_k
            k3_v = fuerza_arrastre(v3_k) / m_t

            v4_k = v + dt * k3_v
            k4_x = v4_k
            k4_v = fuerza_arrastre(v4_k) / m_t

            dx = (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            dv = (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

            if x + dx >= d:
                frac = (d - x) / dx if dx > 0 else 0.0
                t += frac * dt
                v += frac * dv
                x = d
                ts.append(t)
                xs.append(x)
                vs.append(v)
                Fs.append(fuerza_arrastre(v))
                break

            x += dx
            v += dv
            t += dt
            ts.append(t)
            xs.append(x)
            vs.append(v)
            Fs.append(fuerza_arrastre(v))

        t_rk4 = ts[-1]
        vf_rk4 = vs[-1]

    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    ts_arr = np.array(ts)
    xs_arr = np.array(xs)
    vs_arr = np.array(vs)
    Fs_arr = np.array(Fs)

    W_trap = float(trap_fn(Fs_arr, xs_arr)) if len(xs_arr) > 1 else 0.0
    Ec0 = 0.5 * m_t * v0**2
    Ecf = 0.5 * m_t * vf_rk4**2
    dEc = Ecf - Ec0
    err_teorema = abs(W_trap - dEc)

    # Modelo MRUA equivalente (fuerza constante)
    if d > 0 and (vf_rk4**2 >= v0**2):
        a_mrua = (vf_rk4**2 - v0**2) / (2.0 * d)
        F_mrua = m_t * a_mrua
        t_mrua = (vf_rk4 - v0) / a_mrua if a_mrua > 0 else (d / v0 if v0 > 0 else 0.0)
        W_mrua = F_mrua * d
    else:
        a_mrua, F_mrua, t_mrua, W_mrua = 0.0, 0.0, 0.0, 0.0

    dif_t_mrua_pct = abs(t_rk4 - t_mrua) / t_mrua * 100.0 if t_mrua > 0 else 0.0

    return TroncoResult(
        m_t=m_t,
        V_t=params.V_t,
        W_t=params.W_t,
        A_sub_t=A_sub_t,
        u_corriente=u_w,
        v0=v0,
        vf_rk4=vf_rk4,
        t_rk4=t_rk4,
        W_trapecio=W_trap,
        Ec0=Ec0,
        Ecf=Ecf,
        dEc=dEc,
        err_teorema=err_teorema,
        a_mrua=a_mrua,
        t_mrua=t_mrua,
        F_mrua=F_mrua,
        W_mrua=W_mrua,
        dif_t_mrua_pct=dif_t_mrua_pct,
        ts=ts_arr,
        xs=xs_arr,
        vs=vs_arr,
        Fs=Fs_arr
    )
