"""
Módulo de Flujo en el tramo (Etapas 1-2).
Física mecánica aplicada al box culvert con contracción geométrica por obstrucción parcial.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from .parameters import FlujoParams


@dataclass
class FlujoResult:
    """Contenedor de resultados del análisis de flujo."""
    f: float
    A1: float
    A2: float
    v1: float
    v2: float
    Q: float
    a_unif: float
    dt_analit: float
    dt_rk4: float
    dif_dt_pct: float
    V_trap: float
    m_trap: float
    m_unif: float
    F_vc: float            # Fuerza neta Sigma Fx = rho*Q*(v2 - v1) (N)
    P1: float
    P2: float
    Rx: float              # Reacción axial horizontal sobre la estructura (N)
    w_masa: float          # Trabajo por unidad de masa w = 0.5*(v2^2 - v1^2) (J/kg)
    potencia: float        # Potencia mecanica = rho*Q*w (W)
    Ec1_u: float           # Energia cinetica inicial con masa m_unif (J)
    Ec2_u: float           # Energia cinetica final con masa m_unif (J)
    W_u: float             # Trabajo neto Delta Ec con masa m_unif (J)
    ts_sim: np.ndarray
    xs_sim: np.ndarray
    vs_sim: np.ndarray
    xs_grid: np.ndarray
    Ax_grid: np.ndarray
    dFdx_analit: np.ndarray
    dFdx_sim: np.ndarray
    err_teorema_trap: float
    F_net_integral: float
    w_integral: float


def calcular_flujo_analitico(params: FlujoParams, f: float) -> Dict[str, float]:
    """
    Cálculo analítico cerrado bajo el modelo de aceleración espacial uniforme.
    """
    A1 = params.A1
    A2 = (1.0 - f) * A1
    v1 = params.v1
    v2 = params.Q / A2 if A2 > 0 else float("inf")
    Q = params.Q
    d = params.d
    rho = params.rho
    P1 = params.P1

    # Cinemática uniforme equivalente
    a_unif = (v2**2 - v1**2) / (2.0 * d)
    dt_analit = 2.0 * d / (v1 + v2) if (v1 + v2) > 0 else 0.0

    # Volumen trapezoidal y masas de referencia
    V_trap = 0.5 * (A1 + A2) * d
    m_trap = rho * V_trap
    m_unif = rho * Q * dt_analit

    # Dinámica de volumen de control e integral
    F_vc = rho * Q * (v2 - v1)   # Sigma Fx independiente del perfil A(x)
    P2 = P1 - 0.5 * rho * (v2**2 - v1**2)
    Rx = P1 * A1 - P2 * A2 - F_vc

    # Trabajo por unidad de masa y potencia
    w = 0.5 * (v2**2 - v1**2)
    potencia = rho * Q * w

    # Energías de referencia (con m_unif)
    Ec1_u = 0.5 * m_unif * v1**2
    Ec2_u = 0.5 * m_unif * v2**2
    W_u = Ec2_u - Ec1_u

    return {
        "f": f,
        "A1": A1,
        "A2": A2,
        "v1": v1,
        "v2": v2,
        "Q": Q,
        "a_unif": a_unif,
        "dt_analit": dt_analit,
        "V_trap": V_trap,
        "m_trap": m_trap,
        "m_unif": m_unif,
        "F_vc": F_vc,
        "P1": P1,
        "P2": P2,
        "Rx": Rx,
        "w": w,
        "potencia": potencia,
        "Ec1_u": Ec1_u,
        "Ec2_u": Ec2_u,
        "W_u": W_u,
    }


def simular_flujo_rk4(params: FlujoParams, f: float, dt: float = 0.001) -> FlujoResult:
    """
    Integra la trayectoria de una porción de agua dx/dt = v(x) = Q/A(x)
    con contracción lineal de área A(x) = A1 + (A2 - A1)*(x/d).
    Aplica Runge-Kutta de 4to orden (RK4) e integración por regla del trapecio.
    """
    analit = calcular_flujo_analitico(params, f)
    A1 = analit["A1"]
    A2 = analit["A2"]
    Q = analit["Q"]
    d = params.d
    rho = params.rho

    def v_local(x_pos: float) -> float:
        s = np.clip(x_pos / d, 0.0, 1.0)
        Ax = A1 + (A2 - A1) * s
        return Q / Ax

    t, x = 0.0, 0.0
    ts, xs, vs = [0.0], [0.0], [v_local(0.0)]

    while x < d:
        k1 = v_local(x)
        k2 = v_local(x + 0.5 * dt * k1)
        k3 = v_local(x + 0.5 * dt * k2)
        k4 = v_local(x + dt * k3)
        dx = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if x + dx >= d:
            frac = (d - x) / dx
            t += frac * dt
            x = d
            ts.append(t)
            xs.append(x)
            vs.append(v_local(d))
            break

        x += dx
        t += dt
        ts.append(t)
        xs.append(x)
        vs.append(v_local(x))

    dt_rk4 = t
    dif_dt_pct = abs(dt_rk4 - analit["dt_analit"]) / analit["dt_analit"] * 100.0 if analit["dt_analit"] > 0 else 0.0

    # Perfiles espaciales en malla de 1001 puntos
    xs_grid = np.linspace(0.0, d, 1001)
    Ax_grid = A1 + (A2 - A1) * (xs_grid / d)
    ax_grid = -(Q**2 / Ax_grid**3) * ((A2 - A1) / d)

    # Regla del trapecio
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    w_integral = float(trap_fn(ax_grid, xs_grid))
    F_net_integral = float(trap_fn(rho * Ax_grid * ax_grid, xs_grid))

    # Error en el teorema trabajo-energia con masa trapezoidal
    Fx_trap = analit["m_trap"] * ax_grid
    W_trap = float(trap_fn(Fx_trap, xs_grid))
    dEc_trap = 0.5 * analit["m_trap"] * (analit["v2"]**2 - analit["v1"]**2)
    err_teorema = abs(W_trap - dEc_trap)

    # Fuerza por unidad de longitud dF/dx = rho*Q*dv/dx
    dFdx_sim = rho * Q * (-(Q / Ax_grid**2) * ((A2 - A1) / d))
    
    # Modelo analítico uniforme
    a_u = analit["a_unif"]
    v_u = np.sqrt(analit["v1"]**2 + 2.0 * a_u * xs_grid) if a_u > 0 else np.full_like(xs_grid, analit["v1"])
    dFdx_analit = np.where(v_u > 0, rho * Q * a_u / v_u, 0.0)

    return FlujoResult(
        f=f,
        A1=A1,
        A2=A2,
        v1=analit["v1"],
        v2=analit["v2"],
        Q=Q,
        a_unif=analit["a_unif"],
        dt_analit=analit["dt_analit"],
        dt_rk4=dt_rk4,
        dif_dt_pct=dif_dt_pct,
        V_trap=analit["V_trap"],
        m_trap=analit["m_trap"],
        m_unif=analit["m_unif"],
        F_vc=analit["F_vc"],
        P1=analit["P1"],
        P2=analit["P2"],
        Rx=analit["Rx"],
        w_masa=analit["w"],
        potencia=analit["potencia"],
        Ec1_u=analit["Ec1_u"],
        Ec2_u=analit["Ec2_u"],
        W_u=analit["W_u"],
        ts_sim=np.array(ts),
        xs_sim=np.array(xs),
        vs_sim=np.array(vs),
        xs_grid=xs_grid,
        Ax_grid=Ax_grid,
        dFdx_analit=dFdx_analit,
        dFdx_sim=dFdx_sim,
        err_teorema_trap=err_teorema,
        F_net_integral=F_net_integral,
        w_integral=w_integral
    )
