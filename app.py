"""
Simulador Interactivo de Física Mecánica - Ciénaga de Mallorquín (ABP Etapa 3).
Universidad de la Costa (CUC) - 2026.
Autores: Javith Santana, Kevin Díaz | Tutor: Luis Polo Ariza.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import io
import os
import sys

# Asegurar importación del núcleo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core import (
    SimulatorConfig,
    FlujoParams,
    TroncoParams,
    RejillaParams,
    ImpactoParams,
    OxigenoParams,
    calcular_flujo_analitico,
    simular_flujo_rk4,
    simular_tronco_rk4,
    calcular_estatica_rejilla,
    calcular_f_apertura,
    calcular_rejilla_abp3,
    calcular_impacto_2x2,
    simular_contacto_dinamico,
    calcular_oxigeno_contexto,
    barrido_sensibilidad,
    simulacion_montecarlo,
    mapa_apertura,
    ejecutar_verificaciones_completas
)

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Simulador Físico ABP - Ciénaga de Mallorquín",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado para avisos y semáforo
st.markdown("""
<style>
    .alert-banner {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 12px 18px;
        margin-bottom: 20px;
        border-radius: 4px;
        font-weight: 500;
        color: #856404;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar estado de sesión
if "config" not in st.session_state:
    st.session_state.config = SimulatorConfig()

cfg: SimulatorConfig = st.session_state.config

# -------------------------------------------------------------
# BARRA LATERAL (PANEL DE PARÁMETROS Y CONTROLES)
# -------------------------------------------------------------
st.sidebar.title("⚙️ Parámetros Adoptados")
st.sidebar.caption("📌 Todos los valores corresponden a parámetros ADOPTADOS de referencia (no mediciones de campo).")

# Botón restaurar valores del informe
col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🔄 Restaurar Informe", help="Restaura los parámetros originales adoptados en el informe técnico"):
    st.session_state.config.restaurar_informe()
    st.rerun()

# Accesos rápidos a Escenarios A, B, C
st.sidebar.subheader("🎯 Escenarios Rápidos")
col_scA, col_scB, col_scC = st.sidebar.columns(3)
if col_scA.button("A (0.00)", help="Sección libre f = 0.00"):
    cfg.flujo.f = 0.00
    st.rerun()
if col_scB.button("B (0.30)", help="Obstrucción parcial de diseño f = 0.30"):
    cfg.flujo.f = 0.30
    st.rerun()
if col_scC.button("C (0.60)", help="Obstrucción fuerte / basculamiento f = 0.60"):
    cfg.flujo.f = 0.60
    st.rerun()

st.sidebar.markdown("---")

# Regla de corriente unificada
st.sidebar.subheader("🌊 Velocidad de Corriente")
modo_unif = st.sidebar.checkbox(
    "Unificar velocidad de corriente",
    value=cfg.unificar_velocidad_corriente,
    help="Regla de consistencia: una sola velocidad para el flujo, arrastre del tronco y empuje sobre rejilla."
)
if modo_unif:
    u_comun = st.sidebar.slider(
        "u_corriente (m/s) [UNIFICADA]",
        min_value=0.50, max_value=3.00, value=cfg.u_unificada, step=0.05
    )
    cfg.aplicar_unificacion_corriente(u_comun)
else:
    cfg.unificar_velocidad_corriente = False
    st.sidebar.caption("⚠️ Modo desacoplado (informe original): u_tronco = 2.0 m/s (creciente en aproximación) y u_rejilla = 1.0 m/s (nominal en boca).")
    u_tronco = st.sidebar.slider("u corriente tronco (m/s)", 0.50, 3.00, cfg.tronco.u_corriente, 0.05)
    u_rejilla = st.sidebar.slider("u corriente rejilla (m/s)", 0.50, 3.00, cfg.rejilla.u_corriente, 0.05)
    cfg.tronco.u_corriente = u_tronco
    cfg.rejilla.u_corriente = u_rejilla

# Parámetros del Flujo
st.sidebar.subheader("📐 Flujo y Box Culvert")
cfg.flujo.f = st.sidebar.slider("Fracción obstruida (f)", 0.00, 0.85, cfg.flujo.f, 0.01)
cfg.flujo.v1 = st.sidebar.slider("Velocidad de entrada v1 (m/s)", 0.50, 2.50, cfg.flujo.v1, 0.05)
cfg.flujo.d = st.sidebar.number_input("Longitud tramo d (m)", 1.0, 20.0, cfg.flujo.d, 0.5)

# Parámetros del Tronco
st.sidebar.subheader("🪵 Tronco Flotante")
m_tronco_actual = cfg.tronco.m_t
masa_tronco_in = st.sidebar.slider("Masa tronco mt (kg)", 20.0, 150.0, float(m_tronco_actual), 1.0)
# Ajustar densidad proporcionalmente para mantener D y L fijos si el usuario varía masa
cfg.tronco.rho_mad = masa_tronco_in / cfg.tronco.V_t
cfg.tronco.v0 = st.sidebar.slider("Velocidad inicial tronco v0 (m/s)", 0.00, 2.00, cfg.tronco.v0, 0.05)

# Parámetros de la Rejilla y Contrapeso
st.sidebar.subheader("🛡️ Rejilla y Contrapeso")
cfg.rejilla.m_c = st.sidebar.slider("Masa contrapeso mc (kg)", 50.0, 450.0, cfg.rejilla.m_c, 5.0)
cfg.rejilla.r_c = st.sidebar.slider("Brazo contrapeso rc (m)", 0.30, 1.50, cfg.rejilla.r_c, 0.05)
cfg.rejilla.theta0_deg = st.sidebar.slider("Inclinación theta0 (°)", 0.0, 35.0, cfg.rejilla.theta0_deg, 1.0)

# Parámetros del Impacto
st.sidebar.subheader("💥 Parámetros de Choque")
cfg.impacto.e = st.sidebar.slider("Coef. restitución e", 0.00, 0.60, cfg.impacto.e, 0.05)
cfg.impacto.dt_imp = st.sidebar.slider("Duración impacto dt_imp (s)", 0.02, 0.20, cfg.impacto.dt_imp, 0.005)

st.sidebar.markdown("---")
st.sidebar.info("Versión: **2.1.0-Etapa3-CUC**\nSemilla fijada: **42**\nMotor físico: **NumPy / RK4 / Trapecio**")

# -------------------------------------------------------------
# EJECUCIÓN DE CÁLCULOS DEL NÚCLEO FÍSICO
# -------------------------------------------------------------
u_tro_efectiva = cfg.tronco.u_corriente
u_rej_efectiva = cfg.rejilla.u_corriente

res_flujo = simular_flujo_rk4(cfg.flujo, f=cfg.flujo.f)
res_tronco = simular_tronco_rk4(cfg.tronco, u_corriente=u_tro_efectiva, d=cfg.flujo.d)
res_rejilla = calcular_estatica_rejilla(cfg.rejilla, f=cfg.flujo.f, u_corriente=u_rej_efectiva)
res_impacto = calcular_impacto_2x2(
    cfg.tronco,
    cfg.rejilla,
    cfg.impacto,
    v_impacto=res_tronco.vf_rk4,
    e_override=cfg.impacto.e
)
res_abp3 = calcular_rejilla_abp3(
    Rx=res_flujo.Rx,
    v2=res_flujo.v2,
    f=cfg.flujo.f,
    m_t=cfg.tronco.m_t,
    e=cfg.impacto.e,
    dt=cfg.impacto.dt_imp
)
res_oxigeno = calcular_oxigeno_contexto(cfg.oxigeno, cfg.flujo)

# -------------------------------------------------------------
# CABECERA Y BANNER DE VERIFICACIÓN
# -------------------------------------------------------------
st.title("🌊 Simulador de Física Mecánica: Box Culvert y Rejilla de Retención")
st.markdown("#### *Ciénaga de Mallorquín — Proyecto Integrador de Aprendizaje (ABP, Etapa 3)*")

st.markdown(f"""
<div class="alert-banner">
    ⚠️ <strong>ESTADO DEL MODELO:</strong> {cfg.etiqueta_estado}.<br>
    Este simulador constituye una herramienta de diseño computacional y verificación numérica. Las cifras varían según los parámetros adoptados y no deben interpretarse como datos de campo.
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SEMÁFORO DE OPERACIÓN DE LA REJILLA
# -------------------------------------------------------------
color_map = {
    "green": ("#d4edda", "#155724", "#28a745", "🟢 CERRADA — APOYADA EN TOPE"),
    "yellow": ("#fff3cd", "#856404", "#ffc107", "🟡 CERCA DE APERTURA — ALERTA DE MANTENIMIENTO"),
    "red": ("#f8d7da", "#721c24", "#dc3545", "🔴 SUPERA PAR DE CIERRE — BASCULA PARA ALIVIO PASIVO")
}
bg_c, txt_c, brd_c, sem_title = color_map[res_rejilla.color_semaforo]

st.markdown(f"""
<div style="background-color: {bg_c}; color: {txt_c}; border: 2px solid {brd_c}; border-radius: 8px; padding: 14px 20px; margin-bottom: 20px;">
    <h3 style="margin: 0; color: {txt_c};">{sem_title}</h3>
    <div style="margin-top: 6px; font-size: 1.05rem;">
        <strong>Fuerza en el tope inferior:</strong> N_tope = <strong>{res_rejilla.N_tope:.1f} N</strong> 
        ({res_rejilla.porcentaje_tope_f0:.1f} % del valor con sección limpia f = 0) | 
        <strong>Punto crítico de apertura:</strong> f_ap = <strong>{res_rejilla.f_ap:.3f}</strong> (para u = {u_rej_efectiva:.2f} m/s)
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# PESTAÑAS PRINCIPALES DE VISUALIZACIÓN
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 1. Flujo y Aceleración",
    "🪵 2. Tronco e Impacto",
    "📊 3. Libro de Energía y Rejilla",
    "🔍 4. Incertidumbre y Monte Carlo",
    "✅ 5. Panel de Verificación Numérica",
    "🌱 6. Oxígeno y Ambiente"
])

# -------------------------------------------------------------
# TAB 1: FLUJO EN EL TRAMO Y CINEMÁTICA
# -------------------------------------------------------------
with tab1:
    st.subheader("Dinámica del Flujo en el Tramo Horizontal (Etapas 1 y 2)")
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Velocidad salida v2", f"{res_flujo.v2:.3f} m/s", f"+{(res_flujo.v2/res_flujo.v1 - 1)*100:.1f} %")
    col_m2.metric("Fuerza neta ΣFx", f"{res_flujo.F_vc:.1f} N", "rho·Q·Δv")
    col_m3.metric("Reacción horizontal Rx", f"{res_flujo.Rx:.1f} N", "Paredes y tarulla")
    col_m4.metric("Trabajo por unidad masa", f"{res_flujo.w_masa:.3f} J/kg", "½(v2² - v1²)")
    col_m5.metric("Potencia mecánica", f"{res_flujo.potencia:.1f} W", "rho·Q·w")

    col_g1, col_g2 = st.columns(2)
    
    # Gráfica 1: Cinemática x(t) y v(t)
    with col_g1:
        fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(9, 4.2), dpi=150)
        
        # x(t)
        t_analit_vec = np.linspace(0.0, res_flujo.dt_analit, 200)
        x_analit_vec = res_flujo.v1 * t_analit_vec + 0.5 * res_flujo.a_unif * (t_analit_vec**2)
        ax1a.plot(t_analit_vec, x_analit_vec, "b-", lw=2, label="Analítico (a unif.)")
        ax1a.plot(res_flujo.ts_sim, res_flujo.xs_sim, "r--", lw=2, label="Simulación RK4")
        ax1a.set_xlabel("Tiempo t (s)")
        ax1a.set_ylabel("Posición x (m)")
        ax1a.set_title("Posición vs. Tiempo x(t)")
        ax1a.grid(True, alpha=0.3)
        ax1a.legend(fontsize=8)

        # v(t)
        v_analit_vec = res_flujo.v1 + res_flujo.a_unif * t_analit_vec
        ax1b.plot(t_analit_vec, v_analit_vec, "b-", lw=2, label="Analítico (a unif.)")
        ax1b.plot(res_flujo.ts_sim, res_flujo.vs_sim, "r--", lw=2, label="Simulación RK4")
        ax1b.set_xlabel("Tiempo t (s)")
        ax1b.set_ylabel("Velocidad v (m/s)")
        ax1b.set_title("Velocidad vs. Tiempo v(t)")
        ax1b.grid(True, alpha=0.3)
        ax1b.legend(fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    # Gráfica 2: Fuerza por unidad de longitud y área bajo la curva
    with col_g2:
        fig2, ax2 = plt.subplots(figsize=(7, 4.2), dpi=150)
        ax2.plot(res_flujo.xs_grid, res_flujo.dFdx_analit, "b-", lw=2, label="Analítico: rho·Q·a/v(x)")
        ax2.plot(res_flujo.xs_grid, res_flujo.dFdx_sim, "r--", lw=2, label="Contracción lineal (RK4)")
        ax2.fill_between(res_flujo.xs_grid, 0, res_flujo.dFdx_sim, color="red", alpha=0.15, label=f"Área = ΣFx = {res_flujo.F_vc:.1f} N")
        ax2.set_xlabel("Posición longitudinal x (m)")
        ax2.set_ylabel("Fuerza por unidad de longitud dF/dx (N/m)")
        ax2.set_title("Distribución de Fuerza Axial en el Agua")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown(f"""
    > **Nota de verificación geométrica:** El tiempo de tránsito analítico bajo aceleración uniforme es 
    `dt_analit = {res_flujo.dt_analit:.3f} s`, mientras que la contracción lineal RK4 arroja `dt_sim = {res_flujo.dt_rk4:.3f} s` 
    (diferencia del **{res_flujo.dif_dt_pct:.2f} %**). Esta divergencia es puramente geométrica: la contracción lineal encierra un volumen 
    mayor `V_trap = {res_flujo.V_trap:.2f} m³`, por lo que el tiempo exacto de residencia es `V_trap / Q = {res_flujo.V_trap / res_flujo.Q:.3f} s`.
    """)

# -------------------------------------------------------------
# TAB 2: TRONCO FLOTANTE E IMPACTO
# -------------------------------------------------------------
with tab2:
    st.subheader("Dinámica del Tronco Flotante y Colisión Tronco–Rejilla")
    
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Masa del tronco mt", f"{res_tronco.m_t:.1f} kg", f"Vol = {res_tronco.V_t:.3f} m³")
    col_t2.metric("Velocidad de llegada vf", f"{res_tronco.vf_rk4:.3f} m/s", f"v0 = {res_tronco.v0:.2f} m/s")
    col_t3.metric("Tiempo de tránsito", f"{res_tronco.t_rk4:.3f} s", f"MRUA = {res_tronco.t_mrua:.3f} s")
    col_t4.metric("Trabajo W = ΔEc", f"{res_tronco.W_trapecio:.2f} J", f"|W - ΔEc| = {res_tronco.err_teorema:.2e} J")

    col_g3, col_g4 = st.columns(2)
    
    # Gráfica 3: Trayectoria y fuerza del tronco
    with col_g3:
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(7, 5.0), dpi=150, sharex=True)
        ax3a.plot(res_tronco.xs, res_tronco.vs, "g-", lw=2, label="v(x) Tronco (RK4)")
        ax3a.axhline(u_tro_efectiva, color="blue", ls=":", label=f"Corriente u = {u_tro_efectiva:.2f} m/s")
        ax3a.set_ylabel("Velocidad (m/s)")
        ax3a.set_title("Evolución de Velocidad y Fuerza del Tronco")
        ax3a.grid(True, alpha=0.3)
        ax3a.legend(fontsize=8)

        ax3b.plot(res_tronco.xs, res_tronco.Fs, "m-", lw=2, label="F_arrastre(x)")
        ax3b.fill_between(res_tronco.xs, 0, res_tronco.Fs, color="purple", alpha=0.15, label=f"Área = W = {res_tronco.W_trapecio:.1f} J")
        ax3b.set_xlabel("Desplazamiento x (m)")
        ax3b.set_ylabel("Fuerza arrastre (N)")
        ax3b.grid(True, alpha=0.3)
        ax3b.legend(fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # Gráfica 4: Oscilación rotacional post-impacto
    with col_g4:
        fig4, ax4 = plt.subplots(figsize=(7, 4.6), dpi=150)
        ax4.plot(res_impacto.ts_rot, res_impacto.thetas_rot_deg, "b-", lw=2, label="Deflexión θ(t) [grados]")
        ax4.plot(res_impacto.ts_rot, res_impacto.omegas_rot, "r--", lw=1.5, label="Velocidad angular ω(t) [rad/s]")
        ax4.axhline(res_impacto.dtheta_max_deg, color="darkorange", ls=":", label=f"θ_máx = {res_impacto.dtheta_max_deg:.2f}°")
        ax4.set_xlabel("Tiempo post-impacto t (s)")
        ax4.set_ylabel("Ángulo / Velocidad angular")
        ax4.set_title("Respuesta Rotacional Post-Impacto (Articulada Contrapesada)")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # Comparativa rígida vs articulada
    st.markdown("#### Contraste de Colisión: Rejilla Rígida vs. Articulada (Misma Componente Normal)")
    st.markdown(f"""
    * **Velocidad normal de aproximación:** `vn,i = {res_impacto.v_normal_i:.3f} m/s` (evaluada con `θ0 = {cfg.rejilla.theta0_deg:.1f}°`).
    * **Rejilla rígida (bloqueada):** Fuerza de impacto promedio = **{res_impacto.F_prom_rig:.0f} N** | Impulso = **{abs(res_impacto.J_rig):.1f} N·s** | ΔEc = **{res_impacto.dEc_rig:.1f} J**.
    * **Rejilla articulada (sistema 2x2):** Fuerza de impacto promedio = **{res_impacto.F_prom_art:.0f} N** | ω_f = **{res_impacto.omega_f:.4f} rad/s** | Deflexión máxima = **{res_impacto.dtheta_max_deg:.2f}°**.
    * **Verificación no tautológica (contacto finito dt = {cfg.impacto.dt_imp:.3f} s):** La hipótesis de impulso instantáneo introduce una discrepancia de **{res_impacto.error_impulso_instantaneo_pct:.1f} %** frente a la integración dinámica con torques finitos (confirmando la cota de la Tabla VIII).
    """)

    st.markdown("---")
    st.markdown("#### 🛡️ Modelo Oficial ABP 3 (Informe Final: Bisagra Inferior + Tirante Superior)")
    st.info(
        f"**Supuestos del ABP 3 (Tabla 6 / Diapositiva 13):** Rejilla 2,0 × 1,5 m articulada en la base O (y = 0) con tirante horizontal superior (y_t = {res_abp3.y_t:.2f} m). "
        f"La carga axial hidrodinámica global R_x = {res_abp3.Rx:.1f} N se adopta como cota superior en y_r = h/3 = {res_abp3.y_r:.2f} m. "
        f"En el choque rígido del informe se adopta v_t = v_2 = {res_abp3.v2:.3f} m/s; la rejilla permanece bloqueada sin bascular."
    )
    col_abp1, col_abp2, col_abp3, col_abp4 = st.columns(4)
    col_abp1.metric("Tensión estática T", f"{res_abp3.T_estatica:.0f} N", "Rx / 3")
    col_abp2.metric("Reacción horiz. Ho", f"{res_abp3.H_O:.0f} N", "2·Rx / 3")
    col_abp3.metric("Fuerza media choque", f"{res_abp3.F_impact_media:.0f} N", f"J = {res_abp3.J_rigid:.1f} N·s")
    col_abp4.metric("Tensión total media", f"{res_abp3.T_total_media:.0f} N", "T_est + T_imp")

# -------------------------------------------------------------
# TAB 3: LIBRO DE ENERGÍA Y REJILLA
# -------------------------------------------------------------
with tab3:
    st.subheader("Libro de Energía y Curva de Operación de la Rejilla")

    col_g5, col_g6 = st.columns(2)
    
    # Gráfica 5: Libro de energía
    with col_g5:
        fig5, ax5 = plt.subplots(figsize=(7, 4.3), dpi=150)
        categorias = ["Ec Inicial (Tronco)", "Ec Final Rotacional", "Ec Final Tronco", "Energía Disipada"]
        valores = [
            res_impacto.Ec_i_tronco,
            res_impacto.Ec_f_rot,
            res_impacto.Ec_f_tronco,
            res_impacto.E_disipada
        ]
        colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        bars = ax5.bar(categorias, valores, color=colores, width=0.55)
        ax5.set_ylabel("Energía Mecánica (Joules)")
        ax5.set_title("Libro de Energía: Balance del Impacto Tronco–Rejilla")
        ax5.grid(True, axis="y", alpha=0.3)
        for bar in bars:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + max(valores)*0.02, f"{yval:.1f} J", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_ylim(0, max(valores) * 1.2)
        
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    # Gráfica 6: Curva N_tope vs f con punto f_ap
    with col_g6:
        fig6, ax6 = plt.subplots(figsize=(7, 4.3), dpi=150)
        f_sweep = np.linspace(0.0, 0.85, 300)
        N_sweep = [calcular_estatica_rejilla(cfg.rejilla, f=fv, u_corriente=u_rej_efectiva).N_tope for fv in f_sweep]
        
        ax6.plot(f_sweep, N_sweep, "b-", lw=2, label="N_tope(f)")
        ax6.axhline(0, color="k", ls="--", lw=1)
        
        # Marcar f_ap
        f_ap_val = res_rejilla.f_ap
        ax6.axvline(f_ap_val, color="red", ls=":", lw=1.5, label=f"Apertura pasiva f_ap = {f_ap_val:.3f}")
        ax6.plot([cfg.flujo.f], [res_rejilla.N_tope], "go", ms=8, label=f"Punto actual f = {cfg.flujo.f:.2f}")

        # Zona cerrada y zona abierta
        ax6.fill_between(f_sweep, N_sweep, 0, where=(np.array(N_sweep) >= 0), color="green", alpha=0.1, label="Zona Cerrada (Estable)")
        ax6.fill_between(f_sweep, N_sweep, 0, where=(np.array(N_sweep) < 0), color="red", alpha=0.1, label="Zona Abierta (Alivio)")

        ax6.set_xlabel("Fracción Obstruida del Área f")
        ax6.set_ylabel("Fuerza en el Tope N_tope (N)")
        ax6.set_title("Curva de Estabilidad de la Rejilla frente a Colmatación")
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

# -------------------------------------------------------------
# TAB 4: INCERTIDUMBRE, MONTE CARLO Y MAPA (f, u)
# -------------------------------------------------------------
with tab4:
    st.subheader("Análisis de Incertidumbre y Propagación Paramétrica")
    st.caption("Como los parámetros son adoptados y no provienen de aforos directos, se evalúa su variabilidad.")

    col_mc1, col_mc2 = st.columns([1, 1])

    with col_mc1:
        st.markdown("#### Simulación Monte Carlo (Semilla Fija = 42)")
        n_corridas = st.selectbox("Número de corridas Monte Carlo", [500, 1000, 2000], index=1)
        if st.button("🎲 Ejecutar Monte Carlo", help="Realiza muestreo estocástico uniforme de parámetros adoptados"):
            with st.spinner("Ejecutando simulación Monte Carlo..."):
                res_mc = simulacion_montecarlo(cfg, n_corridas=n_corridas, seed=42)
                st.session_state.res_mc = res_mc

        if "res_mc" in st.session_state:
            mc: MonteCarloSummary = st.session_state.res_mc
            st.success(f"Simulación de {mc.n_corridas} corridas completada.")
            df_mc = pd.DataFrame({
                "Magnitud Física": ["Fuerza en Tope N_tope (N)", "Fuerza Impacto Promedio F_prom (N)", "Deflexión Angular θ_máx (°)"],
                "P5 (Conservador)": [f"{mc.N_tope_p5:.1f}", f"{mc.F_prom_p5:.1f}", f"{mc.dtheta_p5:.2f}°"],
                "P50 (Mediana)": [f"{mc.N_tope_p50:.1f}", f"{mc.F_prom_p50:.1f}", f"{mc.dtheta_p50:.2f}°"],
                "P95 (Extremo)": [f"{mc.N_tope_p95:.1f}", f"{mc.F_prom_p95:.1f}", f"{mc.dtheta_p95:.2f}°"],
                "Media ± Desv": [f"{mc.N_tope_media:.1f} ± {mc.N_tope_std:.1f}", f"{mc.F_prom_media:.1f}", f"{mc.dtheta_media:.2f}°"]
            })
            st.table(df_mc)
            st.metric("Probabilidad de Apertura de Rejilla", f"{mc.prob_apertura_pct:.1f} %")

    with col_mc2:
        st.markdown("#### Mapa Bidimensional de Apertura (f vs. u)")
        mapa_data = mapa_apertura(cfg.rejilla, f_res=50, u_res=50)

        fig_map, ax_map = plt.subplots(figsize=(6.5, 4.2), dpi=150)
        cax = ax_map.contourf(
            mapa_data["F_grid"],
            mapa_data["U_grid"],
            mapa_data["N_tope_grid"],
            levels=20,
            cmap="RdYlGn"
        )
        # Línea de apertura N_tope = 0
        cs = ax_map.contour(
            mapa_data["F_grid"],
            mapa_data["U_grid"],
            mapa_data["N_tope_grid"],
            levels=[0.0],
            colors="black",
            linewidths=2.5
        )
        ax_map.clabel(cs, inline=True, fmt="Apertura (N=0)", fontsize=9)
        # Punto de operación actual
        ax_map.plot([cfg.flujo.f], [u_rej_efectiva], "ko", ms=9, label="Operación actual")
        ax_map.set_xlabel("Fracción Obstruida f")
        ax_map.set_ylabel("Velocidad de Corriente u (m/s)")
        ax_map.set_title("Mapa de Apertura: Verde=Cerrada, Rojo=Abierta")
        fig_map.colorbar(cax, ax=ax_map, label="Fuerza en Tope N_tope (N)")
        ax_map.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_map)
        plt.close(fig_map)

# -------------------------------------------------------------
# TAB 5: PANEL DE VERIFICACIÓN NUMÉRICA Y EXPORTACIÓN
# -------------------------------------------------------------
with tab5:
    st.subheader("Registro de Verificación Numérica y Reproducibilidad")
    st.markdown(f"**Etiqueta de rigor metodológico:** `{cfg.etiqueta_estado}`")

    verifs = ejecutar_verificaciones_completas(cfg)
    datos_tabla = []
    for v in verifs:
        icono = "✅ PASA" if v.estado == "PASA" else "❌ FALLA"
        datos_tabla.append({
            "Tipo": v.tipo,
            "Criterio / Prueba": v.prueba,
            "Resultado Obtenido": v.valor_obtenido,
            "Tolerancia": v.tolerancia,
            "Estado": icono,
            "Observación Física": v.observacion
        })
    df_verif = pd.DataFrame(datos_tabla)
    st.dataframe(df_verif, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("💾 Exportación y Reproducibilidad")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        json_str = cfg.to_json()
        st.download_button(
            label="📥 Descargar Parámetros en JSON",
            data=json_str,
            file_name="parametros_mallorquin.json",
            mime="application/json"
        )

    with col_exp2:
        # Generar CSV de resultados de los 3 escenarios
        esc_res = []
        for nom, f_val in [("A_Libre", 0.0), ("B_Parcial", 0.30), ("C_Fuerte", 0.60)]:
            fl = simular_flujo_rk4(cfg.flujo, f=f_val)
            rj = calcular_estatica_rejilla(cfg.rejilla, f=f_val, u_corriente=u_rej_efectiva)
            esc_res.append({
                "Escenario": nom,
                "f": f_val,
                "v2_m_s": fl.v2,
                "Sigma_Fx_N": fl.F_vc,
                "Rx_N": fl.Rx,
                "Potencia_W": fl.potencia,
                "N_tope_N": rj.N_tope,
                "f_ap": rj.f_ap,
                "Estado_Rejilla": rj.estado
            })
        csv_data = pd.DataFrame(esc_res).to_csv(index=False)
        st.download_button(
            label="📊 Descargar Resumen de Escenarios en CSV",
            data=csv_data,
            file_name="resumen_escenarios_mallorquin.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------
# TAB 6: CONTEXTO AMBIENTAL Y OXÍGENO
# -------------------------------------------------------------
with tab6:
    st.subheader("🌱 Contexto Ambiental: Estimación de Oxígeno Disuelto (ABP Etapa 1)")
    st.markdown(
        """
        > **[ADOPTADO] Orden de Magnitud Ilustrativo (Tabla VI del Informe)**  
        > En la Ciénaga de Mallorquín, el oxígeno disuelto saturado de referencia es de **8,30 mg/L** a 25 °C y salinidad típica.
        > Cada gramo de celulosa vegetal (tarulla o madera retenida) demanda aproximadamente **1,19 g de O₂** para su degradación aerobia bacteriana.
        """
    )
    
    col_ox1, col_ox2, col_ox3 = st.columns(3)
    col_ox1.metric("Concentración O₂ ref.", "8,30 mg/L", "Saturación a 25 °C")
    col_ox2.metric("Demanda mineralización", "1,19 g O₂ / g cel.", "Estequiometría C₆H₁₀O₅")
    col_ox3.metric("Transporte Esc. B (f=0.30)", f"{res_oxigeno.filas[1].aporte_o2_t_dia:.2f} t O₂/día", f"Q = {res_oxigeno.filas[1].caudal_m3_s:.2f} m³/s")

    st.markdown("### Comparativa de Escenarios de Transporte de O₂")
    filas_oxi = []
    for fo in res_oxigeno.filas:
        filas_oxi.append({
            "Escenario": fo.escenario,
            "Fracción f": f"{fo.f:.2f}",
            "Área Útil (1 - f)": f"{fo.fraccion_libre:.2f}",
            "Caudal ilustrativo Q (m³/s)": f"{fo.caudal_m3_s:.2f}",
            "Transporte O₂ (t/día)": f"{fo.aporte_o2_t_dia:.2f}",
            "Celulosa oxidable equivalente (t/día)": f"{fo.materia_organica_t_dia:.2f}",
            "Estado Rejilla": fo.estado_rejilla
        })
    st.table(pd.DataFrame(filas_oxi))
    st.caption(res_oxigeno.nota)
