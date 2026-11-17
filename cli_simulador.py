"""
Línea de Comandos (CLI) y Generador de Reportes y Gráficos PNG para el Simulador.
Permite ejecutar el simulador sin interfaz gráfica, exportar CSV, JSON y figuras científicas en alta resolución (300 dpi).
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from core import (
    SimulatorConfig,
    simular_flujo_rk4,
    simular_tronco_rk4,
    calcular_estatica_rejilla,
    calcular_f_apertura,
    calcular_impacto_2x2,
    calcular_oxigeno_contexto,
    simulacion_montecarlo,
    mapa_apertura,
    ejecutar_verificaciones_completas
)


def generar_graficos_completos(cfg: SimulatorConfig, out_dir: str = "exports"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generando figuras en {out_dir}...")

    # 1. Figuras cinemáticas y dinámicas del flujo (A, B, C)
    res_A = simular_flujo_rk4(cfg.flujo, f=0.00)
    res_B = simular_flujo_rk4(cfg.flujo, f=0.30)
    res_C = simular_flujo_rk4(cfg.flujo, f=0.60)

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    # x-t
    ax1.plot(res_A.ts_sim, res_A.xs_sim, label="A: Libre (f=0.00)", color="#1f77b4", lw=2)
    ax1.plot(res_B.ts_sim, res_B.xs_sim, label="B: Parcial (f=0.30)", color="#ff7f0e", lw=2)
    ax1.plot(res_C.ts_sim, res_C.xs_sim, label="C: Fuerte (f=0.60)", color="#d62728", lw=2)
    ax1.set_xlabel("Tiempo t (s)")
    ax1.set_ylabel("Posición x (m)")
    ax1.set_title("Trayectoria Posición-Tiempo x(t)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # v-t
    ax2.plot(res_A.ts_sim, res_A.vs_sim, label="A: Libre (f=0.00)", color="#1f77b4", lw=2)
    ax2.plot(res_B.ts_sim, res_B.vs_sim, label="B: Parcial (f=0.30)", color="#ff7f0e", lw=2)
    ax2.plot(res_C.ts_sim, res_C.vs_sim, label="C: Fuerte (f=0.60)", color="#d62728", lw=2)
    ax2.set_xlabel("Tiempo t (s)")
    ax2.set_ylabel("Velocidad v (m/s)")
    ax2.set_title("Evolución Velocidad-Tiempo v(t)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "fig1_cinematica_xt_vt.png"))
    plt.close(fig1)

    # 2. Fuerza por unidad de longitud
    fig2, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(res_B.xs_grid, res_B.dFdx_analit, color="#ff7f0e", lw=2, label="B: analítico uniforme")
    ax.plot(res_B.xs_grid, res_B.dFdx_sim, color="#ff7f0e", lw=1.6, ls="--", label="B: simulación RK4")
    ax.fill_between(res_B.xs_grid, 0, res_B.dFdx_sim, color="#ff7f0e", alpha=0.1)

    ax.plot(res_C.xs_grid, res_C.dFdx_analit, color="#d62728", lw=2, label="C: analítico uniforme")
    ax.plot(res_C.xs_grid, res_C.dFdx_sim, color="#d62728", lw=1.6, ls="--", label="C: simulación RK4")
    ax.fill_between(res_C.xs_grid, 0, res_C.dFdx_sim, color="#d62728", alpha=0.1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("dF/dx = rho*Q*dv/dx (N/m)")
    ax.set_title("Fuerza por Unidad de Longitud sobre el Agua")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "fig2_fuerza_longitud_dFdx.png"))
    plt.close(fig2)

    # 3. Tronco F-x y área sombreada
    res_tro = simular_tronco_rk4(cfg.tronco, u_corriente=2.00)
    fig3, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.plot(res_tro.xs, res_tro.Fs, "m-", lw=2.2, label="F_arrastre(x)")
    ax.fill_between(res_tro.xs, 0, res_tro.Fs, color="purple", alpha=0.15, label=f"Área = W = {res_tro.W_trapecio:.1f} J")
    ax.set_xlabel("Desplazamiento x (m)")
    ax.set_ylabel("Fuerza neta sobre el tronco (N)")
    ax.set_title("Trabajo Mecánico sobre el Tronco Flotante (W = ∫ F dx)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "fig3_trabajo_tronco_Fx.png"))
    plt.close(fig3)

    # 4. N_tope vs f con punto de apertura f_ap
    fig4, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    f_grid = np.linspace(0.0, 0.80, 200)
    N_grid = [calcular_estatica_rejilla(cfg.rejilla, f=fv, u_corriente=1.00).N_tope for fv in f_grid]
    f_ap = calcular_f_apertura(cfg.rejilla, u_corriente=1.00)
    ax.plot(f_grid, N_grid, "b-", lw=2, label="N_tope(f)")
    ax.axhline(0, color="k", ls="--")
    ax.axvline(f_ap, color="red", ls=":", lw=2, label=f"Apertura f_ap = {f_ap:.3f}")
    ax.set_xlabel("Fracción de obstrucción f")
    ax.set_ylabel("Fuerza sobre tope N_tope (N)")
    ax.set_title("Estabilidad de la Rejilla Articulada Contrapesada")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, "fig4_N_tope_vs_f.png"))
    plt.close(fig4)

    # 5. Libro de energía del impacto
    res_imp = calcular_impacto_2x2(cfg.tronco, cfg.rejilla, cfg.impacto, v_impacto=res_tro.vf_rk4, e_override=0.0)
    fig5, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    cats = ["Ec Antes", "Ec Rotacional", "Ec Residual Tronco", "Energía Disipada"]
    vals = [res_imp.Ec_i_tronco, res_imp.Ec_f_rot, res_imp.Ec_f_tronco, res_imp.E_disipada]
    ax.bar(cats, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], width=0.55)
    ax.set_ylabel("Energía (J)")
    ax.set_title("Libro de Energía: Conservación y Disipación en el Impacto")
    ax.grid(True, axis="y", alpha=0.3)
    fig5.tight_layout()
    fig5.savefig(os.path.join(out_dir, "fig5_libro_energia.png"))
    plt.close(fig5)

    # 6. Mapa de apertura (f, u)
    map_data = mapa_apertura(cfg.rejilla)
    fig6, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    cax = ax.contourf(map_data["F_grid"], map_data["U_grid"], map_data["N_tope_grid"], levels=20, cmap="RdYlGn")
    cs = ax.contour(map_data["F_grid"], map_data["U_grid"], map_data["N_tope_grid"], levels=[0.0], colors="black", linewidths=2.5)
    ax.clabel(cs, inline=True, fmt="Apertura (N=0)")
    ax.set_xlabel("Fracción Obstruida f")
    ax.set_ylabel("Velocidad de Corriente u (m/s)")
    ax.set_title("Mapa de Apertura Pasiva de la Rejilla (f vs. u)")
    fig6.colorbar(cax, label="N_tope (N)")
    fig6.tight_layout()
    fig6.savefig(os.path.join(out_dir, "fig6_mapa_apertura_fu.png"))
    plt.close(fig6)

    print(f"Todas las figuras se guardaron exitosamente en {out_dir}/.")


def imprimir_reporte_completo(cfg: SimulatorConfig):
    print("=" * 78)
    print("   SIMULADOR DE FÍSICA MECÁNICA - CIÉNAGA DE MALLORQUÍN (CUC 2026)")
    print(f"   {cfg.etiqueta_estado}")
    print("=" * 78)

    print("\n--- 1. TABLA COMPARATIVA DE ESCENARIOS HIDRÁULICOS ---")
    escenarios = [("A (Libre)", 0.00), ("B (Parcial)", 0.30), ("C (Fuerte)", 0.60)]
    for nom, f_val in escenarios:
        fl = simular_flujo_rk4(cfg.flujo, f=f_val)
        rj = calcular_estatica_rejilla(cfg.rejilla, f=f_val, u_corriente=1.00)
        print(f"\n[Escenario {nom} - f = {f_val:.2f}]")
        print(f"  v2: {fl.v2:.3f} m/s | dt_analit: {fl.dt_analit:.3f} s | dt_RK4: {fl.dt_rk4:.3f} s (dif {fl.dif_dt_pct:.1f} %)")
        print(f"  Fuerza neta ΣFx: {fl.F_vc:.1f} N | Rx estructural: {fl.Rx:.1f} N | Potencia: {fl.potencia:.0f} W")
        print(f"  N_tope rejilla: {rj.N_tope:.1f} N | Estado: {rj.estado}")

    print("\n--- 2. DINÁMICA DEL TRONCO FLOTANTE ---")
    tro = simular_tronco_rk4(cfg.tronco, u_corriente=2.00)
    print(f"  Masa mt: {tro.m_t:.2f} kg | vf (RK4): {tro.vf_rk4:.3f} m/s | t_RK4: {tro.t_rk4:.3f} s")
    print(f"  Trabajo Trapecio: {tro.W_trapecio:.2f} J | Delta Ec: {tro.dEc:.2f} J | Error Teorema: {tro.err_teorema:.2e} J")

    print("\n--- 3. IMPACTO TRONCO-REJILLA (SISTEMA 2x2) ---")
    imp = calcular_impacto_2x2(cfg.tronco, cfg.rejilla, cfg.impacto, v_impacto=tro.vf_rk4, e_override=0.0)
    print(f"  Rapidez de aproximación normal: {imp.v_normal_i:.3f} m/s")
    print(f"  Rejilla rígida: F_prom = {imp.F_prom_rig:.0f} N | J = {abs(imp.J_rig):.1f} N·s")
    print(f"  Rejilla articulada: omega_f = {imp.omega_f:.4f} rad/s | F_prom = {imp.F_prom_art:.0f} N | Deflexión máx = {imp.dtheta_max_deg:.2f}°")
    print(f"  Libro de Energía: Inicial = {imp.Ec_i_tronco:.1f} J | Disipada = {imp.E_disipada:.1f} J ({imp.pct_disipada:.1f} %)")
    print(f"  Verificación no tautológica de L_O: discrepancia instantáneo vs. dinámico = {imp.error_impulso_instantaneo_pct:.1f} %")

    print("\n--- 4. REGISTRO DE VERIFICACIÓN NUMÉRICA ---")
    verifs = ejecutar_verificaciones_completas(cfg)
    for v in verifs:
        print(f"  [{v.estado}] {v.prueba}: {v.valor_obtenido} (Tol: {v.tolerancia})")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI del Simulador de Física Mecánica - Ciénaga de Mallorquín")
    parser.add_argument("--reporte", action="store_true", help="Imprime reporte técnico completo en consola")
    parser.add_argument("--graficar", action="store_true", help="Genera y guarda todas las figuras en exports/")
    parser.add_argument("--montecarlo", type=int, default=0, help="Ejecuta simulación Monte Carlo con N corridas")
    parser.add_argument("--exportar_csv", action="store_true", help="Exporta tablas de resultados a CSV")
    parser.add_argument("--exportar_json", action="store_true", help="Exporta configuración a JSON")
    args = parser.parse_args()

    cfg = SimulatorConfig()

    if args.exportar_json:
        with open("parametros_adoptados.json", "w", encoding="utf-8") as f:
            f.write(cfg.to_json())
        print("Configuración exportada a parametros_adoptados.json")

    if args.exportar_csv:
        os.makedirs("exports", exist_ok=True)
        esc_res = []
        for nom, f_val in [("A_Libre", 0.0), ("B_Parcial", 0.30), ("C_Fuerte", 0.60)]:
            fl = simular_flujo_rk4(cfg.flujo, f=f_val)
            rj = calcular_estatica_rejilla(cfg.rejilla, f=f_val, u_corriente=1.00)
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
        pd.DataFrame(esc_res).to_csv("exports/resumen_escenarios.csv", index=False)
        print("Resumen de escenarios exportado a exports/resumen_escenarios.csv")

    if args.montecarlo > 0:
        mc = simulacion_montecarlo(cfg, n_corridas=args.montecarlo, seed=42)
        print(f"\nMonte Carlo ({mc.n_corridas} corridas, semilla 42):")
        print(f"  N_tope: P5={mc.N_tope_p5:.1f} N, P50={mc.N_tope_p50:.1f} N, P95={mc.N_tope_p95:.1f} N | Media={mc.N_tope_media:.1f} N")
        print(f"  F_prom: P5={mc.F_prom_p5:.1f} N, P50={mc.F_prom_p50:.1f} N, P95={mc.F_prom_p95:.1f} N")
        print(f"  θ_máx:  P5={mc.dtheta_p5:.2f}°, P50={mc.dtheta_p50:.2f}°, P95={mc.dtheta_p95:.2f}°")
        print(f"  Probabilidad de apertura: {mc.prob_apertura_pct:.1f} %")

    if args.graficar:
        generar_graficos_completos(cfg)

    # Por defecto imprimir reporte
    if args.reporte or (not args.graficar and args.montecarlo == 0 and not args.exportar_csv and not args.exportar_json):
        imprimir_reporte_completo(cfg)
