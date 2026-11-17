"""
Pruebas de regresión, casos límite y sensibilidad física para el simulador de Física Mecánica.
Verifica la exactitud matemática contra las cifras publicadas en el informe técnico del proyecto.
"""

import pytest
import numpy as np
import sys
import os

# Asegurar importación del núcleo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import (
    SimulatorConfig,
    FlujoParams,
    TroncoParams,
    RejillaParams,
    ImpactoParams,
    simular_flujo_rk4,
    simular_tronco_rk4,
    calcular_estatica_rejilla,
    calcular_f_apertura,
    calcular_impacto_2x2,
    calcular_oxigeno_contexto,
    ejecutar_verificaciones_completas
)


class TestRegresionCifrasInforme:
    """Verifica la reproducción exacta de las cifras centrales del informe."""

    def setup_method(self):
        self.cfg = SimulatorConfig()

    def test_flujo_escenario_b_fuerza_neta(self):
        """Regresión: Sigma Fx = 1285.7 N en Escenario B (f = 0.30)."""
        res = simular_flujo_rk4(self.cfg.flujo, f=0.30)
        assert abs(res.F_vc - 1285.7) < 0.2
        assert abs(res.v2 - 1.4286) < 0.001
        assert abs(res.Rx - 6428.9) < 1.0

    def test_tronco_tiempo_y_trabajo(self):
        """Regresión: t_RK4 = 3.515 s y W = 92.6 J para el tronco flotante."""
        res = simular_tronco_rk4(self.cfg.tronco, u_corriente=2.00)
        assert abs(res.t_rk4 - 3.515) < 0.005
        assert abs(res.W_trapecio - 92.6) < 0.2
        assert abs(res.dEc - 92.6) < 0.2
        assert abs(res.vf_rk4 - 1.899) < 0.005

    def test_rejilla_apertura_y_fuerza_tope(self):
        """Regresión: f_ap = 0.378 y N_tope = 112 N en Escenario B."""
        f_ap = calcular_f_apertura(self.cfg.rejilla, u_corriente=1.00)
        assert abs(f_ap - 0.378) < 0.005

        res_B = calcular_estatica_rejilla(self.cfg.rejilla, f=0.30, u_corriente=1.00)
        assert abs(res_B.N_tope - 111.8) < 0.5  # ~112 N
        assert res_B.estado.startswith("CERRADA")

        # Escenario A (f=0): N_tope ~ 539 N
        res_A = calcular_estatica_rejilla(self.cfg.rejilla, f=0.00, u_corriente=1.00)
        assert abs(res_A.N_tope - 539.3) < 1.0

        # Escenario C (f=0.60): N_tope negativo (-315.7 N), supera par de cierre
        res_C = calcular_estatica_rejilla(self.cfg.rejilla, f=0.60, u_corriente=1.00)
        assert abs(res_C.N_tope - (-315.7)) < 1.0
        assert "ABIERTA" in res_C.estado

    def test_impacto_velocidad_angular_2x2(self):
        """Regresión: omega_f = 0.1434 rad/s en impacto plástico (e = 0.00)."""
        res = calcular_impacto_2x2(
            self.cfg.tronco,
            self.cfg.rejilla,
            self.cfg.impacto,
            v_impacto=1.899,
            e_override=0.00
        )
        assert abs(res.omega_f - 0.1434) < 0.001
        assert abs(res.J_art - 95.7) < 0.5
        assert abs(res.dtheta_max_deg - 1.15) < 0.05


class TestCasosLimite:
    """Verifica que el modelo respete los casos límite analíticos."""

    def setup_method(self):
        self.cfg = SimulatorConfig()

    def test_limite_seccion_libre_f_cero(self):
        """Límite f = 0: no hay contracción ni fuerza neta."""
        res = simular_flujo_rk4(self.cfg.flujo, f=0.00)
        assert abs(res.v2 - self.cfg.flujo.v1) < 1e-12
        assert abs(res.F_vc) < 1e-12
        assert abs(res.w_masa) < 1e-12
        assert abs(res.Rx) < 1e-12

    def test_limite_tronco_velocidad_igual_corriente(self):
        """Límite v0 -> u_w: la fuerza relativa es nula."""
        res = simular_tronco_rk4(self.cfg.tronco, u_corriente=self.cfg.tronco.v0)
        assert abs(res.Fs[0]) < 1e-9
        assert abs(res.dEc) < 1e-9
        assert abs(res.W_trapecio) < 1e-9

    def test_limite_inercia_infinita_impacto(self):
        """Límite I_O -> inf: la rejilla se comporta como pared infinitamente rígida (omega_f -> 0)."""
        r_inf = RejillaParams(
            m_c=1e9  # contrapeso gigantesco
        )
        res = calcular_impacto_2x2(self.cfg.tronco, r_inf, self.cfg.impacto, v_impacto=1.899, e_override=0.20)
        assert abs(res.omega_f) < 1e-5
        # Velocidad de rebote igual a -e * v_n
        assert abs(res.v_t_n_f - (-0.20 * res.v_normal_i)) < 1e-5


class TestSensibilidadFisica:
    """Verifica la sensibilidad física requerida."""

    def setup_method(self):
        self.cfg = SimulatorConfig()

    def test_sensibilidad_u_sube_de_1_a_2(self):
        """
        Regla de prueba de sensibilidad:
        Si u sube de 1 a 2 m/s en Escenario B (f = 0.30),
        el arrastre se cuadruplica y N_tope se vuelve negativo (la rejilla bascula).
        """
        res_u1 = calcular_estatica_rejilla(self.cfg.rejilla, f=0.30, u_corriente=1.00)
        assert res_u1.N_tope > 0.0  # Positivo a 1 m/s (~111.8 N)

        res_u2 = calcular_estatica_rejilla(self.cfg.rejilla, f=0.30, u_corriente=2.00)
        assert res_u2.N_tope < 0.0  # Negativo a 2 m/s (~ -1926 N)
        assert "ABIERTA" in res_u2.estado


class TestVerificacionesPanel:
    """Comprueba que todas las pruebas del panel de verificación pasen."""

    def test_todas_verificaciones_pasan(self):
        cfg = SimulatorConfig()
        items = ejecutar_verificaciones_completas(cfg)
        for it in items:
            assert it.estado == "PASA", f"Falló verificación: {it.prueba} ({it.observacion})"
