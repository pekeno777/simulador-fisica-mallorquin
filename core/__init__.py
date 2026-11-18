"""
Módulo del núcleo de física para el Simulador de Física Mecánica - Ciénaga de Mallorquín.
"""

from .parameters import (
    FlujoParams,
    TroncoParams,
    RejillaParams,
    ImpactoParams,
    OxigenoParams,
    SimulatorConfig
)
from .flujo import calcular_flujo_analitico, simular_flujo_rk4, FlujoResult
from .tronco import simular_tronco_rk4, TroncoResult
from .rejilla import (
    calcular_estatica_rejilla,
    RejillaResult,
    calcular_f_apertura,
    calcular_rejilla_abp3,
    RejillaABP3Result
)
from .impacto import calcular_impacto_2x2, simular_contacto_dinamico, ImpactoResult
from .oxigeno import calcular_oxigeno_contexto, OxigenoResult
from .sensibilidad_montecarlo import barrido_sensibilidad, simulacion_montecarlo, mapa_apertura
from .verificacion import ejecutar_verificaciones_completas

__all__ = [
    "FlujoParams",
    "TroncoParams",
    "RejillaParams",
    "ImpactoParams",
    "OxigenoParams",
    "SimulatorConfig",
    "calcular_flujo_analitico",
    "simular_flujo_rk4",
    "FlujoResult",
    "simular_tronco_rk4",
    "TroncoResult",
    "calcular_estatica_rejilla",
    "RejillaResult",
    "calcular_f_apertura",
    "calcular_rejilla_abp3",
    "RejillaABP3Result",
    "calcular_impacto_2x2",
    "simular_contacto_dinamico",
    "ImpactoResult",
    "calcular_oxigeno_contexto",
    "OxigenoResult",
    "barrido_sensibilidad",
    "simulacion_montecarlo",
    "mapa_apertura",
    "ejecutar_verificaciones_completas",
]
