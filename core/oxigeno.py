"""
Módulo de Oxigenación y Contextualización Ambiental (Química y Recursos Naturales).
Calcula el aporte advectivo de oxígeno disuelto y la biomasa celulósica oxidable equivalente.
Marcado rigurosamente como estimación ilustrativa de ORDEN DE MAGNITUD (no validada en campo).
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from .parameters import OxigenoParams, FlujoParams


@dataclass
class OxigenoRow:
    """Fila para la tabla de contextualización ambiental."""
    escenario: str
    f: float
    fraccion_libre: float
    caudal_m3_s: float
    aporte_o2_t_dia: float
    materia_organica_t_dia: float
    estado_rejilla: str


@dataclass
class OxigenoResult:
    """Resultado del módulo de oxígeno."""
    OD_sat_mg_L: float
    demanda_esteq_celulosa: float
    filas: List[OxigenoRow]
    nota: str


def calcular_oxigeno_contexto(
    oxigeno_params: OxigenoParams,
    flujo_params: FlujoParams = None
) -> OxigenoResult:
    """
    Genera la tabla ilustrativa de orden de magnitud (Tabla VI del proyecto).
    Demuestra que el beneficio radica en interceptar la tarulla antes de su descomposición.
    """
    OD = oxigeno_params.OD_sat  # mg/L = g/m³
    demanda = oxigeno_params.demanda_esteq_celulosa  # g O2 / g materia
    Q_libre = flujo_params.Q if flujo_params is not None else 3.00

    # Conversión: Q (m³/s) * OD (g/m³) * 86400 s/día / 1e6 g/t = t O₂ / día
    factor_t_dia = 86400.0 / 1e6

    escenarios = [
        ("A (Libre)", 0.00, "Retiene limpia (sección íntegra)"),
        ("B (Parcial)", 0.30, "Retiene (alerta de mantenimiento preventivo)"),
        ("C (Fuerte)", 0.60, "Bascula / Alivio pasivo de flujo"),
    ]

    filas = []
    for nom, f_val, estado in escenarios:
        frac_libre = 1.0 - f_val
        # Caudal ilustrativo decreciente si se asume Q proportional a (1 - f)
        Q_efectivo = Q_libre * frac_libre
        aporte_o2 = Q_efectivo * OD * factor_t_dia
        mo_oxidable = aporte_o2 / demanda if demanda > 0 else 0.0

        filas.append(OxigenoRow(
            escenario=nom,
            f=f_val,
            fraccion_libre=frac_libre,
            caudal_m3_s=Q_efectivo,
            aporte_o2_t_dia=aporte_o2,
            materia_organica_t_dia=mo_oxidable,
            estado_rejilla=estado
        ))

    nota = (
        "ORDEN DE MAGNITUD ILUSTRATIVO: El cálculo asume agua dulce a saturación (8,30 mg/L O₂ a 25 °C) "
        "y estequiometría de celulosa (C₆H₁₀O₅ + 6 O₂ → 6 CO₂ + 5 H₂O, 1,19 g O₂/g). "
        "Constituye una proyección conceptual de orden físico-químico, no una validación limnológica de campo."
    )

    return OxigenoResult(
        OD_sat_mg_L=OD,
        demanda_esteq_celulosa=demanda,
        filas=filas,
        nota=nota
    )
