"""
Parámetros y configuraciones del Simulador de Física Mecánica - Ciénaga de Mallorquín.
Todos los valores son ADOPTADOS de referencia y no mediciones de campo.
"""

from dataclasses import dataclass, asdict, field
import json
import numpy as np


@dataclass
class FlujoParams:
    """Parámetros adoptados para el flujo en el tramo del box culvert."""
    d: float = 6.00          # Longitud del tramo horizontal (m) [ADOPTADO]
    b: float = 2.00          # Ancho del box culvert (m) [ADOPTADO]
    h: float = 1.50          # Altura hidráulica (m) [ADOPTADO]
    v1: float = 1.00         # Velocidad media en entrada (m/s) [ADOPTADO]
    rho: float = 1000.0      # Densidad del agua (kg/m³) [ADOPTADO]
    g: float = 9.81          # Aceleración de la gravedad (m/s²) [ADOPTADO]
    P1: float = 7357.5       # Presión manométrica de entrada al centroide = rho*g*(h/2) (Pa) [ADOPTADO]
    f: float = 0.30          # Fracción obstruida del área de salida f in [0, 0.90) [ADOPTADO]

    @property
    def A1(self) -> float:
        return self.b * self.h

    @property
    def Q(self) -> float:
        return self.A1 * self.v1


@dataclass
class TroncoParams:
    """Parámetros adoptados para el tronco flotante (partícula en traslación)."""
    rho_mad: float = 750.0   # Densidad madera tropical semidura (kg/m³) [ADOPTADO]
    D_t: float = 0.250       # Diámetro del tronco (m) [ADOPTADO]
    L_t: float = 1.50        # Longitud del tronco (m) [ADOPTADO]
    C_D_t: float = 1.10      # Coeficiente de arrastre cilindro transversal [ADOPTADO]
    A_sub_ratio: float = 0.70# Fracción sumergida del área frontal proyectada [ADOPTADO]
    v0: float = 0.500        # Velocidad inicial del tronco en x=0 (m/s) [ADOPTADO]
    u_corriente: float = 2.00# Rapidez de corriente adoptada (m/s) [ADOPTADO]

    @property
    def V_t(self) -> float:
        return float(np.pi * (self.D_t / 2.0)**2 * self.L_t)

    @property
    def m_t(self) -> float:
        return self.rho_mad * self.V_t

    @property
    def W_t(self) -> float:
        return self.m_t * 9.81

    @property
    def A_sub_t(self) -> float:
        return self.A_sub_ratio * self.D_t * self.L_t


@dataclass
class RejillaParams:
    """Parámetros adoptados para la rejilla articulada contrapesada (cuerpo rígido en rotación)."""
    rho_s: float = 7850.0    # Densidad del acero estructural (kg/m³) [ADOPTADO]
    theta0_deg: float = 15.0 # Ángulo de inclinación respecto a la vertical (grados) [ADOPTADO]
    H_eje: float = 2.10      # Altura del eje de articulación O sobre el fondo (m) [ADOPTADO]
    h_agua: float = 1.50     # Tirante hidráulico en la rejilla (m) [ADOPTADO]
    b_canal: float = 2.00    # Ancho del canal de acceso (m) [ADOPTADO]
    r_c: float = 0.850       # Longitud del brazo de contrapeso (m) [ADOPTADO]
    m_brazo: float = 18.0    # Masa del brazo del contrapeso (kg) [ADOPTADO]
    m_c: float = 250.0       # Masa del contrapeso (kg) [ADOPTADO]
    C_D_r: float = 1.80      # Coeficiente de arrastre de rejilla con acumulación [ADOPTADO]
    u_corriente: float = 1.00# Rapidez nominal del flujo sobre rejilla (m/s) [ADOPTADO]

    @property
    def theta0(self) -> float:
        return float(np.radians(self.theta0_deg))

    @property
    def L_r(self) -> float:
        return self.H_eje / float(np.cos(self.theta0))

    @property
    def A_canal(self) -> float:
        return self.b_canal * self.h_agua

    @property
    def V_rej(self) -> float:
        # 15 barrotes verticales + 3 soleras horizontales + 2 montantes laterales
        return (15 * (self.L_r * 0.050 * 0.010) +
                3 * (1.90 * 0.060 * 0.012) +
                2 * (self.L_r * 0.060 * 0.010))

    @property
    def m_r(self) -> float:
        return self.rho_s * self.V_rej

    @property
    def W_r(self) -> float:
        return self.m_r * 9.81

    @property
    def r_G(self) -> float:
        return self.L_r / 2.0

    @property
    def L_sum_r(self) -> float:
        return self.h_agua / float(np.cos(self.theta0))

    @property
    def s_libre(self) -> float:
        return self.L_r - self.L_sum_r

    @property
    def r_D(self) -> float:
        return self.s_libre + 0.5 * self.L_sum_r

    @property
    def B_r(self) -> float:
        sub_frac = self.L_sum_r / self.L_r
        return (self.m_r * sub_frac / self.rho_s) * 1000.0 * 9.81

    @property
    def I_rej_O(self) -> float:
        return (1.0 / 3.0) * self.m_r * self.L_r**2

    @property
    def I_brazo_O(self) -> float:
        return (1.0 / 3.0) * self.m_brazo * self.r_c**2

    @property
    def I_c_O(self) -> float:
        return self.m_c * self.r_c**2

    @property
    def I_O_total(self) -> float:
        return self.I_rej_O + self.I_brazo_O + self.I_c_O


@dataclass
class ImpactoParams:
    """Parámetros adoptados para el choque tronco-rejilla."""
    y_imp: float = 1.42      # Altura de impacto del tronco sobre el fondo (m) [ADOPTADO]
    dt_imp: float = 0.060    # Duración finita del impacto de contacto (s) [ADOPTADO]
    e: float = 0.00          # Coeficiente de restitución (0=retenido plástico, 0.3=elástico) [ADOPTADO]
    k_contacto: float = 5.0e5# Rigidez elástica de contacto (N/m) para modelo dinámico [ADOPTADO]
    c_contacto: float = 1.2e3# Amortiguamiento de contacto (N·s/m) para modelo dinámico [ADOPTADO]

    def r_imp(self, H_eje: float, theta0: float) -> float:
        dy = H_eje - self.y_imp
        return dy / float(np.cos(theta0))


@dataclass
class OxigenoParams:
    """Parámetros para contextualización ambiental y orden de magnitud de oxígeno."""
    OD_sat: float = 8.30     # Oxígeno disuelto a saturación a 25 °C agua dulce (mg/L = g/m³) [ADOPTADO]
    demanda_esteq_celulosa: float = 1.19  # g O2 por gramo de celulosa oxidada estequiométrica [ADOPTADO]


@dataclass
class SimulatorConfig:
    """Configuración maestra del simulador."""
    flujo: FlujoParams = field(default_factory=FlujoParams)
    tronco: TroncoParams = field(default_factory=TroncoParams)
    rejilla: RejillaParams = field(default_factory=RejillaParams)
    impacto: ImpactoParams = field(default_factory=ImpactoParams)
    oxigeno: OxigenoParams = field(default_factory=OxigenoParams)
    
    # Regla: una sola velocidad de corriente por defecto (unificada)
    unificar_velocidad_corriente: bool = False  # False para reproducir valores exactos del informe por defecto (u_tronco=2.0, u_rejilla=1.0)
    u_unificada: float = 1.00                   # Velocidad si se activa la unificación
    
    # Metadatos de reproducibilidad
    seed: int = 42
    version: str = "2.1.0-Etapa3-CUC"
    etiqueta_estado: str = "VERIFICACIÓN NUMÉRICA (PARÁMETROS ADOPTADOS - NO VALIDADO EN CAMPO)"

    def aplicar_unificacion_corriente(self, u_val: float):
        """Aplica la regla de una sola velocidad de corriente si se habilita."""
        self.unificar_velocidad_corriente = True
        self.u_unificada = u_val
        self.tronco.u_corriente = u_val
        self.rejilla.u_corriente = u_val

    def restaurar_informe(self):
        """Restaura los valores exactos publicados en el informe del proyecto."""
        self.flujo = FlujoParams()
        self.tronco = TroncoParams()
        self.rejilla = RejillaParams()
        self.impacto = ImpactoParams()
        self.oxigeno = OxigenoParams()
        self.unificar_velocidad_corriente = False
        self.u_unificada = 1.00

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SimulatorConfig":
        data = json.loads(json_str)
        cfg = cls()
        cfg.flujo = FlujoParams(**data.get("flujo", {}))
        cfg.tronco = TroncoParams(**data.get("tronco", {}))
        cfg.rejilla = RejillaParams(**data.get("rejilla", {}))
        cfg.impacto = ImpactoParams(**data.get("impacto", {}))
        cfg.oxigeno = OxigenoParams(**data.get("oxigeno", {}))
        cfg.unificar_velocidad_corriente = data.get("unificar_velocidad_corriente", False)
        cfg.u_unificada = data.get("u_unificada", 1.00)
        cfg.seed = data.get("seed", 42)
        cfg.version = data.get("version", "2.1.0-Etapa3-CUC")
        return cfg
