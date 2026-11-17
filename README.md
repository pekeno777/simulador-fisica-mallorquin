# Simulador Físico-Mecánico: Box Culvert y Rejilla de Retención
### Ciénaga de Mallorquín — Proyecto ABP Etapa 3 (Física Mecánica, Universidad de la Costa - 2026)
**Autores:** Javith Santana, Kevin Díaz | **Tutor:** Luis Polo Ariza  
**Etiqueta metodológica obligatoria:** `VERIFICACIÓN NUMÉRICA (PARÁMETROS ADOPTADOS - NO VALIDADO EN CAMPO)`

---

## 1. Alcance y Principios de Diseño
Este simulador tiene un **único propósito**: permitir que el equipo de trabajo y el docente modifiquen cualquier parámetro adoptado de entrada y observen en tiempo real cómo varían las fuerzas hidrodinámicas, torques gravitatorios y de contrapeso, tiempos de tránsito, energías mecánicas y estados de estabilidad estructural.

* **Núcleo de física desacoplado:** El motor de cálculo reside en funciones puras y tipadas dentro del paquete `core/`, garantizando total reproducibilidad en el informe escrito, la suite de pruebas unitarias (`pytest`), la consola CLI, la aplicación interactiva Streamlit y el panel web autónomo.
* **Transparencia en el rigor:** Todo valor se declara explícitamente como **adoptado** de referencia. Se prohíbe el uso de la palabra "validado" (reservada para aforos experimentales de campo).

---

## 2. Estructura del Proyecto

```text
simulador_mallorquin/
├── app.py                      # Interfaz interactiva completa en Streamlit
├── cli_simulador.py            # Ejecutor CLI y generador de gráficos en alta resolución (300 dpi)
├── parametros_adoptados.json   # Exportación canónica en JSON de parámetros adoptados
├── core/                       # Núcleo de física mecánica pura
│   ├── __init__.py
│   ├── parameters.py           # Dataclasses tipadas de parámetros y configuraciones
│   ├── flujo.py                # Flujo en el tramo: Bernoulli, volumen de control y RK4
│   ├── tronco.py               # Partícula: arrastre hidrodinámico no lineal e integrador RK4
│   ├── rejilla.py              # Cuerpo rígido: equilibrio estático, inercia, torques y N_tope
│   ├── impacto.py              # Colisión tronco-rejilla: sistema 2x2, Libro de Energía y contacto dinámico
│   ├── oxigeno.py              # Contextualización limnológica: orden de magnitud de transporte de O2
│   ├── sensibilidad_montecarlo.py # Barridos paramétricos y simulación Monte Carlo
│   └── verificacion.py         # 13 pruebas de consistencia dimensional, teoremas y convergencia
├── tests/
│   └── test_regresion.py       # Suite pytest con regresión exacta contra las cifras del informe
├── exports/                    # Figuras científicas PNG (300 dpi) y tablas CSV generadas
│   ├── fig1_cinematica_xt_vt.png
│   ├── fig2_fuerza_longitud_dFdx.png
│   ├── fig3_trabajo_tronco_Fx.png
│   ├── fig4_N_tope_vs_f.png
│   ├── fig5_libro_energia.png
│   ├── fig6_mapa_apertura_fu.png
│   └── resumen_escenarios.csv
└── web/
    └── simulador.html          # Panel interactivo web autónomo (HTML/JS/Canvas, sin instalación)
```

---

## 3. Guía de Ejecución Rápida

### A. Ejecutar la Aplicación Interactiva (Streamlit)
```bash
streamlit run app.py
```

### B. Ejecutar la Suite de Pruebas de Regresión (Pytest)
```bash
pytest tests/ -v
```

### C. Generar Gráficas Científicas y Exportaciones (CLI)
```bash
python cli_simulador.py --reporte --graficar --exportar_csv --exportar_json --montecarlo 1000
```

### D. Ejecución Web Autónoma (Sin Instalación)
Basta con abrir el archivo `web/simulador.html` con doble clic en Google Chrome, Edge o Firefox para disponer de controles en tiempo real, semáforo dinámico, gráficas interactivas y simulación Monte Carlo.

---

## 4. Reglas Físicas Implementadas para Corregir Inconsistencias

1. **Unificación de la velocidad de corriente:**
   * En el informe base se empleaba $u_w = 2.0$ m/s para el arrastre del tronco en aproximación y $u_{\text{nom}} = 1.0$ m/s para el flujo en la boca. El simulador incluye una opción de **unificación estricta** (`u_corriente` común) y documenta el desacoplamiento cuando se selecciona.
2. **Misma base de comparación en el impacto:**
   * Tanto para la rejilla rígida bloqueada como para la articulada con contrapeso, la velocidad de choque se proyecta sobre la misma componente normal $v_n = v_1 \cos\theta_0$, garantizando una comparación dimensionalmente equitativa.
3. **Verificación no tautológica de $L_O$ (Hipótesis de impulso instantáneo):**
   * Además del sistema algebraico $2 \times 2$, se evalúa la integral dinámica de los torques externos finitos (peso, contrapeso, empuje hidrostático y arrastre) durante el tiempo finito $\Delta t_{\text{imp}} = 0.060$ s. Se demuestra numéricamente que la hipótesis de impulso instantáneo introduce una discrepancia de entre el 10 % y el 20 %, tal como anticipaba la Tabla VIII del informe.

---

## 5. Tabla de Regresión de Cifras Clave

| Magnitud Física | Valor en Informe | Simulador Numérico | Tolerancia | Estado |
| :--- | :---: | :---: | :---: | :---: |
| **Tiempo tránsito tronco ($t_{\text{RK4}}$)** | 3,515 s | 3,515 s | $< 0,005$ s | **PASA** |
| **Trabajo en tronco ($W = \Delta Ec$)** | 92,6 J | 92,62 J | $< 0,2$ J | **PASA** |
| **Fuerza neta flujo Esc. B ($\Sigma F_x$)** | 1.285,7 N | 1.285,7 N | $< 0,2$ N | **PASA** |
| **Punto de apertura pasiva ($f_{\text{ap}}$)** | 0,378 | 0,378 | $< 0,005$ | **PASA** |
| **Fuerza en tope Esc. B ($N_{\text{tope}}$)** | 111,8 N | 111,8 N | $< 0,5$ N | **PASA** |
| **Vel. angular impacto ($e=0,00$, $\omega_f$)** | 0,1434 rad/s | 0,1435 rad/s | $< 0,001$ rad/s | **PASA** |
| **Prueba sensibilidad ($u = 1 \to 2$ m/s)** | $N_{\text{tope}} < 0$ | $-1.926$ N | Negativo | **PASA** |
| **Límite sección libre ($f = 0$)** | $v_2 = v_1, \Sigma F_x = 0$ | $1,000$ m/s, $0,0$ N | $< 10^{-12}$ | **PASA** |
| **Límite inercia infinita ($I_O \to \infty$)** | $\omega_f \to 0$ | $0,0000$ rad/s | $< 10^{-5}$ | **PASA** |
