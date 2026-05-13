# Predicción de Tráfico Basada en LSTM para Gestión Proactiva de Recursos en Redes 5G

# LSTM-Based Traffic Prediction for Proactive Resource Management in 5G Networks

Suite de simulación en Python para el artículo / Python simulation suite for the paper:

> **"Predicción de Tráfico Basada en LSTM para Gestión Proactiva de Recursos en Redes 5G"**
> IEEE — proposed submission

---

## Marco Propuesto / The Proposed Framework

### Motivación / Motivation

**ES:** La gestión eficiente de recursos en redes 5G representa un desafío crítico debido a la heterogeneidad de servicios (eMBB, URLLC, mMTC), la alta dinámica del tráfico y los requisitos estrictos de calidad de servicio (QoS). Los enfoques tradicionales de gestión de recursos radio (RRM) son principalmente reactivos. Este artículo presenta un estudio exhaustivo sobre técnicas de predicción de tráfico basadas en LSTM para habilitar la gestión proactiva de recursos, anticipando demandas futuras y optimizando la asignación de recursos antes de que se produzcan los picos de carga.

**EN:** Efficient resource management in 5G networks is a critical challenge due to service heterogeneity (eMBB, URLLC, mMTC), high traffic dynamics, and strict QoS requirements. Traditional Radio Resource Management (RRM) approaches are primarily reactive. This paper presents an exhaustive study on LSTM-based traffic prediction techniques to enable proactive resource management, anticipating future demands and optimizing resource allocation before load peaks occur.

### Contribuciones Clave / Key Contributions

1. **Fundamentación teórica rigurosa de la arquitectura LSTM** (Sección II) / Rigorous theoretical foundation of the LSTM architecture (Section II)
2. **Caracterización del tráfico 5G con modelos estocásticos** (Sección III) / 5G traffic characterization with stochastic models (Section III)
3. **Arquitectura LSTM avanzada con atención y procesamiento multi-resolución** (Sección IV) / Advanced LSTM architecture with attention and multi-resolution processing (Section IV)
4. **Estrategias de gestión proactiva de recursos** (Sección V) / Proactive resource management strategies (Section V)
5. **Algoritmos paso a paso para implementación práctica** (Sección VI) — 5 algoritmos / Step-by-step algorithms for practical implementation (Section VI) — 5 algorithms
6. **Evaluación exhaustiva de rendimiento** (Sección VII) / Comprehensive performance evaluation (Section VII)

### Resultados Clave / Key Results

- LSTM alcanza RMSE < 4%, R² > 0.94 en el dataset Telecom Italia Milano / LSTM achieves RMSE < 4%, R² > 0.94 on Telecom Italia Milano dataset
- 54% de reducción de RMSE vs ARIMA, 34% vs LSTM sin atención / 54% RMSE reduction vs ARIMA, 34% vs LSTM without attention
- Gestión proactiva: 35-42% reducción de bloqueo, 28-34% reducción de latencia, 26% ahorro energético / Proactive management: 35-42% blocking reduction, 28-34% latency reduction, 26% energy savings

---

## Estructura del Repositorio / Repository Structure

```
LSTM_Traffic5G/
├── LSTM_Traffic_Prediction_5G_IEEE.md    # Artículo completo / Full article
├── README.MD                             # Este archivo / This file
└── simulations/
    ├── requirements.txt                   # Dependencias Python / Python dependencies
    ├── generate_datasets.py               # Generación de datasets (Dataset 1-3) / Dataset generation
    ├── models.py                          # Todas las arquitecturas de redes neuronales (13 modelos) / All neural network architectures (13 models)
    ├── train.py                           # Pipeline de entrenamiento (Algoritmo 1) / Training pipeline (Algorithm 1)
    ├── predict.py                         # Predicción en tiempo real (Algoritmo 2) / Real-time prediction (Algorithm 2)
    ├── proactive_resource_management.py   # Gestión de recursos (Algoritmos 3-5) / Resource management (Algorithms 3-5)
    ├── run_benchmarks.py                  # Evaluación comparativa completa / Full comparative evaluation
    ├── plot_figures.py                    # Figuras con calidad de publicación / Publication-quality figures
    ├── results/                           # Checkpoints y resultados .npz / Saved model checkpoints and .npz results
    └── figures/                           # Figuras generadas (.png + .pdf) / Generated figures (.png + .pdf)
```

---

## Dependencias / Dependencies

| Paquete / Package | Versión / Version | Notas / Notes |
|-------------------|-------------------|---------------|
| Python | ≥ 3.9 | |
| PyTorch | ≥ 2.0 | CPU suficiente; CUDA opcional / CPU sufficient; CUDA optional |
| NumPy | ≥ 1.23 | |
| SciPy | ≥ 1.9 | |
| Matplotlib | ≥ 3.6 | Generación de figuras / Figure generation |
| scikit-learn | ≥ 1.2 | Baselines: SVR, Random Forest, GradientBoosting |
| pandas | ≥ 1.5 | Manejo de datos / Data handling |
| statsmodels | ≥ 0.14 | Baselines: ARIMA/SARIMA |

### Instalación / Installation

```bash
cd LSTM_Traffic5G/simulations
pip install -r requirements.txt
```

O instalar manualmente / Or install manually:

```bash
pip install torch>=2.0 numpy>=1.23 scipy>=1.9 matplotlib>=3.6 scikit-learn>=1.2 pandas>=1.5 statsmodels>=0.14
```

---

## Datasets

### Dataset 1 — Telecom Italia (Milano)

**ES:** Dataset real de tráfico celular en Milán, Italia.
**EN:** Real dataset of cellular traffic in Milan, Italy.

- **Periodo / Period:** Noviembre–Diciembre 2013 / November–December 2013
- **Granularidad / Granularity:** 10 minutos / 10 minutes
- **Cobertura / Coverage:** 100 celdas, 20 km² / 100 cells, 20 km²
- **Descarga / Download:** https://dandelion.eu/datagems/SpaseTelecomItalia/
- **Alternativo / Alternative:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

### Dataset 2 — Shanghai Telecom

**ES:** Tráfico 3G/4G de Shanghai, China.
**EN:** 3G/4G traffic from Shanghai, China.

- **Periodo / Period:** 6 meses / 6 months
- **Granularidad / Granularity:** 15 minutos / 15 minutes
- **Cobertura / Coverage:** 3233 estaciones base / 3233 base stations
- **Descarga / Download:** https://sguangwang.com/TelescopeData.html
- **Alternativo / Alternative:** Disponible bajo solicitud académica / Available through academic request

### Dataset 3 — 5G Sintético / Synthetic 5G

**ES:** Generado por `generate_datasets.py` con modelos de tráfico realistas incluyendo patrones eMBB, URLLC y mMTC.
**EN:** Generated by `generate_datasets.py` with realistic traffic models including eMBB, URLLC, and mMTC patterns.

---

## Descripción de Scripts / Script Descriptions

### `generate_datasets.py` — Generación de Datos / Data Generation

**ES:** Genera los tres datasets usados en el artículo: descarga/preprocesa datos reales (Milano, Shanghai) y sintetiza tráfico 5G realista con patrones eMBB, URLLC y mMTC.
**EN:** Generates the three datasets used in the article: downloads/preprocesses real data (Milano, Shanghai) and synthesizes realistic 5G traffic with eMBB, URLLC, and mMTC patterns.

- **Sección del artículo / Article section:** Sección III (Caracterización del Tráfico 5G) / Section III (5G Traffic Characterization)
- **Funciones clave / Key functions:** `generate_milano_dataset()`, `generate_shanghai_dataset()`, `generate_synthetic_5g()`, `add_stochastic_patterns()`
- **Ejecución / Run:**
  ```bash
  python generate_datasets.py
  python generate_datasets.py --self-test   # Verificar funcionamiento / Verify functionality
  ```
- **Salida esperada / Expected output:** Archivos `.csv` y `.npz` en `results/` con series temporales de tráfico / `.csv` and `.npz` files in `results/` with traffic time series

### `models.py` — Arquitecturas de Redes Neuronales / Neural Network Architectures

**ES:** Implementa 13 arquitecturas de redes neuronales para predicción de tráfico, incluyendo el modelo LSTM propuesto con mecanismo de atención y procesamiento multi-resolución.
**EN:** Implements 13 neural network architectures for traffic prediction, including the proposed LSTM model with attention mechanism and multi-resolution processing.

- **Sección del artículo / Article section:** Sección IV (Arquitectura LSTM Avanzada) / Section IV (Advanced LSTM Architecture)
- **Modelos / Models:** `SimpleLSTM`, `AttentionLSTM` (propuesto), `MultiResolutionLSTM`, `GRU`, `BiLSTM`, `StackedLSTM`, `ConvLSTM`, `SimpleRNN`, `TransformerPredictor`, `SVRWrapper`, `RandomForestWrapper`, `GradientBoostingWrapper`, `ARIMAWrapper`
- **Ejecución / Run:**
  ```bash
  python models.py --self-test   # Verifica instanciación y forward pass de los 13 modelos / Verify instantiation and forward pass of all 13 models
  ```
- **Verificación / Verification:** Cada modelo debe producir tensores de salida con forma correcta sin errores / Each model must produce output tensors with correct shape without errors

### `train.py` — Pipeline de Entrenamiento / Training Pipeline

**ES:** Implementa el pipeline completo de entrenamiento con early stopping, learning rate scheduling, y validación. Corresponde al Algoritmo 1 del artículo.
**EN:** Implements the complete training pipeline with early stopping, learning rate scheduling, and validation. Corresponds to Algorithm 1 in the article.

- **Sección del artículo / Article section:** Sección VI, Algoritmo 1 (Entrenamiento LSTM) / Section VI, Algorithm 1 (LSTM Training)
- **Funciones clave / Key functions:** `train_model()`, `validate()`, `EarlyStopping`, `LRScheduler`
- **Ejecución / Run:**
  ```bash
  python train.py --self-test
  python train.py --dataset milano --model AttentionLSTM --epochs 200
  python train.py --dataset synthetic --model AttentionLSTM --epochs 100 --lr 0.001
  ```
- **Verificación de calidad / Quality verification:**
  1. La pérdida de entrenamiento debe decrecer a lo largo de las épocas / Training loss must decrease over epochs
  2. R² de validación > 0.85 para el LSTM propuesto / Validation R² > 0.85 for proposed LSTM
  3. Sin NaN en las pérdidas / No NaN in losses
- **Salida / Output:** Checkpoints del modelo en `results/` (e.g., `AttentionLSTM_best.pt`) / Model checkpoints in `results/`

### `predict.py` — Predicción en Tiempo Real / Real-Time Prediction

**ES:** Implementa predicción en tiempo real usando modelos entrenados. Corresponde al Algoritmo 2 del artículo. Soporta predicción multi-horizonte (1-step, 6-step, 24-step).
**EN:** Implements real-time prediction using trained models. Corresponds to Algorithm 2 in the article. Supports multi-horizon prediction (1-step, 6-step, 24-step).

- **Sección del artículo / Article section:** Sección VI, Algoritmo 2 (Predicción en Tiempo Real) / Section VI, Algorithm 2 (Real-Time Prediction)
- **Funciones clave / Key functions:** `load_model()`, `predict_next()`, `predict_horizon()`, `sliding_window_predict()`
- **Ejecución / Run:**
  ```bash
  python predict.py --self-test
  python predict.py --model results/AttentionLSTM_best.pt --horizon 24
  python predict.py --model results/AttentionLSTM_best.pt --input data.csv --output predictions.csv
  ```
- **Salida esperada / Expected output:** Series de predicción con intervalos de confianza / Prediction series with confidence intervals

### `proactive_resource_management.py` — Gestión Proactiva de Recursos / Proactive Resource Management

**ES:** Implementa las estrategias de gestión proactiva de recursos basadas en predicciones LSTM. Incluye los Algoritmos 3 (Asignación Proactiva de RBs), 4 (Control de Potencia Predictivo), y 5 (Reconfiguración Predictiva de Slices).
**EN:** Implements proactive resource management strategies based on LSTM predictions. Includes Algorithm 3 (Proactive RB Allocation), Algorithm 4 (Predictive Power Control), and Algorithm 5 (Predictive Slice Reconfiguration).

- **Sección del artículo / Article section:** Secciones V–VI, Algoritmos 3-5 / Sections V–VI, Algorithms 3-5
- **Funciones clave / Key functions:** `proactive_rb_allocation()`, `predictive_power_control()`, `predictive_slice_reconfiguration()`, `evaluate_kpis()`
- **Ejecución / Run:**
  ```bash
  python proactive_resource_management.py --self-test
  python proactive_resource_management.py --predictions results/predictions.npz
  ```
- **Salida esperada / Expected output:** KPIs de gestión de recursos (tasa de bloqueo, latencia, eficiencia energética) / Resource management KPIs (blocking rate, latency, energy efficiency)

### `run_benchmarks.py` — Evaluación Comparativa / Comparative Evaluation

**ES:** Ejecuta la evaluación comparativa completa entrenando los 13 modelos en los 3 datasets y generando la Tabla I del artículo con métricas RMSE, MAE, MAPE y R².
**EN:** Runs the full comparative evaluation training all 13 models on all 3 datasets and generating Table I from the article with RMSE, MAE, MAPE, and R² metrics.

- **Sección del artículo / Article section:** Sección VII (Evaluación de Rendimiento) / Section VII (Performance Evaluation)
- **Funciones clave / Key functions:** `run_all_benchmarks()`, `generate_table_I()`, `compare_models()`
- **Ejecución / Run:**
  ```bash
  python run_benchmarks.py
  ```
- **Tiempo estimado / Estimated time:** ~30-120 min en CPU / ~30-120 min on CPU
- **Salida / Output:** Tabla I impresa en stdout + resultados en `results/benchmark_results.npz` / Table I printed to stdout + results in `results/benchmark_results.npz`

### `plot_figures.py` — Generación de Figuras / Figure Generation

**ES:** Genera todas las figuras del artículo con calidad de publicación IEEE. Requiere resultados previos de `run_benchmarks.py`.
**EN:** Generates all article figures with IEEE publication quality. Requires previous results from `run_benchmarks.py`.

- **Sección del artículo / Article section:** Todas las figuras del artículo / All article figures
- **Funciones clave / Key functions:** `plot_training_convergence()`, `plot_prediction_comparison()`, `plot_rmse_horizon()`, `plot_proactive_kpis()`, `plot_attention_weights()`
- **Ejecución / Run:**
  ```bash
  python plot_figures.py
  ```
- **Salida / Output:** Figuras `.png` y `.pdf` en `figures/` / `.png` and `.pdf` figures in `figures/`

---

## Ejecución del Pipeline Completo / Running the Full Simulation Pipeline

**ES:** Instrucciones paso a paso para reproducir TODOS los resultados del artículo.
**EN:** Step-by-step instructions to reproduce ALL article results.

```bash
# Paso 0 / Step 0: Instalar dependencias / Install dependencies
cd simulations
pip install -r requirements.txt

# Paso 1 / Step 1: Generar datasets sintéticos (o usar reales)
# Generate synthetic datasets (or use real ones)
python generate_datasets.py

# Paso 2 / Step 2: Verificar funcionamiento de componentes
# Verify component functionality
python generate_datasets.py --self-test
python models.py --self-test
python train.py --self-test
python predict.py --self-test
python proactive_resource_management.py --self-test

# Paso 3 / Step 3: Ejecutar benchmarks completos (entrena todos los modelos, produce Tabla I)
# Run full benchmarks (trains all models, produces Table I)
python run_benchmarks.py

# Paso 4 / Step 4: Generar todas las figuras
# Generate all figures
python plot_figures.py

# Paso 5 / Step 5: Entrenamiento individual (opcional)
# Individual training (optional)
python train.py --dataset milano --model AttentionLSTM --epochs 200

# Paso 6 / Step 6: Demo de predicción en tiempo real
# Real-time prediction demo
python predict.py --model results/AttentionLSTM_best.pt --horizon 24
```

---

## Archivos de Salida / Output Files

**ES:** Todos los resultados se guardan en `simulations/results/` y `simulations/figures/` (creados automáticamente).
**EN:** All outputs are saved to `simulations/results/` and `simulations/figures/` (auto-created).

| Archivo / File | Contenido / Contents |
|----------------|----------------------|
| `results/benchmark_results.npz` | Métricas de todos los modelos en todos los datasets / Metrics for all models on all datasets |
| `results/AttentionLSTM_best.pt` | Mejor checkpoint del modelo propuesto / Best checkpoint of proposed model |
| `results/{Model}_best.pt` | Checkpoints de cada modelo / Checkpoints for each model |
| `results/predictions.npz` | Predicciones generadas / Generated predictions |
| `figures/training_convergence.png` | Curvas de convergencia de entrenamiento / Training convergence curves |
| `figures/prediction_comparison.png` | Comparación de predicciones vs. valores reales / Prediction vs. actual comparison |
| `figures/rmse_vs_horizon.png` | RMSE en función del horizonte de predicción / RMSE vs. prediction horizon |
| `figures/proactive_kpis.png` | KPIs de gestión proactiva / Proactive management KPIs |

---

## Reproducción de Resultados del Artículo / Reproducing Article Results

### Tabla I — Comparación de Precisión de Predicción / Prediction Accuracy Comparison

**ES:** Generada automáticamente por `run_benchmarks.py`. Valores de referencia del artículo:
**EN:** Automatically generated by `run_benchmarks.py`. Reference values from the article:

| Método / Method | RMSE (%) | MAE (%) | MAPE (%) | R² |
|-----------------|----------|---------|----------|-----|
| **LSTM con Atención (propuesto) / LSTM with Attention (proposed)** | **3.82** | **2.91** | **4.15** | **0.945** |
| LSTM Multi-Resolución / Multi-Resolution LSTM | 4.12 | 3.15 | 4.52 | 0.937 |
| LSTM Bidireccional / Bidirectional LSTM | 4.35 | 3.28 | 4.78 | 0.931 |
| LSTM Apilado / Stacked LSTM | 4.51 | 3.42 | 4.95 | 0.926 |
| GRU | 4.68 | 3.55 | 5.12 | 0.920 |
| LSTM Simple / Simple LSTM | 5.12 | 3.89 | 5.58 | 0.905 |
| RNN Simple / Simple RNN | 6.45 | 4.92 | 7.01 | 0.862 |
| ARIMA/SARIMA | 8.31 | 6.33 | 9.05 | 0.781 |
| SVR | 7.15 | 5.44 | 7.82 | 0.834 |

### KPIs de Gestión Proactiva / Proactive Management KPIs

| KPI | Mejora / Improvement |
|-----|----------------------|
| Reducción de tasa de bloqueo / Blocking rate reduction | 35–42% |
| Reducción de latencia / Latency reduction | 28–34% |
| Mejora de utilización de recursos / Resource utilization improvement | 22% |
| Eficiencia energética / Energy efficiency | 26% reducción / reduction |

---

## Verificación de Calidad del Entrenamiento / Training Quality Verification

**ES:** Cómo verificar que cada modelo entrena correctamente:
**EN:** How to verify each model trains correctly:

1. **Pérdida decreciente / Decreasing loss:** La pérdida de entrenamiento debe decrecer a lo largo de las épocas / Training loss must decrease over epochs
2. **R² de validación / Validation R²:** R² > 0.85 para el LSTM propuesto, R² > 0.75 para baselines / R² > 0.85 for proposed LSTM, R² > 0.75 for baselines
3. **Sin NaN / No NaN:** Verificar que no hay NaN en las pérdidas / Check no NaN in losses
4. **Forma de predicciones / Prediction shapes:** Las predicciones deben tener la forma `[batch, horizon]` correcta / Predictions must have correct `[batch, horizon]` shape
5. **Orden de RMSE / RMSE ordering:** El orden de RMSE debe coincidir con el artículo / RMSE ordering must match article:
   ```
   LSTM propuesto < GRU < LSTM sin atención < RNN Simple < ARIMA < ...
   proposed LSTM < GRU < LSTM without attention < Simple RNN < ARIMA < ...
   ```

---

## Reproducibilidad / Reproducibility

**ES:** Todos los scripts usan semillas aleatorias fijas para reproducibilidad.
**EN:** All scripts use fixed random seeds for reproducibility.

```python
torch.manual_seed(42)
np.random.seed(42)
```

- **ES:** Todos los scripts son determinísticos. Los resultados pueden variar ligeramente entre plataformas debido a no-determinismo en operaciones de punto flotante de PyTorch.
- **EN:** All scripts are deterministic. Results may vary slightly across platforms due to floating-point non-determinism in PyTorch.

Para reproducibilidad exacta en el mismo hardware / For exact bitwise reproducibility on the same hardware:

```python
torch.use_deterministic_algorithms(True)
```

---

## Definiciones Matemáticas Clave / Key Mathematical Definitions

### Celda LSTM / LSTM Cell (Eq. 1-6, Sección II / Section II)

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)         (puerta de olvido / forget gate)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)         (puerta de entrada / input gate)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)      (candidato de celda / cell candidate)
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t             (estado de celda / cell state)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)         (puerta de salida / output gate)
h_t = o_t ⊙ tanh(C_t)                        (estado oculto / hidden state)
```

### Mecanismo de Atención / Attention Mechanism (Sección IV / Section IV)

```
e_{t,i} = v^T · tanh(W_h · h_i + W_s · s_t + b_attn)    (puntaje de energía / energy score)
α_{t,i} = exp(e_{t,i}) / Σ_j exp(e_{t,j})                (pesos de atención / attention weights)
c_t = Σ_i α_{t,i} · h_i                                   (vector de contexto / context vector)
```

### Función de Pérdida / Loss Function

```
L = (1/N) Σ_{t=1}^{N} (ŷ_t − y_t)² + λ₁‖θ‖₂² + λ₂ Σ_t |Δŷ_t − Δy_t|
```
Donde / Where: MSE + regularización L2 + penalización de suavidad temporal / MSE + L2 regularization + temporal smoothness penalty

### Predicción Multi-horizonte / Multi-horizon Prediction

```
ŷ_{t+h} = f_θ(x_{t-W+1}, ..., x_t; h)    para / for h = 1, ..., H
```
Donde W es el tamaño de ventana y H el horizonte máximo / Where W is window size and H is max horizon

### Tasa de Bloqueo Proactiva / Proactive Blocking Rate (Sección V / Section V)

```
P_block^proactive = P(D_t > C_allocated(ŷ_t))
                  ≤ P(|D_t − ŷ_t| > margin) + P(ŷ_t error > ε)
```

### Eficiencia Energética / Energy Efficiency

```
η_energy = (E_reactive − E_proactive) / E_reactive × 100%
```

---

## Citación / Citation

**ES:** Si utiliza este framework o código de simulación en su investigación, por favor cite:
**EN:** If you use this framework or simulation code in your research, please cite:

```bibtex
@article{lstm_traffic_5g_2024,
  title   = {Predicción de Tráfico Basada en LSTM para Gestión
             Proactiva de Recursos en Redes 5G},
  author  = {},
  journal = {IEEE},
  year    = {2024},
  note    = {Proposed submission}
}
```
