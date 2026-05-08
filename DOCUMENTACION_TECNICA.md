# Sistema de clasificación de prácticas en investigación de recursos hídricos mediante Modelos de Lenguaje de Gran Escala fine-tuneados

---

Este documento describe el desarrollo, entrenamiento y despliegue de un sistema de procesamiento de lenguaje natural especializado en la identificación, clasificación y extracción de buenas y malas prácticas dentro de literatura científica sobre recursos hídricos, monitoreo ambiental e hidrología. El sistema se basa en un Modelo de Lenguaje de Gran Escala (LLM) fine-tuneado de 7 mil millones de parámetros, desplegado localmente con cuantización a 4 bits, accesible mediante un dashboard interactivo.

El proyecto demuestra un pipeline completo de MLOps que abarca desde el preprocesamiento de datos hasta el despliegue en producción, ejecutado con un presupuesto cloud de aproximadamente USD $6.50, alcanzando un F1-score de 0.864 sobre el conjunto de evaluación.

---

## 1. Aclaración

**Importante**: el modelo está especializado en literatura sobre **recursos hídricos, monitoreo ambiental, hidrología y desarrollo rural relacionado con agua**. Su corpus de entrenamiento (1089 documentos) proviene de revistas como *International Journal of Water Resources Development*, *Water Research*, *Environmental Monitoring*, entre otras. El sistema **no está diseñado** para clasificar prácticas en disciplinas no relacionadas (geología pura, biomedicina, ciencias sociales generales, etc.).

---

## 2. Taxonomía de prácticas

Se definieron 14 categorías de prácticas (7 buenas + 7 malas) basadas en el conocimiento de dominio sobre tecnologías apropiadas para monitoreo hídrico rural.

### 2.1 Categorías de buenas prácticas

| Categoría | Descripción |
|---|---|
| `data_acquisition_technology` | Sensores de bajo costo, validación en campo, loggers offline, energía solar, microcontroladores (Arduino, ESP32), LoRa, SMS, hardware abierto. |
| `data_management` | CSV, JSON, SQLite, datos locales/offline, validación automática, control de versiones, sincronización diferencial, formatos abiertos. |
| `operation_maintenance` | Dashboards minimalistas, apps offline-first, manuales ilustrados, protocolos de mantenimiento, redundancia de sensores, capacitación técnica local. |
| `sustainability` | Paneles solares, reusabilidad, optimización energética, bajo costo, replicabilidad, materiales disponibles localmente, tecnología apropiada. |
| `community_participation` | Capacitación campesina, monitoreo participativo, líderes comunitarios, juntas de agua, apropiación tecnológica, formación local. |
| `local_adaptation` | Adaptación climática, materiales locales, contextos de baja alfabetización, zonas rurales sin conectividad, idioma local. |
| `scalability` | Kits estandarizados, replicabilidad, arquitectura modular, pilotaje progresivo, transferencia de conocimiento, código abierto. |

### 2.2 Categorías de malas prácticas

| Categoría | Descripción |
|---|---|
| `inappropriate_technology` | Equipos industriales sofisticados, instrumentos de laboratorio, hardware de alta gama no adaptado al contexto. |
| `non_adaptable_infrastructure` | Fibra óptica, 4G/5G obligatorio, servidores dedicados, requerimientos de baja latencia, conexión permanente. |
| `cloud_dependency` | Dependencia obligatoria de cloud (AWS, Azure, GCP), suscripciones cloud, dependencia de internet, SaaS. |
| `high_costs` | Inversiones costosas, infraestructura cara, mantenimiento costoso, licencias caras, software propietario. |
| `technical_complexity` | Requerimientos técnicos avanzados, ingeniería especializada, expertos requeridos, calibración compleja, vendor lock-in. |
| `centralization` | Sistemas centralizados, control central, plataformas únicas, monopolios tecnológicos, enfoques top-down. |
| `rural_inaccessibility` | Solo zonas urbanas, requiere electricidad constante, sin opción offline, no funciona sin internet. |

---

## 3. Metodología

El proyecto se ejecutó en seis fases secuenciales, cada una con artefactos verificables.

### 3.1 Fase 1 — Preprocesamiento del corpus

**Insumos**: 5 archivos `.txt` con 1089 documentos científicos separados por delimitador `<|endoftext|>`:

- `corpus_buenas_practicas_350docs_objGen.txt` (350 docs, objetivo general)
- `corpus_buenas_practicas_285docs_Obj1.txt` (285 docs, objetivo 1)
- `corpus_buenas_practicas_184docs_Obj2.txt` (184 docs, objetivo 2)
- `corpus_buenas_practicas_267docs_Obj3.txt` (267 docs, objetivo 3)
- `corpus_malas_practicas_3doc_objGen.txt` (3 docs de prácticas malas)

**Operaciones**:

1. Parseo y separación por delimitador
2. Limpieza: normalización de saltos de línea, colapso de espacios múltiples, preservación de estructura semántica
3. Detección de idioma por heurística (resultado: 99.7% inglés)
4. Análisis estadístico: longitud media de 70k caracteres por documento, máximo de 2.1M caracteres

**Salida**: `data/processed/documents.jsonl` (1089 registros con metadata)

### 3.2 Fase 1b — Resegmentación y muestreo estratificado

Los documentos originales presentaban "párrafos" de hasta 142,555 caracteres debido a inconsistencias en la conversión PDF→texto. Se implementó una resegmentación basada en oraciones reales mediante NLTK Punkt:

**Parámetros de chunking**:
- Tamaño objetivo: 400 tokens (~1600 caracteres)
- Mínimo aceptable: 80 tokens
- Máximo permitido: 600 tokens
- Solapamiento entre chunks consecutivos: 1 oración (preserva contexto cuando una práctica cruza límites)

**Resultado**: 71,170 chunks totales con distribución mediana de 382 tokens por chunk.

**Muestreo estratificado** para etiquetado:
- 37 documentos por cada uno de los 4 objetivos de buenas prácticas (148 docs)
- 3 documentos completos de malas prácticas
- Cap de 80 chunks por documento (mitigación de sesgo por documentos extremadamente largos)

**Salida**: `data/processed/chunks_sample.jsonl` (8,795 chunks → 7,246 tras cap)

### 3.3 Fase 2 — Etiquetado sintético con Qwen 2.5 72B Instruct AWQ

Dado que los documentos no venían pre-etiquetados a nivel fragmento, se empleó **destilación de conocimiento** desde un modelo grande de propósito general hacia el modelo objetivo a fine-tunear.

**Modelo etiquetador**: Qwen 2.5 72B Instruct AWQ (cuantización a 4 bits, ~40GB en VRAM).

**Infraestructura**: NVIDIA A100 SXM4 80GB en TensorDock (Praga, República Checa). Costo: ~$0.92 USD/hora.

**Framework de inferencia**: vLLM 0.8.5 con batching dinámico continuo (continuous batching), aprovechando la elevada concurrencia de la A100 (Maximum concurrency 22.14x).

**Prompt de etiquetado**: prompt extenso (~2000 tokens) que incluye:
- Definición precisa de qué constituye una "práctica" en el dominio
- Ejemplos de qué SÍ y qué NO cuenta
- Ontología completa de las 14 categorías con descripciones
- Schema JSON estricto de salida
- Instrucción explícita de ser generoso con detección (calibrado tras validación con 10 chunks)

**Schema de salida por chunk**:
```json
{
  "contains_practice": true|false,
  "practices": [
    {
      "type": "good"|"bad",
      "categories": ["category_key", ...],
      "subcategory_proposed": "snake_case_name_or_null",
      "span": "verbatim text from chunk",
      "explanation": "brief justification",
      "confidence": 0.0-1.0
    }
  ],
  "summary": "one-sentence summary of the chunk content"
}
```

**Resultados del etiquetado**:

| Métrica | Valor |
|---|---|
| Chunks procesados | 7,246 |
| Inferencias exitosas | 7,242 (99.94%) |
| Errores de parseo | 4 (0.06%) |
| Chunks con práctica detectada | 1,302 (18.0%) |
| Chunks sin práctica | 5,940 (82.0%) |
| Asignaciones a categorías good | 5,523 |
| Asignaciones a categorías bad | 762 |

**Salida**: `data/labeled/labeled_chunks.jsonl`

### 3.4 Fase 3 — Construcción del dataset de Supervised Fine-Tuning

Se transformaron los chunks etiquetados al formato chat de Qwen (`messages` con roles `system`, `user`, `assistant`).

**Operaciones aplicadas**:

1. **Normalización de categorías**: las subcategorías propuestas por el modelo etiquetador (37 instancias de variantes como `low_cost`, `governance`, `funding`, etc.) se mapearon a las 14 categorías oficiales según diccionario de equivalencias.
2. **Filtrado de errores**: descarte de los 4 chunks con error de parseo.
3. **Balanceo del dataset**: submuestreo de chunks negativos a ratio 2:1 respecto a positivos para evitar sesgo extremo a "no práctica".
4. **División train/eval**: 90/10 estratificada por `contains_practice`.

**Resultado**:

| Conjunto | Total | Con práctica | Sin práctica |
|---|---|---|---|
| Train | 3,516 | 1,168 (33%) | 2,348 (67%) |
| Eval | 390 | 134 (34%) | 256 (66%) |

**Salida**: `data/sft/train.jsonl` y `data/sft/eval.jsonl`

### 3.5 Fase 4 — Fine-tuning con LoRA

**Modelo base**: Qwen 2.5 7B Instruct (~7.7B parámetros totales, ~15GB en BF16).

**Justificación de elección**:
- Capacidad de razonamiento adecuada para tareas estructuradas
- Excelente soporte multilingüe (inglés y español)
- Compatible con cuantización GGUF para despliegue consumer
- Tamaño manejable en GPU única

**Método**: LoRA (Low-Rank Adaptation) — entrena solo matrices de adaptación de bajo rango, dejando los pesos del modelo base congelados.

**Configuración LoRA**:

| Parámetro | Valor |
|---|---|
| Rank (r) | 32 |
| Alpha | 64 (escala 2.0) |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Parámetros entrenables | 80,740,352 |
| Parámetros totales | 7,696,356,864 |
| Porcentaje entrenable | 1.05% |

**Hiperparámetros de entrenamiento**:

| Parámetro | Valor |
|---|---|
| Epochs | 3 |
| Batch size por dispositivo | 4 |
| Gradient accumulation steps | 4 |
| Effective batch size | 16 |
| Learning rate | 2e-4 |
| Scheduler | Cosine con warmup ratio 0.05 |
| Weight decay | 0.01 |
| Precisión | BF16 |
| Gradient checkpointing | Activado |
| Atención | SDPA (PyTorch native) |
| Max sequence length | 2048 tokens |

**Stack técnico**: HuggingFace `transformers` 4.51.3, `peft` 0.13.2, `trl` 0.15.2, `accelerate` 1.1.1, `datasets` 3.1.0.

**Hardware**: NVIDIA A100 SXM4 80GB.

**Tiempo de entrenamiento**: 1 hora 1 minuto (657 steps).

### 3.6 Fase 5 — Merge, push y cuantización

**5.1 Merge LoRA + base**: el adapter LoRA se integró a los pesos del modelo base mediante `model.merge_and_unload()`, produciendo un modelo de 15.2 GB en formato safetensors (4 shards de ~5GB cada uno).

**5.2 Publicación en HuggingFace Hub**: el modelo merged se subió como repositorio público en `CaffeineAddict69/qwen7b-water-rural-practices`. Velocidad de subida desde TensorDock: ~131 MB/s. Tiempo de subida: ~2 minutos.

**5.3 Conversión a GGUF Q4_K_M**: para despliegue en hardware de consumo, se utilizó `llama.cpp` para convertir y cuantizar:

1. HuggingFace → GGUF F16 (15.2 GB intermedio) vía `convert_hf_to_gguf.py`
2. GGUF F16 → GGUF Q4_K_M (4.68 GB final) vía `llama-quantize.exe`

**Cuantización Q4_K_M** mantiene aproximadamente 95-97% de la calidad del modelo en BF16, con reducción de tamaño de ~70%.

---

## 4. Evaluación

### 4.1 Metodología de evaluación

Se evaluó el modelo entrenado contra el conjunto de evaluación (390 ejemplos no vistos durante el entrenamiento) usando vLLM con LoRA dinámico, generación greedy (temperature=0) para garantizar determinismo.

### 4.2 Métricas obtenidas

| Métrica | Valor | Interpretación |
|---|---|---|
| **Tasa de parseo JSON exitoso** | 99.74% | El modelo produce JSON válido en prácticamente todos los casos |
| **Accuracy (contains_practice)** | 91.03% | Acuerdo con ground truth en clasificación binaria |
| **Precision** | 90.24% | De los chunks marcados como "con práctica", 90% son correctos |
| **Recall** | 82.84% | Detecta el 83% de los chunks que realmente contienen práctica |
| **F1-score** | 0.864 | Balance preciso entre precision y recall |
| **Type agreement (good/bad)** | 83.78% | Cuando ambos detectan práctica, coinciden en su tipo en 84% |
| **Mean categories Jaccard** | 0.641 | Solapamiento de categorías entre predicción y ground truth |

### 4.3 Matriz de confusión (`contains_practice`)

| | Predicho: True | Predicho: False |
|---|---|---|
| **Real: True** | 111 (TP) | 23 (FN) |
| **Real: False** | 12 (FP) | 244 (TN) |

### 4.4 Análisis cualitativo

Se inspeccionaron manualmente 5 ejemplos representativos del eval set. El modelo demuestra:

- **Fortalezas**: detección precisa de prácticas explícitas con tecnologías específicas, identificación correcta de patrones de cloud dependency y high costs, separación de múltiples prácticas en un mismo chunk en arrays distintos del JSON.
- **Limitaciones identificadas**: tendencia conservadora con prácticas físicas como irrigación, ocasional pérdida de buenas prácticas cuando coexisten con malas en el mismo chunk, sensibilidad a la formulación explícita de la práctica.

---

## 5. Despliegue

### 5.1 Arquitectura de inferencia local

```
┌─────────────────┐    HTTP    ┌──────────────────┐
│  Streamlit      │ ─────────> │  Ollama Server   │
│  Dashboard      │ <───────── │  localhost:11434 │
│  (puerto 8501)  │            │  + GGUF Q4_K_M   │
└─────────────────┘            └──────────────────┘
        │                              │
        ▼                              ▼
   Usuario sube              Modelo en VRAM (RTX)
   PDF/TXT y ve              + offloading parcial
   resultados                a CPU+RAM
```

### 5.2 Componentes

**Ollama**: runtime de LLMs basado en `llama.cpp`, gestiona la carga del modelo, asignación dinámica de capas a GPU/CPU, y expone una API HTTP compatible con OpenAI.

**Modelfile** (configuración del modelo en Ollama):
- Plantilla de chat ChatML (formato Qwen)
- System prompt embebido (alineado con el del entrenamiento)
- Parámetros: temperature=0.0, top_p=1.0, num_ctx=4096, num_predict=1024
- Stop token: `<|im_end|>`

**Cliente Python (`local/ollama_client.py`)**: cliente HTTP que abstrae la comunicación con Ollama y proporciona parseo robusto de respuestas (tolera fences markdown, preámbulos, edge cases).

**Pipeline de procesamiento (`local/pipeline.py`)**:
1. Extracción de texto desde PDF (PyMuPDF) o TXT
2. Chunking por oraciones con NLTK Punkt
3. Inferencia secuencial sobre cada chunk
4. Agregación de resultados a nivel documento

**Dashboard (`dashboard/app.py`)**: interfaz web en Streamlit con:
- Upload múltiple drag-and-drop
- Procesamiento batch con progreso por documento
- Visualizaciones interactivas (Plotly): donut de clasificación, bar charts de scores, distribución de categorías
- Drill-down por documento con tabs (top findings, categorías, chunks completos, resumen)
- Exportación a JSON estructurado y CSV plano

### 5.3 Requerimientos de hardware (inferencia)

| Componente | Mínimo | Recomendado |
|---|---|---|
| GPU VRAM | 4 GB | 8+ GB |
| RAM | 16 GB | 24 GB |
| Disco | 8 GB libres | 15 GB |
| Sistema | Windows 10+, Linux, macOS | — |

Velocidad de inferencia observada en RTX 3050 4GB + 24GB RAM: 5-25 segundos por chunk (depende de longitud del output).

---

## 6. Resultados de uso

### 6.1 Capacidades demostradas

El sistema, dado un documento del dominio, produce:

1. **Clasificación binaria por chunk**: detección de presencia o ausencia de prácticas
2. **Tipificación**: cada práctica se clasifica como "good" o "bad"
3. **Categorización taxonómica**: asignación de 1-3 categorías de las 14 oficiales
4. **Extracción de spans**: cita textual de la práctica detectada
5. **Explicación**: justificación breve de la clasificación
6. **Confidence score**: estimación de certeza del modelo
7. **Resumen por chunk y documento**: síntesis del contenido analizado
8. **Métricas agregadas**: scores, conteos por categoría, ranking de findings

### 6.2 Limitaciones del sistema

1. **Especialización de dominio**: el modelo solo detecta prácticas alineadas con la taxonomía y corpus de entrenamiento. Documentos de otros dominios (geología, biomedicina, ciencias sociales no relacionadas con agua) producirán mayoritariamente respuestas "no práctica".
2. **Idioma**: aunque Qwen base es multilingüe, el corpus de entrenamiento es 99.7% inglés. La precisión en español puede ser inferior.
3. **Cuantización Q4**: implica una pérdida estimada de 3-5% de calidad respecto al modelo BF16 original.
4. **Velocidad en hardware modesto**: con 4GB VRAM, el procesamiento de un paper de 50 chunks toma 10-20 minutos.
5. **Sesgo del corpus**: los 1086 documentos de buenas prácticas vs solo 3 de malas prácticas implican que el modelo aprendió principalmente desde patrones positivos, complementados con detección de malas prácticas mencionadas como contraste dentro de los documentos buenos.

---

## 7. Reproducibilidad

### 7.1 Estructura del proyecto

```
Sophia_LLM_GoodPractices/
├── data/
│   ├── raw/                          # Corpus original (5 .txt)
│   ├── processed/                    # Chunks y muestras
│   ├── labeled/                      # Etiquetas de Qwen 72B
│   └── sft/                          # Dataset SFT
├── scripts/
│   ├── 01_preprocess.py              # Fase 1: parseo
│   ├── 01b_resegment_and_sample.py   # Fase 1b: chunking
│   ├── 02_label_with_qwen_vllm.py    # Fase 2: etiquetado
│   ├── 03_build_sft_dataset.py       # Fase 3: dataset SFT
│   ├── 04_train_lora.py              # Fase 4: fine-tuning
│   ├── 05_evaluate_lora.py           # Evaluación
│   └── 06_merge_and_push.py          # Merge + HF Hub
├── local/
│   ├── ollama_client.py              # Cliente para inferencia
│   └── pipeline.py                   # Pipeline PDF→análisis
├── dashboard/
│   └── app.py                        # Dashboard Streamlit
├── models/
│   └── gguf/qwen7b-water-Q4_K_M.gguf # Modelo cuantizado
├── tools/
│   └── llama.cpp/                    # Convertidor y cuantizador
├── logs/                             # Logs de entrenamiento y eval
├── Modelfile                         # Configuración Ollama
├── requirements.txt                  # Dependencias locales
└── DOCUMENTACION_TECNICA.md
```

### 7.2 Costos del proyecto

| Concepto | Costo USD                   |
|---|-----------------------------|
| TensorDock A100 SXM4 80GB (~7 horas) | $13.50                      |
| Almacenamiento HuggingFace Hub | $0 (tier gratuito)          |
| API de etiquetado | $0 (uso local con Qwen 72B) |
| Despliegue local | $0 (hardware preexistente)  |
| **TOTAL** | **$13.50**                  |

### 7.3 Modelo final disponible

- **Repositorio**: https://huggingface.co/CaffeineAddict69/qwen7b-water-rural-practices
- **Formato BF16**: 15.2 GB (4 shards safetensors)
- **Formato GGUF Q4_K_M**: 4.68 GB (local)
- **Licencia**: Apache 2.0

---

## 8. Stack tecnológico

### 8.1 Modelos

- **Modelo etiquetador**: Qwen 2.5 72B Instruct AWQ
- **Modelo base para fine-tuning**: Qwen 2.5 7B Instruct
- **Modelo final**: Qwen 2.5 7B Instruct + LoRA fine-tuned (cuantizado a Q4_K_M)

### 8.2 Frameworks y librerías

| Categoría | Tecnología |
|---|---|
| Inferencia LLM cloud | vLLM 0.8.5 |
| Fine-tuning | HuggingFace transformers 4.51.3, PEFT 0.13.2, TRL 0.15.2 |
| Aceleración | accelerate 1.1.1, PyTorch 2.6.0 |
| Cuantización | llama.cpp |
| Inferencia local | Ollama |
| Procesamiento de texto | NLTK Punkt, tiktoken |
| PDF | PyMuPDF |
| Dashboard | Streamlit, Plotly |

### 8.3 Infraestructura

- **Cloud GPU**: TensorDock (NVIDIA A100 SXM4 80GB, Praga)
- **Almacenamiento de modelo**: HuggingFace Hub
- **Despliegue producción**: Local (NVIDIA RTX 3050 4GB + 24GB RAM, Windows 11)

---

## Anexos

### A. Archivos generados por el pipeline

| Archivo | Descripción | Tamaño |
|---|---|---|
| `data/processed/documents.jsonl` | 1089 documentos limpios | ~98 MB |
| `data/processed/chunks_all.jsonl` | 71,170 chunks resegmentados | ~250 MB |
| `data/processed/chunks_sample.jsonl` | 8,795 chunks de muestra | ~30 MB |
| `data/labeled/labeled_chunks.jsonl` | 7,242 chunks con etiquetas | ~12 MB |
| `data/sft/train.jsonl` | Dataset de entrenamiento | ~14 MB |
| `data/sft/eval.jsonl` | Dataset de evaluación | ~1.5 MB |
| `models/qwen7b-water-lora/` | Adapter LoRA entrenado | 309 MB |
| `models/gguf/qwen7b-water-Q4_K_M.gguf` | Modelo cuantizado final | 4.68 GB |

### B. Glosario técnico

- **LLM (Large Language Model)**: modelo de lenguaje de gran escala, típicamente con miles de millones de parámetros, entrenado en corpus masivos de texto.
- **LoRA (Low-Rank Adaptation)**: técnica de fine-tuning eficiente que entrena matrices de bajo rango añadidas al modelo base, sin modificar los pesos originales.
- **AWQ (Activation-aware Weight Quantization)**: técnica de cuantización a 4 bits que preserva la calidad del modelo identificando pesos críticos por su impacto en activaciones.
- **GGUF (GPT-Generated Unified Format)**: formato de archivo optimizado para distribución y carga eficiente de LLMs cuantizados, usado por llama.cpp y derivados.
- **Q4_K_M**: variante específica de cuantización a 4 bits con K-quants y rango medio, balance entre tamaño y calidad.
- **Chunking**: división de texto largo en fragmentos más pequeños procesables por el modelo dentro de su context window.
- **vLLM**: motor de inferencia para LLMs con batching dinámico continuo, optimizado para throughput.
- **SFT (Supervised Fine-Tuning)**: fine-tuning supervisado clásico donde el modelo aprende a reproducir respuestas etiquetadas dados los inputs correspondientes.
- **Ollama**: runtime local para LLMs basado en llama.cpp, expone API HTTP y gestiona modelos localmente.