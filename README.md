# KidScript: OCR de cuentos infantiles manuscritos

## Objetivo

El input son cuentos hechos por niños, escritos a mano, que pueden tener dibujos.

- Pueden tener errores ortográficos y gramaticales.
- El texto puede estar desordenado.
- Las páginas del PDF pueden estar rotadas.
- Un PDF puede incluir muchos cuentos.
- Los escaneos de cada página pueden tener bordes de otras páginas.

El objetivo del proyecto es

- Identificar e individualizar cada cuento (comienzo y fin) y separar los resultados por cuento.
- De cada cuento, ordenar las páginas, recortar y rotar cada una para facilitar el OCR.
- Extraer el texto de cada página.
- Corregir los errores.
- Juntar las páginas de un mismo cuento en un solo archivo de texto.

Hoy el repositorio cubre la extracción de texto (recorte de líneas + OCR) y el reentrenamiento del modelo.
La separación por cuento, el ordenado y la rotación de páginas siguen pendientes.

## Estructura del repositorio

| Ruta | Descripción |
| ------ | ------------- |
| `ocr_qwen_lora.py` | Inferencia: convierte un PDF en imágenes, detecta las líneas de texto, las recorta y las transcribe con Qwen3.5-0.8B + adaptador LoRA. |
| `train_qwen_lora.py` | Reentrenamiento (fine-tuning) del adaptador LoRA con pares imagen/transcripción. |
| `crops/` | Recortes de líneas de texto (`page-001-region-XX.png`) y `transcriptions.json` con la transcripción correcta de cada uno, usado para entrenar. |
| `qwen3.5-handwritten-spanish-adapter-v2/` | Adaptador LoRA reentrenado (versión actual para inferir). Incluye `checkpoint-4/` y `checkpoint-6/`. |

## Cómo funciona

### Inferencia (`ocr_qwen_lora.py`)

1. Convierte cada página del PDF en imagen (`pdf2image`, 300 DPI por defecto).
2. Detecta bandas horizontales de escritura con la proyección de píxeles oscuros y descarta fragmentos angostos (bordes y manchas del escaneo).
3. Transcribe cada recorte con el modelo base `unsloth/Qwen3.5-0.8B` más el adaptador LoRA.
4. Escribe el texto de cada página en el archivo de salida.

```bash
uv run ocr_qwen_lora.py paginas.pdf -o salida.txt --save-crops crops_salida
```

Opciones útiles:

- `--adapter RUTA_O_ID`: adaptador a usar. Por defecto `qwen3.5-handwritten-spanish-adapter-v2`; también sirve un checkpoint (`.../checkpoint-4`) o el adaptador original `Ak137/qwen3.5-0.8B-handwritten-spanish-ocr-lora`.
- `--save-crops DIR`: guarda los recortes detectados para revisarlos.
- `--max-new-tokens N`: límite de tokens generados por recorte (256 por defecto).
- `--max-image-pixels N`: tope de píxeles por imagen (100352 por defecto, igual que en el entrenamiento).
- `--dpi N`: resolución de renderizado del PDF.

El prompt de inferencia debe coincidir con el usado en el entrenamiento.

### Reentrenamiento (`train_qwen_lora.py`)

Continúa entrenando un adaptador LoRA existente (por defecto el de Ak137) con tus correcciones y guarda el resultado en otra carpeta, sin modificar el original.

1. Completá el campo `text` de cada entrada en `crops/transcriptions.json`:

   ```json
   [
     {"image": "page-001-region-02.png", "text": "Había una vez, una luna amiga"}
   ]
   ```

   Las rutas son relativas al JSON. Las entradas con `text` vacío se omiten.

2. Ejecutá el entrenamiento:

   ```bash
   uv run train_qwen_lora.py --output-dir qwen3.5-handwritten-spanish-adapter-v2
   ```

Opciones útiles:

- `--train-data` / `--validation-data`: manifiestos JSON de entrenamiento y validación.
- `--adapter`: adaptador de partida.
- `--epochs`, `--learning-rate`, `--gradient-accumulation-steps`: hiperparámetros.
- `--max-steps 1`: prueba rápida para verificar que el entrenamiento corre.
- `--max-image-pixels`: reduce el uso de memoria de la GPU (100352 por defecto).

## Requisitos

- Python 3.12 o superior y [uv](https://docs.astral.sh/uv/).
- GPU NVIDIA recomendada. El entrenamiento fue probado en una RTX 3050 Ti con unos 3.7 GB de VRAM, usando recortes pequeños, lote de 1 y gradient checkpointing. Sin GPU libre suficiente, la inferencia usa CPU.
- `poppler` instalado en el sistema (lo necesita `pdf2image`).

```bash
uv sync
```

Los modelos se descargan de Hugging Face la primera vez y quedan en `~/.cache/huggingface/hub/`.

### Notas

- `causal-conv1d` y `flash-linear-attention` son opcionales y aceleran el modelo. Si `causal-conv1d` falla al importar con `undefined symbol`, hay que recompilarlo contra el Torch instalado:
  `uv pip install --reinstall-package causal-conv1d --no-build-isolation --no-binary causal-conv1d --no-cache causal-conv1d` (tarda unos 15 minutos).
- `uv run` puede quedarse esperando un lock; ejecutar con `.venv/bin/python` lo evita.
- Un modelo de 0.8B parámetros puede inventar texto en recortes con poco contexto. Revisá siempre los recortes con `--save-crops`.
