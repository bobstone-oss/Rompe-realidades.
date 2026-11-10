# NexusXR Engine: Motor de Computación Espacial Políglota, Holográfico y Neuro-Simbólico

**NexusXR** es un motor de computación espacial de última generación diseñado para operar en la convergencia de Realidad Aumentada (AR), Realidad Virtual (VR) y hardware físico embebido. Integra reconocimiento de vectores manuales 3D, un lenguaje homoicónico con metaprogramación en tiempo de ejecución, máquinas virtuales de pila vectorizadas y aprendizaje adaptativo continuo impulsado por redes neuronales y algoritmos genéticos.

---

## 🏛️ Arquitectura del Sistema (5 Capas)

La arquitectura modular de NexusXR está organizada en 5 capas maestras según las especificaciones del diseño:

```
NexusXR-Engine/
├── 01_kernel_os/               # Capa base para correr sobre hardware directo (Rust/C)
│   ├── bootloader/             # Arranque en dispositivos electrónicos propios
│   └── hardware_abstraction/   # Sensores, cámaras, GPU, puente ONNX Runtime (<1ms)
├── 02_polyglot_runtime/        # Traductor universal de lenguajes y runtime propio
│   ├── nexus_lang_compiler/    # Compilador homoicónico, Lexer, Parser, AST reflexivo
│   └── wasm_host / stack_vm/   # Máquina Virtual de Pila vectorizada de 32 bits (FastStackVM)
├── 03_spatial_xr_core/         # Motor de físicas 3D y renderizado espacial
│   ├── scene_graph/            # Geometrías holográficas interactivas
│   ├── physics_solver/         # Gravedad, colisiones por conservación de momento y dipolos magnéticos
│   └── persistence/            # Serializador de proyectos espaciales '.nxs' con integridad SHA-256
├── 04_perception_vectors/      # Percepción y visión artificial
│   ├── hand_tracking_3d/       # Extracción de 21 puntos vectoriales 3D por mano (MediaPipe)
│   └── gesture_math/           # Normalización invariante por Gram-Schmidt y álgebra de contacto
└── 05_self_learning_ai/        # Inteligencia artificial neuro-simbólica y evolutiva
    ├── hyperspherical_vault/   # Bóveda Few-Shot en PyTorch para aprender posturas en vivo
    ├── temporal_transformer/   # Transformer espacio-temporal 4D para trazos dinámicos en el aire
    ├── genetic_programming/    # Cruce de subárboles AST estilo Koza y Salón de la Fama
    └── onnx_bridge/            # Exportación e inferencia nativa en Rust con baja latencia
```

---

## 🚀 Lenguaje NexusLang

NexusLang es un lenguaje espacial de tipo declarativo y homoicónico (el código es una estructura de datos AST que puede modificarse a sí misma en tiempo real).

### Comandos Principales
- `MODO_REALIDAD <MODO>`: Alterna entre `PASSTHROUGH_AR`, `HIBRIDO_CUANTICO_AR_VR` o `VR_TOTAL`.
- `CREAR_HOLOGRAMA <Nombre> <Forma> <X> <Y> <Z>`: Materializa un cuerpo rígido 3D en el espacio.
- `VINCULAR_MANO <Nombre> <PuntoArticulacion>`: Ancla un holograma a un vector corporal (ej. `DEDO_INDICE_PUNTA`).
- `APRENDER_COMANDO <Nombre> <Macro>`: El lenguaje define nuevas palabras clave sobre la marcha (metaprogramación).
- `OPTIMIZAR_AST(<NombreFuncion>)`: Simplifica algebraicamente operaciones redundantes en el árbol sintáctico.
- `MUTAR_FUNCION(<NombreFuncion>, <Variable> -> <NuevoValor>)`: Modifica constantes en memoria de la función sin interrumpir el bucle de ejecución.
- `SINTETIZAR_FUNCION(<Nueva> = <Fn1> + <Fn2>)`: Fusiona dos funciones AST generando una nueva habilidad.

---

## 🔬 Especificación del Formato `.nxs` (Nexus Spatial Project)

El formato `.nxs` preserva el estado completo de la simulación en formato JSON estructurado con firma criptográfica SHA-256 para verificación de integridad:
1. **Cabecera**: Metadatos de versión y timestamp.
2. **Entorno**: Gravedad espacial (`gravity_g`), coeficiente de mezcla AR/VR.
3. **Escena Holográfica**: Posiciones, orientaciones, velocidades, radios colisionadores, masa, coeficientes de restitución elástica y cargas magnéticas.
4. **Macros Aprendidas**: Diccionario de comandos de metaprogramación generados durante la sesión.
5. **Pesos Neuronales**: Bóvedas de prototipos y pesos de PyTorch codificados en Base64 para portabilidad autónoma.
6. **Integridad Criptográfica**: Hash SHA-256 que garantiza que el proyecto no ha sido manipulado.

---

## 💻 Instalación y Requisitos

### Requisitos
- Python 3.10+
- Cargo / Rust (para compilación del Kernel)
- PyTorch (CPU o CUDA)
- NumPy, OpenCV (headless o GUI), MediaPipe

```bash
# Instalar dependencias en Python
pip install torch numpy opencv-python-headless mediapipe
```

---

## 🛠️ Ejecución y Pruebas

### 1. Ejecutar Demostración Completa (CLI Runner)
```bash
python3 main.py --demo-all
```

Opciones específicas:
```bash
python3 main.py --polyglot   # Prueba Lexer, AST, Metaprogramación y VM
python3 main.py --physics    # Prueba Solucionador de Físicas 3D, Imanes y persistencia .nxs
python3 main.py --ai         # Prueba Bóveda Few-Shot PyTorch y Temporal Transformer
python3 main.py --genetic    # Prueba Cruce Koza de Programación Genética Simbólica
```

### 2. Ejecutar Suite de Pruebas Automatizadas
```bash
PYTHONPATH=. python3 -m unittest discover tests
```

### 3. Verificar Compilación del Kernel en Rust
```bash
cargo check --manifest-path nexus_xr/kernel_os/Cargo.toml --lib
```
