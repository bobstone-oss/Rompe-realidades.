use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::time::Instant;
use ndarray::{Array2, Array3};
use ort::{GraphOptimizationLevel, Session};

const SEQ_LEN: usize = 32;
const FEATURE_DIM: usize = 84;
const EMBED_DIM: usize = 128;

/// Representa un prototipo vectorial en la hipersfera unitaria de 128 dimensiones
#[derive(Debug, Clone)]
pub struct GesturePrototype {
    pub name: String,
    pub vector_128d: [f32; EMBED_DIM],
}

/// Motor Neuronal Nativo en Rust para el Kernel de NexusXR
pub struct NexusRustNeuralEngine {
    session: Session,
    prototypes: Vec<GesturePrototype>,
    similarity_threshold: f32,
}

impl NexusRustNeuralEngine {
    /// Inicializa ONNX Runtime con optimización máxima de grafo (Nivel 3)
    /// y carga en RAM la bóveda binaria de prototipos.
    pub fn new(onnx_path: &str, prototypes_bin_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?
            .commit_from_file(onnx_path)?;

        let prototypes = Self::load_binary_prototypes(prototypes_bin_path)?;

        println!(
            "[KERNEL RUST] Modelo ONNX '{}' activo | {} prototipos cargados en caché L1.",
            onnx_path,
            prototypes.len()
        );

        Ok(Self {
            session,
            prototypes,
            similarity_threshold: 0.74,
        })
    }

    /// Lee directamente el archivo binario 'nexus_prototypes.bin' generado desde Python
    fn load_binary_prototypes(path: &str) -> IoResult<Vec<GesturePrototype>> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        assert_eq!(&magic, b"NXPR", "Encabezado binario NXPR inválido");

        let mut count_buf = [0u8; 4];
        file.read_exact(&mut count_buf)?;
        let num_gestures = u32::from_le_bytes(count_buf) as usize;

        let mut list = Vec::with_capacity(num_gestures);

        for _ in 0..num_gestures {
            let mut len_buf = [0u8; 2];
            file.read_exact(&mut len_buf)?;
            let name_len = u16::from_le_bytes(len_buf) as usize;

            let mut name_bytes = vec![0u8; name_len];
            file.read_exact(&mut name_bytes)?;
            let name = String::from_utf8_lossy(&name_bytes).to_string();

            let mut raw_floats = [0u8; EMBED_DIM * 4];
            file.read_exact(&mut raw_floats)?;

            let mut vector_128d = [0.0f32; EMBED_DIM];
            for i in 0..EMBED_DIM {
                let b = [
                    raw_floats[i * 4],
                    raw_floats[i * 4 + 1],
                    raw_floats[i * 4 + 2],
                    raw_floats[i * 4 + 3],
                ];
                vector_128d[i] = f32::from_le_bytes(b);
            }

            list.push(GesturePrototype { name, vector_128d });
        }

        Ok(list)
    }

    /// Producto punto auto-vectorizado por LLVM (equivalente a similitud coseno en la hipersfera)
    #[inline(always)]
    fn fast_cosine_similarity(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
        let mut acc = 0.0f32;
        for i in 0..EMBED_DIM {
            acc += a[i] * b[i];
        }
        acc
    }

    /// Ejecuta el Temporal Transformer en < 1 ms y devuelve (NombreGesto, Confianza, Microsegundos)
    pub fn infer_gesture_realtime(
        &self,
        sequence_32x84: &Array2<f32>,
    ) -> Result<(String, f32, u128), Box<dyn std::error::Error>> {
        let t0 = Instant::now();

        // 1. Expandir dimensión de Batch: (1, 32, 84)
        let input_tensor: Array3<f32> = sequence_32x84
            .view()
            .insert_axis(ndarray::Axis(0))
            .to_owned();

        // 2. Ejecutar el grafo ONNX del Temporal Transformer
        let outputs = self.session.run(ort::inputs![
            "hand_sequence_84d" => input_tensor.view()
        ]?)?;

        let emb_view = outputs["hypersphere_embedding_128d"].try_extract_tensor::<f32>()?;
        let emb_slice = emb_view.as_slice().expect("Salida ONNX contigua");

        let mut live_embedding = [0.0f32; EMBED_DIM];
        live_embedding.copy_from_slice(&emb_slice[..EMBED_DIM]);

        // 3. Comparación instantánea contra todos los prototipos en la hipersfera
        let mut best_name = String::from("TRAZO_DESCONOCIDO");
        let mut best_score = -1.0f32;

        for proto in &self.prototypes {
            let score = Self::fast_cosine_similarity(&live_embedding, &proto.vector_128d);
            if score > best_score {
                best_score = score;
                if score >= self.similarity_threshold {
                    best_name = proto.name.clone();
                }
            }
        }

        let elapsed_us = t0.elapsed().as_micros();
        Ok((best_name, best_score, elapsed_us))
    }

    /// Permite registrar un nuevo prototipo directamente en la RAM de Rust en vivo
    pub fn register_live_prototype_in_rust(&mut self, name: &str, embedding_128d: [f32; EMBED_DIM]) {
        self.prototypes.push(GesturePrototype {
            name: name.to_string(),
            vector_128d: embedding_128d,
        });
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Inicializar el motor neuronal nativo cargando los archivos exportados por Python
    let engine = NexusRustNeuralEngine::new("nexus_transformer.onnx", "nexus_prototypes.bin")?;

    // 2. Simular una ventana deslizante de 32 fotogramas x 84 vectores de la mano
    let simulated_hand_stream = Array2::<f32>::zeros((SEQ_LEN, FEATURE_DIM));

    // Calentamiento de caché del procesador (Warm-up)
    for _ in 0..20 {
        let _ = engine.infer_gesture_realtime(&simulated_hand_stream)?;
    }

    // 3. Benchmark en tiempo real (1,000 ciclos del motor XR)
    let mut total_us: u128 = 0;
    let runs = 1000;
    let mut last_result = (String::new(), 0.0f32, 0u128);

    for _ in 0..runs {
        last_result = engine.infer_gesture_realtime(&simulated_hand_stream)?;
        total_us += last_result.2;
    }

    let avg_us = total_us as f64 / runs as f64;
    let avg_ms = avg_us / 1000.0;

    println!("==================================================================");
    println!(" BENCHMARK KERNEL RUST + ONNX RUNTIME (TEMPORAL TRANSFORMER)");
    println!("------------------------------------------------------------------");
    println!(" Última Predicción     : {} (Similitud: {:.2}%)", last_result.0, last_result.1 * 100.0);
    println!(" Tiempo por Inferencia : {:.1} microsegundos ({:.3} ms)", avg_us, avg_ms);
    println!(" Capacidad Máxima FPS  : {:.0} inferencias por segundo", 1_000_000.0 / avg_us);
    println!("==================================================================");

    Ok(())
}
