// Núcleo del Motor Espacial NexusXR
// Compilable para Windows, Linux, Android, iOS, macOS y embebidos ARM/RISC-V.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RealityBlendMode {
    AugmentedRealityPassthrough, // 100% mundo real + hologramas superpuestos
    MixedRealitySpatialBlend(f32), // Fusión progresiva entre 0.0 (AR) y 1.0 (VR)
    FullVirtualReality,          // 100% entorno sintético
}

#[derive(Debug, Clone, Copy)]
pub struct Vector3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone)]
pub struct HandSkeletonFrame {
    pub hand_id: u8,
    pub joints_3d: [Vector3D; 21],
    pub pinch_strength: f32,
}

/// Interfaz Universal para ejecutar código escrito en CUALQUIER lenguaje
/// (Python, C++, C#, Rust, Go, TypeScript) compilado a módulos Wasm/LLVM.
pub trait PolyglotScriptModule {
    fn language_origin(&self) -> &str;
    fn on_init(&mut self);
    fn on_hand_vector_update(&mut self, hands: &[HandSkeletonFrame]);
    fn on_spatial_tick(&mut self, delta_time: f32, mode: RealityBlendMode);
}

pub struct NexusXREngine {
    pub blend_mode: RealityBlendMode,
    pub active_modules: HashMap<String, Box<dyn PolyglotScriptModule>>,
    pub tracked_hands: Vec<HandSkeletonFrame>,
}

impl NexusXREngine {
    pub fn new() -> Self {
        Self {
            blend_mode: RealityBlendMode::MixedRealitySpatialBlend(0.5),
            active_modules: HashMap::new(),
            tracked_hands: Vec::with_capacity(2),
        }
    }

    /// Registra un módulo sin importar en qué lenguaje fue programado
    pub fn register_universal_module(&mut self, name: &str, mut module: Box<dyn PolyglotScriptModule>) {
        println!(
            "[KERNEL XR] Cargando módulo '{}' escrito en [{}]",
            name,
            module.language_origin()
        );
        module.on_init();
        self.active_modules.insert(name.to_string(), module);
    }

    /// Ciclo principal de tiempo real (90Hz - 120Hz)
    pub fn step_frame(&mut self, delta_time: f32, incoming_hands: Vec<HandSkeletonFrame>) {
        self.tracked_hands = incoming_hands;

        for (_name, module) in self.active_modules.iter_mut() {
            module.on_hand_vector_update(&self.tracked_hands);
            module.on_spatial_tick(delta_time, self.blend_mode);
        }
    }
}
