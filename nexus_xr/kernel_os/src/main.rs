use nexus_xr_kernel::{NexusXREngine, PolyglotScriptModule, RealityBlendMode, HandSkeletonFrame, Vector3D};

struct DemoUniversalModule {
    name: String,
}

impl PolyglotScriptModule for DemoUniversalModule {
    fn language_origin(&self) -> &str {
        "NexusLang / Rust-Native"
    }

    fn on_init(&mut self) {
        println!("[MODULO] '{}' inicializado en el espacio 3D holográfico.", self.name);
    }

    fn on_hand_vector_update(&mut self, hands: &[HandSkeletonFrame]) {
        if !hands.is_empty() {
            println!("[INTERACCIÓN] {} mano(s) detectada(s). Pellizco: {:.2}", hands.len(), hands[0].pinch_strength);
        }
    }

    fn on_spatial_tick(&mut self, delta_time: f32, mode: RealityBlendMode) {
        println!("[TICK XR] dt: {:.4}s | Modo actual: {:?}", delta_time, mode);
    }
}

fn main() {
    println!("==================================================================");
    println!(" NEXUS XR KERNEL OS - AR/VR NATIVE SPATIAL RUNTIME");
    println!("==================================================================");

    let mut engine = NexusXREngine::new();
    let module = Box::new(DemoUniversalModule {
        name: "NexusHologramController".to_string(),
    });

    engine.register_universal_module("HologramCore", module);

    // Simular un fotograma a 90Hz (dt = 0.011s) con una mano detectada
    let mut fake_hand = HandSkeletonFrame {
        hand_id: 1,
        joints_3d: [Vector3D { x: 0.0, y: 0.0, z: 0.0 }; 21],
        pinch_strength: 0.95,
    };
    fake_hand.joints_3d[8] = Vector3D { x: 0.05, y: 1.2, z: -0.4 }; // Dedo índice

    println!("[KERNEL] Ejecutando fotograma espacial...");
    engine.step_frame(0.0111, vec![fake_hand]);
    println!("==================================================================");
    println!(" [KERNEL RUST] FOTOGRAMA ESPACIAL COMPLETADO CON ÉXITO");
    println!("==================================================================");
}
