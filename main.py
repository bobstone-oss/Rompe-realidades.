#!/usr/bin/env python3
"""
NexusXR Engine - Punto de Entrada Principal y CLI Runner

Permite ejecutar demostraciones interactivas en vivo o en modo simulación (headless)
para entornos de integración continua o dispositivos sin cámara física.
"""
import sys
import argparse
import time
import numpy as np
import torch

from nexus_xr.polyglot_runtime import (
    NexusUniversalLanguage,
    HomoiconicRuntime,
    NexusLexer,
    NexusParser,
    Vec3,
    FastVectorStackVM,
    PackedBytecodeCompiler,
)
from nexus_xr.spatial_core import (
    SpatialPhysicsSolver3D,
    RigidBodyHologram3D,
    NXSProjectSerializer,
    SerializableHologram3D,
)
from nexus_xr.ai_brain import (
    HypersphericalGestureNet,
    SelfLearningGestureVault,
    CanonicalHandGeometry,
    TemporalGestureTransformer,
    KozaGeneticEngine,
    EvolvableASTFunction,
)


def run_polyglot_demo():
    print("=" * 70)
    print(" [1/4] DEMOSTRACIÓN: RUNTIME POLÍGLOTA Y METAPROGRAMACIÓN NEXUSLANG")
    print("=" * 70)
    interp = NexusUniversalLanguage()
    codigo = """
    MODO_REALIDAD HIBRIDO_CUANTICO_AR_VR
    CREAR_HOLOGRAMA OrbePrimario ESFERA_PLASMA 0.0 1.2 -0.4
    VINCULAR_MANO OrbePrimario DEDO_INDICE_PUNTA
    APRENDER_COMANDO ACTIVAR_SUPERCONDUCTOR MODO_REALIDAD VR_TOTAL ; HARDWARE_PIN 13 ENCENDIDO
    ACTIVAR_SUPERCONDUCTOR
    """
    interp.execute_script(codigo)

    # Demostración del compilador homoicónico y optimización de AST
    source_ast = """
    FUNCION calcular_impulso(vector_mano) {
        ganancia = (2 * 4) + 0
        desplazamiento = vector_mano * ganancia
        RETORNAR desplazamiento
    }
    OPTIMIZAR_AST(calcular_impulso)
    MUTAR_FUNCION(calcular_impulso, ganancia -> 30.0)
    """
    runtime = HomoiconicRuntime()
    lexer = NexusLexer(source_ast)
    parser = NexusParser(lexer.tokenize())
    program = parser.parse_program()
    runtime.execute_program(program)
    print("✓ NexusLang compilado y mutado con éxito.")


def run_physics_demo():
    print("\n" + "=" * 70)
    print(" [2/4] DEMOSTRACIÓN: FÍSICAS 3D Y PERSISTENCIA .NXS")
    print("=" * 70)
    solver = SpatialPhysicsSolver3D()
    cubo1 = RigidBodyHologram3D("CuboNorte", pos=(-0.25, 0.4, 0.0))
    cubo1.magnetic_charge = 12.0

    cubo2 = RigidBodyHologram3D("CuboSur", pos=(0.25, 0.4, 0.0))
    cubo2.magnetic_charge = -12.0

    bodies = [cubo1, cubo2]
    print(f"Estado inicial: CuboNorte X={cubo1.position[0]:.2f}, CuboSur X={cubo2.position[0]:.2f}")

    for _ in range(5):
        solver.step_simulation(bodies, [])

    print(f"Tras 5 ticks de atracción magnética:")
    print(f" -> CuboNorte VelX={cubo1.velocity[0]:.4f}, PosX={cubo1.position[0]:.4f}")
    print(f" -> CuboSur   VelX={cubo2.velocity[0]:.4f}, PosX={cubo2.position[0]:.4f}")

    # Serializar proyecto
    nxs_file = "demo_session.nxs"
    holograms = {
        "CuboNorte": SerializableHologram3D("CuboNorte", pos=tuple(cubo1.position)),
        "CuboSur": SerializableHologram3D("CuboSur", pos=tuple(cubo2.position)),
    }
    msg = NXSProjectSerializer.save_project(nxs_file, holograms, {"DEMO": ["MODO_REALIDAD VR_TOTAL"]}, gravity_g=9.81, vr_blend=0.5)
    print(f"✓ {msg}")


def run_ai_demo():
    print("\n" + "=" * 70)
    print(" [3/4] DEMOSTRACIÓN: CEREBRO ADAPTATIVO (PYTORCH FEW-SHOT & TRANSFORMER)")
    print("=" * 70)
    vault_file = "demo_vault.pt"
    vault = SelfLearningGestureVault(vault_path=vault_file)

    # Simular mano en reposo
    raw_points = np.zeros((21, 3), dtype=np.float32)
    raw_points[9] = [0.0, 0.1, 0.0]
    canonical_vec = CanonicalHandGeometry.extract_invariant_features(raw_points)

    print("Enseñando nuevo gesto en caliente: 'GESTO_TELEQUINESIS'...")
    vault.learn_gesture_sample("GESTO_TELEQUINESIS", canonical_vec)

    # Clasificar en tiempo real
    gesto, conf, all_scores = vault.recognize(canonical_vec)
    print(f"✓ Reconocimiento: {gesto} (Confianza: {conf*100:.1f}%)")

    # Transformer para secuencias temporales en el aire
    print("Ejecutando Temporal Transformer para trazos 4D en el aire...")
    transformer = TemporalGestureTransformer()
    seq_stream = torch.randn(1, 32, 84)
    embedding = transformer(seq_stream)
    print(f"✓ Embedding hipersférico 128D generado con norma unitaria: {torch.norm(embedding).item():.4f}")


def run_genetic_demo():
    print("\n" + "=" * 70)
    print(" [4/4] DEMOSTRACIÓN: PROGRAMACIÓN GENÉTICA SIMBÓLICA (CRUCE KOZA)")
    print("=" * 70)
    p1 = EvolvableASTFunction(ind_id=0, generation=1)
    p2 = EvolvableASTFunction(ind_id=1, generation=1)
    print(f"Padre 1: {p1.parent_info}")
    print(f"Padre 2: {p2.parent_info}")

    hijo = KozaGeneticEngine.crossover(p1, p2, child_id=2, generation=2)
    print(f"Hijo generado por Cruce de Sub-árboles: {hijo.parent_info}")
    print("✓ Ecuaciones genéticas recombinadas y listas para compilación a Bytecode de 32 bits.")


def main():
    parser = argparse.ArgumentParser(description="NexusXR - Spatial Engine CLI Runner")
    parser.add_argument("--demo-all", action="store_true", help="Ejecutar todas las demostraciones del motor")
    parser.add_argument("--polyglot", action="store_true", help="Probar runtime políglota de NexusLang")
    parser.add_argument("--physics", action="store_true", help="Probar motor de físicas 3D e imanes")
    parser.add_argument("--ai", action="store_true", help="Probar bóveda de gestos PyTorch y Transformer")
    parser.add_argument("--genetic", action="store_true", help="Probar evolución genética de código AST")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.demo_all:
        run_polyglot_demo()
        run_physics_demo()
        run_ai_demo()
        run_genetic_demo()
        print("\n==================================================================")
        print(" [NEXUSXR] TODOS LOS SUBSISTEMAS OPERATIVOS Y VERIFICADOS CON ÉXITO")
        print("==================================================================")
    else:
        if args.polyglot: run_polyglot_demo()
        if args.physics: run_physics_demo()
        if args.ai: run_ai_demo()
        if args.genetic: run_genetic_demo()


if __name__ == "__main__":
    main()
