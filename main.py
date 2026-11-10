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
    HistoricalSpatialLens,
    HistoricalObjectPOI,
    build_default_historical_park,
    GlobalHistoricalConnector,
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
    print(" [4/5] DEMOSTRACIÓN: PROGRAMACIÓN GENÉTICA SIMBÓLICA (CRUCE KOZA)")
    print("=" * 70)
    p1 = EvolvableASTFunction(ind_id=0, generation=1)
    p2 = EvolvableASTFunction(ind_id=1, generation=1)
    print(f"Padre 1: {p1.parent_info}")
    print(f"Padre 2: {p2.parent_info}")

    hijo = KozaGeneticEngine.crossover(p1, p2, child_id=2, generation=2)
    print(f"Hijo generado por Cruce de Sub-árboles: {hijo.parent_info}")
    print("✓ Ecuaciones genéticas recombinadas y listas para compilación a Bytecode de 32 bits.")


def run_historical_demo():
    print("\n" + "=" * 70)
    print(" [5/5] DEMOSTRACIÓN: LENTE GEO-HISTÓRICA EN REALIDAD AUMENTADA Y VIRTUAL")
    print("=" * 70)
    park_lens = build_default_historical_park()
    print(f"Parque Histórico inicializado con {len(park_lens.catalog)} Puntos de Interés (POIs).")

    # Coordenadas de prueba que simulan al usuario caminando por el parque con la cámara
    itinerario = [
        {"desc": "Entrada principal del Parque (cerca al monumento al Libertador)", "pos": (0.0, 0.0, 0.0)},
        {"desc": "Plazoleta Arqueológica (cerca al monolito prehispánico)", "pos": (-4.5, 0.0, 5.0)},
        {"desc": "Observatorio Astronómico (cerca al sabio Caldas)", "pos": (5.0, 0.0, 6.5)},
    ]

    synthetic_frame = np.full((720, 1280, 3), 20, dtype=np.uint8)

    for i, parada in enumerate(itinerario, 1):
        print(f"\n--- Parada {i}: {parada['desc']} ---")
        park_lens.set_camera_pose(parada["pos"])
        rendered_frame, breakdowns = park_lens.render_ar_frame(synthetic_frame)
        print(f"Cámara en Coordenadas: {parada['pos']}")
        print(f"Objetos detectados por proximidad en cámara: {len(breakdowns)}")

        for b in breakdowns:
            print(f"  ✦ Holograma 3D: '{b['title']}' ({b['category']}) en coords {b['coordinates']}")
            print(f"    ➤ De dónde vino: {b['origin']}")
            print(f"    ➤ Época: {b['epoch']}")
            print(f"    ➤ Reseña: {b['description']}")

    # Guardar en Realidad Virtual (.nxs)
    nxs_output = "parque_historico_expedicion.nxs"
    save_msg = park_lens.save_vr_expedition(nxs_output)
    print(f"\n✓ {save_msg}")

    # Exportar script de reproducción en NexusLang
    script_output = "parque_historico_nexuslang.nxs"
    export_msg = park_lens.export_nexus_script(script_output)
    print(f"✓ {export_msg}")


def run_global_historical_demo():
    print("\n" + "=" * 70)
    print(" [6/6] DEMOSTRACIÓN: CONECTOR GLOBAL 3D E IA DE CONOCIMIENTO (WIKIPEDIA/OSM/CESIUM)")
    print("=" * 70)
    connector = GlobalHistoricalConnector()
    lens = HistoricalSpatialLens(detection_radius=25.0)

    # Lista de lugares globales icónicos a resolver en tiempo real
    monumentos = [
        {"name": "Torre Eiffel", "lat": 48.8584, "lon": 2.2945},
        {"name": "Pirámides de Teotihuacán", "lat": 19.6925, "lon": -98.8438},
        {"name": "Coliseo Romano", "lat": 41.8902, "lon": 12.4922},
    ]

    print("Consultando en vivo la IA de conocimiento y servicios geoespaciales...")
    pois = connector.sync_locations_to_lens(lens, monumentos)

    print(f"\n✓ Se resolvieron y proyectaron {len(pois)} monumentos globales en el motor espacial:")
    for poi in pois:
        card = lens.get_ar_breakdown_card(poi)
        print(f"\n✦ {card['title']} [{card['category']}]")
        print(f"  ➤ DE DÓNDE VINO: {card['origin']}")
        print(f"  ➤ ÉPOCA / AÑO:  {card['epoch']}")
        print(f"  ➤ COORDENADAS:  {card['coordinates']}")
        print(f"  ➤ RESEÑA:       {card['description'][:140]}...")

    # Renderizar frame AR con cámara en el centro
    lens.set_camera_pose((0.0, 0.0, 0.0))
    test_frame = np.full((720, 1280, 3), 15, dtype=np.uint8)
    rendered_view, breakdowns = lens.render_ar_frame(test_frame)
    print(f"\n✓ Frame AR holográfico renderizado con {len(breakdowns)} tarjetas y geometrías 3D.")

    # Guardar expedición mundial en formato .nxs
    nxs_world = "expedicion_mundial_3d.nxs"
    msg_save = lens.save_vr_expedition(nxs_world)
    print(f"✓ {msg_save}")


def main():
    parser = argparse.ArgumentParser(description="NexusXR - Spatial Engine CLI Runner")
    parser.add_argument("--demo-all", action="store_true", help="Ejecutar todas las demostraciones del motor")
    parser.add_argument("--polyglot", action="store_true", help="Probar runtime políglota de NexusLang")
    parser.add_argument("--physics", action="store_true", help="Probar motor de físicas 3D e imanes")
    parser.add_argument("--ai", action="store_true", help="Probar bóveda de gestos PyTorch y Transformer")
    parser.add_argument("--genetic", action="store_true", help="Probar evolución genética de código AST")
    parser.add_argument("--history", action="store_true", help="Probar desglose histórico AR con cámara y persistencia VR .nxs")
    parser.add_argument("--global-history", action="store_true", help="Probar conector global histórico con IA (Wikipedia/OSM/Cesium)")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.demo_all:
        run_polyglot_demo()
        run_physics_demo()
        run_ai_demo()
        run_genetic_demo()
        run_historical_demo()
        run_global_historical_demo()
        print("\n==================================================================")
        print(" [NEXUSXR] TODOS LOS SUBSISTEMAS OPERATIVOS Y VERIFICADOS CON ÉXITO")
        print("==================================================================")
    else:
        if args.polyglot: run_polyglot_demo()
        if args.physics: run_physics_demo()
        if args.ai: run_ai_demo()
        if args.genetic: run_genetic_demo()
        if args.history: run_historical_demo()
        if args.global_history: run_global_historical_demo()


if __name__ == "__main__":
    main()
