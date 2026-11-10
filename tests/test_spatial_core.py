"""
Pruebas unitarias para el Núcleo Espacial 3D, Físicas y Serialización de Proyectos (.nxs)
"""
import unittest
import numpy as np
import os
from nexus_xr.spatial_core import (
    SpatialPhysicsSolver3D,
    RigidBodyHologram3D,
    HandColliderNode,
    NXSProjectSerializer,
    SerializableHologram3D,
)


class TestSpatialPhysics3D(unittest.TestCase):
    def setUp(self):
        self.solver = SpatialPhysicsSolver3D()

    def test_add_hologram_and_gravity(self):
        holo = RigidBodyHologram3D("CuboPrueba", pos=(0.0, 0.5, 0.0))
        holo.velocity = np.zeros(3, dtype=np.float32)
        holo.mass = 1.0
        holo.restitution = 0.8
        holo.magnetic_charge = 0.0

        bodies = [holo]
        initial_y = float(holo.position[1])

        # En la convención de este motor, self.gravity = +0.0018
        # Un paso de física integra la aceleración sobre la velocidad y posición
        self.solver.step_simulation(bodies, [])
        self.assertNotEqual(holo.position[1], initial_y)
        self.assertNotEqual(holo.velocity[1], 0.0)

    def test_magnetic_dipole_interaction(self):
        # Dos cuerpos magnéticos con cargas interactúan entre sí
        h1 = RigidBodyHologram3D("ImanNorte", pos=(-0.3, 0.0, 0.0))
        h1.velocity = np.zeros(3, dtype=np.float32)
        h1.magnetic_charge = 10.0

        h2 = RigidBodyHologram3D("ImanSur", pos=(0.3, 0.0, 0.0))
        h2.velocity = np.zeros(3, dtype=np.float32)
        h2.magnetic_charge = -10.0

        bodies = [h1, h2]

        # Desactivar gravedad temporalmente
        self.solver.gravity = 0.0
        self.solver.step_simulation(bodies, [])

        # La fuerza magnética modifica las velocidades en el eje X en sentidos opuestos
        self.assertNotEqual(h1.velocity[0], 0.0)
        self.assertNotEqual(h2.velocity[0], 0.0)
        self.assertAlmostEqual(h1.velocity[0] + h2.velocity[0], 0.0, places=5)


class TestNXSSerializer(unittest.TestCase):
    def test_save_and_load_project(self):
        filepath = "test_workspace.nxs"
        if os.path.exists(filepath):
            os.remove(filepath)

        holograms = {
            "CuboAlfa": SerializableHologram3D(
                name="CuboAlfa",
                shape="CUBO",
                pos=(0.0, 1.2, -0.5),
                color_name="CIAN"
            )
        }
        macros = {
            "ACTIVAR_ESCUDO": ["MODO_REALIDAD VR_TOTAL", "HARDWARE_PIN 13 ENCENDIDO"]
        }

        # Guardar proyecto
        msg = NXSProjectSerializer.save_project(filepath, holograms, macros, gravity_g=9.81, vr_blend=0.5)
        self.assertIn("guardado", msg)
        self.assertTrue(os.path.exists(filepath))

        # Cargar y verificar
        loaded_scene, loaded_macros, loaded_g, loaded_blend, loaded_msg = NXSProjectSerializer.load_project(filepath)
        self.assertIn("CuboAlfa", loaded_scene)
        self.assertIn("ACTIVAR_ESCUDO", loaded_macros)
        self.assertEqual(loaded_macros["ACTIVAR_ESCUDO"], ["MODO_REALIDAD VR_TOTAL", "HARDWARE_PIN 13 ENCENDIDO"])
        self.assertAlmostEqual(loaded_g, 9.81)
        self.assertAlmostEqual(loaded_blend, 0.5)

        # Limpiar
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    unittest.main()
