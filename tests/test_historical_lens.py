"""
Pruebas Unitarias para la Lente Espacial Geo-Histórica en Realidad Aumentada y Virtual (NexusXR)
"""
import unittest
import os
import numpy as np
from nexus_xr.spatial_core import (
    HistoricalObjectPOI,
    HistoricalSpatialLens,
    build_default_historical_park,
    NXSProjectSerializer,
)


class TestHistoricalSpatialLens(unittest.TestCase):
    def setUp(self):
        self.lens = HistoricalSpatialLens(detection_radius=10.0)
        self.poi1 = HistoricalObjectPOI(
            poi_id="estatua_heroe",
            name="General Francisco de Miranda",
            category="PERSONAJE",
            coordinates=(0.0, 0.0, 5.0),
            origin="Caracas, Venezuela (1750)",
            epoch="Siglo XVIII - XIX",
            description="El Precursor de la Emancipación Americana y participante en 3 grandes revoluciones.",
            hologram_shape="PIRAMIDE",
            color_name="ORO",
            scale=1.0,
        )
        self.poi2 = HistoricalObjectPOI(
            poi_id="vasija_ancestral",
            name="Urna Funeraria Quimbaya",
            category="ARTEFACTO",
            coordinates=(20.0, 0.0, 20.0),
            origin="Cauca Medio / Cultura Quimbaya",
            epoch="Período Clásico (500 a.C. - 600 d.C.)",
            description="Orfebrería y alfarería ceremonial de alta precisión metalúrgica.",
            hologram_shape="ESFERA",
            color_name="MAGENTA",
            scale=0.8,
        )
        self.lens.register_poi(self.poi1)
        self.lens.register_poi(self.poi2)

    def test_poi_serialization_dict(self):
        d = self.poi1.to_dict()
        self.assertEqual(d["poi_id"], "estatua_heroe")
        self.assertEqual(d["origin"], "Caracas, Venezuela (1750)")
        self.assertEqual(d["hologram_shape"], "PIRAMIDE")

        reconstructed = HistoricalObjectPOI.from_dict(d)
        self.assertEqual(reconstructed.name, self.poi1.name)
        self.assertEqual(reconstructed.coordinates, (0.0, 0.0, 5.0))

    def test_proximity_scanning_and_discovery(self):
        # Cámara ubicada en el origen (0, 0, 0)
        self.lens.set_camera_pose((0.0, 0.0, 0.0))
        nearby = self.lens.scan_nearby_objects()

        # poi1 está a 5 metros (dentro de radio 10), poi2 está a ~28 metros (fuera)
        self.assertEqual(len(nearby), 1)
        self.assertEqual(nearby[0].poi_id, "estatua_heroe")
        self.assertIn("estatua_heroe", self.lens.discovered_pois)
        self.assertNotIn("vasija_ancestral", self.lens.discovered_pois)

        # Mover cámara cerca a poi2
        self.lens.set_camera_pose((19.0, 0.0, 19.0))
        nearby_step2 = self.lens.scan_nearby_objects()
        self.assertEqual(len(nearby_step2), 1)
        self.assertEqual(nearby_step2[0].poi_id, "vasija_ancestral")
        # Ambos ya deben estar descubiertos en la bitácora
        self.assertEqual(len(self.lens.discovered_pois), 2)

    def test_ar_breakdown_card_generation(self):
        self.lens.set_camera_pose((0.0, 0.0, 0.0))
        card = self.lens.get_ar_breakdown_card(self.poi1)
        self.assertEqual(card["title"], "General Francisco de Miranda")
        self.assertEqual(card["origin"], "Caracas, Venezuela (1750)")
        self.assertEqual(card["epoch"], "Siglo XVIII - XIX")
        self.assertAlmostEqual(card["distance_m"], 5.0)

        # Verificar contenido de overlay_text
        text_joined = " ".join(card["overlay_text"])
        self.assertIn("DE DÓNDE VINO: Caracas, Venezuela (1750)", text_joined)
        self.assertIn("ÉPOCA / AÑO: Siglo XVIII - XIX", text_joined)

    def test_3d_projection_and_render_frame(self):
        self.lens.set_camera_pose((0.0, 0.0, 0.0))
        proj = self.lens.project_poi_to_screen(self.poi1, frame_width=1280, frame_height=720)
        self.assertIsNotNone(proj)
        self.assertIn("screen_pos", proj)
        self.assertIn("panel_pos", proj)
        self.assertGreater(proj["z_depth"], 0.0)

        # Renderizar sobre frame sintético
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered, breakdowns = self.lens.render_ar_frame(frame)
        self.assertEqual(rendered.shape, (720, 1280, 3))
        self.assertEqual(len(breakdowns), 1)
        self.assertEqual(breakdowns[0]["poi_id"], "estatua_heroe")

    def test_default_historical_park_environment(self):
        park = build_default_historical_park()
        self.assertGreaterEqual(len(park.catalog), 4)
        self.assertIn("estatua_libertador", park.catalog)
        self.assertIn("monolito_solar", park.catalog)
        self.assertIn("sabio_caldas", park.catalog)

    def test_save_vr_expedition_nxs_and_integrity(self):
        filepath = "test_historical_expedition.nxs"
        if os.path.exists(filepath):
            os.remove(filepath)

        # Descubrir los objetos
        self.lens.set_camera_pose((0.0, 0.0, 0.0))
        self.lens.scan_nearby_objects()

        msg = self.lens.save_vr_expedition(filepath)
        self.assertIn("guardado", msg)
        self.assertTrue(os.path.exists(filepath))

        # Cargar con el serializador oficial de NexusXR
        scene, macros, gravity, vr_blend, load_msg = NXSProjectSerializer.load_project(filepath)
        self.assertIn("estatua_heroe", scene)
        self.assertIn("RECONSTRUIR_PARQUE_VR", macros)
        self.assertEqual(vr_blend, 1.0)  # Modo Realidad Virtual activado para persistencia

        if os.path.exists(filepath):
            os.remove(filepath)

    def test_export_nexus_script(self):
        filepath = "test_historical_script.nxs"
        if os.path.exists(filepath):
            os.remove(filepath)

        self.lens.set_camera_pose((0.0, 0.0, 0.0))
        self.lens.scan_nearby_objects()

        msg = self.lens.export_nexus_script(filepath)
        self.assertTrue(os.path.exists(filepath))
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("MODO_REALIDAD HIBRIDO_CUANTICO_AR_VR", content)
        self.assertIn("CREAR_HOLOGRAMA estatua_heroe PIRAMIDE", content)
        self.assertIn("APRENDER_COMANDO INSPECCIONAR_ESTATUA_HEROE", content)

        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    unittest.main()
