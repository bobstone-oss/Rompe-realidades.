"""
Pruebas Unitarias para el Conector Geo-Histórico Global con IA (NexusXR)
"""
import unittest
import os
import numpy as np
from nexus_xr.spatial_core import (
    GlobalHistoricalConnector,
    HistoricalSpatialLens,
    HistoricalObjectPOI,
    NXSProjectSerializer,
    EMBEDDED_KNOWLEDGE_BASE,
)


class TestGlobalHistoricalConnector(unittest.TestCase):
    def setUp(self):
        self.connector = GlobalHistoricalConnector(timeout_sec=4.0)
        self.lens = HistoricalSpatialLens(detection_radius=20.0)

    def test_embedded_knowledge_base_integrity(self):
        """Verifica que los monumentos clave existan en la base de conocimiento."""
        self.assertIn("eiffel_tower", EMBEDDED_KNOWLEDGE_BASE)
        self.assertIn("colosseum", EMBEDDED_KNOWLEDGE_BASE)
        self.assertIn("teotihuacan", EMBEDDED_KNOWLEDGE_BASE)
        self.assertIn("machu_picchu", EMBEDDED_KNOWLEDGE_BASE)

        col = EMBEDDED_KNOWLEDGE_BASE["colosseum"]
        self.assertEqual(col["category"], "EDIFICACION")
        self.assertIn("Roma", col["origin"])
        self.assertIn("Flavia", col["origin"])

    def test_haversine_distance_calculation(self):
        """Verifica el cálculo de distancias sobre la esfera terrestre."""
        # Distancia entre París (Torre Eiffel) y Roma (Coliseo) ~ 1100-1120 km
        paris_lat, paris_lon = 48.8584, 2.2945
        rome_lat, rome_lon = 41.8902, 12.4922
        dist_km = GlobalHistoricalConnector.haversine_distance_km(paris_lat, paris_lon, rome_lat, rome_lon)
        self.assertGreater(dist_km, 1050.0)
        self.assertLess(dist_km, 1150.0)

    def test_resolve_by_coordinates_embedded_or_online(self):
        """Resuelve coordenadas de Teotihuacán y verifica metadatos culturales."""
        lat, lon = 19.6925, -98.8438
        poi = self.connector.resolve_by_coordinates(lat, lon, spatial_local_coords=(0.0, 0.0, 8.0))

        self.assertIsNotNone(poi)
        self.assertIn("Sol", poi.name)
        self.assertIn("Mesoamérica", poi.origin)
        self.assertEqual(poi.hologram_shape, "PIRAMIDE")
        self.assertEqual(poi.coordinates, (0.0, 0.0, 8.0))

    def test_resolve_by_name_eiffel_tower(self):
        """Resuelve por nombre y verifica desglose 'de dónde vino', época y descripción."""
        poi = self.connector.resolve_by_name("Torre Eiffel", spatial_local_coords=(2.0, 0.0, 6.0))

        self.assertIsNotNone(poi)
        self.assertIn("Eiffel", poi.name)
        self.assertTrue("Francia" in poi.origin or "París" in poi.origin)
        self.assertIn("1889", poi.epoch)
        self.assertGreater(len(poi.description), 20)

    def test_offline_fallback_for_unknown_coords(self):
        """Verifica que coordenadas en alta mar o sin red produzcan un POI válido sin romper."""
        # Punto en medio del océano Pacífico
        poi = self.connector.resolve_by_coordinates(-10.0, -140.0)
        self.assertIsNotNone(poi)
        self.assertIn("poi_lat", poi.poi_id)
        self.assertEqual(poi.hologram_shape, "CUBO")
        self.assertIn("Coordenadas", poi.origin)

    def test_cultural_metadata_extraction_heuristics(self):
        """Prueba directa de las reglas heurísticas de extracción neuro-simbólica."""
        title = "Gran Pirámide de Guiza"
        extract = "Construida durante la Dinastía IV en 2570 a.C. como tumba del faraón Keops en Egipto."
        cat, origin, epoch, shape, col = self.connector._extract_cultural_metadata(title, extract)

        self.assertEqual(cat, "SITIO_ARQUEOLOGICO")
        self.assertIn("Egipto", origin)
        self.assertIn("a.C.", epoch)
        self.assertEqual(shape, "PIRAMIDE")

    def test_sync_locations_to_lens_and_ar_render(self):
        """Prueba sincronización de lista de sitios globales con la lente AR y persistencia VR."""
        sites = [
            {"name": "Coliseo Romano", "lat": 41.8902, "lon": 12.4922},
            {"name": "Partenón de Atenas", "lat": 37.9715, "lon": 23.7266},
        ]

        pois = self.connector.sync_locations_to_lens(self.lens, sites)
        self.assertEqual(len(pois), 2)
        self.assertEqual(len(self.lens.catalog), 2)

        # Ubicar cámara cerca de los objetos y renderizar AR frame
        self.lens.set_camera_pose((0.0, 0.0, 0.0))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered, breakdowns = self.lens.render_ar_frame(frame)

        self.assertEqual(rendered.shape, (720, 1280, 3))
        self.assertGreaterEqual(len(breakdowns), 1)

        # Guardar en formato espacial .nxs
        test_nxs = "test_global_expedition.nxs"
        if os.path.exists(test_nxs):
            os.remove(test_nxs)

        msg = self.lens.save_vr_expedition(test_nxs)
        self.assertIn("guardado", msg)
        self.assertTrue(os.path.exists(test_nxs))

        # Verificar integridad criptográfica al cargar
        scene, macros, gravity, vr_blend, load_msg = NXSProjectSerializer.load_project(test_nxs)
        self.assertEqual(vr_blend, 1.0)
        self.assertTrue(len(scene) >= 1)

        if os.path.exists(test_nxs):
            os.remove(test_nxs)


if __name__ == "__main__":
    unittest.main()
