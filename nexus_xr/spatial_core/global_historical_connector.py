"""
NexusXR - Conector Geo-Histórico Global con IA y Servicios de Conocimiento Abierto.

Permite consultar cualquier lugar del mundo mediante coordenadas GPS (lat, lon)
o por nombre de monumento/ciudad histórica, sincronizándolo con:
1. Proveedores de Terreno y Edificios 3D (Cesium 3D Tiles, OpenStreetMap 3D).
2. Agente de Búsqueda Web & IA de Conocimiento (Wikipedia Geosearch, Wikidata API).
3. Motor Espacial NexusXR (HistoricalSpatialLens) para proyectar hologramas AR,
   desplegar tarjetas flotantes con origen ("de dónde vino") e historia,
   y persistir la sesión completa en formato '.nxs' para Realidad Virtual.
"""

import math
import re
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .historical_spatial_lens import HistoricalObjectPOI, HistoricalSpatialLens


@dataclass
class GlobalLocationQuery:
    """Información estructurada de consulta de una ubicación global."""
    name: str
    latitude: float
    longitude: float
    category: str = "EDIFICACION"
    elevation_m: float = 0.0
    tileset_ref: str = "Cesium_WorldTerrain_OSM3D"


# Base de conocimiento offline integrada con monumentos y sitios arqueológicos universales
EMBEDDED_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "eiffel_tower": {
        "name": "Torre Eiffel",
        "category": "MONUMENTO",
        "latitude": 48.8584,
        "longitude": 2.2945,
        "origin": "París, Francia (Ingeniero Gustave Eiffel)",
        "epoch": "Belle Époque / Exposición Universal de 1889",
        "description": "Estructura de hierro pudelado de 330 metros construida como arco de entrada de la Exposición Universal de 1889.",
        "hologram_shape": "PIRAMIDE",
        "color_name": "ORO",
        "scale": 1.5,
    },
    "colosseum": {
        "name": "Coliseo Romano (Anfiteatro Flavio)",
        "category": "EDIFICACION",
        "latitude": 41.8902,
        "longitude": 12.4922,
        "origin": "Roma Antigua, Imperio Romano (Dinastía Flavia / Emperador Vespasiano)",
        "epoch": "Siglo I d.C. (70-80 d.C.)",
        "description": "El anfiteatro más grande del mundo clásico, erigido en travertino y toba para espectáculos de gladiadores.",
        "hologram_shape": "CUBO",
        "color_name": "ORO",
        "scale": 1.4,
    },
    "teotihuacan": {
        "name": "Pirámide del Sol (Teotihuacán)",
        "category": "SITIO_ARQUEOLOGICO",
        "latitude": 19.6925,
        "longitude": -98.8438,
        "origin": "Valle de Teotihuacán, Mesoamérica (Cultura Teotihuacana)",
        "epoch": "Período Clásico Mesoamericano (c. 100-200 d.C.)",
        "description": "Monumento cívico-religioso de 65 metros de altura alineado astronómicamente con los solsticios y el cenit solar.",
        "hologram_shape": "PIRAMIDE",
        "color_name": "MAGENTA",
        "scale": 1.6,
    },
    "machu_picchu": {
        "name": "Santuario Histórico de Machu Picchu",
        "category": "SITIO_ARQUEOLOGICO",
        "latitude": -13.1631,
        "longitude": -72.5450,
        "origin": "Cusco, Imperio Incaico / Tahuantinsuyo (Inca Pachacútec)",
        "epoch": "Siglo XV (c. 1450 d.C.)",
        "description": "Ciudadela y obra maestra de arquitectura e ingeniería inca emplazada a 2.430 metros sobre el valle del Urubamba.",
        "hologram_shape": "OCTAEDRO",
        "color_name": "VERDE",
        "scale": 1.3,
    },
    "taj_mahal": {
        "name": "Taj Mahal",
        "category": "EDIFICACION",
        "latitude": 27.1751,
        "longitude": 78.0421,
        "origin": "Agra, Imperio Mogol (Emperador Shah Jahan)",
        "epoch": "Siglo XVII (1631-1648 d.C.)",
        "description": "Mausoleo de mármol blanco construido en homenaje a Mumtaz Mahal, cúspide del arte y la simetría mogol.",
        "hologram_shape": "ESFERA",
        "color_name": "BLANCO",
        "scale": 1.3,
    },
    "great_pyramid_giza": {
        "name": "Gran Pirámide de Guiza",
        "category": "MONUMENTO",
        "latitude": 29.9792,
        "longitude": 31.1342,
        "origin": "Guiza, Imperio Antiguo de Egipto (Faraón Keops)",
        "epoch": "Dinastía IV (c. 2570 a.C.)",
        "description": "La más antigua de las siete maravillas del mundo antiguo y la única que permanece en pie, construida con 2.3 millones de bloques de piedra.",
        "hologram_shape": "PIRAMIDE",
        "color_name": "ORO",
        "scale": 1.7,
    },
    "parthenon": {
        "name": "Partenón de la Acrópolis de Atenas",
        "category": "EDIFICACION",
        "latitude": 37.9715,
        "longitude": 23.7266,
        "origin": "Atenas, Grecia Antigua (Pericles, Fidias, Ictino y Calícrates)",
        "epoch": "Siglo de Pericles (447-432 a.C.)",
        "description": "Templo dórico consagrado a la diosa Atenea Pártenos, símbolo universal de la democracia y la estética helénica.",
        "hologram_shape": "CUBO",
        "color_name": "CIAN",
        "scale": 1.2,
    },
    "chichen_itza": {
        "name": "Templo de Kukulcán (Chichén Itzá)",
        "category": "SITIO_ARQUEOLOGICO",
        "latitude": 20.6843,
        "longitude": -88.5678,
        "origin": "Península de Yucatán, Civilización Maya (Período Posclásico)",
        "epoch": "Siglos IX - XII d.C.",
        "description": "Pirámide escalonada con precisión astronómica donde las sombras equinocciales proyectan la serpiente emplumada.",
        "hologram_shape": "PIRAMIDE",
        "color_name": "MAGENTA",
        "scale": 1.4,
    }
}


class GlobalHistoricalConnector:
    """
    Conector inteligente que sincroniza motores 3D mundiales (Cesium / OpenStreetMap)
    con IA y APIs abiertas de conocimiento (Wikipedia / Wikidata) para resolver
    cualquier punto del planeta y proyectarlo en la Lente Espacial de NexusXR.
    """
    def __init__(self, user_agent: str = "NexusXR-GlobalSpatial/1.0 (spatial-engine@nexusxr.org)", timeout_sec: float = 6.0):
        self.user_agent = user_agent
        self.timeout_sec = float(timeout_sec)
        self.cache: Dict[str, HistoricalObjectPOI] = {}
        self.embedded_kb = EMBEDDED_KNOWLEDGE_BASE

    def _http_get_json(self, url: str) -> Optional[Dict[str, Any]]:
        """Realiza una petición HTTP segura con cabecera User-Agent obligatoria."""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                if resp.status == 200:
                    raw_data = resp.read().decode("utf-8")
                    return json.loads(raw_data)
        except Exception:
            return None
        return None

    def find_nearest_embedded_kb(self, lat: float, lon: float, max_dist_km: float = 35.0) -> Optional[Dict[str, Any]]:
        """Busca si las coordenadas coinciden con un monumento de la base integrada."""
        closest_entry = None
        min_dist = float("inf")
        for key, entry in self.embedded_kb.items():
            dist = self.haversine_distance_km(lat, lon, entry["latitude"], entry["longitude"])
            if dist <= max_dist_km and dist < min_dist:
                min_dist = dist
                closest_entry = entry
        return closest_entry

    def search_embedded_kb_by_name(self, query: str) -> Optional[Dict[str, Any]]:
        """Busca en la base embebida por coincidencia de texto."""
        norm_q = query.lower().strip()
        for key, entry in self.embedded_kb.items():
            if norm_q in entry["name"].lower() or key in norm_q or any(w in entry["name"].lower() for w in norm_q.split()):
                return entry
        return None

    @staticmethod
    def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula la distancia sobre la esfera terrestre en kilómetros."""
        R = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return R * c

    def _extract_cultural_metadata(self, title: str, extract: str, desc: str = "") -> Tuple[str, str, str, str, str]:
        """
        Extractor neuro-simbólico:
        Determina origen ('de dónde vino'), constructores/origen, época histórica,
        categoría y forma holográfica 3D a partir del texto resumen.
        """
        combined = f"{title}. {desc}. {extract}".lower()

        # 1. Categoría
        category = "MONUMENTO"
        if any(w in combined for w in ["pirámide", "pyramid", "templo", "ruinas", "arqueológico", "ancient"]):
            category = "SITIO_ARQUEOLOGICO"
        elif any(w in combined for w in ["anfiteatro", "castillo", "catedral", "edificio", "tower", "torre", "basílica"]):
            category = "EDIFICACION"
        elif any(w in combined for w in ["estatua", "escultura", "monumento", "memorial", "busto"]):
            category = "MONUMENTO"
        elif any(w in combined for w in ["faraón", "general", "rey", "prócer", "líder", "filósofo", "arquitecto"]):
            category = "PERSONAJE"

        # 2. Forma holográfica 3D sugerida
        hologram_shape = "CUBO"
        color_name = "ORO"
        if "pirámide" in combined or "pyramid" in combined or "obelisco" in combined:
            hologram_shape = "PIRAMIDE"
            color_name = "ORO"
        elif "torre" in combined or "tower" in combined:
            hologram_shape = "PIRAMIDE"
            color_name = "ORO"
        elif "cúpula" in combined or "anfiteatro" in combined or "esfera" in combined or "templo" in combined:
            hologram_shape = "ESFERA"
            color_name = "CIAN"
        elif "castillo" in combined or "palacio" in combined or "fortaleza" in combined:
            hologram_shape = "CUBO"
            color_name = "VERDE"
        elif "ruina" in combined or "arqueológico" in combined:
            hologram_shape = "OCTAEDRO"
            color_name = "MAGENTA"

        # 3. Época aproximada mediante expresiones regulares
        epoch = "Época Histórica Clásica"
        siglo_match = re.search(r"siglo\s+([xvi|ivxlc]+|[0-9]{1,2})\s*(?:a\.?c\.?|d\.?c\.?|ne|ane)?", combined)
        year_match = re.search(r"(?:en|de|entre|hacia|c\.)\s*([0-9]{3,4})\s*(?:a\.?c\.?|a\.?d\.?|d\.?c\.?)?", combined)
        bc_match = re.search(r"([0-9]{3,4})\s*(?:a\.?c\.?|a\.?e\.?c\.?|bce|bc)", combined)

        if bc_match:
            epoch = f"c. {bc_match.group(1)} a.C. (Antigüedad)"
        elif siglo_match:
            epoch = f"Siglo {siglo_match.group(1).upper()}"
        elif year_match:
            epoch = f"Año {year_match.group(1)}"
        elif "medieval" in combined:
            epoch = "Edad Media (Siglos V - XV)"
        elif "renacimiento" in combined or "renaissance" in combined:
            epoch = "Renacimiento (Siglos XV - XVI)"
        elif "colonial" in combined:
            epoch = "Época Virreinal / Colonial"

        # 4. Origen geográfico y cultural: "De dónde vino"
        origin = "Civilización e Historia Universal"
        if "méxico" in combined or "azteca" in combined or "maya" in combined or "teotihuac" in combined:
            origin = "Mesoamérica / México Prehispánico"
        elif "roma" in combined or "romano" in combined or "italia" in combined:
            origin = "Imperio Romano / Península Itálica"
        elif "francia" in combined or "parís" in combined:
            origin = "París, República Francesa"
        elif "egipto" in combined or "faraón" in combined or "nilo" in combined:
            origin = "Antiguo Egipto / Valle del Nilo"
        elif "grecia" in combined or "atenas" in combined or "helen" in combined:
            origin = "Antigua Grecia / Mundo Helénico"
        elif "inca" in combined or "perú" in combined or "cusco" in combined:
            origin = "Imperio Incaico / Región Andina"
        elif "españa" in combined or "madrid" in combined or "hispano" in combined:
            origin = "Reino de España / Península Ibérica"
        elif desc:
            origin = desc

        return category, origin, epoch, hologram_shape, color_name

    def resolve_by_coordinates(
        self,
        latitude: float,
        longitude: float,
        spatial_local_coords: Optional[Tuple[float, float, float]] = None
    ) -> HistoricalObjectPOI:
        """
        Resuelve un lugar histórico dado su punto GPS (Lat, Lon) usando:
        1. Wikipedia Geosearch en vivo (español o inglés).
        2. Extracción de resumen y metadatos históricos.
        3. Fallback inteligente a base de conocimiento embebida si no hay conexión.
        """
        cache_key = f"coord_{latitude:.4f}_{longitude:.4f}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        local_xyz = spatial_local_coords if spatial_local_coords else (0.0, 0.0, 5.0)

        # 1. Intentar consultar APIs abiertas de Wikipedia (Español primero, luego Inglés)
        languages = ["es", "en"]
        found_title = None

        for lang in languages:
            geosearch_url = (
                f"https://{lang}.wikipedia.org/w/api.php?action=query&list=geosearch&"
                f"gscoord={latitude}|{longitude}&gsradius=5000&gslimit=3&format=json"
            )
            data = self._http_get_json(geosearch_url)
            if data and "query" in data and "geosearch" in data["query"] and len(data["query"]["geosearch"]) > 0:
                found_title = data["query"]["geosearch"][0]["title"]
                break

        if found_title:
            for lang in languages:
                summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(found_title)}"
                sum_data = self._http_get_json(summary_url)
                if sum_data and "extract" in sum_data and len(sum_data["extract"]) > 30:
                    extract = sum_data.get("extract", "")
                    desc = sum_data.get("description", "")
                    title = sum_data.get("title", found_title)

                    cat, origin, epoch, shape, col = self._extract_cultural_metadata(title, extract, desc)
                    clean_id = re.sub(r"[^a-zA-Z0-9_]+", "_", title.lower()).strip("_")
                    poi = HistoricalObjectPOI(
                        poi_id=clean_id,
                        name=title,
                        category=cat,
                        coordinates=local_xyz,
                        origin=origin,
                        epoch=epoch,
                        description=extract[:360] + ("..." if len(extract) > 360 else ""),
                        hologram_shape=shape,
                        color_name=col,
                        scale=1.2
                    )
                    self.cache[cache_key] = poi
                    return poi

        # 2. Fallback a la Base de Conocimiento Embebida
        closest = self.find_nearest_embedded_kb(latitude, longitude)
        if closest:
            clean_id = re.sub(r"[^a-zA-Z0-9_]+", "_", closest["name"].lower()).strip("_")
            poi = HistoricalObjectPOI(
                poi_id=clean_id,
                name=closest["name"],
                category=closest["category"],
                coordinates=local_xyz,
                origin=closest["origin"],
                epoch=closest["epoch"],
                description=closest["description"],
                hologram_shape=closest.get("hologram_shape", "CUBO"),
                color_name=closest.get("color_name", "ORO"),
                scale=closest.get("scale", 1.0)
            )
            self.cache[cache_key] = poi
            return poi

        # 3. Fallback genérico para coordenadas aisladas
        poi = HistoricalObjectPOI(
            poi_id=f"poi_lat{latitude:.2f}_lon{longitude:.2f}",
            name=f"Sitio Geográfico ({latitude:.3f}, {longitude:.3f})",
            category="EDIFICACION",
            coordinates=local_xyz,
            origin=f"Coordenadas Globales ({latitude:.4f}° N, {longitude:.4f}° E)",
            epoch="Registro Geoespacial Activo",
            description=f"Punto geográfico detectado en la cuadrícula de Cesium 3D Tiles / OpenStreetMap.",
            hologram_shape="CUBO",
            color_name="CIAN",
            scale=1.0
        )
        self.cache[cache_key] = poi
        return poi

    def resolve_by_name(
        self,
        place_name: str,
        spatial_local_coords: Optional[Tuple[float, float, float]] = None
    ) -> HistoricalObjectPOI:
        """
        Resuelve un lugar del mundo por su nombre o término histórico:
        (ej: 'Torre Eiffel', 'Pirámides de Teotihuacán', 'Coliseo Romano', 'Machu Picchu').
        """
        cache_key = f"name_{place_name.lower().strip()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        local_xyz = spatial_local_coords if spatial_local_coords else (0.0, 0.0, 5.0)

        # 1. Comprobar si coincide directamente con la base de conocimiento integrada
        emb_match = self.search_embedded_kb_by_name(place_name)

        # 2. Intentar consultar en línea en Wikipedia
        languages = ["es", "en"]
        online_poi = None
        for lang in languages:
            search_url = (
                f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&"
                f"srsearch={urllib.parse.quote(place_name)}&format=json"
            )
            data = self._http_get_json(search_url)
            if data and "query" in data and "search" in data["query"] and len(data["query"]["search"]) > 0:
                best_title = data["query"]["search"][0]["title"]
                sum_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(best_title)}"
                sum_data = self._http_get_json(sum_url)
                if sum_data and "extract" in sum_data and len(sum_data["extract"]) > 30:
                    extract = sum_data.get("extract", "")
                    desc = sum_data.get("description", "")
                    title = sum_data.get("title", best_title)

                    cat, origin, epoch, shape, col = self._extract_cultural_metadata(title, extract, desc)
                    clean_id = re.sub(r"[^a-zA-Z0-9_]+", "_", title.lower()).strip("_")

                    # Si la base embebida tiene detalles específicos para este lugar, combinarlos
                    if emb_match and (emb_match["name"].lower() in title.lower() or title.lower() in emb_match["name"].lower()):
                        origin = emb_match["origin"]
                        epoch = emb_match["epoch"]
                        shape = emb_match["hologram_shape"]
                        col = emb_match["color_name"]

                    online_poi = HistoricalObjectPOI(
                        poi_id=clean_id,
                        name=title,
                        category=cat,
                        coordinates=local_xyz,
                        origin=origin,
                        epoch=epoch,
                        description=extract[:360] + ("..." if len(extract) > 360 else ""),
                        hologram_shape=shape,
                        color_name=col,
                        scale=1.2
                    )
                    break

        if online_poi:
            self.cache[cache_key] = online_poi
            return online_poi

        # Si falló la red o no hubo respuesta, usar base embebida
        if emb_match:
            clean_id = re.sub(r"[^a-zA-Z0-9_]+", "_", emb_match["name"].lower()).strip("_")
            poi = HistoricalObjectPOI(
                poi_id=clean_id,
                name=emb_match["name"],
                category=emb_match["category"],
                coordinates=local_xyz,
                origin=emb_match["origin"],
                epoch=emb_match["epoch"],
                description=emb_match["description"],
                hologram_shape=emb_match.get("hologram_shape", "CUBO"),
                color_name=emb_match.get("color_name", "ORO"),
                scale=emb_match.get("scale", 1.0)
            )
            self.cache[cache_key] = poi
            return poi

        # Si todo falla, POI sintético basado en el término de búsqueda
        clean_id = re.sub(r"[^a-zA-Z0-9_]+", "_", place_name.lower()).strip("_")
        poi = HistoricalObjectPOI(
            poi_id=clean_id,
            name=place_name.title(),
            category="MONUMENTO",
            coordinates=local_xyz,
            origin="Historia y Geografía Mundial",
            epoch="Registro Histórico Global",
            description=f"Entidad holográfica modelada para {place_name}.",
            hologram_shape="CUBO",
            color_name="ORO",
            scale=1.0
        )
        self.cache[cache_key] = poi
        return poi

    def sync_locations_to_lens(
        self,
        lens: HistoricalSpatialLens,
        locations: List[Dict[str, Any]],
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        meters_per_degree: float = 111320.0
    ) -> List[HistoricalObjectPOI]:
        """
        Sincroniza una lista de ubicaciones globales hacia una instancia de HistoricalSpatialLens.
        Mapea coordenadas esféricas GPS a coordenadas espaciales relativas euclidianas 3D (X, Y, Z)
        centradas en la posición del usuario en el espacio AR/VR.
        """
        if not locations:
            return []

        # Determinar centro si no se especificó
        if center_lat is None or center_lon is None:
            lats = [loc["lat"] for loc in locations if "lat" in loc]
            lons = [loc["lon"] for loc in locations if "lon" in loc]
            center_lat = float(np.mean(lats)) if lats else 0.0
            center_lon = float(np.mean(lons)) if lons else 0.0

        resolved_pois: List[HistoricalObjectPOI] = []
        cos_center = math.cos(math.radians(center_lat))

        for idx, item in enumerate(locations):
            # Calcular offset relativo en metros desde el centro GPS
            if "lat" in item and "lon" in item:
                lat = float(item["lat"])
                lon = float(item["lon"])
                delta_x = (lon - center_lon) * meters_per_degree * cos_center
                delta_z = (lat - center_lat) * meters_per_degree
                # Escalar para espacio de visión confortable en AR (ej. radio de 5 a 20 metros)
                mag = math.sqrt(delta_x**2 + delta_z**2)
                if mag > 0.001:
                    norm_dist = min(max(mag * 0.05, 3.5), 18.0)
                    local_x = round((delta_x / mag) * norm_dist, 2)
                    local_z = round((delta_z / mag) * norm_dist, 2)
                else:
                    local_x = float(idx * 3.0)
                    local_z = 5.0
                local_y = float(item.get("altitude", 0.0))
                spatial_coords = (local_x, local_y, local_z)

                poi = self.resolve_by_coordinates(lat, lon, spatial_local_coords=spatial_coords)
            elif "name" in item:
                spatial_coords = (float((idx - len(locations) / 2) * 4.0), 0.0, 5.0 + idx * 2.0)
                poi = self.resolve_by_name(item["name"], spatial_local_coords=spatial_coords)
            else:
                continue

            lens.register_poi(poi)
            resolved_pois.append(poi)

        return resolved_pois
