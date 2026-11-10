"""
NexusXR - Módulo de Lente Espacial Geo-Histórica en Realidad Aumentada y Virtual.

Permite colocar la cámara en un entorno (como un parque histórico) para:
1. Detectar coordenadas geográficas / espaciales en el mapa.
2. Materializar hologramas 3D de personajes, estatuas, edificaciones o artefactos históricos.
3. Desglosar al lado de cada objeto su origen ("De dónde vino"), época y descripción histórica.
4. Persistir toda la experiencia y objetos descubiertos en el formato criptográfico .nxs (Realidad Virtual).
5. Decompilar o ejecutar secuencias de descubrimiento en NexusLang.
"""

import math
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .nexus_workspace_project import SerializableHologram3D, NXSProjectSerializer, rotation_matrix_3d


@dataclass
class HistoricalObjectPOI:
    """
    Punto de Interés (POI) histórico ubicado en coordenadas espaciales/mapa.
    Representa personajes, monumentos o artefactos históricos con su desglose cultural.
    """
    poi_id: str
    name: str
    category: str              # ej: "PERSONAJE", "MONUMENTO", "ARTEFACTO", "EDIFICACION"
    coordinates: Tuple[float, float, float]  # Coordenadas en mapa (x, y, z) o (este, norte, alt)
    origin: str                # Procedencia geográfica/cultural: de dónde vino
    epoch: str                 # Época / Siglo / Año histórico
    description: str           # Reseña histórica detallada
    hologram_shape: str = "CUBO"   # "CUBO", "ESFERA", "PIRAMIDE", "OCTAEDRO", "DIAMANTE"
    color_name: str = "ORO"
    scale: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "poi_id": self.poi_id,
            "name": self.name,
            "category": self.category,
            "coordinates": list(self.coordinates),
            "origin": self.origin,
            "epoch": self.epoch,
            "description": self.description,
            "hologram_shape": self.hologram_shape,
            "color_name": self.color_name,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoricalObjectPOI":
        return cls(
            poi_id=data["poi_id"],
            name=data["name"],
            category=data.get("category", "ARTEFACTO"),
            coordinates=tuple(data.get("coordinates", [0.0, 0.0, 0.0])),
            origin=data.get("origin", "Desconocido"),
            epoch=data.get("epoch", "Época no especificada"),
            description=data.get("description", ""),
            hologram_shape=data.get("hologram_shape", "CUBO"),
            color_name=data.get("color_name", "ORO"),
            scale=float(data.get("scale", 1.0)),
        )

    def to_hologram_3d(self) -> SerializableHologram3D:
        """Crea la entidad holográfica 3D física a partir del POI histórico."""
        holo = SerializableHologram3D(
            name=self.poi_id,
            shape=self.hologram_shape,
            pos=self.coordinates,
            color_name=self.color_name
        )
        holo.scale = self.scale
        return holo


class HistoricalSpatialLens:
    """
    Motor de Cámara AR / VR con Lente Histórica:
    - Rastrea la posición y orientación de la cámara en el mapa histórico.
    - Proyecta en Realidad Aumentada los personajes/objetos 3D según coordenadas.
    - Genera carteles/HUDs con la procedencia histórica ("De dónde vino") y detalles.
    - Guarda todos los descubrimientos a proyectos espaciales '.nxs' para Realidad Virtual.
    """
    def __init__(self, detection_radius: float = 15.0):
        self.detection_radius = float(detection_radius)
        self.catalog: Dict[str, HistoricalObjectPOI] = {}
        self.discovered_pois: Dict[str, HistoricalObjectPOI] = {}
        self.camera_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_orientation = np.array([0.0, 0.0, 0.0], dtype=np.float32) # yaw, pitch, roll
        self.vr_blend: float = 0.0  # 0.0: AR (Cámara directa), 1.0: VR (Inmersión total)

    def register_poi(self, poi: HistoricalObjectPOI):
        """Registra un nuevo personaje u objeto histórico en el mapa."""
        self.catalog[poi.poi_id] = poi

    def set_camera_pose(self, pos: Tuple[float, float, float], angles: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """Actualiza las coordenadas actuales del usuario/cámara en el parque."""
        self.camera_position = np.array(pos, dtype=np.float32)
        self.camera_orientation = np.array(angles, dtype=np.float32)

    def scan_nearby_objects(self) -> List[HistoricalObjectPOI]:
        """
        Escanea y retorna los objetos o personajes históricos en el rango de visión de la cámara.
        Automáticamente los marca como descubiertos en la bitácora.
        """
        in_range: List[HistoricalObjectPOI] = []
        for poi in self.catalog.values():
            poi_pos = np.array(poi.coordinates, dtype=np.float32)
            dist = float(np.linalg.norm(poi_pos - self.camera_position))
            if dist <= self.detection_radius:
                in_range.append(poi)
                if poi.poi_id not in self.discovered_pois:
                    self.discovered_pois[poi.poi_id] = poi
        return in_range

    def get_ar_breakdown_card(self, poi: HistoricalObjectPOI) -> Dict[str, Any]:
        """
        Genera la ficha explicativa que aparece flotando al lado del objeto 3D en AR:
        Nombre, clasificación, 'De dónde vino' (origen), época, distancia y descripción.
        """
        poi_pos = np.array(poi.coordinates, dtype=np.float32)
        dist = float(np.linalg.norm(poi_pos - self.camera_position))
        return {
            "poi_id": poi.poi_id,
            "title": poi.name,
            "category": poi.category,
            "origin": poi.origin,
            "epoch": poi.epoch,
            "description": poi.description,
            "distance_m": round(dist, 2),
            "coordinates": [round(float(c), 3) for c in poi.coordinates],
            "overlay_text": [
                f"✦ {poi.name.upper()} ({poi.category})",
                f"➤ DE DÓNDE VINO: {poi.origin}",
                f"➤ ÉPOCA / AÑO: {poi.epoch}",
                f"➤ UBICACIÓN: Coords {poi.coordinates} | Dist: {dist:.1f}m",
                f"➤ HISTORIA: {poi.description}",
            ]
        }

    def project_poi_to_screen(
        self,
        poi: HistoricalObjectPOI,
        frame_width: int = 1280,
        frame_height: int = 720,
        focal_length: float = 650.0
    ) -> Optional[Dict[str, Any]]:
        """
        Calcula la proyección matemática 3D a pantalla 2D del objeto y su panel lateral de texto.
        Transforma desde el espacio del mapa al sistema de coordenadas de la cámara.
        """
        # Posición relativa a la cámara
        rel_pos = np.array(poi.coordinates, dtype=np.float32) - self.camera_position

        # Rotar por la orientación de la cámara
        cy, sy = math.cos(self.camera_orientation[0]), math.sin(self.camera_orientation[0])
        # Rotación simple en eje Y (yaw)
        cam_x = rel_pos[0] * cy - rel_pos[2] * sy
        cam_y = rel_pos[1]
        cam_z = rel_pos[0] * sy + rel_pos[2] * cy

        # Si el objeto está detrás de la cámara, no se proyecta
        if cam_z <= 0.1:
            return None

        screen_x = int((cam_x * focal_length) / cam_z + (frame_width * 0.5))
        screen_y = int((cam_y * focal_length) / cam_z + (frame_height * 0.5))

        # Panel lateral histórico flotante al lado derecho del objeto 3D
        panel_x = screen_x + int(80.0 / max(cam_z * 0.5, 0.2))
        panel_y = screen_y - 40

        breakdown = self.get_ar_breakdown_card(poi)
        return {
            "poi_id": poi.poi_id,
            "screen_pos": (screen_x, screen_y),
            "panel_pos": (panel_x, panel_y),
            "z_depth": float(cam_z),
            "breakdown": breakdown,
        }

    def render_ar_frame(
        self,
        frame: np.ndarray,
        focal_length: float = 650.0
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Renderiza la vista en Realidad Aumentada / Mixta / Virtual sobre una imagen dada:
        - Si vr_blend > 0, dibuja rejilla sintética VR.
        - Dibuja el objeto 3D holográfico proyectado en sus coordenadas.
        - Dibuja el panel flotante lateral que muestra de dónde vino y su historia.
        """
        import cv2

        h, w, _ = frame.shape
        active_render = frame.copy()

        # Si está activo el modo Realidad Virtual, mezclar con entorno virtual
        if self.vr_blend > 0.01:
            vr_bg = np.zeros_like(frame)
            horizon = int(h * 0.52)
            for i in range(-15, 16):
                cv2.line(vr_bg, (int(w * 0.5 + i * 90), h), (int(w * 0.5 + i * 15), horizon), (50, 40, 20), 1)
            for d in range(1, 10):
                y = int(horizon + (d ** 1.8) * 6.0)
                if y < h:
                    cv2.line(vr_bg, (0, y), (w, y), (60, 45, 20), 1)
            active_render = cv2.addWeighted(frame, 1.0 - self.vr_blend, vr_bg, self.vr_blend, 0)

        nearby = self.scan_nearby_objects()
        rendered_breakdowns = []

        for poi in nearby:
            proj = self.project_poi_to_screen(poi, w, h, focal_length)
            if proj is None:
                continue

            sx, sy = proj["screen_pos"]
            px, py = proj["panel_pos"]
            breakdown = proj["breakdown"]
            rendered_breakdowns.append(breakdown)

            # 1. Dibujar holograma 3D proyectado
            holo_3d = poi.to_hologram_3d()
            # Rotar ligeramente según la escala
            R = rotation_matrix_3d(0.2, 0.4, 0.1)
            cam_pts_3d = (holo_3d.vertices * holo_3d.scale) + np.array(poi.coordinates, dtype=np.float32)

            projected_verts = []
            for v in cam_pts_3d:
                rel = v - self.camera_position
                if rel[2] > 0.1:
                    vx = int((rel[0] * focal_length) / rel[2] + (w * 0.5))
                    vy = int((rel[1] * focal_length) / rel[2] + (h * 0.5))
                    projected_verts.append((vx, vy))

            # Dibujar aristas si se proyectaron los vértices
            if len(projected_verts) == len(holo_3d.vertices):
                for e1, e2 in holo_3d.edges:
                    cv2.line(active_render, projected_verts[e1], projected_verts[e2], holo_3d.color, 2, cv2.LINE_AA)

            # Punto central y halo holográfico
            cv2.circle(active_render, (sx, sy), 8, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(active_render, (sx, sy), 14, holo_3d.color, 2, cv2.LINE_AA)

            # 2. Línea guía que conecta el objeto 3D con la tarjeta informativa lateral
            cv2.line(active_render, (sx, sy), (px, py + 15), (0, 255, 255), 1, cv2.LINE_AA)

            # 3. Panel flotante lateral (Tarjeta de Desglose Histórico)
            card_lines = breakdown["overlay_text"]
            panel_w = 460
            panel_h = 24 + len(card_lines) * 22

            # Clamp panel para que no se salga de pantalla
            clamp_px = min(max(px, 10), w - panel_w - 10)
            clamp_py = min(max(py, 10), h - panel_h - 10)

            # Fondo translúcido del panel de información
            overlay = active_render.copy()
            cv2.rectangle(overlay, (clamp_px, clamp_py), (clamp_px + panel_w, clamp_py + panel_h), (18, 20, 26), -1)
            cv2.rectangle(active_render, (clamp_px, clamp_py), (clamp_px + panel_w, clamp_py + panel_h), (0, 215, 255), 1)
            cv2.addWeighted(overlay, 0.75, active_render, 0.25, 0, active_render)

            # Escribir textos en la tarjeta
            for idx, text in enumerate(card_lines):
                font_color = (0, 255, 220) if idx == 0 else ((255, 210, 100) if idx == 1 else (225, 235, 245))
                font_scale = 0.44 if idx <= 1 else 0.38
                thickness = 2 if idx <= 1 else 1
                cv2.putText(
                    active_render,
                    text,
                    (clamp_px + 12, clamp_py + 22 + idx * 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    font_color,
                    thickness,
                    cv2.LINE_AA
                )

        # HUD superior con estado de la expedición
        cv2.rectangle(active_render, (15, 15), (560, 80), (12, 14, 20), -1)
        cv2.rectangle(active_render, (15, 15), (560, 80), (0, 255, 160), 1)
        mode_str = "AR (CÁMARA REAL)" if self.vr_blend < 0.3 else ("VR (VIRTUAL TOTAL)" if self.vr_blend > 0.7 else "HÍBRIDO AR/VR")
        cv2.putText(active_render, f"NEXUS-XR: LENTE HISTÓRICA GEOESPACIAL | MODO: {mode_str}",
                    (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 160), 2)
        cv2.putText(active_render, f"Coords Cámara: X={self.camera_position[0]:.1f}, Y={self.camera_position[1]:.1f}, Z={self.camera_position[2]:.1f} | Descubiertos: {len(self.discovered_pois)}/{len(self.catalog)}",
                    (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 240, 250), 1)

        return active_render, rendered_breakdowns

    def save_vr_expedition(self, filepath: str = "expedicion_historica.nxs") -> str:
        """
        Guarda toda la sesión y los objetos 3D descubiertos en Realidad Virtual
        en formato de proyecto espacial '.nxs' con integridad criptográfica SHA-256.
        """
        scene: Dict[str, SerializableHologram3D] = {}
        for poi in self.discovered_pois.values():
            holo = poi.to_hologram_3d()
            scene[poi.poi_id] = holo

        macros = {
            "RECONSTRUIR_PARQUE_VR": [
                f"MODO_REALIDAD VR_TOTAL",
                f"# Reconstrucción de expedición histórica: {len(self.discovered_pois)} figuras registradas.",
            ]
        }
        for poi in self.discovered_pois.values():
            macros["RECONSTRUIR_PARQUE_VR"].append(
                f"CREAR_HOLOGRAMA {poi.poi_id} {poi.hologram_shape} {poi.coordinates[0]} {poi.coordinates[1]} {poi.coordinates[2]}"
            )

        msg = NXSProjectSerializer.save_project(
            filepath=filepath,
            scene=scene,
            macros=macros,
            gravity_g=0.0,
            vr_blend=1.0  # Al guardarse para VR, el factor de mezcla se establece en 1.0 (VR Total)
        )
        return msg

    def export_nexus_script(self, filepath: str = "expedicion_historica_script.nxs") -> str:
        """
        Decompila la experiencia histórica en un script editable de NexusLang
        con metadatos históricos comentados y comandos ejecutables de holografía.
        """
        full_path = NXSProjectSerializer.ensure_extension(filepath)
        lines = [
            "# ===================================================================",
            f"# EXPEDICIÓN HISTÓRICA GEOESPACIAL NEXUS-XR ({datetime.now().isoformat()})",
            "# ===================================================================",
            "MODO_REALIDAD HIBRIDO_CUANTICO_AR_VR",
            ""
        ]

        for poi in self.discovered_pois.values():
            lines.append(f"# [POI HISTÓRICO] {poi.name} ({poi.category})")
            lines.append(f"# Origen: {poi.origin}")
            lines.append(f"# Época: {poi.epoch}")
            lines.append(f"# Descripción: {poi.description}")
            px, py, pz = poi.coordinates
            lines.append(f"CREAR_HOLOGRAMA {poi.poi_id} {poi.hologram_shape} {px:.2f} {py:.2f} {pz:.2f}")
            lines.append(f"APRENDER_COMANDO INSPECCIONAR_{poi.poi_id.upper()} IMPRIMIR \"{poi.name}: {poi.origin} ({poi.epoch})\"")
            lines.append("")

        with open(full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return f"Script histórico NexusLang exportado exitosamente en '{full_path}'."


def build_default_historical_park() -> HistoricalSpatialLens:
    """
    Construye un entorno predeterminado de Parque Histórico con diferentes personajes,
    monumentos y artefactos en coordenadas espaciales clave.
    """
    lens = HistoricalSpatialLens(detection_radius=12.0)

    # 1. Monumento al Libertador / Personaje en la entrada principal
    lens.register_poi(HistoricalObjectPOI(
        poi_id="estatua_libertador",
        name="General Simón Bolívar",
        category="PERSONAJE",
        coordinates=(0.0, 0.2, 4.0),
        origin="Caracas, Capitanía General de Venezuela (1783)",
        epoch="Siglo XIX (1810-1830)",
        description="Líder fundamental de la independencia hispanoamericana frente al Imperio Español.",
        hologram_shape="PIRAMIDE",
        color_name="ORO",
        scale=1.2
    ))

    # 2. Artefacto prehispánico en la plazoleta arqueológica
    lens.register_poi(HistoricalObjectPOI(
        poi_id="monolito_solar",
        name="Monolito del Calendario Solar",
        category="ARTEFACTO",
        coordinates=(-5.0, 0.0, 6.0),
        origin="Valle de Anáhuac / Civilización Azteca-Mexica",
        epoch="Período Posclásico Tardío (c. 1479)",
        description="Talla basáltica con inscripciones cosmogónicas sobre los ciclos solares y las eras astronómicas.",
        hologram_shape="OCTAEDRO",
        color_name="MAGENTA",
        scale=1.1
    ))

    # 3. Personaje científico en el observatorio histórico
    lens.register_poi(HistoricalObjectPOI(
        poi_id="sabio_caldas",
        name="Francisco José de Caldas",
        category="PERSONAJE",
        coordinates=(6.0, 0.4, 7.5),
        origin="Popayán, Virreinato de la Nueva Granada (1768)",
        epoch="Ilustración / Expedición Botánica (1801-1816)",
        description="Científico, astrónomo e ingeniero pionero que inventó el hipsómetro para medir alturas con agua hirviendo.",
        hologram_shape="ESFERA",
        color_name="CIAN",
        scale=1.0
    ))

    # 4. Cañón colonial en el baluarte de defensa
    lens.register_poi(HistoricalObjectPOI(
        poi_id="canon_defensa",
        name="Cañón de Bronce de 12 Libras",
        category="MONUMENTO",
        coordinates=(-8.0, -0.2, 10.0),
        origin="Reales Fábricas de Artillería de Sevilla, España",
        epoch="Período Virreinal / Siglo XVIII (1742)",
        description="Pieza de artillería naval fundida para la defensa costera contra flotas corsarias.",
        hologram_shape="CUBO",
        color_name="VERDE",
        scale=0.95
    ))

    return lens
