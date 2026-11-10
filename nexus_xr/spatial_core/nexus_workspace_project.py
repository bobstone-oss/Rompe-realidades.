import os
import glob
import json
import time
import math
import random
import hashlib
import threading
import queue
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import cv2
import mediapipe as mp
import numpy as np


# =====================================================================
# 1. MATEMÁTICA 3D Y PALETA ESPECTRAL
# =====================================================================

def rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "CIAN": (255, 230, 0),
    "VERDE": (90, 255, 90),
    "MAGENTA": (255, 80, 255),
    "ORO": (0, 215, 255),
    "ROJO": (60, 60, 255),
    "AZUL": (255, 140, 40),
    "BLANCO": (255, 255, 255),
}


# =====================================================================
# 2. ENTIDAD HOLOGRÁFICA SERIALIZABLE (TO_DICT / FROM_DICT)
# =====================================================================

class SerializableHologram3D:
    """
    Cuerpo rígido 3D que sabe serializar su estado físico completo
    a estructuras nativas de archivos '.nxs' y reconstruirse desde cero.
    """
    def __init__(
        self,
        name: str,
        shape: str = "CUBO",
        pos=(0.0, 0.0, 0.0),
        color_name: str = "CIAN"
    ):
        self.name = name
        self.shape = shape.upper()
        self.color_name = color_name.upper() if color_name.upper() in COLOR_PALETTE else "CIAN"
        self.color = COLOR_PALETTE[self.color_name]

        self.position = np.array(pos, dtype=np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.25, 0.45, 0.1], dtype=np.float32)
        self.angular_vel = np.array([0.008, 0.018, 0.005], dtype=np.float32)

        self.scale = 0.85
        self.base_radius = 0.36
        self.mass = 1.0
        self.restitution = 0.88
        self.magnetic_charge = 0.0
        self.bound_to_hand = False

        self.vertices, self.edges, self.faces = self._build_mesh(self.shape)

    @property
    def collider_radius(self) -> float:
        return self.base_radius * self.scale

    def set_color(self, color_name: str):
        cname = color_name.upper()
        if cname in COLOR_PALETTE:
            self.color_name = cname
            self.color = COLOR_PALETTE[cname]

    def to_nxs_dict(self) -> Dict[str, Any]:
        """Empaqueta todos los vectores físicos e identidad del holograma."""
        return {
            "name": self.name,
            "shape": self.shape,
            "color_name": self.color_name,
            "position": [round(float(x), 5) for x in self.position],
            "velocity": [round(float(x), 5) for x in self.velocity],
            "rotation": [round(float(x), 5) for x in self.rotation],
            "angular_vel": [round(float(x), 5) for x in self.angular_vel],
            "scale": round(float(self.scale), 4),
            "mass": round(float(self.mass), 4),
            "restitution": round(float(self.restitution), 4),
            "magnetic_charge": float(self.magnetic_charge),
            "bound_to_hand": bool(self.bound_to_hand),
        }

    @classmethod
    def from_nxs_dict(cls, data: Dict[str, Any]) -> "SerializableHologram3D":
        """Reconstruye el objeto 3D exacto a partir de un nodo del archivo '.nxs'."""
        obj = cls(
            name=data["name"],
            shape=data.get("shape", "CUBO"),
            pos=tuple(data.get("position", [0.0, 0.0, 0.0])),
            color_name=data.get("color_name", "CIAN")
        )
        obj.velocity = np.array(data.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        obj.rotation = np.array(data.get("rotation", [0.25, 0.45, 0.1]), dtype=np.float32)
        obj.angular_vel = np.array(data.get("angular_vel", [0.008, 0.018, 0.005]), dtype=np.float32)
        obj.scale = float(data.get("scale", 0.85))
        obj.mass = float(data.get("mass", 1.0))
        obj.restitution = float(data.get("restitution", 0.88))
        obj.magnetic_charge = float(data.get("magnetic_charge", 0.0))
        obj.bound_to_hand = bool(data.get("bound_to_hand", False))
        return obj

    def _build_mesh(self, shape: str):
        if shape == "PIRAMIDE":
            verts = np.array([
                [-0.45, 0.4, -0.45], [0.45, 0.4, -0.45],
                [0.45, 0.4, 0.45], [-0.45, 0.4, 0.45], [0.0, -0.5, 0.0]
            ], dtype=np.float32)
            edges = [(0,1), (1,2), (2,3), (3,0), (0,4), (1,4), (2,4), (3,4)]
            faces = [(0,1,4), (1,2,4), (2,3,4), (3,0,4)]
            return verts, edges, faces
        elif shape in ("OCTAEDRO", "DIAMANTE"):
            verts = np.array([
                [0.5, 0, 0], [-0.5, 0, 0], [0, 0.58, 0],
                [0, -0.58, 0], [0, 0, 0.5], [0, 0, -0.5]
            ], dtype=np.float32)
            edges = [(0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,4), (4,3), (3,5), (5,2)]
            faces = [(0,2,4), (0,4,3), (1,2,4), (1,4,3)]
            return verts, edges, faces
        elif shape == "ESFERA":
            verts, edges = [], []
            rings, sectors = 5, 10
            for r in range(rings):
                phi = math.pi * (r + 1) / (rings + 1) - (math.pi / 2)
                for s in range(sectors):
                    theta = 2.0 * math.pi * s / sectors
                    verts.append([0.46 * math.cos(phi) * math.cos(theta), 0.46 * math.sin(phi), 0.46 * math.cos(phi) * math.sin(theta)])
                    curr = r * sectors + s
                    edges.append((curr, r * sectors + ((s + 1) % sectors)))
                    if r > 0:
                        edges.append((curr, (r - 1) * sectors + s))
            return np.array(verts, dtype=np.float32), edges, []
        else:
            verts = np.array([
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
            ], dtype=np.float32) * 0.36
            edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
            faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6)]
            return verts, edges, faces


# =====================================================================
# 3. MOTOR DE SERIALIZACIÓN Y DECOMPILACIÓN DE ARCHIVOS '.NXS'
# =====================================================================

class NXSProjectSerializer:
    """
    Gestiona la lectura/escritura de archivos de proyecto '.nxs':
    - Incluye verificación criptográfica SHA-256 de integridad de escena.
    - Soporta exportación como script ejecutable de NexusLang.
    """
    FORMAT_VERSION = "NEXUS_SPATIAL_PROJECT_V1"

    @staticmethod
    def ensure_extension(filename: str) -> str:
        clean = filename.strip().replace('"', "").replace("'", "")
        if not clean.lower().endswith(".nxs"):
            clean += ".nxs"
        return clean

    @classmethod
    def save_project(
        cls,
        filepath: str,
        scene: Dict[str, SerializableHologram3D],
        macros: Dict[str, List[str]],
        gravity_g: float,
        vr_blend: float
    ) -> str:
        full_path = cls.ensure_extension(filepath)

        payload_core = {
            "environment": {
                "gravity_g": round(float(gravity_g), 4),
                "vr_blend": round(float(vr_blend), 4),
            },
            "learned_macros": macros,
            "holograms": [obj.to_nxs_dict() for obj in scene.values()],
        }

        # Calcular huella SHA-256 sobre el contenido determinista
        canonical_json = json.dumps(payload_core, sort_keys=True)
        checksum = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]

        document = {
            "header": {
                "magic": cls.FORMAT_VERSION,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "object_count": len(scene),
                "macro_count": len(macros),
                "sha256_short": checksum,
            },
            "data": payload_core,
        }

        # Guardado atómico mediante archivo temporal
        tmp_path = full_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, full_path)

        return f"Proyecto '{full_path}' guardado ({len(scene)} hologramas, {len(macros)} macros, SHA:{checksum[:8]})."

    @classmethod
    def load_project(
        cls,
        filepath: str
    ) -> Tuple[Dict[str, SerializableHologram3D], Dict[str, List[str]], float, float, str]:
        full_path = cls.ensure_extension(filepath)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No existe el archivo '{full_path}'")

        with open(full_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Si el archivo empieza con '#' o comandos, es un script '.nxs' en texto plano
        if not raw_text.lstrip().startswith("{"):
            raise ValueError("SCRIPT_MODE")

        document = json.loads(raw_text)
        header = document.get("header", {})
        data = document.get("data", {})

        # Verificar integridad SHA-256
        expected_sha = header.get("sha256_short", "")
        canonical_json = json.dumps(data, sort_keys=True)
        actual_sha = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]
        if expected_sha and expected_sha != actual_sha:
            raise ValueError(f"Checksum inválido en '{full_path}': archivo alterado.")

        # Reconstruir diccionario de hologramas 3D
        new_scene: Dict[str, SerializableHologram3D] = {}
        for h_dict in data.get("holograms", []):
            obj = SerializableHologram3D.from_nxs_dict(h_dict)
            new_scene[obj.name] = obj

        learned_macros: Dict[str, List[str]] = data.get("learned_macros", {})
        env = data.get("environment", {})
        gravity_g = float(env.get("gravity_g", 0.0))
        vr_blend = float(env.get("vr_blend", 0.25))

        msg = f"Cargado '{full_path}' ({len(new_scene)} hologramas, {len(learned_macros)} macros)."
        return new_scene, learned_macros, gravity_g, vr_blend, msg

    @classmethod
    def export_executable_script(
        cls,
        filepath: str,
        scene: Dict[str, SerializableHologram3D],
        macros: Dict[str, List[str]],
        gravity_g: float,
        vr_blend: float
    ) -> str:
        """
        Decompila la escena 3D viva a un archivo de código fuente NexusLang
        que cualquiera puede abrir, editar o ejecutar línea por línea.
        """
        full_path = cls.ensure_extension(filepath)
        lines = [
            f"# ===================================================================",
            f"# PROYECTO ESPACIAL NEXUS-XR DECOMPILADO ({datetime.now().isoformat(timespec='seconds')})",
            f"# ===================================================================",
            f"BORRAR TODOS",
            f"REALIDAD {int(vr_blend * 100)}",
            f"GRAVEDAD {gravity_g:.2f}",
            ""
        ]

        if macros:
            lines.append("# --- MACROS DE METAPROGRAMACIÓN APRENDIDAS ---")
            for m_name, m_cmds in macros.items():
                joined = " ; ".join(m_cmds)
                lines.append(f"MACRO {m_name} = {joined}")
            lines.append("")

        lines.append("# --- ENTIDADES HOLOGRÁFICAS 3D Y ESTADO FÍSICO ---")
        for obj in scene.values():
            px, py, pz = obj.position
            vx, vy, vz = obj.velocity
            lines.append(f"CREAR {obj.name} {obj.shape} [{px:.3f}, {py:.3f}, {pz:.3f}]")
            lines.append(f"COLOR {obj.name} {obj.color_name}")
            lines.append(f"ESCALAR {obj.name} {obj.scale:.3f}")
            lines.append(f"REBOTE {obj.name} {obj.restitution:.2f}")
            if obj.magnetic_charge != 0.0:
                pol = "POSITIVO" if obj.magnetic_charge > 0 else "NEGATIVO"
                lines.append(f"IMAN {obj.name} {pol}")
            if np.linalg.norm(obj.velocity) > 0.002:
                lines.append(f"LANZAR {obj.name} [{vx:.4f}, {vy:.4f}, {vz:.4f}]")
            if obj.bound_to_hand:
                lines.append(f"ANCLAR_MANO {obj.name}")
            lines.append("")

        with open(full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return f"Script NexusLang exportado en '{full_path}' ({len(lines)} líneas)."


# =====================================================================
# 4. SOLUCIONADOR FÍSICO 3D + INTÉRPRETE CON SOPORTE DE PROYECTOS '.NXS'
# =====================================================================

class ProjectPhysicsEngine:
    def __init__(self):
        self.bounds = np.array([1.65, 0.95, 0.85], dtype=np.float32)
        self.gravity_g = 0.0
        self.vr_blend = 0.25
        self.magnetic_k = 0.0022
        self.magnetic_links: List[Tuple[np.ndarray, np.ndarray, bool]] = []

    def step(self, scene: Dict[str, SerializableHologram3D], hand_pos_3d: Optional[np.ndarray], hand_vel_3d: np.ndarray, is_pinching: bool):
        self.magnetic_links.clear()
        bodies = list(scene.values())
        n = len(bodies)

        # 1. Fuerza magnética entre hologramas
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = bodies[i], bodies[j]
                if b1.magnetic_charge != 0.0 and b2.magnetic_charge != 0.0:
                    delta = b2.position - b1.position
                    dist_sq = float(np.dot(delta, delta)) + 0.04
                    dist = math.sqrt(dist_sq)
                    u_dir = delta / dist
                    prod = b1.magnetic_charge * b2.magnetic_charge
                    f_vec = u_dir * ((self.magnetic_k * prod) / dist_sq)
                    b1.velocity -= f_vec / b1.mass
                    b2.velocity += f_vec / b2.mass
                    self.magnetic_links.append((b1.position.copy(), b2.position.copy(), prod < 0))

        # 2. Movimiento, gravedad y muros de contención
        for b in bodies:
            if b.bound_to_hand and hand_pos_3d is not None:
                b.position += (hand_pos_3d - b.position) * 0.35
                b.velocity = hand_vel_3d * 0.8
            else:
                b.velocity[1] += self.gravity_g * 0.0024
                b.velocity *= 0.992
                b.position += b.velocity

            b.rotation += b.angular_vel
            r = b.collider_radius
            for axis in range(3):
                limit = self.bounds[axis]
                if b.position[axis] - r < -limit:
                    b.position[axis] = -limit + r
                    b.velocity[axis] = abs(b.velocity[axis]) * b.restitution
                elif b.position[axis] + r > limit:
                    b.position[axis] = limit - r
                    b.velocity[axis] = -abs(b.velocity[axis]) * b.restitution

        # 3. Colisiones elásticas entre hologramas
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = bodies[i], bodies[j]
                delta = b2.position - b1.position
                dist = float(np.linalg.norm(delta))
                min_d = b1.collider_radius + b2.collider_radius
                if dist < min_d and dist > 1e-6:
                    normal = delta / dist
                    overlap = min_d - dist
                    b1.position -= normal * (overlap * 0.5)
                    b2.position += normal * (overlap * 0.5)
                    rel_v = b2.velocity - b1.velocity
                    vn = float(np.dot(rel_v, normal))
                    if vn < 0:
                        e = min(b1.restitution, b2.restitution)
                        imp = normal * (-(1.0 + e) * vn * 0.5)
                        b1.velocity -= imp
                        b2.velocity += imp

        # 4. Interacción física con la mano
        if hand_pos_3d is not None:
            for b in bodies:
                delta = b.position - hand_pos_3d
                dist = float(np.linalg.norm(delta))
                if is_pinching and dist < (b.collider_radius + 0.25):
                    b.position = hand_pos_3d.copy()
                    b.velocity = hand_vel_3d * 0.85
                elif dist < (b.collider_radius + 0.22) and dist > 1e-6:
                    normal = delta / dist
                    b.position += normal * ((b.collider_radius + 0.22) - dist)
                    b.velocity += normal * 0.04 + hand_vel_3d * 0.5


class NXSWorkspaceInterpreter:
    def __init__(self, physics: ProjectPhysicsEngine):
        self.physics = physics
        self.scene: Dict[str, SerializableHologram3D] = {
            "cubo_central": SerializableHologram3D("cubo_central", "CUBO", (-0.55, 0.0, 0.0), "CIAN"),
            "nucleo_solar": SerializableHologram3D("nucleo_solar", "ESFERA", (0.55, 0.0, 0.0), "ORO"),
        }
        self.macros: Dict[str, List[str]] = {
            "MODO_GALAXIA": [
                "REALIDAD 80",
                "IMAN cubo_central POSITIVO",
                "IMAN nucleo_solar NEGATIVO",
                "CREAR p1 PIRAMIDE [0.0, -0.4, 0.0]",
                "COLOR p1 MAGENTA"
            ]
        }
        self.last_project_file = "proyecto_espacial.nxs"
        self.logs: List[str] = [
            "SERIALIZADOR DE PROYECTOS '.NXS' ACTIVO.",
            "Usa GUARDAR <nombre>, CARGAR <nombre> o teclas [F5] / [F9]."
        ]

    def log(self, msg: str):
        print(f"[NXS-PROJECT] {msg}")
        self.logs.append(msg)
        if len(self.logs) > 4:
            self.logs.pop(0)

    def _parse_vec3(self, text: str) -> Tuple[float, float, float]:
        clean = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        parts = [float(x) for x in clean.replace(",", " ").split() if x]
        return (parts[0], parts[1], parts[2]) if len(parts) >= 3 else (0.0, 0.0, 0.0)

    def execute(self, raw_line: str):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            return

        if line.upper().startswith("MACRO "):
            rest = line[6:].strip()
            if "=" in rest:
                m_name, m_body = rest.split("=", 1)
                m_name = m_name.strip().upper()
                self.macros[m_name] = [c.strip() for c in m_body.split(";") if c.strip()]
                self.log(f"Macro '{m_name}' registrada ({len(self.macros[m_name])} pasos).")
            return

        if ";" in line:
            for sub in line.split(";"):
                self.execute(sub)
            return

        tokens = line.split()
        cmd = tokens[0].upper()
        args = tokens[1:]

        try:
            if cmd in self.macros:
                self.log(f"Ejecutando Macro '{cmd}'...")
                for sub_cmd in self.macros[cmd]:
                    self.execute(sub_cmd)
                return

            # --- COMANDOS DE SERIALIZACIÓN DE PROYECTOS (.NXS) ---
            if cmd == "GUARDAR":
                fname = args[0] if args else self.last_project_file
                self.last_project_file = NXSProjectSerializer.ensure_extension(fname)
                msg = NXSProjectSerializer.save_project(
                    self.last_project_file, self.scene, self.macros,
                    self.physics.gravity_g, self.physics.vr_blend
                )
                self.log(msg)

            elif cmd == "CARGAR":
                fname = args[0] if args else self.last_project_file
                full_p = NXSProjectSerializer.ensure_extension(fname)
                self.last_project_file = full_p
                try:
                    new_sc, new_mac, g_val, vr_val, msg = NXSProjectSerializer.load_project(full_p)
                    self.scene = new_sc
                    self.macros.update(new_mac)
                    self.physics.gravity_g = g_val
                    self.physics.vr_blend = vr_val
                    self.log(msg)
                except ValueError as ve:
                    if str(ve) == "SCRIPT_MODE":
                        # Si el archivo .nxs es un script exportado, ejecutarlo línea por línea
                        with open(full_p, "r", encoding="utf-8") as sf:
                            for script_line in sf:
                                self.execute(script_line)
                        self.log(f"Script '{full_p}' ejecutado y reconstruido en vivo.")
                    else:
                        raise ve

            elif cmd == "EXPORTAR_SCRIPT":
                fname = args[0] if args else "escena_decompilada.nxs"
                msg = NXSProjectSerializer.export_executable_script(
                    fname, self.scene, self.macros,
                    self.physics.gravity_g, self.physics.vr_blend
                )
                self.log(msg)

            elif cmd == "LISTAR_PROYECTOS":
                files = glob.glob("*.nxs")
                self.log(f"Archivos .nxs en disco: {files if files else 'Ninguno aún'}")

            # --- COMANDOS DE CONSTRUCCIÓN ESPACIAL Y FÍSICAS ---
            elif cmd == "CREAR":
                name = args[0].lower()
                shape = args[1].upper() if len(args) > 1 else "CUBO"
                pos = self._parse_vec3(" ".join(args[2:])) if len(args) > 2 else (
                    round(random.uniform(-0.85, 0.85), 2), round(random.uniform(-0.4, 0.3), 2), 0.0
                )
                self.scene[name] = SerializableHologram3D(name, shape, pos)
                self.log(f"Creado '{name}' ({shape}) en {pos}")

            elif cmd == "COLOR":
                target = args[0].lower()
                cname = args[1].upper()
                if target == "todos":
                    for o in self.scene.values(): o.set_color(cname)
                elif target in self.scene:
                    self.scene[target].set_color(cname)

            elif cmd == "ESCALAR":
                target = args[0].lower()
                sc = float(np.clip(float(args[1]), 0.3, 2.5))
                if target == "todos":
                    for o in self.scene.values(): o.scale = sc
                elif target in self.scene:
                    self.scene[target].scale = sc

            elif cmd == "MOVER":
                target = args[0].lower()
                pos = self._parse_vec3(" ".join(args[1:]))
                if target in self.scene:
                    self.scene[target].position = np.array(pos, dtype=np.float32)

            elif cmd == "LANZAR":
                target = args[0].lower()
                vel = self._parse_vec3(" ".join(args[1:]))
                if target in self.scene:
                    self.scene[target].velocity = np.array(vel, dtype=np.float32)

            elif cmd == "REBOTE":
                target = args[0].lower()
                val = float(np.clip(float(args[1]), 0.1, 1.2))
                if target == "todos":
                    for o in self.scene.values(): o.restitution = val
                elif target in self.scene:
                    self.scene[target].restitution = val

            elif cmd == "IMAN":
                target = args[0].lower()
                pol = args[1].upper()
                charge = 1.0 if pol in ("POSITIVO", "+") else (-1.0 if pol in ("NEGATIVO", "-") else 0.0)
                if target == "todos":
                    for o in self.scene.values(): o.magnetic_charge = charge
                elif target in self.scene:
                    self.scene[target].magnetic_charge = charge

            elif cmd == "ANCLAR_MANO":
                target = args[0].lower()
                if target in self.scene:
                    self.scene[target].bound_to_hand = True

            elif cmd == "SOLTAR_MANO":
                for o in self.scene.values():
                    o.bound_to_hand = False

            elif cmd == "GRAVEDAD":
                self.physics.gravity_g = float(args[0])

            elif cmd == "REALIDAD":
                self.physics.vr_blend = float(np.clip(float(args[0]) / 100.0, 0.0, 1.0))

            elif cmd == "BORRAR":
                if args[0].lower() == "todos":
                    self.scene.clear()
                elif args[0].lower() in self.scene:
                    del self.scene[args[0].lower()]

        except Exception as e:
            self.log(f"Error: {e}")


# =====================================================================
# 5. APLICACIÓN VISUAL CON HUD DE PROYECTOS '.NXS'
# =====================================================================

class NexusProjectWorkspaceApp:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.physics = ProjectPhysicsEngine()
        self.workspace = NXSWorkspaceInterpreter(self.physics)
        self.cmd_queue: queue.Queue = queue.Queue()

        self.prev_hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.onscreen_input = ""
        self.running = True

    def _console_worker(self):
        while self.running:
            try:
                line = input("NexusProject (.nxs) >> ")
                if line.strip():
                    self.cmd_queue.put(line.strip())
            except EOFError:
                break

    def project_3d(self, pts_3d, w, h, focal=640.0, cam_dist=3.6):
        out = []
        for pt in pts_3d:
            z = max(cam_dist + pt[2], 0.35)
            out.append((int((pt[0] * focal) / z + w * 0.5), int((pt[1] * focal) / z + h * 0.5)))
        return out

    def render_workspace(self, frame):
        h, w, _ = frame.shape
        vr_blend = self.physics.vr_blend

        if vr_blend > 0.02:
            vr_bg = np.zeros_like(frame)
            horizon = int(h * 0.50)
            for i in range(-10, 11):
                cv2.line(vr_bg, (int(w*0.5 + i*115), h), (int(w*0.5 + i*18), horizon), (55, 30, 15), 1)
            for d in range(1, 9):
                y = int(horizon + (d**1.75) * 6.5)
                if y < h: cv2.line(vr_bg, (0, y), (w, y), (65, 35, 15), 1)
            frame = cv2.addWeighted(frame, 1.0 - vr_blend, vr_bg, vr_blend, 0)

        for p1, p2, attr in self.physics.magnetic_links:
            pts_m = self.project_3d([p1, p2], w, h)
            cv2.line(frame, pts_m[0], pts_m[1], (80, 255, 120) if attr else (70, 70, 255), 2, cv2.LINE_AA)

        overlay = frame.copy()
        for name, b in list(self.workspace.scene.items()):
            R = rotation_matrix_3d(*b.rotation)
            world_v = (b.vertices * b.scale) @ R.T + b.position
            pts_2d = self.project_3d(world_v, w, h)

            for face in b.faces:
                poly = np.array([pts_2d[i] for i in face], dtype=np.int32)
                c = b.color
                cv2.fillPoly(overlay, [poly], (c[0]//4, c[1]//4, c[2]//4))

            for e1, e2 in b.edges:
                cv2.line(frame, pts_2d[e1], pts_2d[e2], b.color, 2, cv2.LINE_AA)

            c2d = self.project_3d([b.position], w, h)[0]
            mag = " [+]" if b.magnetic_charge > 0 else (" [-]" if b.magnetic_charge < 0 else "")
            cv2.putText(frame, f"{b.name} ({b.shape}){mag}", (c2d[0] - 50, c2d[1] - int(48 * b.scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)
        return frame

    def draw_serializer_hud(self, frame):
        h, w, _ = frame.shape

        # Panel Izquierdo: Estado del Proyecto .NXS y Logs
        cv2.rectangle(frame, (15, 15), (675, 175), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (675, 175), (0, 230, 255), 1)
        cv2.putText(frame, f"PROYECTO ACTIVO: '{self.workspace.last_project_file}' | Objetos: {len(self.workspace.scene)}",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 160), 2)

        for i, line in enumerate(self.workspace.logs[-4:]):
            cv2.putText(frame, f"> {line}", (26, 65 + i * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.41, (215, 230, 245), 1)

        cv2.putText(frame, "Atajos: [1] Guardar .nxs  [2] Cargar .nxs  [3] Exportar Script .nxs  [4] Ejecutar MODO_GALAXIA",
                    (26, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 215, 80), 1)

        # Panel Derecho: Macros Aprendidas y Archivos .nxs en Disco
        nxs_files = glob.glob("*.nxs")[:4]
        cv2.rectangle(frame, (w - 430, 15), (w - 15, 175), (18, 16, 26), -1)
        cv2.rectangle(frame, (w - 430, 15), (w - 15, 175), (255, 150, 0), 1)
        cv2.putText(frame, f"MACROS ({len(self.workspace.macros)}) & ARCHIVOS .NXS EN DISCO",
                    (w - 416, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 80), 1)

        y_cursor = 64
        for m_name in list(self.workspace.macros.keys())[:2]:
            cv2.putText(frame, f"[MACRO] {m_name}", (w - 416, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 255, 150), 1)
            y_cursor += 20

        for f_name in nxs_files:
            cv2.putText(frame, f"[ARCHIVO] {f_name}", (w - 416, y_cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1)
            y_cursor += 20

        # Barra Inferior de Comandos
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (12, 14, 20), -1)
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (0, 255, 140), 1)
        cursor = "_" if int(time.time() * 2.5) % 2 == 0 else " "
        cv2.putText(frame, f"NXS-CMD >> {self.onscreen_input}{cursor}",
                    (28, h - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 140), 2)
        cv2.putText(frame, "Escribe: GUARDAR mi_mundo.nxs  |  BORRAR TODOS  |  CARGAR mi_mundo.nxs  |  EXPORTAR_SCRIPT codigo.nxs",
                    (28, h - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (175, 195, 215), 1)

    def run(self):
        threading.Thread(target=self._console_worker, daemon=True).start()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened() and self.running:
            success, raw = cap.read()
            if not success:
                break

            frame = cv2.flip(raw, 1)
            h, w, _ = frame.shape
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            while not self.cmd_queue.empty():
                self.workspace.execute(self.cmd_queue.get_nowait())

            hand_pos_3d = None
            hand_vel_3d = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            is_pinching = False

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                mid = (pts[4] + pts[8]) * 0.5
                is_pinching = float(np.linalg.norm(pts[8] - pts[4])) < 0.060

                hand_pos_3d = np.array([(mid[0] - 0.5) * 3.3, (mid[1] - 0.5) * 1.9, mid[2] * 1.2], dtype=np.float32)
                hand_vel_3d = hand_pos_3d - self.prev_hand_pos
                self.prev_hand_pos = hand_pos_3d.copy()

                cv2.circle(frame, (int(mid[0] * w), int(mid[1] * h)), 14,
                           (0, 255, 140) if is_pinching else (255, 200, 0), 2, cv2.LINE_AA)

            self.physics.step(self.workspace.scene, hand_pos_3d, hand_vel_3d, is_pinching)
            scene_frame = self.render_workspace(frame)
            self.draw_serializer_hud(scene_frame)

            cv2.imshow("NexusXR - Serializador de Mundos Holograficos (.nxs)", scene_frame)

            key = cv2.waitKey(1)
            if key == -1:
                continue
            elif key == 27:  # ESC
                self.running = False
                break
            elif key in (13, 10):  # ENTER
                if self.onscreen_input.strip():
                    self.cmd_queue.put(self.onscreen_input.strip())
                    self.onscreen_input = ""
            elif key in (8, 127):  # BACKSPACE
                self.onscreen_input = self.onscreen_input[:-1]
            # Atajos instantáneos cuando la barra de comandos está vacía
            elif not self.onscreen_input and (key & 0xFF) == ord('1'):
                self.cmd_queue.put("GUARDAR mundo_alfa.nxs")
            elif not self.onscreen_input and (key & 0xFF) == ord('2'):
                self.cmd_queue.put("CARGAR mundo_alfa.nxs")
            elif not self.onscreen_input and (key & 0xFF) == ord('3'):
                self.cmd_queue.put("EXPORTAR_SCRIPT script_editable.nxs")
            elif not self.onscreen_input and (key & 0xFF) == ord('4'):
                self.cmd_queue.put("MODO_GALAXIA")
            elif 32 <= (key & 0xFF) <= 126:
                self.onscreen_input += chr(key & 0xFF)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = NexusProjectWorkspaceApp()
    app.run()
