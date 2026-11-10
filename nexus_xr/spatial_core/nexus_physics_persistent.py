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
# 1. MATEMÁTICA 3D, PALETA Y ONDAS DE CHOQUE DE COLISIÓN
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


class CollisionShockwave:
    """Anillo visual expansivo generado en el punto 3D exacto de cada impacto."""
    def __init__(self, pos_3d: np.ndarray, color=(255, 255, 140)):
        self.pos_3d = pos_3d.copy()
        self.radius_px = 6
        self.life = 1.0
        self.color = color

    def update(self) -> bool:
        self.radius_px += 5
        self.life -= 0.12
        return self.life > 0.0


# =====================================================================
# 2. CUERPO RÍGIDO HOLOGRÁFICO 3D CON SERIALIZACIÓN '.NXS'
# =====================================================================

class RigidBodyHologram3D:
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
        self.velocity = np.array([
            random.uniform(-0.015, 0.015),
            random.uniform(-0.015, 0.015),
            random.uniform(-0.008, 0.008)
        ], dtype=np.float32)

        self.rotation = np.array([0.25, 0.45, 0.1], dtype=np.float32)
        self.angular_vel = np.array([0.01, 0.02, 0.006], dtype=np.float32)

        self.scale = 0.85
        self.base_radius = 0.36
        self.mass = 1.0
        self.restitution = 0.90  # Elasticidad de rebote (0.1 a 1.25)
        self.magnetic_charge = 0.0  # +1.0 (POSITIVO), -1.0 (NEGATIVO), 0.0 (NEUTRO)
        self.is_grabbed = False
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
    def from_nxs_dict(cls, data: Dict[str, Any]) -> "RigidBodyHologram3D":
        obj = cls(
            name=data["name"],
            shape=data.get("shape", "CUBO"),
            pos=tuple(data.get("position", [0.0, 0.0, 0.0])),
            color_name=data.get("color_name", "CIAN")
        )
        obj.velocity = np.array(data.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        obj.rotation = np.array(data.get("rotation", [0.25, 0.45, 0.1]), dtype=np.float32)
        obj.angular_vel = np.array(data.get("angular_vel", [0.01, 0.02, 0.006]), dtype=np.float32)
        obj.scale = float(data.get("scale", 0.85))
        obj.mass = float(data.get("mass", 1.0))
        obj.restitution = float(data.get("restitution", 0.90))
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
# 3. COLISIONADORES CINEMÁTICOS DE MANO Y MOTOR DE FÍSICAS 3D
# =====================================================================

class HandColliderNode:
    def __init__(self, pos_3d: np.ndarray, vel_3d: np.ndarray, radius: float = 0.22, is_pinching: bool = False):
        self.pos_3d = pos_3d
        self.vel_3d = vel_3d
        self.radius = radius
        self.is_pinching = is_pinching


class SpatialPhysicsSolver3D:
    def __init__(self):
        self.bounds_x = 1.65
        self.bounds_y = 0.95
        self.bounds_z = 0.85

        self.gravity_g = 0.0        # En Gs (1.0G = caída libre terrestre escalada)
        self.vr_blend = 0.25        # Mezcla AR/VR
        self.air_damping = 0.992
        self.magnetic_k = 0.0024
        self.hand_magnet_charge = 0.0

        self.shockwaves: List[CollisionShockwave] = []
        self.magnetic_links: List[Tuple[np.ndarray, np.ndarray, bool]] = []

    def step_simulation(self, bodies: List[RigidBodyHologram3D], hand_colliders: List[HandColliderNode]):
        self.magnetic_links.clear()
        n = len(bodies)

        # 1. Fuerzas Dipolares Magnéticas (Ley de Coulomb 3D entre hologramas)
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = bodies[i], bodies[j]
                if b1.magnetic_charge != 0.0 and b2.magnetic_charge != 0.0:
                    delta = b2.position - b1.position
                    dist_sq = float(np.dot(delta, delta)) + 0.04
                    dist = math.sqrt(dist_sq)
                    unit_dir = delta / dist

                    charge_prod = b1.magnetic_charge * b2.magnetic_charge
                    force_vec = unit_dir * ((self.magnetic_k * charge_prod) / dist_sq)

                    if not b1.is_grabbed:
                        b1.velocity -= force_vec / b1.mass
                    if not b2.is_grabbed:
                        b2.velocity += force_vec / b2.mass

                    self.magnetic_links.append((b1.position.copy(), b2.position.copy(), charge_prod < 0))

        # 2. Integración de Gravedad, Velocidad y Rebotes con las 6 Paredes del Cuarto 3D
        gravity_acc = self.gravity_g * 0.0022
        for b in bodies:
            if not b.is_grabbed:
                b.velocity[1] += gravity_acc
                b.velocity *= self.air_damping
                speed = float(np.linalg.norm(b.velocity))
                if speed > 0.24:
                    b.velocity = (b.velocity / speed) * 0.24
                b.position += b.velocity

            b.rotation += b.angular_vel
            b.angular_vel *= 0.985 + 0.01 * min(float(np.linalg.norm(b.velocity)) * 10.0, 1.0)

            r = b.collider_radius
            if b.position[0] - r < -self.bounds_x:
                b.position[0] = -self.bounds_x + r
                b.velocity[0] = abs(b.velocity[0]) * b.restitution
                b.angular_vel[1] += 0.04
            elif b.position[0] + r > self.bounds_x:
                b.position[0] = self.bounds_x - r
                b.velocity[0] = -abs(b.velocity[0]) * b.restitution
                b.angular_vel[1] -= 0.04

            if b.position[1] - r < -self.bounds_y:
                b.position[1] = -self.bounds_y + r
                b.velocity[1] = abs(b.velocity[1]) * b.restitution
            elif b.position[1] + r > self.bounds_y:
                b.position[1] = self.bounds_y - r
                b.velocity[1] = -abs(b.velocity[1]) * b.restitution
                b.velocity[0] *= 0.96

            if b.position[2] - r < -self.bounds_z:
                b.position[2] = -self.bounds_z + r
                b.velocity[2] = abs(b.velocity[2]) * b.restitution
            elif b.position[2] + r > self.bounds_z:
                b.position[2] = self.bounds_z - r
                b.velocity[2] = -abs(b.velocity[2]) * b.restitution

        # 3. Colisiones Elásticas 3D Holograma contra Holograma
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = bodies[i], bodies[j]
                delta = b2.position - b1.position
                dist = float(np.linalg.norm(delta))
                min_dist = b1.collider_radius + b2.collider_radius

                if dist < min_dist and dist > 1e-6:
                    normal = delta / dist
                    overlap = min_dist - dist

                    inv_m1 = 0.0 if b1.is_grabbed else (1.0 / b1.mass)
                    inv_m2 = 0.0 if b2.is_grabbed else (1.0 / b2.mass)
                    inv_sum = inv_m1 + inv_m2
                    if inv_sum > 0:
                        b1.position -= normal * (overlap * (inv_m1 / inv_sum))
                        b2.position += normal * (overlap * (inv_m2 / inv_sum))

                    rel_vel = b2.velocity - b1.velocity
                    vel_along_normal = float(np.dot(rel_vel, normal))

                    if vel_along_normal < 0:
                        e = min(b1.restitution, b2.restitution)
                        j_imp = -(1.0 + e) * vel_along_normal / max(inv_sum, 1e-5)
                        impulse_vec = normal * j_imp

                        if not b1.is_grabbed:
                            b1.velocity -= impulse_vec * inv_m1
                            b1.angular_vel += np.cross(normal, b1.velocity) * 0.6
                        if not b2.is_grabbed:
                            b2.velocity += impulse_vec * inv_m2
                            b2.angular_vel -= np.cross(normal, b2.velocity) * 0.6

                        contact_pt = b1.position + normal * b1.collider_radius
                        self.shockwaves.append(CollisionShockwave(contact_pt, (255, 255, 120)))

        # 4. Colisión Física, Golpeo y Agarre contra los Colisionadores de tus Manos
        for b in bodies:
            b.is_grabbed = False

        for hc in hand_colliders:
            for b in bodies:
                if b.bound_to_hand and hc.is_pinching:
                    b.is_grabbed = True
                    b.position = hc.pos_3d.copy()
                    b.velocity = hc.vel_3d * 0.85
                    continue

                delta = b.position - hc.pos_3d
                dist = float(np.linalg.norm(delta))
                combined_r = b.collider_radius + hc.radius

                # Imán en la mano
                if self.hand_magnet_charge != 0.0 and dist > 0.12:
                    u_dir = delta / dist
                    mag_f = (self.magnetic_k * 1.8 * self.hand_magnet_charge) / (dist * dist + 0.05)
                    b.velocity += u_dir * mag_f
                    self.magnetic_links.append((hc.pos_3d.copy(), b.position.copy(), self.hand_magnet_charge < 0))

                # Agarre directo por pellizco (Pinch)
                if hc.is_pinching and dist < (combined_r * 1.15):
                    b.is_grabbed = True
                    b.position = hc.pos_3d.copy()
                    b.velocity = hc.vel_3d * 0.85
                    continue

                # Golpe físico elástico con la palma o los dedos
                if dist < combined_r and dist > 1e-6:
                    normal = delta / dist
                    overlap = combined_r - dist
                    b.position += normal * overlap

                    rel_v = b.velocity - hc.vel_3d
                    v_norm = float(np.dot(rel_v, normal))
                    if v_norm < 0:
                        bounce = (1.0 + b.restitution) * (-v_norm)
                        b.velocity += normal * bounce + hc.vel_3d * 0.48
                        b.angular_vel += np.array([
                            random.uniform(-0.07, 0.07),
                            random.uniform(-0.07, 0.07),
                            random.uniform(-0.07, 0.07)
                        ], dtype=np.float32)
                        self.shockwaves.append(CollisionShockwave(b.position - normal * b.collider_radius, (0, 255, 160)))

        self.shockwaves = [sw for sw in self.shockwaves if sw.update()]


# =====================================================================
# 4. SERIALIZADOR CRIPTOGRÁFICO DE PROYECTOS '.NXS'
# =====================================================================

class NXSProjectSerializer:
    FORMAT_VERSION = "NEXUS_PHYSICS_PROJECT_V2"

    @staticmethod
    def ensure_ext(filename: str) -> str:
        clean = filename.strip().replace('"', "").replace("'", "")
        return clean if clean.lower().endswith(".nxs") else clean + ".nxs"

    @classmethod
    def save_project(
        cls,
        filepath: str,
        scene: Dict[str, RigidBodyHologram3D],
        macros: Dict[str, List[str]],
        solver: SpatialPhysicsSolver3D
    ) -> str:
        full_path = cls.ensure_ext(filepath)
        payload = {
            "environment": {
                "gravity_g": round(float(solver.gravity_g), 4),
                "vr_blend": round(float(solver.vr_blend), 4),
                "hand_magnet_charge": float(solver.hand_magnet_charge),
            },
            "learned_macros": macros,
            "holograms": [obj.to_nxs_dict() for obj in scene.values()],
        }

        canonical = json.dumps(payload, sort_keys=True)
        sha_short = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

        doc = {
            "header": {
                "magic": cls.FORMAT_VERSION,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "object_count": len(scene),
                "macro_count": len(macros),
                "sha256_short": sha_short,
            },
            "data": payload,
        }

        tmp = full_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        os.replace(tmp, full_path)
        return f"Guardado '{full_path}' ({len(scene)} cuerpos 3D, SHA:{sha_short[:8]})."

    @classmethod
    def load_project(
        cls,
        filepath: str,
        solver: SpatialPhysicsSolver3D
    ) -> Tuple[Dict[str, RigidBodyHologram3D], Dict[str, List[str]], str]:
        full_path = cls.ensure_ext(filepath)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No existe '{full_path}'")

        with open(full_path, "r", encoding="utf-8") as f:
            raw = f.read()

        if not raw.lstrip().startswith("{"):
            raise ValueError("SCRIPT_MODE")

        doc = json.loads(raw)
        header = doc.get("header", {})
        data = doc.get("data", {})

        expected_sha = header.get("sha256_short", "")
        actual_sha = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        if expected_sha and expected_sha != actual_sha:
            raise ValueError(f"Archivo '{full_path}' corrupto (SHA no coincide).")

        new_scene: Dict[str, RigidBodyHologram3D] = {}
        for h_dict in data.get("holograms", []):
            obj = RigidBodyHologram3D.from_nxs_dict(h_dict)
            new_scene[obj.name] = obj

        env = data.get("environment", {})
        solver.gravity_g = float(env.get("gravity_g", 0.0))
        solver.vr_blend = float(env.get("vr_blend", 0.25))
        solver.hand_magnet_charge = float(env.get("hand_magnet_charge", 0.0))

        macros = data.get("learned_macros", {})
        return new_scene, macros, f"Recargado '{full_path}' ({len(new_scene)} cuerpos, {len(macros)} macros)."

    @classmethod
    def export_script(
        cls,
        filepath: str,
        scene: Dict[str, RigidBodyHologram3D],
        macros: Dict[str, List[str]],
        solver: SpatialPhysicsSolver3D
    ) -> str:
        full_path = cls.ensure_ext(filepath)
        lines = [
            f"# SCRIPT NEXUS-LANG EXPORTADO ({datetime.now().isoformat(timespec='seconds')})",
            "BORRAR TODOS",
            f"REALIDAD {int(solver.vr_blend * 100)}",
            f"GRAVEDAD {solver.gravity_g:.2f}",
            ""
        ]
        for m_name, m_cmds in macros.items():
            lines.append(f"MACRO {m_name} = {' ; '.join(m_cmds)}")
        lines.append("")

        for obj in scene.values():
            px, py, pz = obj.position
            vx, vy, vz = obj.velocity
            lines.append(f"CREAR {obj.name} {obj.shape} [{px:.3f}, {py:.3f}, {pz:.3f}]")
            lines.append(f"COLOR {obj.name} {obj.color_name}")
            lines.append(f"ESCALAR {obj.name} {obj.scale:.2f}")
            lines.append(f"REBOTE {obj.name} {obj.restitution:.2f}")
            lines.append(f"MASA {obj.name} {obj.mass:.2f}")
            if obj.magnetic_charge != 0.0:
                pol = "POSITIVO" if obj.magnetic_charge > 0 else "NEGATIVO"
                lines.append(f"IMAN {obj.name} {pol}")
            if np.linalg.norm(obj.velocity) > 0.002:
                lines.append(f"LANZAR {obj.name} [{vx:.4f}, {vy:.4f}, {vz:.4f}]")
            lines.append("")

        with open(full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return f"Exportado script editable '{full_path}'."


# =====================================================================
# 5. INTÉRPRETE UNIFICADO (FÍSICAS + METAPROGRAMACIÓN + '.NXS')
# =====================================================================

class UnifiedPhysicsNXSInterpreter:
    def __init__(self, solver: SpatialPhysicsSolver3D):
        self.solver = solver
        self.scene: Dict[str, RigidBodyHologram3D] = {
            "cubo1": RigidBodyHologram3D("cubo1", "CUBO", (-0.65, -0.15, 0.0), "CIAN"),
            "esfera1": RigidBodyHologram3D("esfera1", "ESFERA", (0.0, -0.25, 0.0), "MAGENTA"),
            "piramide1": RigidBodyHologram3D("piramide1", "PIRAMIDE", (0.65, -0.15, 0.0), "ORO"),
        }
        self.macros: Dict[str, List[str]] = {
            "COLISIONADOR_CUANTICO": [
                "GRAVEDAD 0.0",
                "REBOTE TODOS 1.05",
                "IMAN cubo1 POSITIVO",
                "IMAN esfera1 NEGATIVO",
                "IMAN piramide1 POSITIVO"
            ]
        }
        self.active_project = "laboratorio_fisico.nxs"
        self.logs: List[str] = [
            "MOTOR 3D DE FISICAS, IMANES Y ARCHIVOS '.NXS' ACTIVO.",
            "Batea los hologramas con tus manos o usa los atajos [1-6]."
        ]

    def log(self, msg: str):
        print(f"[NEXUS-3D] {msg}")
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
                self.macros[m_name.strip().upper()] = [c.strip() for c in m_body.split(";") if c.strip()]
                self.log(f"Macro '{m_name.strip().upper()}' aprendida.")
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
                self.log(f"Disparando Macro '{cmd}'...")
                for sub_cmd in self.macros[cmd]:
                    self.execute(sub_cmd)
                return

            if cmd == "GUARDAR":
                fname = args[0] if args else self.active_project
                self.active_project = NXSProjectSerializer.ensure_ext(fname)
                self.log(NXSProjectSerializer.save_project(self.active_project, self.scene, self.macros, self.solver))

            elif cmd == "CARGAR":
                fname = args[0] if args else self.active_project
                self.active_project = NXSProjectSerializer.ensure_ext(fname)
                try:
                    new_sc, new_mac, msg = NXSProjectSerializer.load_project(self.active_project, self.solver)
                    self.scene = new_sc
                    self.macros.update(new_mac)
                    self.log(msg)
                except ValueError as ve:
                    if str(ve) == "SCRIPT_MODE":
                        with open(self.active_project, "r", encoding="utf-8") as sf:
                            for s_line in sf:
                                self.execute(s_line)
                        self.log(f"Script '{self.active_project}' reconstruido.")
                    else:
                        raise ve

            elif cmd == "EXPORTAR_SCRIPT":
                fname = args[0] if args else "escena_fisica_codigo.nxs"
                self.log(NXSProjectSerializer.export_script(fname, self.scene, self.macros, self.solver))

            elif cmd == "CREAR":
                name = args[0].lower()
                shape = args[1].upper() if len(args) > 1 else "CUBO"
                pos = self._parse_vec3(" ".join(args[2:])) if len(args) > 2 else (
                    round(random.uniform(-0.9, 0.9), 2), round(random.uniform(-0.45, 0.25), 2), 0.0
                )
                self.scene[name] = RigidBodyHologram3D(name, shape, pos)
                self.log(f"Creado '{name}' ({shape}) en {pos}")

            elif cmd == "GRAVEDAD":
                self.solver.gravity_g = float(args[0])
                self.log(f"Gravedad -> {self.solver.gravity_g:.1f}G")

            elif cmd == "REBOTE":
                target = args[0].lower()
                val = float(np.clip(float(args[1]), 0.1, 1.25))
                if target == "todos":
                    for b in self.scene.values(): b.restitution = val
                elif target in self.scene:
                    self.scene[target].restitution = val
                self.log(f"Rebote de '{target}' -> {val}")

            elif cmd == "MASA":
                target = args[0].lower()
                val = max(0.1, float(args[1]))
                if target in self.scene:
                    self.scene[target].mass = val
                    self.log(f"Masa de '{target}' -> {val} kg")

            elif cmd == "IMAN":
                target = args[0].lower()
                pol = args[1].upper()
                charge = 1.0 if pol in ("POSITIVO", "NORTE", "+") else (-1.0 if pol in ("NEGATIVO", "SUR", "-") else 0.0)
                if target == "mano":
                    self.solver.hand_magnet_charge = charge
                    self.log(f"Guante Magnético en Mano -> {pol}")
                elif target == "todos":
                    for b in self.scene.values(): b.magnetic_charge = charge
                    self.log(f"Todos los hologramas cargados -> {pol}")
                elif target in self.scene:
                    self.scene[target].magnetic_charge = charge
                    self.log(f"Imán '{target}' -> {pol} ({charge:+.0f})")

            elif cmd == "LANZAR":
                target = args[0].lower()
                vel = self._parse_vec3(" ".join(args[1:]))
                if target in self.scene:
                    self.scene[target].velocity = np.array(vel, dtype=np.float32)

            elif cmd == "EXPLOSION":
                for b in self.scene.values():
                    d = b.position + np.array([random.uniform(-0.15, 0.15), -0.22, random.uniform(-0.1, 0.1)], dtype=np.float32)
                    b.velocity = (d / (float(np.linalg.norm(d)) + 1e-5)) * 0.15
                self.log("¡Impulso radial explosivo aplicado!")

            elif cmd == "COLOR":
                target = args[0].lower()
                cname = args[1].upper()
                if target == "todos":
                    for b in self.scene.values(): b.set_color(cname)
                elif target in self.scene:
                    self.scene[target].set_color(cname)

            elif cmd == "ESCALAR":
                target = args[0].lower()
                sc = float(np.clip(float(args[1]), 0.35, 2.2))
                if target == "todos":
                    for b in self.scene.values(): b.scale = sc
                elif target in self.scene:
                    self.scene[target].scale = sc

            elif cmd == "REALIDAD":
                self.solver.vr_blend = float(np.clip(float(args[0]) / 100.0, 0.0, 1.0))

            elif cmd == "BORRAR":
                if args[0].lower() == "todos":
                    self.scene.clear()
                elif args[0].lower() in self.scene:
                    del self.scene[args[0].lower()]

        except Exception as e:
            self.log(f"Error: {e}")


# =====================================================================
# 6. MOTOR INTERACTIVO EN TIEMPO REAL (MEDIAPIPE + FÍSICAS + HUD '.NXS')
# =====================================================================

class NexusMasterPhysicsWorkspace:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.solver = SpatialPhysicsSolver3D()
        self.interpreter = UnifiedPhysicsNXSInterpreter(self.solver)
        self.cmd_queue: queue.Queue = queue.Queue()

        self.prev_hand_centers: Dict[int, np.ndarray] = {}
        self.onscreen_input = ""
        self.running = True

    def _console_worker(self):
        while self.running:
            try:
                line = input("NexusPhysics (.nxs) >> ")
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

    def extract_hand_colliders(self, multi_hand_landmarks, frame) -> List[HandColliderNode]:
        h, w, _ = frame.shape
        colliders: List[HandColliderNode] = []
        current_centers: Dict[int, np.ndarray] = {}

        if not multi_hand_landmarks:
            self.prev_hand_centers.clear()
            return colliders

        for idx, hand_lms in enumerate(multi_hand_landmarks):
            self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

            pinch_mid = (pts[4] + pts[8]) * 0.5
            is_pinching = float(np.linalg.norm(pts[8] - pts[4])) < 0.060

            finger_3d = np.array([(pinch_mid[0] - 0.5) * 3.3, (pinch_mid[1] - 0.5) * 1.9, pinch_mid[2] * 1.2], dtype=np.float32)
            palm_pt = pts[9]
            palm_3d = np.array([(palm_pt[0] - 0.5) * 3.3, (palm_pt[1] - 0.5) * 1.9, palm_pt[2] * 1.2], dtype=np.float32)

            prev_pos = self.prev_hand_centers.get(idx, finger_3d)
            hand_vel_3d = finger_3d - prev_pos
            current_centers[idx] = finger_3d

            colliders.append(HandColliderNode(finger_3d, hand_vel_3d, radius=0.22, is_pinching=is_pinching))
            if not is_pinching:
                colliders.append(HandColliderNode(palm_3d, hand_vel_3d, radius=0.26, is_pinching=False))

            fx, fy = int(pinch_mid[0] * w), int(pinch_mid[1] * h)
            cv2.circle(frame, (fx, fy), 18 if is_pinching else 26,
                       (0, 255, 140) if is_pinching else (255, 200, 0), 2, cv2.LINE_AA)

        self.prev_hand_centers = current_centers
        return colliders

    def render_world(self, frame):
        h, w, _ = frame.shape
        vr_blend = self.solver.vr_blend

        if vr_blend > 0.02:
            vr_bg = np.zeros_like(frame)
            horizon = int(h * 0.50)
            for i in range(-10, 11):
                cv2.line(vr_bg, (int(w*0.5 + i*115), h), (int(w*0.5 + i*18), horizon), (55, 30, 15), 1)
            for d in range(1, 9):
                y = int(horizon + (d**1.75) * 6.5)
                if y < h: cv2.line(vr_bg, (0, y), (w, y), (65, 35, 15), 1)
            frame = cv2.addWeighted(frame, 1.0 - vr_blend, vr_bg, vr_blend, 0)

        # Piso del recinto de contención 3D
        floor_3d = [
            [-self.solver.bounds_x, self.solver.bounds_y, -self.solver.bounds_z],
            [ self.solver.bounds_x, self.solver.bounds_y, -self.solver.bounds_z],
            [ self.solver.bounds_x, self.solver.bounds_y,  self.solver.bounds_z],
            [-self.solver.bounds_x, self.solver.bounds_y,  self.solver.bounds_z],
        ]
        cv2.polylines(frame, [np.array(self.project_3d(floor_3d, w, h), dtype=np.int32)], True, (90, 70, 40), 1, cv2.LINE_AA)

        # Líneas de fuerza magnética (Verde = Atracción, Rojo = Repulsión)
        for p1, p2, attracting in self.solver.magnetic_links:
            l_2d = self.project_3d([p1, p2], w, h)
            cv2.line(frame, l_2d[0], l_2d[1], (80, 255, 120) if attracting else (70, 70, 255), 2, cv2.LINE_AA)

        overlay = frame.copy()
        for name, b in list(self.interpreter.scene.items()):
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
            mag_tag = " [+]" if b.magnetic_charge > 0 else (" [-]" if b.magnetic_charge < 0 else "")
            cv2.putText(frame, f"{b.name}{mag_tag}", (c2d[0] - 38, c2d[1] - int(48 * b.scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

        # Ondas de choque en impactos
        for sw in self.solver.shockwaves:
            pt_2d = self.project_3d([sw.pos_3d], w, h)[0]
            cv2.circle(frame, pt_2d, sw.radius_px, sw.color, 2, cv2.LINE_AA)

        return frame

    def draw_hud(self, frame):
        h, w, _ = frame.shape
        cv2.rectangle(frame, (15, 15), (760, 170), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (760, 170), (0, 230, 255), 1)

        cv2.putText(frame, f"NEXUS 3D PHYSICS + NXS SERIALIZER | Cuerpos: {len(self.interpreter.scene)} | Gravedad: {self.solver.gravity_g:.1f}G",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 160), 2)

        for i, line in enumerate(self.interpreter.logs[-4:]):
            cv2.putText(frame, f"> {line}", (26, 64 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.41, (215, 230, 245), 1)

        cv2.putText(frame, "Teclas: [1] Gravedad  [2] Iman Atractivo  [3] Iman Repulsivo  [4] Explosion  [5] Guardar .nxs  [6] Cargar .nxs",
                    (26, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 215, 80), 1)

        # Barra Inferior de Código
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (12, 14, 20), -1)
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (0, 255, 140), 1)
        cursor = "_" if int(time.time() * 2.5) % 2 == 0 else " "
        cv2.putText(frame, f"NEXUS >> {self.onscreen_input}{cursor}",
                    (28, h - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 140), 2)
        cv2.putText(frame, "Comandos: CREAR e2 ESFERA  |  IMAN cubo1 POSITIVO ; IMAN esfera1 NEGATIVO  |  GUARDAR mundo.nxs  |  CARGAR mundo.nxs",
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
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            while not self.cmd_queue.empty():
                self.interpreter.execute(self.cmd_queue.get_nowait())

            hand_colliders = self.extract_hand_colliders(results.multi_hand_landmarks, frame)
            self.solver.step_simulation(list(self.interpreter.scene.values()), hand_colliders)

            scene_frame = self.render_world(frame)
            self.draw_hud(scene_frame)
            cv2.imshow("NexusXR - Fisicas 3D, Imanes y Proyectos (.nxs)", scene_frame)

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
            elif not self.onscreen_input and (key & 0xFF) == ord('1'):
                new_g = 0.0 if self.solver.gravity_g > 0 else 1.0
                self.cmd_queue.put(f"GRAVEDAD {new_g}")
            elif not self.onscreen_input and (key & 0xFF) == ord('2'):
                for idx, obj_name in enumerate(list(self.interpreter.scene.keys())):
                    pol = "POSITIVO" if idx % 2 == 0 else "NEGATIVO"
                    self.cmd_queue.put(f"IMAN {obj_name} {pol}")
            elif not self.onscreen_input and (key & 0xFF) == ord('3'):
                self.cmd_queue.put("IMAN TODOS POSITIVO")
            elif not self.onscreen_input and (key & 0xFF) == ord('4'):
                self.cmd_queue.put("EXPLOSION")
            elif not self.onscreen_input and (key & 0xFF) == ord('5'):
                self.cmd_queue.put("GUARDAR laboratorio_fisico.nxs")
            elif not self.onscreen_input and (key & 0xFF) == ord('6'):
                self.cmd_queue.put("CARGAR laboratorio_fisico.nxs")
            elif 32 <= (key & 0xFF) <= 126:
                self.onscreen_input += chr(key & 0xFF)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = NexusMasterPhysicsWorkspace()
    app.run()
