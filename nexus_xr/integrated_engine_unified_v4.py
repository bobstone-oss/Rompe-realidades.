import io
import os
import glob
import json
import time
import math
import base64
import random
import hashlib
import threading
import queue
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =====================================================================
# 1. GEOMETRÍA INVARIANTE DE LA MANO (GRAM-SCHMIDT 78D)
# =====================================================================

class CanonicalHandGeometry:
    FINGERTIPS = [4, 8, 12, 16, 20]

    @staticmethod
    def extract_invariant_features(hand_landmarks) -> np.ndarray:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        wrist = pts[0].copy()
        centered = pts - wrist
        palm_scale = np.linalg.norm(centered[9]) + 1e-6
        scaled = centered / palm_scale

        v_y = scaled[9].copy()
        y_axis = v_y / (np.linalg.norm(v_y) + 1e-6)
        v_across = scaled[5] - scaled[17]
        x_proj = v_across - np.dot(v_across, y_axis) * y_axis
        x_axis = x_proj / (np.linalg.norm(x_proj) + 1e-6)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= (np.linalg.norm(z_axis) + 1e-6)

        rotation_basis = np.stack([x_axis, y_axis, z_axis], axis=1)
        canonical_pts = scaled @ rotation_basis

        extra_features = []
        for i in range(len(CanonicalHandGeometry.FINGERTIPS)):
            tip_i = canonical_pts[CanonicalHandGeometry.FINGERTIPS[i]]
            extra_features.append(np.linalg.norm(tip_i))
            for j in range(i + 1, len(CanonicalHandGeometry.FINGERTIPS)):
                tip_j = canonical_pts[CanonicalHandGeometry.FINGERTIPS[j]]
                extra_features.append(np.linalg.norm(tip_i - tip_j))

        return np.concatenate([canonical_pts.flatten(), np.array(extra_features, dtype=np.float32)])


# =====================================================================
# 2. BÓVEDA NEURONAL PYTORCH SERIALIZABLE EN MEMORIA (BASE64)
# =====================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm2(self.fc2(F.gelu(self.norm1(self.fc1(x))))) + x)


class HypersphericalGestureNet(nn.Module):
    def __init__(self, input_dim: int = 78, hidden_dim: int = 256, embed_dim: int = 128):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.head = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        return F.normalize(self.head(self.res2(self.res1(h))), p=2, dim=-1)


class SerializableNeuralVault:
    """
    Bóveda de gestos en PyTorch capaz de empaquetar sus pesos neuronales y
    prototipos directamente dentro del JSON de un archivo '.nxs'.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = HypersphericalGestureNet().to(self.device)
        self.optimizer = optim.AdamW(self.encoder.parameters(), lr=1.5e-3)
        self.triplet_loss = nn.TripletMarginLoss(margin=0.45, p=2)

        self.gestures: Dict[str, Dict] = {}
        self.gesture_bindings: Dict[str, str] = {
            "GESTO_1": "MODO_ORBITAL",
            "GESTO_2": "EXPLOSION ; COLOR TODOS ORO",
            "GESTO_3": "ESCALAR_DELTA TODOS 0.03",
            "GESTO_4": "CREAR satelite ESFERA ; IMAN satelite NEGATIVO"
        }
        self.last_loss = 0.0
        self.training_steps = 0

    def encode(self, feature_vec: np.ndarray) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            t = torch.from_numpy(feature_vec).float().unsqueeze(0).to(self.device)
            return self.encoder(t).squeeze(0)

    def bind_command(self, gesture_name: str, command_str: str):
        self.gesture_bindings[gesture_name.upper()] = command_str.strip()

    def learn_sample(self, name: str, feature_vec: np.ndarray):
        name = name.upper()
        emb = self.encode(feature_vec)
        if name not in self.gestures:
            self.gestures[name] = {"prototype": emb.clone(), "samples": deque(maxlen=80), "count": 0}

        entry = self.gestures[name]
        entry["samples"].append(feature_vec.copy())
        entry["count"] += 1
        alpha = 0.25 if entry["count"] > 5 else (1.0 / entry["count"])
        entry["prototype"] = F.normalize((1.0 - alpha) * entry["prototype"] + alpha * emb, p=2, dim=-1)

        if len(self.gestures) >= 2:
            self._contrastive_step()

    def _contrastive_step(self, batch_size: int = 16):
        eligible = [k for k, v in self.gestures.items() if len(v["samples"]) >= 2]
        if len(eligible) < 2:
            return
        self.encoder.train()
        anchors, positives, negatives = [], [], []
        for _ in range(batch_size):
            p_cls, n_cls = random.sample(eligible, 2)
            a, p = random.sample(list(self.gestures[p_cls]["samples"]), 2)
            n = random.choice(list(self.gestures[n_cls]["samples"]))
            anchors.append(a); positives.append(p); negatives.append(n)

        loss = self.triplet_loss(
            self.encoder(torch.FloatTensor(np.array(anchors)).to(self.device)),
            self.encoder(torch.FloatTensor(np.array(positives)).to(self.device)),
            self.encoder(torch.FloatTensor(np.array(negatives)).to(self.device))
        )
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.last_loss = float(loss.item())
        self.training_steps += 1

        self.encoder.eval()
        with torch.no_grad():
            for name, data in self.gestures.items():
                if data["samples"]:
                    mat = torch.FloatTensor(np.array(list(data["samples"]))).to(self.device)
                    data["prototype"] = F.normalize(torch.mean(self.encoder(mat), dim=0), p=2, dim=-1)

    def recognize(self, feature_vec: np.ndarray, threshold: float = 0.78) -> Tuple[str, float, Dict[str, float]]:
        if not self.gestures:
            return "SIN_GESTOS", 0.0, {}
        emb = self.encode(feature_vec)
        scores = {name: float(torch.dot(emb, data["prototype"]).item()) for name, data in self.gestures.items()}
        best = max(scores, key=scores.get)
        score = scores[best]
        return (best, score, scores) if score >= threshold else ("DESCONOCIDO", score, scores)

    def export_to_nxs_dict(self) -> Dict[str, Any]:
        """Serializa los pesos de PyTorch a un buffer Base64 y los prototipos a listas JSON."""
        buffer = io.BytesIO()
        torch.save(self.encoder.state_dict(), buffer)
        weights_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        prototypes_dict = {}
        for g_name, g_data in self.gestures.items():
            # Guardamos hasta 25 muestras representativas por gesto para mantener el archivo .nxs ligero
            recent_samples = [s.tolist() for s in list(g_data["samples"])[-25:]]
            prototypes_dict[g_name] = {
                "prototype_128d": [round(float(x), 6) for x in g_data["prototype"].detach().cpu().numpy()],
                "count": int(g_data["count"]),
                "samples_78d": recent_samples,
            }

        return {
            "training_steps": self.training_steps,
            "gesture_bindings": self.gesture_bindings,
            "learned_gestures": prototypes_dict,
            "encoder_weights_b64": weights_b64,
        }

    def import_from_nxs_dict(self, ai_data: Dict[str, Any]):
        """Restaura pesos neuronales, prototipos y vínculos de comandos desde un archivo '.nxs'."""
        if not ai_data:
            return
        self.training_steps = int(ai_data.get("training_steps", 0))
        if "gesture_bindings" in ai_data:
            self.gesture_bindings = dict(ai_data["gesture_bindings"])

        weights_b64 = ai_data.get("encoder_weights_b64", "")
        if weights_b64:
            raw_bytes = base64.b64decode(weights_b64.encode("ascii"))
            buffer = io.BytesIO(raw_bytes)
            state_dict = torch.load(buffer, map_location=self.device, weights_only=False)
            self.encoder.load_state_dict(state_dict)

        self.gestures.clear()
        for g_name, g_info in ai_data.get("learned_gestures", {}).items():
            proto_tensor = torch.tensor(g_info["prototype_128d"], dtype=torch.float32, device=self.device)
            samples_list = [np.array(s, dtype=np.float32) for s in g_info.get("samples_78d", [])]
            self.gestures[g_name] = {
                "prototype": F.normalize(proto_tensor, p=2, dim=-1),
                "samples": deque(samples_list, maxlen=80),
                "count": int(g_info.get("count", len(samples_list))),
            }


# =====================================================================
# 3. CUERPOS RÍGIDOS 3D Y MOTOR DE FÍSICAS ESPACIALES
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


class RigidBodyHologram3D:
    def __init__(self, name: str, shape: str = "CUBO", pos=(0.0, 0.0, 0.0), color_name: str = "CIAN"):
        self.name = name
        self.shape = shape.upper()
        self.color_name = color_name.upper() if color_name.upper() in COLOR_PALETTE else "CIAN"
        self.color = COLOR_PALETTE[self.color_name]

        self.position = np.array(pos, dtype=np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.25, 0.45, 0.1], dtype=np.float32)
        self.angular_vel = np.array([0.01, 0.02, 0.006], dtype=np.float32)

        self.scale = 0.85
        self.base_radius = 0.36
        self.mass = 1.0
        self.restitution = 0.90
        self.magnetic_charge = 0.0
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
            return verts, [(0,1), (1,2), (2,3), (3,0), (0,4), (1,4), (2,4), (3,4)], [(0,1,4), (1,2,4), (2,3,4), (3,0,4)]
        elif shape in ("OCTAEDRO", "DIAMANTE"):
            verts = np.array([
                [0.5, 0, 0], [-0.5, 0, 0], [0, 0.58, 0],
                [0, -0.58, 0], [0, 0, 0.5], [0, 0, -0.5]
            ], dtype=np.float32)
            return verts, [(0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,4), (4,3), (3,5), (5,2)], [(0,2,4), (0,4,3), (1,2,4), (1,4,3)]
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
            return verts, [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)], [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6)]


class SpatialPhysicsSolver3D:
    def __init__(self):
        self.bounds = np.array([1.65, 0.95, 0.85], dtype=np.float32)
        self.gravity_g = 0.0
        self.vr_blend = 0.25
        self.magnetic_k = 0.0024
        self.magnetic_links: List[Tuple[np.ndarray, np.ndarray, bool]] = []

    def step(self, bodies: List[RigidBodyHologram3D], hand_pos_3d: Optional[np.ndarray], hand_vel_3d: np.ndarray, is_pinching: bool):
        self.magnetic_links.clear()
        n = len(bodies)

        # 1. Fuerzas dipolares magnéticas
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

        # 2. Integración de movimiento y rebote con paredes
        for b in bodies:
            if b.bound_to_hand and hand_pos_3d is not None:
                b.position += (hand_pos_3d - b.position) * 0.35
                b.velocity = hand_vel_3d * 0.8
            else:
                b.velocity[1] += self.gravity_g * 0.0022
                b.velocity *= 0.992
                b.position += b.velocity

            b.rotation += b.angular_vel
            r = b.collider_radius
            for axis in range(3):
                lim = self.bounds[axis]
                if b.position[axis] - r < -lim:
                    b.position[axis] = -lim + r
                    b.velocity[axis] = abs(b.velocity[axis]) * b.restitution
                elif b.position[axis] + r > lim:
                    b.position[axis] = lim - r
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


# =====================================================================
# 4. MOTOR MAESTRO DE SERIALIZACIÓN '.NXS' (ESCENA + MACROS + IA PYTORCH)
# =====================================================================

class NXSCompleteSerializer:
    FORMAT_MAGIC = "NEXUS_UNIFIED_ARCHIVE_V4"

    @staticmethod
    def ensure_ext(filename: str) -> str:
        clean = filename.strip().replace('"', "").replace("'", "")
        return clean if clean.lower().endswith(".nxs") else clean + ".nxs"

    @classmethod
    def save_nxs_archive(
        cls,
        filepath: str,
        scene: Dict[str, RigidBodyHologram3D],
        macros: Dict[str, List[str]],
        vault: SerializableNeuralVault,
        solver: SpatialPhysicsSolver3D
    ) -> str:
        full_path = cls.ensure_ext(filepath)

        payload = {
            "environment": {
                "gravity_g": round(float(solver.gravity_g), 4),
                "vr_blend": round(float(solver.vr_blend), 4),
            },
            "learned_macros": macros,
            "holograms": [obj.to_nxs_dict() for obj in scene.values()],
            "neural_vault": vault.export_to_nxs_dict(),
        }

        canonical = json.dumps(payload, sort_keys=True)
        sha256_hex = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

        archive = {
            "header": {
                "magic": cls.FORMAT_MAGIC,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "hologram_count": len(scene),
                "macro_count": len(macros),
                "trained_gestures_count": len(vault.gestures),
                "sha256_integrity": sha256_hex,
            },
            "data": payload,
        }

        tmp_path = full_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(archive, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, full_path)

        return (f"Guardado '{full_path}' ({len(scene)} hologramas, "
                f"{len(macros)} macros, {len(vault.gestures)} gestos IA, SHA:{sha256_hex[:8]})")

    @classmethod
    def load_nxs_archive(
        cls,
        filepath: str,
        solver: SpatialPhysicsSolver3D,
        vault: SerializableNeuralVault
    ) -> Tuple[Dict[str, RigidBodyHologram3D], Dict[str, List[str]], str]:
        full_path = cls.ensure_ext(filepath)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No existe el archivo '{full_path}'")

        with open(full_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # Si no comienza con '{', es un script de código NexusLang exportado
        if not raw_content.lstrip().startswith("{"):
            raise ValueError("SCRIPT_FORMAT")

        archive = json.loads(raw_content)
        header = archive.get("header", {})
        data = archive.get("data", {})

        expected_sha = header.get("sha256_integrity", "")
        actual_sha = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        if expected_sha and expected_sha != actual_sha:
            raise ValueError(f"Alerta de integridad: el archivo '{full_path}' fue alterado.")

        new_scene: Dict[str, RigidBodyHologram3D] = {}
        for h_dict in data.get("holograms", []):
            obj = RigidBodyHologram3D.from_nxs_dict(h_dict)
            new_scene[obj.name] = obj

        env = data.get("environment", {})
        solver.gravity_g = float(env.get("gravity_g", 0.0))
        solver.vr_blend = float(env.get("vr_blend", 0.25))

        macros = data.get("learned_macros", {})
        if "neural_vault" in data:
            vault.import_from_nxs_dict(data["neural_vault"])

        msg = (f"Cargado '{full_path}' ({len(new_scene)} hologramas, "
               f"{len(macros)} macros, {len(vault.gestures)} gestos IA).")
        return new_scene, macros, msg

    @classmethod
    def export_readable_script(
        cls,
        filepath: str,
        scene: Dict[str, RigidBodyHologram3D],
        macros: Dict[str, List[str]],
        vault: SerializableNeuralVault,
        solver: SpatialPhysicsSolver3D
    ) -> str:
        full_path = cls.ensure_ext(filepath)
        lines = [
            f"# ===================================================================",
            f"# SCRIPT EJECUTABLE NEXUS-LANG EXPORTADO ({datetime.now().isoformat(timespec='seconds')})",
            f"# ===================================================================",
            "BORRAR TODOS",
            f"REALIDAD {int(solver.vr_blend * 100)}",
            f"GRAVEDAD {solver.gravity_g:.2f}",
            ""
        ]

        if macros:
            lines.append("# --- 1. MACROS DE METAPROGRAMACION ---")
            for m_name, m_cmds in macros.items():
                lines.append(f"MACRO {m_name} = {' ; '.join(m_cmds)}")
            lines.append("")

        if vault.gesture_bindings:
            lines.append("# --- 2. VINCULOS NEURONALES DE GESTOS (PYTORCH -> NEXUS-LANG) ---")
            for g_name, g_cmd in vault.gesture_bindings.items():
                lines.append(f"VINCULAR_GESTO {g_name} = {g_cmd}")
            lines.append("")

        lines.append("# --- 3. POSICIONES Y ESTADO FISICO DE LOS HOLOGRAMAS 3D ---")
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

        return f"Decompilado a código fuente en '{full_path}'."


# =====================================================================
# 5. INTÉRPRETE DE NEXUS-LANG CON GESTIÓN DE PROYECTOS '.NXS'
# =====================================================================

class NXSCompleteInterpreter:
    def __init__(self, solver: SpatialPhysicsSolver3D, vault: SerializableNeuralVault, learn_cb):
        self.solver = solver
        self.vault = vault
        self.learn_cb = learn_cb

        self.scene: Dict[str, RigidBodyHologram3D] = {
            "cubo1": RigidBodyHologram3D("cubo1", "CUBO", (-0.6, -0.1, 0.0), "CIAN"),
            "esfera1": RigidBodyHologram3D("esfera1", "ESFERA", (0.0, -0.2, 0.0), "MAGENTA"),
            "piramide1": RigidBodyHologram3D("piramide1", "PIRAMIDE", (0.6, -0.1, 0.0), "ORO"),
        }
        self.macros: Dict[str, List[str]] = {
            "MODO_ORBITAL": [
                "GRAVEDAD 0.0",
                "IMAN cubo1 POSITIVO",
                "IMAN esfera1 NEGATIVO",
                "IMAN piramide1 POSITIVO",
                "REALIDAD 75"
            ]
        }
        self.current_project = "proyecto_maestro.nxs"
        self.logs: List[str] = [
            "SISTEMA DE SERIALIZACION '.NXS' V4 ACTIVO.",
            "Guarda escena 3D, fisicas, macros y pesos PyTorch en un solo archivo."
        ]

    def log(self, msg: str):
        print(f"[NXS-V4] {msg}")
        self.logs.append(msg)
        if len(self.logs) > 4:
            self.logs.pop(0)

    def _parse_vec3(self, text: str) -> Tuple[float, float, float]:
        clean = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        parts = [float(x) for x in clean.replace(",", " ").split() if x]
        return (parts[0], parts[1], parts[2]) if len(parts) >= 3 else (0.0, 0.0, 0.0)

    def execute(self, raw_line: str, from_gesture: bool = False):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            return

        upper = line.upper()
        if upper.startswith("APRENDER_GESTO "):
            rest = line[15:].strip()
            if "=" in rest:
                g_name, cmd_body = rest.split("=", 1)
                g_name = g_name.strip().upper()
                self.vault.bind_command(g_name, cmd_body.strip())
                self.learn_cb(g_name)
                self.log(f"Grabando gesto '{g_name}' -> {cmd_body.strip()}")
            return

        if upper.startswith("VINCULAR_GESTO "):
            rest = line[15:].strip()
            if "=" in rest:
                g_name, cmd_body = rest.split("=", 1)
                self.vault.bind_command(g_name.strip().upper(), cmd_body.strip())
                self.log(f"Vínculo actualizado: {g_name.strip().upper()} -> {cmd_body.strip()}")
            return

        if upper.startswith("MACRO "):
            rest = line[6:].strip()
            if "=" in rest:
                m_name, m_body = rest.split("=", 1)
                self.macros[m_name.strip().upper()] = [c.strip() for c in m_body.split(";") if c.strip()]
                self.log(f"Macro '{m_name.strip().upper()}' registrada.")
            return

        if ";" in line:
            for sub in line.split(";"):
                self.execute(sub, from_gesture=from_gesture)
            return

        tokens = line.split()
        cmd = tokens[0].upper()
        args = tokens[1:]

        try:
            if cmd in self.macros:
                if not from_gesture:
                    self.log(f"Ejecutando Macro '{cmd}'...")
                for sub_cmd in self.macros[cmd]:
                    self.execute(sub_cmd, from_gesture=from_gesture)
                return

            # --- COMANDOS DE ARCHIVOS DE PROYECTO '.NXS' ---
            if cmd == "GUARDAR":
                fname = args[0] if args else self.current_project
                self.current_project = NXSCompleteSerializer.ensure_ext(fname)
                self.log(NXSCompleteSerializer.save_nxs_archive(
                    self.current_project, self.scene, self.macros, self.vault, self.solver
                ))

            elif cmd == "SNAPSHOT":
                snap_name = f"snapshot_{datetime.now().strftime('%H%M%S')}.nxs"
                self.log(NXSCompleteSerializer.save_nxs_archive(
                    snap_name, self.scene, self.macros, self.vault, self.solver
                ))

            elif cmd == "CARGAR":
                fname = args[0] if args else self.current_project
                self.current_project = NXSCompleteSerializer.ensure_ext(fname)
                try:
                    new_sc, new_mac, msg = NXSCompleteSerializer.load_nxs_archive(
                        self.current_project, self.solver, self.vault
                    )
                    self.scene = new_sc
                    self.macros.update(new_mac)
                    self.log(msg)
                except ValueError as ve:
                    if str(ve) == "SCRIPT_FORMAT":
                        with open(self.current_project, "r", encoding="utf-8") as sf:
                            for s_line in sf:
                                self.execute(s_line)
                        self.log(f"Script '{self.current_project}' ejecutado y reconstruido.")
                    else:
                        raise ve

            elif cmd == "EXPORTAR_SCRIPT":
                fname = args[0] if args else "escena_codigo.nxs"
                self.log(NXSCompleteSerializer.export_readable_script(
                    fname, self.scene, self.macros, self.vault, self.solver
                ))

            # --- COMANDOS DE ESCENA 3D Y FÍSICAS ---
            elif cmd == "CREAR":
                name = args[0].lower()
                if from_gesture and name in self.scene:
                    name = f"{name}_{len(self.scene)+1}"
                shape = args[1].upper() if len(args) > 1 else "CUBO"
                pos = self._parse_vec3(" ".join(args[2:])) if len(args) > 2 else (
                    round(random.uniform(-0.85, 0.85), 2), round(random.uniform(-0.4, 0.25), 2), 0.0
                )
                self.scene[name] = RigidBodyHologram3D(name, shape, pos)
                self.log(f"Creado '{name}' ({shape}) en {pos}")

            elif cmd == "COLOR":
                target = args[0].lower()
                cname = args[1].upper()
                if target == "todos":
                    for b in self.scene.values(): b.set_color(cname)
                elif target in self.scene:
                    self.scene[target].set_color(cname)

            elif cmd == "ESCALAR":
                target = args[0].lower()
                sc = float(np.clip(float(args[1]), 0.35, 2.4))
                if target == "todos":
                    for b in self.scene.values(): b.scale = sc
                elif target in self.scene:
                    self.scene[target].scale = sc

            elif cmd == "ESCALAR_DELTA":
                target = args[0].lower()
                delta = float(args[1])
                if target == "todos":
                    for b in self.scene.values():
                        b.scale = float(np.clip(b.scale + delta, 0.35, 2.4))

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
                val = float(np.clip(float(args[1]), 0.1, 1.25))
                if target == "todos":
                    for b in self.scene.values(): b.restitution = val
                elif target in self.scene:
                    self.scene[target].restitution = val

            elif cmd == "IMAN":
                target = args[0].lower()
                pol = args[1].upper()
                charge = 1.0 if pol in ("POSITIVO", "+") else (-1.0 if pol in ("NEGATIVO", "-") else 0.0)
                if target == "todos":
                    for b in self.scene.values(): b.magnetic_charge = charge
                elif target in self.scene:
                    self.scene[target].magnetic_charge = charge

            elif cmd == "EXPLOSION":
                for b in self.scene.values():
                    d = b.position + np.array([random.uniform(-0.15, 0.15), -0.22, random.uniform(-0.1, 0.1)], dtype=np.float32)
                    b.velocity = (d / (float(np.linalg.norm(d)) + 1e-5)) * 0.15
                self.log("¡Impulso explosivo aplicado!")

            elif cmd == "ANCLAR_MANO":
                if args[0].lower() in self.scene:
                    self.scene[args[0].lower()].bound_to_hand = True

            elif cmd == "SOLTAR_MANO":
                for b in self.scene.values():
                    b.bound_to_hand = False

            elif cmd == "GRAVEDAD":
                self.solver.gravity_g = float(args[0])

            elif cmd == "REALIDAD":
                self.solver.vr_blend = float(np.clip(float(args[0]) / 100.0, 0.0, 1.0))

            elif cmd == "BORRAR":
                if args[0].lower() == "todos":
                    self.scene.clear()
                elif args[0].lower() in self.scene:
                    del self.scene[args[0].lower()]

        except Exception as e:
            if not from_gesture:
                self.log(f"Error: {e}")


# =====================================================================
# 6. APLICACIÓN PRINCIPAL EN TIEMPO REAL
# =====================================================================

class NexusCompleteProjectEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.solver = SpatialPhysicsSolver3D()
        self.vault = SerializableNeuralVault()
        self.interpreter = NXSCompleteInterpreter(self.solver, self.vault, self.start_gesture_recording)
        self.cmd_queue: queue.Queue = queue.Queue()

        self.recording_gesture: Optional[str] = None
        self.recording_frames = 0
        self.last_discrete_gesture = ""
        self.last_trigger_time = 0.0

        self.prev_hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.onscreen_input = ""
        self.running = True

    def start_gesture_recording(self, gesture_name: str, frames: int = 25):
        self.recording_gesture = gesture_name.upper()
        self.recording_frames = frames

    def _console_worker(self):
        while self.running:
            try:
                line = input("NexusProject (.nxs) >> ")
                if line.strip():
                    self.cmd_queue.put(line.strip())
            except EOFError:
                break

    def dispatch_gesture(self, gesture_name: str):
        bound_cmd = self.vault.gesture_bindings.get(gesture_name)
        if not bound_cmd:
            return
        if "ESCALAR_DELTA" in bound_cmd.upper():
            self.interpreter.execute(bound_cmd, from_gesture=True)
        else:
            now = time.time()
            if gesture_name != self.last_discrete_gesture or (now - self.last_trigger_time) > 1.5:
                self.interpreter.log(f"Gesto [{gesture_name}] -> '{bound_cmd}'")
                self.interpreter.execute(bound_cmd, from_gesture=True)
                self.last_discrete_gesture = gesture_name
                self.last_trigger_time = now

    def project_3d(self, pts_3d, w, h, focal=640.0, cam_dist=3.6):
        out = []
        for pt in pts_3d:
            z = max(cam_dist + pt[2], 0.35)
            out.append((int((pt[0] * focal) / z + w * 0.5), int((pt[1] * focal) / z + h * 0.5)))
        return out

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
            cv2.putText(frame, f"{b.name}{mag_tag}", (c2d[0] - 35, c2d[1] - int(48 * b.scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)
        return frame

    def draw_hud(self, frame, predicted: str, conf: float):
        h, w, _ = frame.shape

        # Panel Izquierdo: Estado del Archivo .NXS y Consola
        cv2.rectangle(frame, (15, 15), (675, 170), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (675, 170), (0, 230, 255), 1)
        cv2.putText(frame, f"ARCHIVO .NXS: '{self.interpreter.current_project}' | Gesto IA: {predicted} ({conf*100:.0f}%)",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 160), 2)

        for i, line in enumerate(self.interpreter.logs[-4:]):
            cv2.putText(frame, f"> {line}", (26, 64 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (215, 230, 245), 1)

        cv2.putText(frame, "Teclas: [1-4] Entrenar Gesto  [5] GUARDAR .nxs  [6] CARGAR .nxs  [7] EXPORTAR_SCRIPT  [8] SNAPSHOT",
                    (26, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 215, 80), 1)

        # Panel Derecho: Archivos .nxs en Disco y Macros/Gestos
        nxs_files = glob.glob("*.nxs")[:4]
        cv2.rectangle(frame, (w - 450, 15), (w - 15, 170), (18, 16, 26), -1)
        cv2.rectangle(frame, (w - 450, 15), (w - 15, 170), (255, 150, 0), 1)
        cv2.putText(frame, f"PROYECTOS .NXS EN DISCO ({len(nxs_files)}) | MACROS ({len(self.interpreter.macros)})",
                    (w - 436, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 80), 1)

        y_pos = 64
        for f_name in nxs_files:
            cv2.putText(frame, f"[.NXS] {f_name}", (w - 436, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 230, 255), 1)
            y_pos += 20
        for m_name in list(self.interpreter.macros.keys())[:2]:
            cv2.putText(frame, f"[MACRO] {m_name}", (w - 436, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 255, 150), 1)
            y_pos += 20

        # Barra Inferior de Comandos
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (12, 14, 20), -1)
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (0, 255, 140), 1)
        cursor = "_" if int(time.time() * 2.5) % 2 == 0 else " "
        cv2.putText(frame, f"NXS-CMD >> {self.onscreen_input}{cursor}",
                    (28, h - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 140), 2)
        cv2.putText(frame, "Comandos: GUARDAR mundo1.nxs  |  BORRAR TODOS  |  CARGAR mundo1.nxs  |  EXPORTAR_SCRIPT codigo.nxs",
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
                self.interpreter.execute(self.cmd_queue.get_nowait())

            hand_pos_3d = None
            hand_vel_3d = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            is_pinching = False
            predicted = "NINGUNO"
            conf = 0.0

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                inv_vec = CanonicalHandGeometry.extract_invariant_features(hand_lms)
                if self.recording_frames > 0 and self.recording_gesture:
                    self.vault.learn_sample(self.recording_gesture, inv_vec)
                    self.recording_frames -= 1
                    if self.recording_frames == 0:
                        self.interpreter.log(f"Gesto '{self.recording_gesture}' listo para guardarse en .nxs")

                predicted, conf, _ = self.vault.recognize(inv_vec)
                if predicted not in ("DESCONOCIDO", "SIN_GESTOS") and self.recording_frames == 0:
                    self.dispatch_gesture(predicted)
                else:
                    self.last_discrete_gesture = ""

                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                mid = (pts[4] + pts[8]) * 0.5
                is_pinching = float(np.linalg.norm(pts[8] - pts[4])) < 0.060
                hand_pos_3d = np.array([(mid[0] - 0.5) * 3.3, (mid[1] - 0.5) * 1.9, mid[2] * 1.2], dtype=np.float32)
                hand_vel_3d = hand_pos_3d - self.prev_hand_pos
                self.prev_hand_pos = hand_pos_3d.copy()

                cv2.circle(frame, (int(mid[0] * w), int(mid[1] * h)), 16,
                           (0, 255, 140) if is_pinching else (255, 200, 0), 2, cv2.LINE_AA)

            self.solver.step(list(self.interpreter.scene.values()), hand_pos_3d, hand_vel_3d, is_pinching)
            scene_frame = self.render_world(frame)
            self.draw_hud(scene_frame, predicted, conf)
            cv2.imshow("NexusXR - Serializador Total de Proyectos (.nxs)", scene_frame)

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
            elif not self.onscreen_input and (key & 0xFF) in (ord('1'), ord('2'), ord('3'), ord('4')):
                self.start_gesture_recording(f"GESTO_{chr(key & 0xFF)}")
            elif not self.onscreen_input and (key & 0xFF) == ord('5'):
                self.cmd_queue.put("GUARDAR proyecto_maestro.nxs")
            elif not self.onscreen_input and (key & 0xFF) == ord('6'):
                self.cmd_queue.put("CARGAR proyecto_maestro.nxs")
            elif not self.onscreen_input and (key & 0xFF) == ord('7'):
                self.cmd_queue.put("EXPORTAR_SCRIPT escena_editable.nxs")
            elif not self.onscreen_input and (key & 0xFF) == ord('8'):
                self.cmd_queue.put("SNAPSHOT")
            elif 32 <= (key & 0xFF) <= 126:
                self.onscreen_input += chr(key & 0xFF)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = NexusCompleteProjectEngine()
    app.run()
