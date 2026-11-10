import os
import glob
import json
import time
import math
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
# 2. RED NEURONAL HIPERSFÉRICA EN PYTORCH + BÓVEDA DE COMANDOS
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


class NeuralGestureVault:
    def __init__(self, vault_path: str = "nexus_supreme_vault.pt"):
        self.vault_path = vault_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = HypersphericalGestureNet().to(self.device)
        self.optimizer = optim.AdamW(self.encoder.parameters(), lr=1.5e-3)
        self.triplet_loss = nn.TripletMarginLoss(margin=0.45, p=2)

        self.gestures: Dict[str, Dict] = {}
        # Mapa vivo que une cada Gesto de PyTorch con Comandos o MACROS de NexusLang
        self.gesture_bindings: Dict[str, str] = {
            "GESTO_1": "COLISIONADOR_CUANTICO",
            "GESTO_2": "EXPLOSION ; COLOR TODOS ORO",
            "GESTO_3": "ESCALAR_DELTA TODOS 0.03",
            "GESTO_4": "CREAR satelite ESFERA ; IMAN satelite NEGATIVO"
        }
        self.last_loss = 0.0
        self.training_steps = 0
        self.load_vault()

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
            self.gestures[name] = {"prototype": emb.clone(), "samples": deque(maxlen=100), "count": 0}

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

    def save_vault(self):
        payload = {
            "model": self.encoder.state_dict(),
            "bindings": self.gesture_bindings,
            "steps": self.training_steps,
            "gestures": {
                k: {"proto": v["prototype"].cpu(), "samples": list(v["samples"]), "count": v["count"]}
                for k, v in self.gestures.items()
            }
        }
        torch.save(payload, self.vault_path)

    def load_vault(self):
        if not os.path.exists(self.vault_path):
            return
        try:
            ckpt = torch.load(self.vault_path, map_location=self.device, weights_only=False)
            self.encoder.load_state_dict(ckpt["model"])
            self.gesture_bindings.update(ckpt.get("bindings", {}))
            self.training_steps = ckpt.get("steps", 0)
            for k, v in ckpt["gestures"].items():
                self.gestures[k] = {
                    "prototype": v["proto"].to(self.device),
                    "samples": deque(v["samples"], maxlen=100),
                    "count": v["count"]
                }
        except Exception as e:
            print(f"[AVISO] Error cargando bóveda PyTorch: {e}")


# =====================================================================
# 3. CUERPOS RÍGIDOS 3D, COLISIONES, IMANES Y ONDAS DE CHOQUE
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
    def __init__(self, pos_3d: np.ndarray, color=(255, 255, 140)):
        self.pos_3d = pos_3d.copy()
        self.radius_px = 6
        self.life = 1.0
        self.color = color

    def update(self) -> bool:
        self.radius_px += 5
        self.life -= 0.12
        return self.life > 0.0


class RigidBodyHologram3D:
    def __init__(self, name: str, shape: str = "CUBO", pos=(0.0, 0.0, 0.0), color_name: str = "CIAN"):
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
        self.gravity_g = 0.0
        self.vr_blend = 0.25
        self.air_damping = 0.992
        self.magnetic_k = 0.0024
        self.hand_magnet_charge = 0.0

        self.shockwaves: List[CollisionShockwave] = []
        self.magnetic_links: List[Tuple[np.ndarray, np.ndarray, bool]] = []

    def step_simulation(self, bodies: List[RigidBodyHologram3D], hand_colliders: List[HandColliderNode]):
        self.magnetic_links.clear()
        n = len(bodies)

        # 1. Dipolos magnéticos entre hologramas
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

        # 2. Integración de velocidad, gravedad y paredes 3D
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
            for axis, limit in enumerate((self.bounds_x, self.bounds_y, self.bounds_z)):
                if b.position[axis] - r < -limit:
                    b.position[axis] = -limit + r
                    b.velocity[axis] = abs(b.velocity[axis]) * b.restitution
                elif b.position[axis] + r > limit:
                    b.position[axis] = limit - r
                    b.velocity[axis] = -abs(b.velocity[axis]) * b.restitution

        # 3. Colisiones elásticas 3D Holograma contra Holograma
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
                        if not b2.is_grabbed:
                            b2.velocity += impulse_vec * inv_m2
                        self.shockwaves.append(CollisionShockwave(b1.position + normal * b1.collider_radius, (255, 255, 120)))

        # 4. Colisión física y agarre contra las manos
        for b in bodies:
            b.is_grabbed = False

        for hc in hand_colliders:
            for b in bodies:
                if b.bound_to_hand:
                    b.is_grabbed = True
                    b.position = hc.pos_3d.copy()
                    b.velocity = hc.vel_3d * 0.85
                    continue

                delta = b.position - hc.pos_3d
                dist = float(np.linalg.norm(delta))
                combined_r = b.collider_radius + hc.radius

                if hc.is_pinching and dist < (combined_r * 1.15):
                    b.is_grabbed = True
                    b.position = hc.pos_3d.copy()
                    b.velocity = hc.vel_3d * 0.85
                    continue

                if dist < combined_r and dist > 1e-6:
                    normal = delta / dist
                    b.position += normal * (combined_r - dist)
                    rel_v = b.velocity - hc.vel_3d
                    v_norm = float(np.dot(rel_v, normal))
                    if v_norm < 0:
                        b.velocity += normal * ((1.0 + b.restitution) * (-v_norm)) + hc.vel_3d * 0.48
                        self.shockwaves.append(CollisionShockwave(b.position - normal * b.collider_radius, (0, 255, 160)))

        self.shockwaves = [sw for sw in self.shockwaves if sw.update()]


# =====================================================================
# 4. SERIALIZADOR '.NXS' CON VÍNCULOS DE GESTOS PYTORCH INCLUIDOS
# =====================================================================

class NXSProjectSerializer:
    FORMAT_VERSION = "NEXUS_SUPREME_PROJECT_V3"

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
        gesture_bindings: Dict[str, str],
        solver: SpatialPhysicsSolver3D
    ) -> str:
        full_path = cls.ensure_ext(filepath)
        payload = {
            "environment": {
                "gravity_g": round(float(solver.gravity_g), 4),
                "vr_blend": round(float(solver.vr_blend), 4),
            },
            "learned_macros": macros,
            "gesture_bindings": gesture_bindings,
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
        return f"Guardado '{full_path}' ({len(scene)} cuerpos, {len(gesture_bindings)} vínculos IA)."

    @classmethod
    def load_project(
        cls,
        filepath: str,
        solver: SpatialPhysicsSolver3D,
        vault: NeuralGestureVault
    ) -> Tuple[Dict[str, RigidBodyHologram3D], Dict[str, List[str]], str]:
        full_path = cls.ensure_ext(filepath)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No existe '{full_path}'")

        with open(full_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        data = doc.get("data", {})
        new_scene: Dict[str, RigidBodyHologram3D] = {}
        for h_dict in data.get("holograms", []):
            obj = RigidBodyHologram3D.from_nxs_dict(h_dict)
            new_scene[obj.name] = obj

        env = data.get("environment", {})
        solver.gravity_g = float(env.get("gravity_g", 0.0))
        solver.vr_blend = float(env.get("vr_blend", 0.25))

        if "gesture_bindings" in data:
            vault.gesture_bindings.update(data["gesture_bindings"])

        macros = data.get("learned_macros", {})
        return new_scene, macros, f"Recargado '{full_path}' ({len(new_scene)} cuerpos, {len(macros)} macros)."


# =====================================================================
# 5. INTÉRPRETE DE NEXUS-LANG UNIDO A LA BÓVEDA DE GESTOS PYTORCH
# =====================================================================

class SupremeNexusInterpreter:
    def __init__(self, solver: SpatialPhysicsSolver3D, vault: NeuralGestureVault, learn_gesture_cb):
        self.solver = solver
        self.vault = vault
        self.learn_gesture_cb = learn_gesture_cb

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
            ],
            "LLUVIA_METEOROS": [
                "GRAVEDAD 1.2",
                "CREAR m1 ESFERA [-0.5, -0.7, 0.0]",
                "COLOR m1 ROJO",
                "REBOTE TODOS 0.95"
            ]
        }
        self.active_project = "mundo_supremo.nxs"
        self.logs: List[str] = [
            "PUENTE PYTORCH <-> NEXUS-LANG + FISICAS 3D ACTIVO.",
            "Presiona [1-4] para grabar un gesto y disparar MACROS con tu mano."
        ]

    def log(self, msg: str):
        print(f"[NEXUS-SUPREME] {msg}")
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

        # --- COMANDOS DE VINCULACIÓN NEURONAL (PYTORCH <-> NEXUS-LANG) ---
        if upper.startswith("APRENDER_GESTO "):
            rest = line[15:].strip()
            if "=" in rest:
                g_name, cmd_body = rest.split("=", 1)
                g_name = g_name.strip().upper()
                self.vault.bind_command(g_name, cmd_body.strip())
                self.learn_gesture_cb(g_name)
                self.log(f"Grabando en PyTorch '{g_name}' -> {cmd_body.strip()}")
            return

        if upper.startswith("VINCULAR_GESTO "):
            rest = line[15:].strip()
            if "=" in rest:
                g_name, cmd_body = rest.split("=", 1)
                g_name = g_name.strip().upper()
                self.vault.bind_command(g_name, cmd_body.strip())
                self.vault.save_vault()
                self.log(f"Gesto '{g_name}' revinculado a -> {cmd_body.strip()}")
            return

        if upper.startswith("MACRO "):
            rest = line[6:].strip()
            if "=" in rest:
                m_name, m_body = rest.split("=", 1)
                self.macros[m_name.strip().upper()] = [c.strip() for c in m_body.split(";") if c.strip()]
                self.log(f"Macro '{m_name.strip().upper()}' compilada.")
            return

        if ";" in line:
            for sub in line.split(";"):
                self.execute(sub, from_gesture=from_gesture)
            return

        tokens = line.split()
        cmd = tokens[0].upper()
        args = tokens[1:]

        try:
            # Si el comando es una MACRO aprendida, expandirla y ejecutarla
            if cmd in self.macros:
                if not from_gesture:
                    self.log(f"Ejecutando Macro '{cmd}'...")
                for sub_cmd in self.macros[cmd]:
                    self.execute(sub_cmd, from_gesture=from_gesture)
                return

            if cmd == "OLVIDAR_GESTO":
                target = args[0].upper()
                if target == "TODOS":
                    self.vault.gestures.clear()
                    self.log("Bóveda PyTorch vaciada.")
                elif target in self.vault.gestures:
                    del self.vault.gestures[target]
                    self.log(f"Gesto '{target}' borrado.")
                self.vault.save_vault()

            elif cmd == "GUARDAR":
                fname = args[0] if args else self.active_project
                self.active_project = NXSProjectSerializer.ensure_ext(fname)
                self.vault.save_vault()
                self.log(NXSProjectSerializer.save_project(
                    self.active_project, self.scene, self.macros, self.vault.gesture_bindings, self.solver
                ))

            elif cmd == "CARGAR":
                fname = args[0] if args else self.active_project
                self.active_project = NXSProjectSerializer.ensure_ext(fname)
                new_sc, new_mac, msg = NXSProjectSerializer.load_project(self.active_project, self.solver, self.vault)
                self.scene = new_sc
                self.macros.update(new_mac)
                self.log(msg)

            elif cmd == "CREAR":
                name = args[0].lower()
                if from_gesture and name in self.scene:
                    name = f"{name}_{len(self.scene)+1}"
                shape = args[1].upper() if len(args) > 1 else "CUBO"
                pos = self._parse_vec3(" ".join(args[2:])) if len(args) > 2 else (
                    round(random.uniform(-0.85, 0.85), 2), round(random.uniform(-0.4, 0.25), 2), 0.0
                )
                self.scene[name] = RigidBodyHologram3D(name, shape, pos)
                self.log(f"Holograma '{name}' ({shape}) materializado.")

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
                elif target in self.scene:
                    self.scene[target].scale = float(np.clip(self.scene[target].scale + delta, 0.35, 2.4))

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

            elif cmd == "IMAN":
                target = args[0].lower()
                pol = args[1].upper()
                charge = 1.0 if pol in ("POSITIVO", "NORTE", "+") else (-1.0 if pol in ("NEGATIVO", "SUR", "-") else 0.0)
                if target == "todos":
                    for b in self.scene.values(): b.magnetic_charge = charge
                elif target in self.scene:
                    self.scene[target].magnetic_charge = charge

            elif cmd == "EXPLOSION":
                for b in self.scene.values():
                    d = b.position + np.array([random.uniform(-0.15, 0.15), -0.22, random.uniform(-0.1, 0.1)], dtype=np.float32)
                    b.velocity = (d / (float(np.linalg.norm(d)) + 1e-5)) * 0.15
                self.log("¡Explosión 3D activada!")

            elif cmd == "COLOR":
                target = args[0].lower()
                cname = args[1].upper()
                if target == "todos":
                    for b in self.scene.values(): b.set_color(cname)
                elif target in self.scene:
                    self.scene[target].set_color(cname)

            elif cmd == "ANCLAR_MANO":
                target = args[0].lower()
                if target in self.scene:
                    self.scene[target].bound_to_hand = True

            elif cmd == "SOLTAR_MANO":
                for b in self.scene.values():
                    b.bound_to_hand = False

            elif cmd == "BORRAR":
                if args[0].lower() == "todos":
                    self.scene.clear()
                elif args[0].lower() in self.scene:
                    del self.scene[args[0].lower()]

        except Exception as e:
            if not from_gesture:
                self.log(f"Error: {e}")


# =====================================================================
# 6. MOTOR INTERACTIVO MAESTRO (PYTORCH + FÍSICAS 3D + CONSOLA EN VIVO)
# =====================================================================

class NexusSupremeEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.solver = SpatialPhysicsSolver3D()
        self.vault = NeuralGestureVault()
        self.interpreter = SupremeNexusInterpreter(self.solver, self.vault, self.start_gesture_recording)
        self.cmd_queue: queue.Queue = queue.Queue()

        self.recording_gesture: Optional[str] = None
        self.recording_frames_left = 0

        # Histéresis y Cooldown para disparar comandos/MACROS limpios al detectar un gesto
        self.last_discrete_gesture = ""
        self.last_trigger_timestamp = 0.0

        self.prev_hand_centers: Dict[int, np.ndarray] = {}
        self.onscreen_input = ""
        self.running = True

    def start_gesture_recording(self, gesture_name: str, frames: int = 25):
        self.recording_gesture = gesture_name.upper()
        self.recording_frames_left = frames

    def _console_worker(self):
        while self.running:
            try:
                line = input("NexusSupreme >> ")
                if line.strip():
                    self.cmd_queue.put(line.strip())
            except EOFError:
                break

    def dispatch_neural_gesture(self, gesture_name: str):
        """
        Puente directo PyTorch -> NexusLang:
        Busca el comando o MACRO asociado al gesto reconocido y lo ejecuta.
        """
        bound_cmd = self.vault.gesture_bindings.get(gesture_name)
        if not bound_cmd:
            return

        if "ESCALAR_DELTA" in bound_cmd.upper():
            self.interpreter.execute(bound_cmd, from_gesture=True)
        else:
            now = time.time()
            if gesture_name != self.last_discrete_gesture or (now - self.last_trigger_timestamp) > 1.6:
                self.interpreter.log(f"IA Gesto [{gesture_name}] -> '{bound_cmd}'")
                self.interpreter.execute(bound_cmd, from_gesture=True)
                self.last_discrete_gesture = gesture_name
                self.last_trigger_timestamp = now

    def project_3d(self, pts_3d, w, h, focal=640.0, cam_dist=3.6):
        out = []
        for pt in pts_3d:
            z = max(cam_dist + pt[2], 0.35)
            out.append((int((pt[0] * focal) / z + w * 0.5), int((pt[1] * focal) / z + h * 0.5)))
        return out

    def process_hands_and_ai(self, multi_hand_landmarks, frame) -> Tuple[List[HandColliderNode], str, float]:
        h, w, _ = frame.shape
        colliders: List[HandColliderNode] = []
        current_centers: Dict[int, np.ndarray] = {}
        predicted_gesture = "NINGUNO"
        confidence = 0.0

        if not multi_hand_landmarks:
            self.prev_hand_centers.clear()
            self.last_discrete_gesture = ""
            return colliders, predicted_gesture, confidence

        for idx, hand_lms in enumerate(multi_hand_landmarks):
            self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

            # Evaluar la IA en PyTorch con la primera mano visible
            if idx == 0:
                invariant_vec = CanonicalHandGeometry.extract_invariant_features(hand_lms)
                if self.recording_frames_left > 0 and self.recording_gesture:
                    self.vault.learn_sample(self.recording_gesture, invariant_vec)
                    self.recording_frames_left -= 1
                    if self.recording_frames_left == 0:
                        self.vault.save_vault()
                        self.interpreter.log(f"Gesto '{self.recording_gesture}' entrenado en PyTorch.")

                predicted_gesture, confidence, _ = self.vault.recognize(invariant_vec)
                if predicted_gesture not in ("DESCONOCIDO", "SIN_GESTOS") and self.recording_frames_left == 0:
                    self.dispatch_neural_gesture(predicted_gesture)
                else:
                    self.last_discrete_gesture = ""

            # Construir colisionadores físicos 3D para golpear o agarrar hologramas
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
        return colliders, predicted_gesture, confidence

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
            cv2.putText(frame, f"{b.name}{mag_tag}", (c2d[0] - 38, c2d[1] - int(48 * b.scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

        for sw in self.solver.shockwaves:
            pt_2d = self.project_3d([sw.pos_3d], w, h)[0]
            cv2.circle(frame, pt_2d, sw.radius_px, sw.color, 2, cv2.LINE_AA)

        return frame

    def draw_supreme_hud(self, frame, predicted: str, conf: float):
        h, w, _ = frame.shape

        # Panel Izquierdo: Telemetría de Físicas + Consola NexusLang
        cv2.rectangle(frame, (15, 15), (660, 168), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (660, 168), (0, 230, 255), 1)
        cv2.putText(frame, f"NEXUS SUPREME | IA Gesto: {predicted} ({conf*100:.0f}%) | Cuerpos: {len(self.interpreter.scene)}",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (0, 255, 160), 2)

        for i, line in enumerate(self.interpreter.logs[-4:]):
            cv2.putText(frame, f"> {line}", (26, 64 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (215, 230, 245), 1)

        cv2.putText(frame, "Teclas [1-4]: Ensenar Gesto_1..4 en PyTorch | [5]: Guardar .nxs | [6]: Cargar .nxs",
                    (26, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 215, 80), 1)

        # Panel Derecho: Mapa de Vínculos Gestuales (PyTorch -> Macros/Comandos NexusLang)
        cv2.rectangle(frame, (w - 495, 15), (w - 15, 168), (18, 16, 26), -1)
        cv2.rectangle(frame, (w - 495, 15), (w - 15, 168), (255, 150, 0), 1)
        cv2.putText(frame, f"VINCULOS PYTORCH -> NEXUS-LANG ({len(self.vault.gestures)} entrenados)",
                    (w - 482, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 80), 1)

        for idx, (g_name, cmd_str) in enumerate(list(self.vault.gesture_bindings.items())[:5]):
            tag = "[ACTIVO]" if g_name in self.vault.gestures else "[Vacio]"
            col = (0, 255, 140) if g_name == predicted else (205, 205, 215)
            short_cmd = cmd_str if len(cmd_str) < 34 else cmd_str[:31] + "..."
            cv2.putText(frame, f"{tag} {g_name} => {short_cmd}",
                        (w - 482, 62 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.39, col, 1)

        # Barra Inferior de Código
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (12, 14, 20), -1)
        bar_col = (255, 80, 255) if self.recording_frames_left > 0 else (0, 255, 140)
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), bar_col, 1)

        if self.recording_frames_left > 0:
            cv2.putText(frame, f">> ENTRENANDO GESTO '{self.recording_gesture}' EN PYTORCH ({self.recording_frames_left} muestras)...",
                        (28, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 120, 255), 2)
        else:
            cursor = "_" if int(time.time() * 2.5) % 2 == 0 else " "
            cv2.putText(frame, f"NEXUS >> {self.onscreen_input}{cursor}",
                        (28, h - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 140), 2)
            cv2.putText(frame, "Escribe: APRENDER_GESTO PUÑO = LLUVIA_METEOROS  |  VINCULAR_GESTO GESTO_1 = EXPLOSION",
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

            hand_colliders, predicted_gesture, confidence = self.process_hands_and_ai(
                results.multi_hand_landmarks, frame
            )
            self.solver.step_simulation(list(self.interpreter.scene.values()), hand_colliders)

            scene_frame = self.render_world(frame)
            self.draw_supreme_hud(scene_frame, predicted_gesture, confidence)
            cv2.imshow("NexusXR - Puente PyTorch + NexusLang + Fisicas 3D", scene_frame)

            key = cv2.waitKey(1)
            if key == -1:
                continue
            elif key == 27:  # ESC
                self.vault.save_vault()
                self.running = False
                break
            elif key in (13, 10):  # ENTER
                if self.onscreen_input.strip():
                    self.cmd_queue.put(self.onscreen_input.strip())
                    self.onscreen_input = ""
            elif key in (8, 127):  # BACKSPACE
                self.onscreen_input = self.onscreen_input[:-1]
            # Atajos rápidos cuando la barra de código está vacía
            elif not self.onscreen_input and (key & 0xFF) in (ord('1'), ord('2'), ord('3'), ord('4')):
                slot = chr(key & 0xFF)
                self.start_gesture_recording(f"GESTO_{slot}")
            elif not self.onscreen_input and (key & 0xFF) == ord('5'):
                self.cmd_queue.put("GUARDAR mundo_supremo.nxs")
            elif not self.onscreen_input and (key & 0xFF) == ord('6'):
                self.cmd_queue.put("CARGAR mundo_supremo.nxs")
            elif 32 <= (key & 0xFF) <= 126:
                self.onscreen_input += chr(key & 0xFF)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = NexusSupremeEngine()
    app.run()
