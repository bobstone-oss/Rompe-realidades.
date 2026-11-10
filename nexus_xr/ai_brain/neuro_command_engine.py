import os
import time
import math
import random
import threading
import queue
from collections import deque
from typing import Dict, List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =====================================================================
# 1. EXTRACTOR GEOMÉTRICO INVARIANTE 3D (GRAM-SCHMIDT 78D)
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
# 2. RED NEURONAL HIPERSFÉRICA EN PYTORCH + VÍNCULOS DE COMANDOS
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


class NeuralCommandVault:
    """
    Almacena tanto los prototipos vectoriales en PyTorch como los comandos
    de NexusLang asociados a cada gesto aprendido.
    """
    def __init__(self, vault_path: str = "nexus_unified_vault.pt"):
        self.vault_path = vault_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = HypersphericalGestureNet().to(self.device)
        self.optimizer = optim.AdamW(self.encoder.parameters(), lr=1.5e-3)
        self.triplet_loss = nn.TripletMarginLoss(margin=0.45, p=2)

        self.gestures: Dict[str, Dict] = {}
        self.gesture_bindings: Dict[str, str] = {
            "GESTO_1": "ESCALAR_DELTA TODOS 0.04",
            "GESTO_2": "ESCALAR_DELTA TODOS -0.04",
            "GESTO_3": "COLOR TODOS MAGENTA ; VELOCIDAD TODOS 3.0",
            "GESTO_4": "CREAR satelite ESFERA [0.6, 0.0, 0.0] ; COLOR satelite ORO"
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
            print(f"[AVISO] No se cargó bóveda previa: {e}")


# =====================================================================
# 3. MOTOR MULTI-HOLOGRAMA 3D
# =====================================================================

def rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


COLOR_PALETTE = {
    "CIAN": (255, 230, 0),
    "VERDE": (100, 255, 80),
    "MAGENTA": (255, 80, 255),
    "ORO": (0, 215, 255),
    "ROJO": (60, 60, 255),
    "AZUL": (255, 130, 40),
    "BLANCO": (255, 255, 255),
}


class HologramEntity3D:
    def __init__(self, name: str, shape: str = "CUBO", pos=(0.0, 0.0, 0.0), color_name="CIAN"):
        self.name = name
        self.shape = shape.upper()
        self.position = np.array(pos, dtype=np.float32)
        self.target_position = np.array(pos, dtype=np.float32)
        self.rotation = np.array([0.3, 0.5, 0.0], dtype=np.float32)
        self.spin_speed = np.array([0.008, 0.018, 0.004], dtype=np.float32)
        self.scale = 1.0
        self.target_scale = 1.0
        self.color = COLOR_PALETTE.get(color_name.upper(), (255, 230, 0))
        self.bound_to_hand = False
        self.vertices, self.edges, self.faces = self._build_mesh(self.shape)

    def _build_mesh(self, shape: str):
        if shape == "PIRAMIDE":
            verts = np.array([
                [-0.5, 0.45, -0.5], [0.5, 0.45, -0.5],
                [0.5, 0.45, 0.5], [-0.5, 0.45, 0.5], [0.0, -0.55, 0.0]
            ], dtype=np.float32)
            edges = [(0,1), (1,2), (2,3), (3,0), (0,4), (1,4), (2,4), (3,4)]
            faces = [(0,1,4), (1,2,4), (2,3,4), (3,0,4)]
            return verts, edges, faces
        elif shape in ("OCTAEDRO", "DIAMANTE"):
            verts = np.array([
                [0.55, 0, 0], [-0.55, 0, 0], [0, 0.65, 0],
                [0, -0.65, 0], [0, 0, 0.55], [0, 0, -0.55]
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
                    verts.append([0.52 * math.cos(phi) * math.cos(theta), 0.52 * math.sin(phi), 0.52 * math.cos(phi) * math.sin(theta)])
                    curr = r * sectors + s
                    edges.append((curr, r * sectors + ((s + 1) % sectors)))
                    if r > 0:
                        edges.append((curr, (r - 1) * sectors + s))
            return np.array(verts, dtype=np.float32), edges, []
        else:
            verts = np.array([
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
            ], dtype=np.float32) * 0.45
            edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
            faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6)]
            return verts, edges, faces

    def tick(self, smoothing: float = 0.25):
        self.position += (self.target_position - self.position) * smoothing
        self.scale += (self.target_scale - self.scale) * smoothing
        self.target_scale = float(np.clip(self.target_scale, 0.25, 3.8))
        self.scale = float(np.clip(self.scale, 0.25, 3.8))
        self.rotation += self.spin_speed


# =====================================================================
# 4. INTÉRPRETE DE NEXUS-LANG CON PUENTE NEURonal
# =====================================================================

class UnifiedNexusInterpreter:
    def __init__(self, scene: Dict[str, HologramEntity3D], vault: NeuralCommandVault, trigger_learn_cb):
        self.scene = scene
        self.vault = vault
        self.trigger_learn_cb = trigger_learn_cb
        self.macros: Dict[str, List[str]] = {}
        self.vr_blend_ref = [0.25]
        self.logs: List[str] = [
            "NUCLEO UNIFICADO ACTIVO (PyTorch + NexusLang + 3D).",
            "Presiona [1-4] o escribe APRENDER_GESTO NOMBRE = COMANDO"
        ]

    def log(self, msg: str):
        print(f"[NEXUS-CORE] {msg}")
        self.logs.append(msg)
        if len(self.logs) > 5:
            self.logs.pop(0)

    def _parse_vec3(self, text: str) -> Tuple[float, float, float]:
        clean = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        parts = [float(x) for x in clean.replace(",", " ").split() if x]
        return (parts[0], parts[1], parts[2]) if len(parts) >= 3 else (0.0, 0.0, 0.0)

    def execute(self, raw_command: str, from_gesture: bool = False):
        raw = raw_command.strip()
        if not raw:
            return

        # Comandos especiales de nivel superior (con '=' propio)
        upper_raw = raw.upper()
        if upper_raw.startswith("APRENDER_GESTO "):
            rest = raw[15:].strip()
            if "=" in rest:
                g_name, cmd_body = rest.split("=", 1)
                g_name = g_name.strip().upper()
                self.vault.bind_command(g_name, cmd_body.strip())
                self.trigger_learn_cb(g_name)
                self.log(f"Grabando gesto '{g_name}' -> {cmd_body.strip()}")
            return

        if upper_raw.startswith("VINCULAR_GESTO "):
            rest = raw[15:].strip()
            if "=" in rest:
                g_name, cmd_body = rest.split("=", 1)
                g_name = g_name.strip().upper()
                self.vault.bind_command(g_name, cmd_body.strip())
                self.vault.save_vault()
                self.log(f"Gesto '{g_name}' revinculado a -> {cmd_body.strip()}")
            return

        if upper_raw.startswith("MACRO "):
            rest = raw[6:].strip()
            if "=" in rest:
                m_name, m_body = rest.split("=", 1)
                m_name = m_name.strip().upper()
                self.macros[m_name] = [c.strip() for c in m_body.split(";") if c.strip()]
                self.log(f"Macro '{m_name}' compilada en memoria.")
            return

        # Si hay múltiples instrucciones separadas por ';'
        if ";" in raw:
            for sub in raw.split(";"):
                self.execute(sub, from_gesture=from_gesture)
            return

        tokens = raw.split()
        cmd = tokens[0].upper()
        args = tokens[1:]

        try:
            if cmd in self.macros:
                for sub_cmd in self.macros[cmd]:
                    self.execute(sub_cmd, from_gesture=from_gesture)
                return

            if cmd == "OLVIDAR_GESTO":
                target = args[0].upper()
                if target == "TODOS":
                    self.vault.gestures.clear()
                    self.log("Bóveda neuronal de gestos vaciada.")
                elif target in self.vault.gestures:
                    del self.vault.gestures[target]
                    self.log(f"Gesto '{target}' eliminado de PyTorch.")
                self.vault.save_vault()

            elif cmd == "CREAR":
                name = args[0].lower()
                # Si viene de un gesto y el nombre ya existe, generar un sufijo único
                if from_gesture and name in self.scene:
                    name = f"{name}_{len(self.scene)+1}"
                shape = args[1].upper() if len(args) > 1 else "CUBO"
                pos = self._parse_vec3(" ".join(args[2:])) if len(args) > 2 else (
                    round(random.uniform(-0.7, 0.7), 2), round(random.uniform(-0.35, 0.35), 2), 0.0
                )
                self.scene[name] = HologramEntity3D(name, shape, pos)
                self.log(f"Holograma '{name}' ({shape}) creado en {pos}")

            elif cmd == "ESCALAR":
                target = args[0].lower()
                val = float(args[1])
                if target == "todos":
                    for obj in self.scene.values(): obj.target_scale = val
                elif target in self.scene:
                    self.scene[target].target_scale = val

            elif cmd == "ESCALAR_DELTA":
                target = args[0].lower()
                delta = float(args[1])
                if target == "todos":
                    for obj in self.scene.values(): obj.target_scale += delta
                elif target in self.scene:
                    self.scene[target].target_scale += delta

            elif cmd == "COLOR":
                target = args[0].lower()
                col = COLOR_PALETTE.get(args[1].upper(), (255, 255, 255))
                if target == "todos":
                    for obj in self.scene.values(): obj.color = col
                elif target in self.scene:
                    self.scene[target].color = col
                if not from_gesture:
                    self.log(f"Color de '{target}' -> {args[1].upper()}")

            elif cmd == "MOVER":
                target = args[0].lower()
                pos = self._parse_vec3(" ".join(args[1:]))
                if target in self.scene:
                    self.scene[target].target_position = np.array(pos, dtype=np.float32)
                    self.scene[target].bound_to_hand = False

            elif cmd == "VELOCIDAD":
                target = args[0].lower()
                mult = float(args[1])
                if target == "todos":
                    for obj in self.scene.values():
                        obj.spin_speed = np.array([0.008, 0.018, 0.004], dtype=np.float32) * mult
                elif target in self.scene:
                    self.scene[target].spin_speed = np.array([0.008, 0.018, 0.004], dtype=np.float32) * mult

            elif cmd == "ANCLAR_MANO":
                target = args[0].lower()
                if target == "todos":
                    for obj in self.scene.values(): obj.bound_to_hand = True
                elif target in self.scene:
                    self.scene[target].bound_to_hand = True
                self.log(f"'{target}' anclado a los vectores de la mano.")

            elif cmd == "SOLTAR_MANO":
                for obj in self.scene.values():
                    obj.bound_to_hand = False
                self.log("Hologramas desacoplados de la mano.")

            elif cmd == "BORRAR":
                target = args[0].lower()
                if target == "todos":
                    self.scene.clear()
                elif target in self.scene:
                    del self.scene[target]

            elif cmd == "REALIDAD":
                pct = float(args[0])
                self.vr_blend_ref[0] = float(np.clip(pct / 100.0, 0.0, 1.0))

        except Exception as e:
            if not from_gesture:
                self.log(f"Error en '{raw}': {e}")


# =====================================================================
# 5. MOTOR EJECUTIVO UNIFICADO (TIEMPO REAL)
# =====================================================================

class NexusUnifiedEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.scene: Dict[str, HologramEntity3D] = {
            "cubo1": HologramEntity3D("cubo1", "CUBO", (0.0, 0.0, 0.0), "CIAN")
        }
        self.vault = NeuralCommandVault()
        self.interpreter = UnifiedNexusInterpreter(self.scene, self.vault, self.start_gesture_recording)
        self.cmd_queue: queue.Queue = queue.Queue()

        # Estado de enseñanza de gestos en vivo
        self.recording_gesture_name: Optional[str] = None
        self.recording_frames_left = 0

        # Control de enfriamiento (Cooldown) para comandos discretos disparados por gestos
        self.last_discrete_gesture = ""
        self.last_discrete_trigger_time = 0.0

        self.onscreen_buffer = ""
        self.running = True

    def start_gesture_recording(self, gesture_name: str, frames: int = 25):
        self.recording_gesture_name = gesture_name.upper()
        self.recording_frames_left = frames

    def _console_worker(self):
        while self.running:
            try:
                line = input("NexusUnified >> ")
                if line.strip():
                    self.cmd_queue.put(line.strip())
            except EOFError:
                break

    def dispatch_recognized_gesture(self, gesture_name: str):
        """
        Ejecuta el comando de NexusLang vinculado al gesto reconocido por PyTorch.
        Distingue entre órdenes continuas (ESCALAR_DELTA) y órdenes de disparo único (CREAR, COLOR).
        """
        bound_cmd = self.vault.gesture_bindings.get(gesture_name)
        if not bound_cmd:
            return

        is_continuous = "ESCALAR_DELTA" in bound_cmd.upper()
        now = time.time()

        if is_continuous:
            self.interpreter.execute(bound_cmd, from_gesture=True)
        else:
            # Cooldown de 1.5 segundos para comandos discretos (evita crear 60 objetos por segundo)
            if gesture_name != self.last_discrete_gesture or (now - self.last_discrete_trigger_time) > 1.5:
                self.interpreter.log(f"IA Disparó [{gesture_name}] -> {bound_cmd}")
                self.interpreter.execute(bound_cmd, from_gesture=True)
                self.last_discrete_gesture = gesture_name
                self.last_discrete_trigger_time = now

    def project_3d(self, points_3d, w, h, focal=640.0, cam_dist=3.6):
        pts_2d = []
        for pt in points_3d:
            z = max(cam_dist + pt[2], 0.35)
            pts_2d.append((int((pt[0] * focal) / z + (w * 0.5)), int((pt[1] * focal) / z + (h * 0.5))))
        return pts_2d

    def render_3d_world(self, frame, pinch_3d: Optional[np.ndarray]):
        h, w, _ = frame.shape
        vr_blend = self.interpreter.vr_blend_ref[0]

        if vr_blend > 0.02:
            vr_bg = np.zeros_like(frame)
            horizon = int(h * 0.50)
            for i in range(-10, 11):
                cv2.line(vr_bg, (int(w*0.5 + i*115), h), (int(w*0.5 + i*18), horizon), (55, 30, 15), 1)
            for d in range(1, 9):
                y = int(horizon + (d**1.75) * 6.5)
                if y < h: cv2.line(vr_bg, (0, y), (w, y), (65, 35, 15), 1)
            frame = cv2.addWeighted(frame, 1.0 - vr_blend, vr_bg, vr_blend, 0)

        overlay = frame.copy()
        for name, obj in list(self.scene.items()):
            if pinch_3d is not None:
                if obj.bound_to_hand or np.linalg.norm(pinch_3d - obj.position) < (0.65 * obj.scale):
                    obj.target_position = pinch_3d.copy()

            obj.tick()
            R = rotation_matrix_3d(*obj.rotation)
            world_verts = (obj.vertices * obj.scale) @ R.T + obj.position
            pts_2d = self.project_3d(world_verts, w, h)

            for face in obj.faces:
                poly = np.array([pts_2d[idx] for idx in face], dtype=np.int32)
                c = obj.color
                cv2.fillPoly(overlay, [poly], (c[0]//4, c[1]//4, c[2]//4))

            for e1, e2 in obj.edges:
                cv2.line(frame, pts_2d[e1], pts_2d[e2], obj.color, 2, cv2.LINE_AA)

            c2d = self.project_3d([obj.position], w, h)[0]
            cv2.putText(frame, f"{obj.name} ({obj.scale:.1f}x)", (c2d[0] - 42, c2d[1] - int(52 * obj.scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)
        return frame

    def draw_unified_hud(self, frame, predicted: str, conf: float):
        h, w, _ = frame.shape

        # Panel Izquierdo: Consola e Intérprete
        cv2.rectangle(frame, (15, 15), (650, 165), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (650, 165), (0, 220, 255), 1)
        cv2.putText(frame, f"NEXUS UNIFIED ENGINE | IA GESTO: {predicted} ({conf*100:.0f}%)",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 160), 2)

        for idx, line in enumerate(self.interpreter.logs[-4:]):
            cv2.putText(frame, f"> {line}", (26, 65 + idx * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 230, 245), 1)

        # Panel Derecho: Vínculos Gestuales en Memoria PyTorch
        cv2.rectangle(frame, (w - 470, 15), (w - 15, 165), (18, 16, 26), -1)
        cv2.rectangle(frame, (w - 470, 15), (w - 15, 165), (255, 140, 0), 1)
        cv2.putText(frame, f"MEMORIA PYTORCH ({len(self.vault.gestures)} gestos | Loss: {self.vault.last_loss:.3f})",
                    (w - 458, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 80), 1)

        shown = 0
        for g_name, cmd_str in list(self.vault.gesture_bindings.items())[:4]:
            trained_tag = "[OK]" if g_name in self.vault.gestures else "[Tecla 1-4]"
            color_row = (0, 255, 140) if g_name == predicted else (200, 200, 200)
            short_cmd = cmd_str if len(cmd_str) < 34 else cmd_str[:31] + "..."
            cv2.putText(frame, f"{trained_tag} {g_name} -> {short_cmd}",
                        (w - 458, 65 + shown * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color_row, 1)
            shown += 1

        # Barra Inferior: Entrada de código en vivo
        cv2.rectangle(frame, (15, h - 68), (w - 15, h - 15), (12, 14, 20), -1)
        bar_col = (255, 80, 255) if self.recording_frames_left > 0 else (0, 255, 140)
        cv2.rectangle(frame, (15, h - 68), (w - 15, h - 15), bar_col, 1)

        if self.recording_frames_left > 0:
            status = f">> GRABANDO GESTO '{self.recording_gesture_name}' EN PYTORCH ({self.recording_frames_left} muestras)..."
            cv2.putText(frame, status, (28, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 120, 255), 2)
        else:
            cursor = "_" if int(time.time() * 2.5) % 2 == 0 else " "
            cv2.putText(frame, f"NEXUS-LANG >> {self.onscreen_buffer}{cursor}",
                        (28, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 140), 2)
            cv2.putText(frame, "Escribe: APRENDER_GESTO PODER = CREAR e1 ESFERA ; COLOR e1 ORO   |   Teclas [F1-F4] o [1-4] Grabacion Rapida   |   [ESC] Salir",
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

            predicted_gesture = "NINGUNO"
            confidence = 0.0
            pinch_pos_3d = None

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                # 1. Extraer vector invariante de 78 dimensiones
                invariant_vec = CanonicalHandGeometry.extract_invariant_features(hand_lms)

                # 2. Si estamos enseñando un gesto nuevo en vivo, alimentar PyTorch
                if self.recording_frames_left > 0 and self.recording_gesture_name:
                    self.vault.learn_sample(self.recording_gesture_name, invariant_vec)
                    self.recording_frames_left -= 1
                    if self.recording_frames_left == 0:
                        self.vault.save_vault()
                        self.interpreter.log(f"Gesto '{self.recording_gesture_name}' aprendido y guardado.")

                # 3. Clasificar el gesto actual con la red hipersférica
                predicted_gesture, confidence, _ = self.vault.recognize(invariant_vec)
                if predicted_gesture not in ("DESCONOCIDO", "SIN_GESTOS") and self.recording_frames_left == 0:
                    self.dispatch_recognized_gesture(predicted_gesture)
                else:
                    self.last_discrete_gesture = ""

                # 4. Detectar agarre directo (Pinch) para mover hologramas con los dedos
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                mid = (pts[4] + pts[8]) * 0.5
                if float(np.linalg.norm(pts[8] - pts[4])) < 0.058 or any(o.bound_to_hand for o in self.scene.values()):
                    pinch_pos_3d = np.array([(mid[0] - 0.5) * 3.4, (mid[1] - 0.5) * 2.0, mid[2] * 1.4], dtype=np.float32)

            scene_frame = self.render_3d_world(frame, pinch_pos_3d)
            self.draw_unified_hud(scene_frame, predicted_gesture, confidence)
            cv2.imshow("NexusXR - Motor Unificado (PyTorch + NexusLang + 3D)", scene_frame)

            key = cv2.waitKey(1)
            if key == -1:
                continue
            elif key == 27:  # ESC
                self.vault.save_vault()
                self.running = False
                break
            elif key in (13, 10):  # ENTER
                if self.onscreen_buffer.strip():
                    self.cmd_queue.put(self.onscreen_buffer.strip())
                    self.onscreen_buffer = ""
            elif key in (8, 127):  # BACKSPACE
                self.onscreen_buffer = self.onscreen_buffer[:-1]
            # Si la barra de texto está vacía, las teclas 1-4 graban rápido los 4 gestos base
            elif not self.onscreen_buffer and (key & 0xFF) in (ord('1'), ord('2'), ord('3'), ord('4')):
                slot = chr(key & 0xFF)
                self.start_gesture_recording(f"GESTO_{slot}")
            elif 32 <= (key & 0xFF) <= 126:
                self.onscreen_buffer += chr(key & 0xFF)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = NexusUnifiedEngine()
    engine.run()
