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
# 1. EXTRACTOR GEOMÉTRICO INVARIANTE (GRAM-SCHMIDT 78D)
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
# 2. BÓVEDA NEURONAL HIPERSFÉRICA EN PYTORCH
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
    def __init__(self, vault_path: str = "nexus_lang_gesture_vault.pt"):
        self.vault_path = vault_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = HypersphericalGestureNet().to(self.device)
        self.optimizer = optim.AdamW(self.encoder.parameters(), lr=1.5e-3)
        self.triplet_loss = nn.TripletMarginLoss(margin=0.45, p=2)

        self.gestures: Dict[str, Dict] = {}
        # Vínculos iniciales entre el nombre del gesto y comandos o MACROS de NexusLang
        self.gesture_bindings: Dict[str, str] = {
            "GESTO_1": "ESCALAR TODOS 1.8 ; COLOR TODOS CIAN",
            "GESTO_2": "ESCALAR TODOS 0.6 ; COLOR TODOS MAGENTA",
            "GESTO_3": "MODO_FUSION",
            "GESTO_4": "EXPLOSION"
        }
        self.last_loss = 0.0
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
            for k, v in ckpt["gestures"].items():
                self.gestures[k] = {
                    "prototype": v["proto"].to(self.device),
                    "samples": deque(v["samples"], maxlen=100),
                    "count": v["count"]
                }
        except Exception as e:
            print(f"[AVISO] No se cargó bóveda previa: {e}")


# =====================================================================
# 3. MOTOR HOLOGRÁFICO 3D Y FÍSICAS
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
    "BLANCO": (255, 255, 255),
}


class HologramEntity3D:
    def __init__(self, name: str, shape: str = "CUBO", pos=(0.0, 0.0, 0.0), color_name="CIAN"):
        self.name = name
        self.shape = shape.upper()
        self.position = np.array(pos, dtype=np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.3, 0.5, 0.0], dtype=np.float32)
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
            return verts, [(0,1), (1,2), (2,3), (3,0), (0,4), (1,4), (2,4), (3,4)], [(0,1,4), (1,2,4)]
        elif shape == "ESFERA":
            verts, edges = [], []
            for r in range(4):
                phi = math.pi * (r + 1) / 5 - (math.pi / 2)
                for s in range(8):
                    theta = 2.0 * math.pi * s / 8
                    verts.append([0.5 * math.cos(phi) * math.cos(theta), 0.5 * math.sin(phi), 0.5 * math.cos(phi) * math.sin(theta)])
                    curr = r * 8 + s
                    edges.append((curr, r * 8 + ((s + 1) % 8)))
            return np.array(verts, dtype=np.float32), edges, []
        else:
            verts = np.array([
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
            ], dtype=np.float32) * 0.45
            return verts, [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)], [(0,1,2,3)]

    def tick(self):
        self.scale += (self.target_scale - self.scale) * 0.25
        self.position += self.velocity
        self.velocity *= 0.95
        self.rotation += np.array([0.01, 0.018, 0.004], dtype=np.float32)


# =====================================================================
# 4. INTÉRPRETE DE NEXUS-LANG CONECTADO A PYTORCH
# =====================================================================

class GestureDrivenInterpreter:
    def __init__(self, scene: Dict[str, HologramEntity3D], vault: NeuralCommandVault, trigger_learn_cb):
        self.scene = scene
        self.vault = vault
        self.trigger_learn_cb = trigger_learn_cb
        self.macros: Dict[str, List[str]] = {
            "MODO_FUSION": [
                "CREAR satelite ESFERA [0.6, 0.0, 0.0]",
                "COLOR satelite ORO",
                "ESCALAR satelite 1.5"
            ]
        }
        self.logs: List[str] = [
            "PUENTE PYTORCH <-> NEXUS-LANG ACTIVO.",
            "Al hacer un gesto aprendido, se ejecuta su comando asociado."
        ]

    def log(self, msg: str):
        print(f"[BRIDGE-LANG] {msg}")
        self.logs.append(msg)
        if len(self.logs) > 4:
            self.logs.pop(0)

    def _parse_vec3(self, text: str) -> Tuple[float, float, float]:
        clean = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        parts = [float(x) for x in clean.replace(",", " ").split() if x]
        return (parts[0], parts[1], parts[2]) if len(parts) >= 3 else (0.0, 0.0, 0.0)

    def execute(self, raw_command: str):
        raw = raw_command.strip()
        if not raw:
            return

        upper_raw = raw.upper()
        if upper_raw.startswith("APRENDER_GESTO "):
            rest = raw[15:].strip()
            if "=" in rest:
                g_name, cmd_body = rest.split("=", 1)
                g_name = g_name.strip().upper()
                self.vault.bind_command(g_name, cmd_body.strip())
                self.trigger_learn_cb(g_name)
                self.log(f"Asociando nuevo gesto '{g_name}' -> {cmd_body.strip()}")
            return

        if upper_raw.startswith("MACRO "):
            rest = raw[6:].strip()
            if "=" in rest:
                m_name, m_body = rest.split("=", 1)
                m_name = m_name.strip().upper()
                self.macros[m_name] = [c.strip() for c in m_body.split(";") if c.strip()]
                self.log(f"Macro '{m_name}' compilada en memoria.")
            return

        if ";" in raw:
            for sub in raw.split(";"):
                self.execute(sub)
            return

        tokens = raw.split()
        cmd = tokens[0].upper()
        args = tokens[1:]

        try:
            if cmd in self.macros:
                self.log(f"Ejecutando Macro '{cmd}' por gesto/consola...")
                for sub_cmd in self.macros[cmd]:
                    self.execute(sub_cmd)
                return

            if cmd == "CREAR":
                name = args[0].lower()
                shape = args[1].upper() if len(args) > 1 else "CUBO"
                pos = self._parse_vec3(" ".join(args[2:])) if len(args) > 2 else (
                    round(random.uniform(-0.7, 0.7), 2), round(random.uniform(-0.35, 0.35), 2), 0.0
                )
                self.scene[name] = HologramEntity3D(name, shape, pos)
                self.log(f"Holograma '{name}' creado.")

            elif cmd == "ESCALAR":
                target = args[0].lower()
                val = float(args[1])
                if target == "todos":
                    for obj in self.scene.values(): obj.target_scale = val
                elif target in self.scene:
                    self.scene[target].target_scale = val

            elif cmd == "COLOR":
                target = args[0].lower()
                col_name = args[1].upper()
                col = COLOR_PALETTE.get(col_name, (255, 255, 255))
                if target == "todos":
                    for obj in self.scene.values(): obj.color = col
                elif target in self.scene:
                    self.scene[target].color = col
                self.log(f"Color de '{target}' -> {col_name}")

            elif cmd == "EXPLOSION":
                for obj in self.scene.values():
                    obj.velocity = np.array([random.uniform(-0.08, 0.08), 0.12, random.uniform(-0.05, 0.05)], dtype=np.float32)
                self.log("¡Explosión de hologramas ejecutada!")

            elif cmd == "BORRAR":
                if args[0].lower() == "todos":
                    self.scene.clear()

        except Exception as e:
            self.log(f"Error ejecutando '{raw}': {e}")


# =====================================================================
# 5. MOTOR UNIFICADO EN TIEMPO REAL
# =====================================================================

class NexusBridgeWorkspace:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.scene: Dict[str, HologramEntity3D] = {
            "cubo1": HologramEntity3D("cubo1", "CUBO", (0.0, 0.0, 0.0), "CIAN")
        }
        self.vault = NeuralCommandVault()
        self.interpreter = GestureDrivenInterpreter(self.scene, self.vault, self.start_recording)
        self.cmd_queue: queue.Queue = queue.Queue()

        self.recording_name: Optional[str] = None
        self.recording_frames = 0
        self.last_triggered_gesture = ""
        self.last_trigger_time = 0.0

        self.onscreen_input = ""
        self.running = True

    def start_recording(self, gesture_name: str, frames: int = 25):
        self.recording_name = gesture_name.upper()
        self.recording_frames = frames

    def _console_worker(self):
        while self.running:
            try:
                line = input("NexusBridge >> ")
                if line.strip():
                    self.cmd_queue.put(line.strip())
            except EOFError:
                break

    def dispatch_gesture_command(self, gesture_name: str):
        """Ejecuta automáticamente el comando vinculado al gesto de PyTorch con un Cooldown."""
        bound_cmd = self.vault.gesture_bindings.get(gesture_name)
        if not bound_cmd:
            return

        now = time.time()
        # Cooldown de 1.2 segundos para evitar disparar comandos en bucle continuo
        if gesture_name != self.last_triggered_gesture or (now - self.last_trigger_time) > 1.2:
            self.interpreter.log(f"Gesto [{gesture_name}] detectado -> Ejecutando: {bound_cmd}")
            self.interpreter.execute(bound_cmd)
            self.last_triggered_gesture = gesture_name
            self.last_trigger_time = now

    def project_3d(self, pts_3d, w, h, focal=640.0, cam_dist=3.6):
        out = []
        for pt in pts_3d:
            z = max(cam_dist + pt[2], 0.35)
            out.append((int((pt[0] * focal) / z + w * 0.5), int((pt[1] * focal) / z + h * 0.5)))
        return out

    def render_world(self, frame):
        h, w, _ = frame.shape
        overlay = frame.copy()
        for obj in self.scene.values():
            obj.tick()
            R = rotation_matrix_3d(*obj.rotation)
            verts = obj.vertices @ R.T + obj.position
            pts_2d = self.project_3d(verts, w, h)

            for face in obj.faces:
                poly = np.array([pts_2d[i] for i in face], dtype=np.int32)
                c = obj.color
                cv2.fillPoly(overlay, [poly], (c[0]//4, c[1]//4, c[2]//4))

            for e1, e2 in obj.edges:
                cv2.line(frame, pts_2d[e1], pts_2d[e2], obj.color, 2, cv2.LINE_AA)

            c2d = self.project_3d([obj.position], w, h)[0]
            cv2.putText(frame, obj.name, (c2d[0] - 25, c2d[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        return frame

    def draw_hud(self, frame, predicted: str, conf: float):
        h, w, _ = frame.shape
        cv2.rectangle(frame, (15, 15), (680, 155), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (680, 155), (0, 230, 255), 1)

        cv2.putText(frame, f"PYTORCH GESTURE BRIDGE | Gesto Activo: {predicted} ({conf*100:.0f}%)",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 160), 2)

        for i, line in enumerate(self.interpreter.logs[-3:]):
            cv2.putText(frame, f"> {line}", (26, 65 + i * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 230, 245), 1)

        cv2.putText(frame, "Teclas [1-4]: Grabacion Rapida de Gestos  |  Vincular: APRENDER_GESTO PODER = EXPLOSION",
                    (26, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 210, 80), 1)

        # Barra inferior
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (12, 14, 20), -1)
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (0, 255, 140), 1)
        cursor = "_" if int(time.time() * 2.5) % 2 == 0 else " "
        cv2.putText(frame, f"BRIDGE-LANG >> {self.onscreen_input}{cursor}",
                    (28, h - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 140), 2)
        cv2.putText(frame, "Escribe: APRENDER_GESTO SALUDO = EXPLOSION  |  VINCULAR_GESTO GESTO_1 = ESCALAR TODOS 2.0",
                    (28, h - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (175, 195, 215), 1)

    def run(self):
        threading.Thread(target=self._console_worker, daemon=True).start()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened() and self.running:
            success, raw = cap.read()
            if not success: break

            frame = cv2.flip(raw, 1)
            h, w, _ = frame.shape
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            while not self.cmd_queue.empty():
                self.interpreter.execute(self.cmd_queue.get_nowait())

            predicted = "NINGUNO"
            confidence = 0.0

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                feature_vec = CanonicalHandGeometry.extract_invariant_features(hand_lms)

                if self.recording_frames > 0 and self.recording_name:
                    self.vault.learn_sample(self.recording_name, feature_vec)
                    self.recording_frames -= 1
                    if self.recording_frames == 0:
                        self.vault.save_vault()
                        self.interpreter.log(f"Gesto '{self.recording_name}' guardado en disco.")

                predicted, confidence, _ = self.vault.recognize(feature_vec)
                if predicted not in ("DESCONOCIDO", "SIN_GESTOS") and self.recording_frames == 0:
                    # ¡Disparar automáticamente el comando de NexusLang!
                    self.dispatch_gesture_command(predicted)
                else:
                    self.last_triggered_gesture = ""

            scene_frame = self.render_world(frame)
            self.draw_hud(scene_frame, predicted, confidence)
            cv2.imshow("NexusXR - Puentear PyTorch con NexusLang", scene_frame)

            key = cv2.waitKey(1)
            if key == -1: continue
            elif key == 27:
                self.vault.save_vault()
                self.running = False
                break
            elif key in (13, 10):
                if self.onscreen_input.strip():
                    self.cmd_queue.put(self.onscreen_input.strip())
                    self.onscreen_input = ""
            elif key in (8, 127):
                self.onscreen_input = self.onscreen_input[:-1]
            elif not self.onscreen_input and (key & 0xFF) == ord('1'):
                self.start_recording("GESTO_1")
            elif not self.onscreen_input and (key & 0xFF) == ord('2'):
                self.start_recording("GESTO_2")
            elif not self.onscreen_input and (key & 0xFF) == ord('3'):
                self.start_recording("GESTO_3")
            elif not self.onscreen_input and (key & 0xFF) == ord('4'):
                self.start_recording("GESTO_4")
            elif 32 <= (key & 0xFF) <= 126:
                self.onscreen_input += chr(key & 0xFF)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = NexusBridgeWorkspace()
    app.run()
