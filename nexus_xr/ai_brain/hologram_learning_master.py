import os
import time
import math
import random
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
# 1. MATEMÁTICA Y GEOMETRÍA 3D DEL HOLOGRAMA
# =====================================================================

def rotation_matrix_3d(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


class SpatialHologram3D:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.target_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.4, 0.6, 0.1], dtype=np.float32)
        self.target_rotation = np.array([0.4, 0.6, 0.1], dtype=np.float32)
        self.scale = 1.0
        self.target_scale = 1.0

        # Vértices del Cubo Holográfico Exterior
        self.cube_vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ], dtype=np.float32) * 0.55

        self.cube_faces = [
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)
        ]

        # Núcleo de Energía Interior
        self.core_vertices = np.array([
            [0.45, 0, 0], [-0.45, 0, 0],
            [0, 0.45, 0], [0, -0.45, 0],
            [0, 0, 0.45], [0, 0, -0.45]
        ], dtype=np.float32)
        self.core_edges = [
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 4), (4, 3), (3, 5), (5, 2)
        ]

    def update_physics(self, smoothing=0.3):
        self.position += (self.target_position - self.position) * smoothing
        self.rotation += (self.target_rotation - self.rotation) * smoothing
        self.scale += (self.target_scale - self.scale) * smoothing
        self.scale = float(np.clip(self.scale, 0.3, 3.5))


# =====================================================================
# 2. EXTRACTOR GEOMÉTRICO INVARIANTE (78D)
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
# 3. RED NEURONAL HIPERSFÉRICA EN PYTORCH (IA DE APRENDIZAJE)
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


class AdaptiveGestureVault:
    def __init__(self, vault_path: str = "nexus_master_vault.pt"):
        self.vault_path = vault_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = HypersphericalGestureNet().to(self.device)
        self.optimizer = optim.AdamW(self.encoder.parameters(), lr=1.5e-3)
        self.triplet_loss = nn.TripletMarginLoss(margin=0.45, p=2)
        self.gestures: Dict[str, Dict] = {}
        self.last_loss = 0.0
        self.load_vault()

    def encode(self, feature_vec: np.ndarray) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            t = torch.from_numpy(feature_vec).float().unsqueeze(0).to(self.device)
            return self.encoder(t).squeeze(0)

    def learn_sample(self, name: str, feature_vec: np.ndarray):
        emb = self.encode(feature_vec)
        if name not in self.gestures:
            self.gestures[name] = {"prototype": emb.clone(), "samples": deque(maxlen=100), "count": 0}
            print(f"[IA] Nuevo gesto aprendido: '{name}'")

        entry = self.gestures[name]
        entry["samples"].append(feature_vec.copy())
        entry["count"] += 1
        alpha = 0.25 if entry["count"] > 5 else (1.0 / entry["count"])
        entry["prototype"] = F.normalize((1.0 - alpha) * entry["prototype"] + alpha * emb, p=2, dim=-1)

        if len(self.gestures) >= 2:
            self._contrastive_step()

    def _contrastive_step(self, batch_size=16):
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

    def recognize(self, feature_vec: np.ndarray, threshold=0.77) -> Tuple[str, float, Dict[str, float]]:
        if not self.gestures:
            return "SIN_GESTOS", 0.0, {}
        emb = self.encode(feature_vec)
        scores = {name: float(torch.dot(emb, data["prototype"]).item()) for name, data in self.gestures.items()}
        best = max(scores, key=scores.get)
        score = scores[best]
        return (best, score, scores) if score >= threshold else ("DESCONOCIDO", score, scores)

    def save_vault(self):
        torch.save({"model": self.encoder.state_dict(), "gestures": {k: {"proto": v["prototype"].cpu(), "samples": list(v["samples"]), "count": v["count"]} for k, v in self.gestures.items()}}, self.vault_path)
        print("[PERSISTENCIA] Bóveda guardada en disco.")

    def load_vault(self):
        if not os.path.exists(self.vault_path):
            return
        try:
            ckpt = torch.load(self.vault_path, map_location=self.device, weights_only=False)
            self.encoder.load_state_dict(ckpt["model"])
            for k, v in ckpt["gestures"].items():
                self.gestures[k] = {"prototype": v["proto"].to(self.device), "samples": deque(v["samples"], maxlen=100), "count": v["count"]}
            print(f"[PERSISTENCIA] Cargados {len(self.gestures)} gestos previos.")
        except Exception as e:
            print(f"[AVISO] Error cargando bóveda: {e}")


# =====================================================================
# 4. MOTOR MAESTRO INTEGRADO (IA + HOLOGRAMA 3D + AR/VR)
# =====================================================================

class NexusMasterEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.hologram = SpatialHologram3D()
        self.vault = AdaptiveGestureVault()
        self.vr_blend = 0.3

        # Control de grabación de gestos en vivo
        self.teaching_target: Optional[str] = None
        self.teaching_frames_left = 0

    def trigger_teaching(self, gesture_name: str):
        self.teaching_target = gesture_name
        self.teaching_frames_left = 25
        print(f"[ENSEÑANZA] Grabando muestras para: '{gesture_name}'...")

    def execute_command_by_gesture(self, gesture_name: str, hand_landmarks):
        """Asocia dinámicamente gestos aprendidos por la IA a acciones del Holograma 3D"""
        pts = hand_landmarks.landmark
        palm_center = np.array([pts[0].x, pts[0].y, pts[0].z])

        # Mapeo de comandos basados en el nombre del gesto aprendido
        if "EXPANDIR" in gesture_name or gesture_name == "GESTO_1":
            self.hologram.target_scale += 0.04
        elif "COMPRIMIR" in gesture_name or gesture_name == "GESTO_2":
            self.hologram.target_scale -= 0.04
        elif "GIRAR" in gesture_name or gesture_name == "GESTO_3":
            self.hologram.target_rotation[1] += 0.12
        elif "MOVER" in gesture_name or gesture_name == "GESTO_4":
            # Seguir la posición 3D de la palma en pantalla
            self.hologram.target_position[0] = (palm_center[0] - 0.5) * 3.0
            self.hologram.target_position[1] = (palm_center[1] - 0.5) * 2.0
        else:
            # Gesto genérico: rota suavemente según la inclinación de la mano
            self.hologram.target_rotation[0] = (palm_center[1] - 0.5) * 2.5

    def project_3d_to_2d(self, points_3d, w, h, focal=650.0, cam_dist=3.6):
        projected = []
        for pt in points_3d:
            z = cam_dist + pt[2]
            if z < 0.3: z = 0.3
            px = int((pt[0] * focal) / z + (w * 0.5))
            py = int((pt[1] * focal) / z + (h * 0.5))
            projected.append((px, py))
        return projected

    def render_environment(self, frame):
        h, w, _ = frame.shape
        vr_world = np.zeros_like(frame)
        horizon = int(h * 0.52)
        for i in range(-10, 11):
            cv2.line(vr_world, (int(w*0.5 + i*120), h), (int(w*0.5 + i*20), horizon), (50, 30, 10), 1)
        for d in range(1, 8):
            y = int(horizon + (d**1.8) * 7)
            if y < h: cv2.line(vr_world, (0, y), (w, y), (60, 35, 10), 1)
        return cv2.addWeighted(frame, 1.0 - self.vr_blend, vr_world, self.vr_blend, 0)

    def draw_hologram(self, frame):
        h, w, _ = frame.shape
        R = rotation_matrix_3d(*self.hologram.rotation)
        transformed = (self.hologram.cube_vertices * self.hologram.scale) @ R.T + self.hologram.position
        pts_2d = self.project_3d_to_2d(transformed, w, h)

        color = (0, 255, 160) if self.teaching_frames_left == 0 else (255, 0, 255)
        overlay = frame.copy()
        for face in self.hologram.cube_faces:
            poly = np.array([pts_2d[v] for v in face], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (color[0]//4, color[1]//4, color[2]//4))
            cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Núcleo interior
        R_core = rotation_matrix_3d(-self.hologram.rotation[0]*1.5, -self.hologram.rotation[1]*1.5, self.hologram.rotation[2])
        core_trans = (self.hologram.core_vertices * self.hologram.scale) @ R_core.T + self.hologram.position
        core_2d = self.project_3d_to_2d(core_trans, w, h)
        for e1, e2 in self.hologram.core_edges:
            cv2.line(frame, core_2d[e1], core_2d[e2], (255, 255, 255), 1, cv2.LINE_AA)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("======================================================================")
        print(" NEXUS-MASTER: IA + MOTOR HOLOGRÁFICO 3D + REALIDAD MIXTA")
        print(" - Pon tu mano frente a la cámara y presiona [1], [2], [3] o [4]")
        print("   para enseñar un gesto nuevo y asociarlo al holograma.")
        print(" - Presiona [m] para alternar AR/VR | [s] Guardar bóveda | [q] Salir")
        print("======================================================================")

        while cap.isOpened():
            success, raw = cap.read()
            if not success: break

            frame = cv2.flip(raw, 1)
            h, w, _ = frame.shape
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            predicted = "NINGUNA_MANO"
            confidence = 0.0
            scores = {}

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                invariant_vec = CanonicalHandGeometry.extract_invariant_features(hand_lms)

                # Si el usuario presionó una tecla para enseñar este gesto
                if self.teaching_frames_left > 0 and self.teaching_target:
                    self.vault.learn_sample(self.teaching_target, invariant_vec)
                    self.teaching_frames_left -= 1
                    if self.teaching_frames_left == 0:
                        self.vault.save_vault()

                # Reconocer gesto en tiempo real
                predicted, confidence, scores = self.vault.recognize(invariant_vec)

                # Ejecutar comando en el holograma según el gesto reconocido
                if predicted != "DESCONOCIDO":
                    self.execute_command_by_gesture(predicted, hand_lms)

            # Actualizar físicas del holograma
            self.hologram.update_physics()

            # Renderizar escena
            scene = self.render_environment(frame)
            self.draw_hologram(scene)

            # HUD de Telemetría e IA
            cv2.rectangle(scene, (15, 15), (560, 130), (20, 20, 25), -1)
            cv2.rectangle(scene, (15, 15), (560, 130), (0, 220, 255), 1)
            cv2.putText(scene, f"IA GESTO: {predicted} ({confidence*100:.1f}%)", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 150), 2)
            cv2.putText(scene, f"Holograma Escala: {self.hologram.scale:.2f}x | Gestos en Memoria: {len(self.vault.gestures)}", (28, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            teach_msg = f">> GRABANDO '{self.teaching_target}'..." if self.teaching_frames_left > 0 else "Teclas [1-4]: Ensenar Gesto | [M]: AR/VR | [S]: Guardar"
            cv2.putText(scene, teach_msg, (28, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 200, 100), 1)

            cv2.imshow("NexusXR - Motor Maestro Espacial", scene)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.vault.save_vault()
                break
            elif key == ord('1'): self.trigger_teaching("EXPANDIR_1")
            elif key == ord('2'): self.trigger_teaching("COMPRIMIR_2")
            elif key == ord('3'): self.trigger_teaching("GIRAR_3")
            elif key == ord('4'): self.trigger_teaching("MOVER_4")
            elif key == ord('m'): self.vr_blend = (self.vr_blend + 0.4) if self.vr_blend < 0.8 else 0.0
            elif key == ord('s'): self.vault.save_vault()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    master = NexusMasterEngine()
    master.run()
