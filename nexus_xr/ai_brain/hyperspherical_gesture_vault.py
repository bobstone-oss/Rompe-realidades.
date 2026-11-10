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
# 1. EXTRACTOR GEOMÉTRICO INVARIANTE A POSICIÓN, ESCALA Y ROTACIÓN 3D
# =====================================================================

class CanonicalHandGeometry:
    """
    Transforma los 21 puntos crudos de MediaPipe al sistema de coordenadas
    intrínseco de la palma mediante ortonormalización de Gram-Schmidt.
    Genera un vector invariante de 78 dimensiones.
    """
    FINGERTIPS = [4, 8, 12, 16, 20]

    @staticmethod
    def extract_invariant_features(hand_landmarks) -> np.ndarray:
        # 1. Obtener matriz cruda (21, 3)
        if hasattr(hand_landmarks, "landmark"):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        elif isinstance(hand_landmarks, np.ndarray):
            pts = hand_landmarks.astype(np.float32)
        else:
            pts = np.array(hand_landmarks, dtype=np.float32)

        # 2. Traslación al origen de la muñeca (Punto 0)
        wrist = pts[0].copy()
        centered = pts - wrist

        # 3. Escala invariante basada en el tamaño físico de la palma
        palm_scale = np.linalg.norm(centered[9]) + 1e-6  # Distancia Muñeca -> Base dedo medio
        scaled = centered / palm_scale

        # 4. Base ortonormal de Gram-Schmidt anclada a la palma
        # Eje Y local: desde la muñeca (0) hacia la base del dedo medio (9)
        v_y = scaled[9].copy()
        y_axis = v_y / (np.linalg.norm(v_y) + 1e-6)

        # Vector auxiliar hacia la base del índice (5) desde el meñique (17)
        v_across = scaled[5] - scaled[17]
        # Componente ortogonal a y_axis -> Eje X local
        x_proj = v_across - np.dot(v_across, y_axis) * y_axis
        x_axis = x_proj / (np.linalg.norm(x_proj) + 1e-6)

        # Eje Z local: normal perpendicular a la palma (Producto Cruz)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= (np.linalg.norm(z_axis) + 1e-6)

        # Si es mano izquierda/derecha, alinear el signo del eje Z para invarianza de lateralidad
        rotation_basis = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3, 3)
        canonical_pts = scaled @ rotation_basis  # (21, 3) -> 63 valores

        # 5. Distancias par-a-par entre las 5 puntas de los dedos (10 valores)
        #    y distancias de las 5 puntas a la muñeca (5 valores) -> Total +15 = 78 dimensiones
        extra_features = []
        for i in range(len(CanonicalHandGeometry.FINGERTIPS)):
            tip_i = canonical_pts[CanonicalHandGeometry.FINGERTIPS[i]]
            extra_features.append(np.linalg.norm(tip_i))  # Distancia punta -> muñeca
            for j in range(i + 1, len(CanonicalHandGeometry.FINGERTIPS)):
                tip_j = canonical_pts[CanonicalHandGeometry.FINGERTIPS[j]]
                extra_features.append(np.linalg.norm(tip_i - tip_j))  # Distancia entre puntas

        feature_vector = np.concatenate([canonical_pts.flatten(), np.array(extra_features, dtype=np.float32)])
        return feature_vector  # Vector de 78 dimensiones


# =====================================================================
# 2. RED NEURONAL HIPERSFÉRICA EN PYTORCH (PROTOTYPICAL RESNET)
# =====================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return F.gelu(out + residual)


class HypersphericalGestureNet(nn.Module):
    """
    Transforma el descriptor geométrico de 78D en un embedding de 128D
    normalizado sobre la hipersfera unitaria (L2 = 1.0).
    """
    def __init__(self, input_dim: int = 78, hidden_dim: int = 256, embed_dim: int = 128):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        self.head = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.res_block1(h)
        h = self.res_block2(h)
        raw_embedding = self.head(h)
        # Normalización L2 para proyectar sobre la hipersfera unitaria
        return F.normalize(raw_embedding, p=2, dim=-1)


# =====================================================================
# 3. BÓVEDA DE MEMORIA PROTOTÍPICA Y ENTRENAMIENTO EN VIVO
# =====================================================================

class SelfLearningGestureVault:
    """
    Administra la memoria a corto y largo plazo de los gestos:
    - Aprende gestos nuevos al instante sin entrenamiento previo (One-Shot).
    - Re-entrena la red neuronal en segundo plano con Triplet Margin Loss.
    - Guarda y carga automáticamente los conocimientos en disco.
    """
    def __init__(self, vault_path: str = "nexus_gesture_vault.pt"):
        self.vault_path = vault_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = HypersphericalGestureNet(input_dim=78, hidden_dim=256, embed_dim=128).to(self.device)
        self.optimizer = optim.AdamW(self.encoder.parameters(), lr=1.5e-3, weight_decay=1e-4)
        self.triplet_loss = nn.TripletMarginLoss(margin=0.45, p=2)

        # Diccionario de gestos aprendidos:
        # nombre_gesto -> {"prototype": Tensor(128), "samples": deque de vectores 78D, "count": int}
        self.gestures: Dict[str, Dict] = {}
        self.last_loss: float = 0.0
        self.training_steps: int = 0

        self.load_vault()

    def encode_features(self, feature_vec: np.ndarray) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            t = torch.from_numpy(feature_vec).float().unsqueeze(0).to(self.device)
            emb = self.encoder(t).squeeze(0)
        return emb

    def learn_gesture_sample(self, gesture_name: str, feature_vec: np.ndarray):
        """
        Registra una muestra en vivo de un gesto (nuevo o existente),
        actualiza su centroide prototípico y ejecuta un paso de optimización neuronal.
        """
        emb = self.encode_features(feature_vec)

        if gesture_name not in self.gestures:
            self.gestures[gesture_name] = {
                "prototype": emb.clone(),
                "samples": deque(maxlen=120),
                "count": 0
            }
            print(f"[IA MEMORIA] Nuevo gesto creado en la bóveda: '{gesture_name}'")

        entry = self.gestures[gesture_name]
        entry["samples"].append(feature_vec.copy())
        entry["count"] += 1

        # Actualización exponencial móvil (EMA) del prototipo en la hipersfera
        alpha = 0.25 if entry["count"] > 5 else (1.0 / entry["count"])
        updated_proto = (1.0 - alpha) * entry["prototype"] + alpha * emb
        entry["prototype"] = F.normalize(updated_proto, p=2, dim=-1)

        # Si ya existen al menos 2 gestos distintos, entrenar la red para separarlos aún más
        if len(self.gestures) >= 2:
            self._online_contrastive_step()

    def _online_contrastive_step(self, batch_size: int = 16):
        """
        Toma muestras de la memoria episódica (Ancla, Positivo, Negativo),
        ajusta los pesos de PyTorch en milisegundos y recalcula los prototipos.
        """
        eligible = [k for k, v in self.gestures.items() if len(v["samples"]) >= 2]
        if len(eligible) < 2:
            return

        self.encoder.train()
        anchors, positives, negatives = [], [], []

        for _ in range(batch_size):
            pos_class, neg_class = random.sample(eligible, 2)
            a_sample, p_sample = random.sample(list(self.gestures[pos_class]["samples"]), 2)
            n_sample = random.choice(list(self.gestures[neg_class]["samples"]))

            anchors.append(a_sample)
            positives.append(p_sample)
            negatives.append(n_sample)

        t_a = torch.from_numpy(np.array(anchors, dtype=np.float32)).to(self.device)
        t_p = torch.from_numpy(np.array(positives, dtype=np.float32)).to(self.device)
        t_n = torch.from_numpy(np.array(negatives, dtype=np.float32)).to(self.device)

        emb_a = self.encoder(t_a)
        emb_p = self.encoder(t_p)
        emb_n = self.encoder(t_n)

        loss = self.triplet_loss(emb_a, emb_p, emb_n)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_loss = float(loss.item())
        self.training_steps += 1

        # Actualizar todos los prototipos con la red recién refinada
        self.encoder.eval()
        with torch.no_grad():
            for name, data in self.gestures.items():
                if len(data["samples"]) > 0:
                    mat = torch.from_numpy(np.array(list(data["samples"]), dtype=np.float32)).to(self.device)
                    all_embs = self.encoder(mat)
                    mean_proto = torch.mean(all_embs, dim=0)
                    data["prototype"] = F.normalize(mean_proto, p=2, dim=-1)

    def recognize(self, feature_vec: np.ndarray, threshold: float = 0.76) -> Tuple[str, float, Dict[str, float]]:
        """
        Compara el vector actual de la mano contra todos los prototipos en memoria
        usando similitud coseno en la hipersfera.
        """
        if not self.gestures:
            return "SIN_GESTOS_EN_MEMORIA", 0.0, {}

        emb = self.encode_features(feature_vec)
        scores: Dict[str, float] = {}

        for name, data in self.gestures.items():
            # Producto punto de vectores L2-normalizados = Similitud Coseno [-1.0 a 1.0]
            cos_sim = float(torch.dot(emb, data["prototype"]).item())
            scores[name] = cos_sim

        best_gesture = max(scores, key=scores.get)
        best_score = scores[best_gesture]

        if best_score < threshold:
            return "GESTO_DESCONOCIDO", best_score, scores
        return best_gesture, best_score, scores

    def save_vault(self):
        """Guarda pesos neuronales, prototipos y memoria episódica en disco."""
        serializable_gestures = {}
        for name, data in self.gestures.items():
            serializable_gestures[name] = {
                "prototype": data["prototype"].cpu(),
                "samples": list(data["samples"]),
                "count": data["count"]
            }

        checkpoint = {
            "model_state": self.encoder.state_dict(),
            "gestures": serializable_gestures,
            "training_steps": self.training_steps
        }
        torch.save(checkpoint, self.vault_path)
        print(f"[PERSISTENCIA] Bóveda guardada en '{self.vault_path}' ({len(self.gestures)} gestos).")

    def load_vault(self):
        """Carga automáticamente los gestos aprendidos en sesiones anteriores."""
        if not os.path.exists(self.vault_path):
            return
        try:
            checkpoint = torch.load(self.vault_path, map_location=self.device, weights_only=False)
            self.encoder.load_state_dict(checkpoint["model_state"])
            self.training_steps = checkpoint.get("training_steps", 0)

            for name, data in checkpoint["gestures"].items():
                self.gestures[name] = {
                    "prototype": data["prototype"].to(self.device),
                    "samples": deque(data["samples"], maxlen=120),
                    "count": data["count"]
                }
            print(f"[PERSISTENCIA] Cargados {len(self.gestures)} gestos previos desde '{self.vault_path}'.")
        except Exception as e:
            print(f"[AVISO] No se pudo cargar la bóveda previa: {e}")


# =====================================================================
# 4. INTERFAZ INTERACTIVA DE APRENDIZAJE EN VIVO (MEDIAPIPE + HUD)
# =====================================================================

class LiveGestureLearningSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.vault = SelfLearningGestureVault()

        # Modo Auto-Descubrimiento: si el usuario mantiene un gesto desconocido quieto, lo aprende solo
        self.auto_discover = False
        self.unknown_hold_frames = 0
        self.auto_gesture_counter = len(self.vault.gestures) + 1

        # Estado de grabación activa por teclado
        self.recording_target: Optional[str] = None
        self.recording_frames_left: int = 0

    def trigger_teaching(self, gesture_name: str, frames: int = 22):
        self.recording_target = gesture_name
        self.recording_frames_left = frames

    def draw_neural_hud(self, frame, predicted_name: str, confidence: float, all_scores: Dict[str, float]):
        h, w, _ = frame.shape

        # Panel superior de estado neuronal
        cv2.rectangle(frame, (15, 15), (620, 155), (18, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (620, 155), (0, 220, 255), 1)

        status_color = (0, 255, 128) if predicted_name not in ("GESTO_DESCONOCIDO", "SIN_GESTOS_EN_MEMORIA") else (0, 180, 255)
        cv2.putText(frame, f"DETECCION IA: {predicted_name} ({confidence*100:.1f}%)",
                    (28, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        cv2.putText(frame, f"Gestos en Memoria: {len(self.vault.gestures)} | Pasos Optimizacion PyTorch: {self.vault.training_steps} | Loss: {self.vault.last_loss:.4f}",
                    (28, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1)

        auto_txt = "ACTIVO" if self.auto_discover else "INACTIVO"
        cv2.putText(frame, f"Teclas [1-9]: Ensenar Gesto_1..9 | [A]: Auto-Descubrir ({auto_txt}) | [S]: Guardar | [C]: Borrar",
                    (28, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 230, 255), 1)

        if self.recording_frames_left > 0:
            cv2.putText(frame, f">> APRENDIENDO '{self.recording_target}' EN VIVO ({self.recording_frames_left} muestras restantes)...",
                        (28, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 255, 80), 2)
        else:
            cv2.putText(frame, "Muestra cualquier posicion de mano y presiona un numero (1-9) para ensenarla al instante.",
                        (28, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

        # Barras laterales de similitud coseno en tiempo real para cada gesto en memoria
        bar_x, bar_y = 20, 185
        for idx, (g_name, score) in enumerate(sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:8]):
            norm_score = max(0.0, min(1.0, score))
            bar_w = int(norm_score * 200)
            y_pos = bar_y + idx * 34

            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + 200, y_pos + 22), (35, 35, 35), -1)
            bar_color = (0, 255, 140) if g_name == predicted_name else (255, 170, 0)
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_w, y_pos + 22), bar_color, -1)
            cv2.putText(frame, f"{g_name}: {score*100:.1f}% ({self.vault.gestures[g_name]['count']} m.)",
                        (bar_x + 8, y_pos + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("======================================================================")
        print(" NEXUS-XR: CEREBRO PYTORCH DE APRENDIZAJE DE GESTOS EN VIVO")
        print(" - Pon tu mano frente a la cámara en cualquier postura nueva.")
        print(" - Presiona las teclas '1' al '9' para enseñar ese gesto al instante.")
        print(" - Presiona 'a' para activar el Auto-Descubrimiento autónomo.")
        print(" - Presiona 's' para guardar la bóveda en disco | 'q' para salir.")
        print("======================================================================")

        while cap.isOpened():
            success, raw_frame = cap.read()
            if not success:
                break

            frame = cv2.flip(raw_frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            predicted_name = "MANO_NO_VISIBLE"
            confidence = 0.0
            all_scores = {}

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                # 1. Extraer el vector geométrico invariante de 78 dimensiones
                invariant_vec = CanonicalHandGeometry.extract_invariant_features(hand_lms)

                # 2. Si estamos en modo de enseñanza activa, alimentar la red en vivo
                if self.recording_frames_left > 0 and self.recording_target:
                    self.vault.learn_gesture_sample(self.recording_target, invariant_vec)
                    self.recording_frames_left -= 1
                    if self.recording_frames_left == 0:
                        self.vault.save_vault()

                # 3. Reconocer el gesto actual comparando en la hipersfera
                predicted_name, confidence, all_scores = self.vault.recognize(invariant_vec)

                # 4. Modo Auto-Descubrimiento autónomo:
                #    Si la mano hace un gesto desconocido durante 35 fotogramas seguidos, lo bautiza y aprende solo
                if self.auto_discover and self.recording_frames_left == 0:
                    if predicted_name in ("GESTO_DESCONOCIDO", "SIN_GESTOS_EN_MEMORIA"):
                        self.unknown_hold_frames += 1
                        if self.unknown_hold_frames >= 35:
                            new_auto_name = f"AUTO_GESTO_{self.auto_gesture_counter}"
                            self.auto_gesture_counter += 1
                            self.unknown_hold_frames = 0
                            self.trigger_teaching(new_auto_name, frames=20)
                    else:
                        self.unknown_hold_frames = 0

            self.draw_neural_hud(frame, predicted_name, confidence, all_scores)
            cv2.imshow("NexusXR - IA Auto-Adaptativa de Gestos (PyTorch)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.vault.save_vault()
                break
            elif ord('1') <= key <= ord('9'):
                slot_num = chr(key)
                self.trigger_teaching(f"GESTO_COMANDO_{slot_num}", frames=22)
            elif key == ord('a'):
                self.auto_discover = not self.auto_discover
                self.unknown_hold_frames = 0
            elif key == ord('s'):
                self.vault.save_vault()
            elif key == ord('c'):
                self.vault.gestures.clear()
                print("[IA MEMORIA] Bóveda de gestos reiniciada en RAM.")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    system = LiveGestureLearningSystem()
    system.run()
