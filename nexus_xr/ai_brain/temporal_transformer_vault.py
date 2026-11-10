import os
import math
import time
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
# 1. EXTRACTOR ESPACIO-TEMPORAL INVARIANTE (POSTURA 78D + TRAZO 6D = 84D)
# =====================================================================

class SpatioTemporalHandExtractor:
    FINGERTIPS = [4, 8, 12, 16, 20]

    @staticmethod
    def extract_posture_78d(hand_landmarks) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve:
        1) El descriptor invariante de postura de 78 dimensiones (Gram-Schmidt).
        2) La coordenada 3D cruda de la punta del dedo índice (landmark 8) para trazar en el aire.
        """
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        index_tip_3d = pts[8].copy()

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
        for i in range(len(SpatioTemporalHandExtractor.FINGERTIPS)):
            tip_i = canonical_pts[SpatioTemporalHandExtractor.FINGERTIPS[i]]
            extra_features.append(np.linalg.norm(tip_i))
            for j in range(i + 1, len(SpatioTemporalHandExtractor.FINGERTIPS)):
                tip_j = canonical_pts[SpatioTemporalHandExtractor.FINGERTIPS[j]]
                extra_features.append(np.linalg.norm(tip_i - tip_j))

        posture_78d = np.concatenate([canonical_pts.flatten(), np.array(extra_features, dtype=np.float32)])
        return posture_78d, index_tip_3d

    @staticmethod
    def normalize_sequence(posture_seq: List[np.ndarray], tip_seq: List[np.ndarray], target_len: int = 32) -> np.ndarray:
        """
        Convierte una secuencia de longitud variable (ej. 18 o 55 fotogramas)
        en un tensor canónico de forma (target_len=32, 84) invariante a la velocidad,
        al tamaño del dibujo y a la posición en la pantalla.
        """
        n_frames = len(posture_seq)
        postures = np.array(posture_seq, dtype=np.float32)  # (N, 78)
        tips = np.array(tip_seq, dtype=np.float32)          # (N, 3)

        # 1. Interpolación temporal uniforme a exactamente 32 pasos
        orig_t = np.linspace(0.0, 1.0, n_frames)
        new_t = np.linspace(0.0, 1.0, target_len)

        resampled_postures = np.zeros((target_len, 78), dtype=np.float32)
        for d in range(78):
            resampled_postures[:, d] = np.interp(new_t, orig_t, postures[:, d])

        resampled_tips = np.zeros((target_len, 3), dtype=np.float32)
        for d in range(3):
            resampled_tips[:, d] = np.interp(new_t, orig_t, tips[:, d])

        # 2. Centrar y escalar la trayectoria 3D del trazo (Invarianza de posición y tamaño del dibujo)
        centroid = np.mean(resampled_tips, axis=0, keepdims=True)
        centered_tips = resampled_tips - centroid
        max_span = np.max(np.linalg.norm(centered_tips, axis=1))
        # Si es un gesto dinámico (hay desplazamiento), normalizar su radio; si es estático, mantener cerca de 0
        if max_span > 0.035:
            norm_tips = centered_tips / (max_span + 1e-6)
        else:
            norm_tips = centered_tips * 2.0

        # 3. Vector de velocidad instantánea (derivada temporal de la trayectoria)
        velocities = np.gradient(norm_tips, axis=0) * 3.0  # (32, 3)

        # Concatenar: 78D (Postura) + 3D (Posición del trazo) + 3D (Velocidad) = 84 Dimensiones por fotograma
        sequence_84d = np.concatenate([resampled_postures, norm_tips, velocities], axis=1)
        return sequence_84d.astype(np.float32)


# =====================================================================
# 2. ARQUITECTURA TEMPORAL TRANSFORMER EN PYTORCH
# =====================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Codificación posicional matemática para que el Transformer conozca el orden del tiempo."""
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TemporalGestureTransformer(nn.Module):
    """
    Recibe una secuencia de forma (Batch, SeqLen=32, Features=84),
    aplica Self-Attention Multi-Cabeza con un token [CLS] y proyecta
    el gesto entero en una hipersfera unitaria de 128 dimensiones.
    """
    def __init__(
        self,
        input_dim: int = 84,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        embed_dim: int = 128,
        seq_len: int = 32
    ):
        super().__init__()
        self.frame_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        # Token [CLS] aprendible que resume toda la trayectoria temporal
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=seq_len + 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, SeqLen=32, 84)
        batch_size = x.size(0)
        h = self.frame_proj(x)  # (B, 32, d_model)

        # Anteponer el token [CLS] al inicio de la secuencia -> (B, 33, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls_tokens, h], dim=1)
        h = self.pos_encoder(h)

        # Auto-Atención sobre todos los instantes de tiempo
        attended = self.transformer(h)

        # Extraer la salida acumulada en el token [CLS] (índice 0)
        cls_out = attended[:, 0, :]
        embedding = self.head(cls_out)

        # Normalización L2 hacia la hipersfera unitaria
        return F.normalize(embedding, p=2, dim=-1)


# =====================================================================
# 3. BÓVEDA NEURONAL SECUENCIAL CON AUTO-APRENDIZAJE CONTRASTIVO
# =====================================================================

class TemporalGestureVault:
    def __init__(self, vault_path: str = "nexus_temporal_transformer_vault.pt"):
        self.vault_path = vault_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TemporalGestureTransformer().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1.2e-3, weight_decay=1e-4)
        self.triplet_loss = nn.TripletMarginLoss(margin=0.42, p=2)

        # Diccionario: nombre_gesto -> {"prototype": Tensor(128), "sequences": deque de arrays (32, 84), "count": int}
        self.gestures: Dict[str, Dict] = {}
        self.last_loss: float = 0.0
        self.training_steps: int = 0
        self.load_vault()

    def _augment_sequence(self, seq: np.ndarray) -> np.ndarray:
        """
        Genera variaciones sintéticas (ruido gaussiano leve y escalado elástico)
        para que con una sola demostración en vivo (One-Shot) el Transformer
        tenga suficientes muestras para entrenar con Triplet Loss.
        """
        noise = np.random.normal(0.0, 0.018, size=seq.shape).astype(np.float32)
        scale_jitter = np.random.uniform(0.92, 1.08)
        aug = seq.copy()
        aug[:, 78:] *= scale_jitter  # Escalar ligeramente la parte de trayectoria
        return aug + noise

    def encode_sequence(self, seq_32x84: np.ndarray) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(seq_32x84).float().unsqueeze(0).to(self.device)  # (1, 32, 84)
            return self.model(t).squeeze(0)

    def learn_dynamic_gesture(self, name: str, seq_32x84: np.ndarray):
        """
        Asimila un gesto en movimiento o estático en el acto, genera 4 variaciones
        de entrenamiento y ejecuta pasos de optimización en el Transformer.
        """
        name = name.upper()
        emb = self.encode_sequence(seq_32x84)

        if name not in self.gestures:
            self.gestures[name] = {
                "prototype": emb.clone(),
                "sequences": deque(maxlen=60),
                "count": 0
            }
            print(f"[TRANSFORMER IA] Nuevo gesto temporal registrado: '{name}'")

        entry = self.gestures[name]
        entry["sequences"].append(seq_32x84.copy())
        # Añadir 3 variaciones sintéticas para acelerar el aprendizaje One-Shot
        for _ in range(3):
            entry["sequences"].append(self._augment_sequence(seq_32x84))
        entry["count"] += 1

        alpha = 0.30 if entry["count"] > 3 else (1.0 / entry["count"])
        entry["prototype"] = F.normalize((1.0 - alpha) * entry["prototype"] + alpha * emb, p=2, dim=-1)

        if len(self.gestures) >= 2:
            for _ in range(3):
                self._online_transformer_step()

    def _online_transformer_step(self, batch_size: int = 12):
        eligible = [k for k, v in self.gestures.items() if len(v["sequences"]) >= 2]
        if len(eligible) < 2:
            return

        self.model.train()
        anchors, positives, negatives = [], [], []
        for _ in range(batch_size):
            pos_c, neg_c = random.sample(eligible, 2)
            a, p = random.sample(list(self.gestures[pos_c]["sequences"]), 2)
            n = random.choice(list(self.gestures[neg_c]["sequences"]))
            anchors.append(a); positives.append(p); negatives.append(n)

        t_a = torch.from_numpy(np.array(anchors, dtype=np.float32)).to(self.device)
        t_p = torch.from_numpy(np.array(positives, dtype=np.float32)).to(self.device)
        t_n = torch.from_numpy(np.array(negatives, dtype=np.float32)).to(self.device)

        loss = self.triplet_loss(self.model(t_a), self.model(t_p), self.model(t_n))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_loss = float(loss.item())
        self.training_steps += 1

        # Recalcular los prototipos en la hipersfera tras actualizar las capas de atención
        self.model.eval()
        with torch.no_grad():
            for g_name, data in self.gestures.items():
                if data["sequences"]:
                    mat = torch.from_numpy(np.array(list(data["sequences"]), dtype=np.float32)).to(self.device)
                    all_embs = self.model(mat)
                    data["prototype"] = F.normalize(torch.mean(all_embs, dim=0), p=2, dim=-1)

    def recognize_sequence(self, seq_32x84: np.ndarray, threshold: float = 0.74) -> Tuple[str, float, Dict[str, float]]:
        if not self.gestures:
            return "SIN_GESTOS_EN_MEMORIA", 0.0, {}

        emb = self.encode_sequence(seq_32x84)
        scores = {name: float(torch.dot(emb, data["prototype"]).item()) for name, data in self.gestures.items()}
        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]

        if best_score < threshold:
            return "TRAZO_DESCONOCIDO", best_score, scores
        return best_name, best_score, scores

    def save_vault(self):
        payload = {
            "model_state": self.model.state_dict(),
            "steps": self.training_steps,
            "gestures": {
                k: {"proto": v["prototype"].cpu(), "sequences": list(v["sequences"]), "count": v["count"]}
                for k, v in self.gestures.items()
            }
        }
        torch.save(payload, self.vault_path)
        print(f"[PERSISTENCIA] Transformer y trazos guardados en '{self.vault_path}'.")

    def load_vault(self):
        if not os.path.exists(self.vault_path):
            return
        try:
            ckpt = torch.load(self.vault_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            self.training_steps = ckpt.get("steps", 0)
            for k, v in ckpt["gestures"].items():
                self.gestures[k] = {
                    "prototype": v["proto"].to(self.device),
                    "sequences": deque(v["sequences"], maxlen=60),
                    "count": v["count"]
                }
            print(f"[PERSISTENCIA] Cargados {len(self.gestures)} gestos temporales desde '{self.vault_path}'.")
        except Exception as e:
            print(f"[AVISO] No se pudo cargar bóveda temporal previa: {e}")


# =====================================================================
# 4. MOTOR INTERACTIVO DE TRAZADO EN EL AIRE + TEMPORAL TRANSFORMER
# =====================================================================

class LiveTemporalGestureEngine:
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
        self.vault = TemporalGestureVault()

        # Ventana deslizante continua para evaluación en tiempo real (32 fotogramas)
        self.rolling_postures: deque = deque(maxlen=32)
        self.rolling_tips_3d: deque = deque(maxlen=32)

        # Estela luminosa 2D en pantalla del dedo índice
        self.screen_trail_2d: deque = deque(maxlen=42)

        # Estado de grabación de un trazo nuevo (36 fotogramas ~ 1.1 segundos de dibujo)
        self.recording_target: Optional[str] = None
        self.recording_frames_total = 36
        self.recorded_postures: List[np.ndarray] = []
        self.recorded_tips: List[np.ndarray] = []

    def start_recording_gesture(self, gesture_name: str):
        self.recording_target = gesture_name
        self.recorded_postures.clear()
        self.recorded_tips.clear()
        self.screen_trail_2d.clear()
        print(f"[GRABANDO TRAZO] Dibuja en el aire el gesto '{gesture_name}' ahora...")

    def draw_neon_air_trail(self, frame):
        """Dibuja la estela holográfica que deja la punta de tu dedo índice en el aire."""
        pts = list(self.screen_trail_2d)
        if len(pts) < 2:
            return

        for i in range(1, len(pts)):
            thickness = max(1, int((i / len(pts)) * 7))
            # Color magenta cuando está aprendiendo un símbolo nuevo, cian cuando reconoce en vivo
            color = (255, 80, 255) if self.recording_target else (255, 230, 0)
            cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

        # Punto brillante en la punta actual del dedo
        cv2.circle(frame, pts[-1], 9, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_transformer_hud(self, frame, predicted: str, conf: float, scores: Dict[str, float]):
        h, w, _ = frame.shape

        # Panel Superior: Telemetría del Temporal Transformer
        cv2.rectangle(frame, (15, 15), (675, 160), (16, 18, 26), -1)
        border_col = (255, 80, 255) if self.recording_target else (0, 230, 255)
        cv2.rectangle(frame, (15, 15), (675, 160), border_col, 1)

        status_col = (0, 255, 140) if predicted not in ("TRAZO_DESCONOCIDO", "SIN_GESTOS_EN_MEMORIA", "ESPERANDO_SECUENCIA") else (0, 190, 255)
        cv2.putText(frame, f"TEMPORAL TRANSFORMER: {predicted} ({conf*100:.1f}%)",
                    (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_col, 2)

        cv2.putText(frame, f"Gestos Dinamicos/Estaticos: {len(self.vault.gestures)} | Pasos Self-Attention: {self.vault.training_steps} | Loss: {self.vault.last_loss:.4f}",
                    (28, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (215, 225, 235), 1)

        cv2.putText(frame, "Teclas [1]: CIRCULO  [2]: TRIANGULO  [3]: RAYO_Z  [4]: ONDA  [5-9]: Gesto Libre  |  [S]: Guardar  [C]: Borrar",
                    (28, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (170, 230, 255), 1)

        if self.recording_target:
            progress = len(self.recorded_postures)
            pct = int((progress / self.recording_frames_total) * 100)
            cv2.putText(frame, f">> DIBUJA EN EL AIRE '{self.recording_target}': {pct}% ({progress}/{self.recording_frames_total} frames)",
                        (28, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 110, 255), 2)
        else:
            cv2.putText(frame, "Mueve tu dedo indice dibujando figuras en el aire. Presiona [1-9] para ensenar un trazo nuevo.",
                        (28, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1)

        # Barras de similitud en la hipersfera temporal
        bx, by = 20, 185
        for idx, (g_name, sc) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:7]):
            norm_s = max(0.0, min(1.0, sc))
            bw = int(norm_s * 210)
            y = by + idx * 32
            cv2.rectangle(frame, (bx, y), (bx + 210, y + 22), (32, 32, 38), -1)
            b_col = (0, 255, 140) if g_name == predicted else (255, 165, 0)
            cv2.rectangle(frame, (bx, y), (bx + bw, y + 22), b_col, -1)
            cv2.putText(frame, f"{g_name}: {sc*100:.1f}% (x{self.vault.gestures[g_name]['count']})",
                        (bx + 8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        preset_names = {
            ord('1'): "TRAZO_CIRCULO",
            ord('2'): "TRAZO_TRIANGULO",
            ord('3'): "SIMBOLO_Z",
            ord('4'): "ONDA_ESPACIAL",
        }

        print("==========================================================================")
        print(" NEXUS-XR: TEMPORAL TRANSFORMER PARA GESTOS EN MOVIMIENTO Y TRAZOS")
        print(" - Presiona '1' y dibuja un CÍRCULO en el aire con tu dedo índice.")
        print(" - Presiona '2' y dibuja un TRIÁNGULO o línea vertical.")
        print(" - Presiona '3' y dibuja una 'Z' en el aire.")
        print(" - El Transformer aprenderá tanto la postura como la trayectoria temporal.")
        print("==========================================================================")

        while cap.isOpened():
            success, raw = cap.read()
            if not success:
                break

            frame = cv2.flip(raw, 1)
            h, w, _ = frame.shape
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            predicted = "ESPERANDO_SECUENCIA"
            confidence = 0.0
            all_scores: Dict[str, float] = {}

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                # 1. Extraer postura invariante (78D) y posición 3D de la punta del índice
                posture_78d, tip_3d = SpatioTemporalHandExtractor.extract_posture_78d(hand_lms)

                # Añadir a la estela visual 2D en pantalla
                px, py = int(tip_3d[0] * w), int(tip_3d[1] * h)
                self.screen_trail_2d.append((px, py))

                # 2. Si estamos grabando un gesto dinámico nuevo
                if self.recording_target is not None:
                    self.recorded_postures.append(posture_78d)
                    self.recorded_tips.append(tip_3d)

                    if len(self.recorded_postures) >= self.recording_frames_total:
                        seq_tensor_np = SpatioTemporalHandExtractor.normalize_sequence(
                            self.recorded_postures, self.recorded_tips, target_len=32
                        )
                        self.vault.learn_dynamic_gesture(self.recording_target, seq_tensor_np)
                        self.vault.save_vault()
                        self.recording_target = None

                # 3. Alimentar la ventana deslizante en tiempo real y evaluar con el Transformer
                self.rolling_postures.append(posture_78d)
                self.rolling_tips_3d.append(tip_3d)

                if len(self.rolling_postures) >= 16 and self.recording_target is None:
                    live_seq = SpatioTemporalHandExtractor.normalize_sequence(
                        list(self.rolling_postures), list(self.rolling_tips_3d), target_len=32
                    )
                    predicted, confidence, all_scores = self.vault.recognize_sequence(live_seq)
            else:
                self.screen_trail_2d.clear()

            self.draw_neon_air_trail(frame)
            self.draw_transformer_hud(frame, predicted, confidence, all_scores)
            cv2.imshow("NexusXR - Temporal Transformer (Gestos Dinamicos en el Aire)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                self.vault.save_vault()
                break
            elif key in preset_names:
                self.start_recording_gesture(preset_names[key])
            elif ord('5') <= key <= ord('9'):
                self.start_recording_gesture(f"GESTO_DINAMICO_{chr(key)}")
            elif key == ord('s'):
                self.vault.save_vault()
            elif key == ord('c'):
                self.vault.gestures.clear()
                print("[TRANSFORMER IA] Memoria de secuencias reiniciada.")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = LiveTemporalGestureEngine()
    engine.run()
