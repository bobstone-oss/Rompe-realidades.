import cv2
import mediapipe as mp
import numpy as np
import math
import socket
import json
import time

# =====================================================================
# 1. MATEMÁTICA 3D Y GEOMETRÍA HOLOGRÁFICA
# =====================================================================

def rotation_matrix_3d(rx, ry, rz):
    """Genera la matriz de rotación 3D combinada ( ángulos de Euler X, Y, Z )."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


class SpatialHologram3D:
    """
    Representa un objeto holográfico compuesto por un Cubo Táctico exterior
    y un Octaedro de Energía interior con caras, aristas y físicas de suavizado.
    """
    def __init__(self):
        # Posición en el espacio 3D (X, Y, Z)
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.target_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Rotación en radianes (Pitch, Yaw, Roll)
        self.rotation = np.array([0.4, 0.6, 0.1], dtype=np.float32)
        self.target_rotation = np.array([0.4, 0.6, 0.1], dtype=np.float32)

        # Factor de escala
        self.scale = 1.0
        self.target_scale = 1.0

        # Geometría 1: Vértices del Cubo Exterior
        self.cube_vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ], dtype=np.float32) * 0.55

        self.cube_faces = [
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)
        ]

        # Geometría 2: Núcleo Octaédrico Interior (gira en sentido inverso)
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

    def update_physics(self, smoothing=0.28):
        """Interpolación lineal (LERP) para movimientos fluidos sin temblor."""
        self.position += (self.target_position - self.position) * smoothing
        self.rotation += (self.target_rotation - self.rotation) * smoothing
        self.scale += (self.target_scale - self.scale) * smoothing
        self.scale = float(np.clip(self.scale, 0.35, 3.2))


# =====================================================================
# 2. EXTRACCIÓN DE VECTORES DE MANO Y ORIENTACIÓN 3D
# =====================================================================

class HandVectorState:
    def __init__(self, pinch_pos_3d, is_pinching, palm_angles, landmarks_2d):
        self.pinch_pos_3d = pinch_pos_3d  # Coordenadas 3D en el mundo (X, Y, Z)
        self.is_pinching = is_pinching    # Bool: ¿Índice y pulgar juntos?
        self.palm_angles = palm_angles    # Rotación 3D de la mano (rx, ry, rz)
        self.landmarks_2d = landmarks_2d  # Puntos en pantalla para dibujar HUD


def analyze_hand_vectors(hand_landmarks, frame_w, frame_h):
    """
    Convierte los 21 puntos de MediaPipe en vectores espaciales 3D,
    detecta el agarre (pinch) y calcula el vector normal de la palma.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

    # 1. Punto de agarre: Punto medio entre Punta del Pulgar (4) e Índice (8)
    thumb_tip = pts[4]
    index_tip = pts[8]
    pinch_mid = (thumb_tip + index_tip) * 0.5
    pinch_dist = float(np.linalg.norm(index_tip - thumb_tip))
    is_pinching = pinch_dist < 0.058

    # Transformar coordenadas normalizadas [0,1] al espacio 3D del motor
    world_x = (pinch_mid[0] - 0.5) * 3.6
    world_y = (pinch_mid[1] - 0.5) * 2.2
    world_z = pinch_mid[2] * 1.5
    pinch_pos_3d = np.array([world_x, world_y, world_z], dtype=np.float32)

    # 2. Orientación 3D de la mano usando producto cruz de la palma
    wrist = pts[0]
    index_mcp = pts[5]
    pinky_mcp = pts[17]

    v_index = index_mcp - wrist
    v_pinky = pinky_mcp - wrist
    v_index /= (np.linalg.norm(v_index) + 1e-8)
    v_pinky /= (np.linalg.norm(v_pinky) + 1e-8)

    # Vector normal perpendicular a la palma
    palm_normal = np.cross(v_index, v_pinky)
    palm_normal /= (np.linalg.norm(palm_normal) + 1e-8)

    # Ángulos de Euler extraídos de los vectores de la mano
    pitch = math.atan2(v_index[1], -v_index[2]) * 1.8
    yaw = math.atan2(palm_normal[0], -palm_normal[2]) * 1.8
    roll = math.atan2(v_pinky[1] - v_index[1], v_pinky[0] - v_index[0]) * 1.8
    palm_angles = np.array([pitch, yaw, roll], dtype=np.float32)

    # Coordenadas 2D para efectos visuales
    landmarks_2d = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand_landmarks.landmark]

    return HandVectorState(pinch_pos_3d, is_pinching, palm_angles, landmarks_2d)


# =====================================================================
# 3. MOTOR ESPACIAL AR/VR Y RENDERIZADOR HOLOGRÁFICO
# =====================================================================

class NexusSpatialInteractionEngine:
    def __init__(self, udp_port=5055):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.hologram = SpatialHologram3D()

        # Nivel de mezcla AR/VR: 0.0 = Realidad Aumentada Pura, 1.0 = Realidad Virtual Pura
        self.vr_blend = 0.25
        self.interaction_mode = "LIBRE"

        # Estados para recordar la referencia al iniciar un agarre o escalado
        self.single_grab_offset = None
        self.initial_hand_angles = None
        self.initial_obj_angles = None

        self.bimanual_initial_dist = None
        self.bimanual_initial_scale = 1.0
        self.bimanual_initial_angle = 0.0
        self.bimanual_base_roll = 0.0

        # Socket UDP para transmitir en vivo a Unity / Godot / Unreal / Rust
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_target = ("127.0.0.1", udp_port)

    def project_3d_to_2d(self, points_3d, w, h, focal_length=650.0, cam_dist=3.6):
        """Proyección de perspectiva 3D real a píxeles 2D de la pantalla."""
        projected_2d = []
        depths = []
        for pt in points_3d:
            z_depth = cam_dist + pt[2]
            if z_depth < 0.4:
                z_depth = 0.4
            px = int((pt[0] * focal_length) / z_depth + (w * 0.5))
            py = int((pt[1] * focal_length) / z_depth + (h * 0.5))
            projected_2d.append((px, py))
            depths.append(z_depth)
        return projected_2d, depths

    def render_vr_environment(self, frame):
        """Funde la cámara real (AR) con un entorno sintético cibernético (VR)."""
        h, w, _ = frame.shape
        vr_world = np.zeros_like(frame)

        # Horizonte y rejilla 3D sintética para el modo Realidad Virtual
        horizon_y = int(h * 0.52)
        for i in range(-12, 13):
            x_bottom = int(w * 0.5 + i * 110)
            x_top = int(w * 0.5 + i * 18)
            cv2.line(vr_world, (x_bottom, h), (x_top, horizon_y), (65, 35, 15), 1)

        for d in range(1, 10):
            y_line = int(horizon_y + (d ** 1.75) * 6.5)
            if y_line < h:
                cv2.line(vr_world, (0, y_line), (w, y_line), (75, 40, 15), 1)

        # Mezcla continua entre el video real (AR) y el mundo sintético (VR)
        return cv2.addWeighted(frame, 1.0 - self.vr_blend, vr_world, self.vr_blend, 0)

    def update_spatial_logic(self, detected_hands):
        """
        Máquina de estados espacial:
        - 2 manos en pinch: Escala bimanual + rotación tipo volante + traslación central.
        - 1 mano en pinch: Agarre directo + traslación 3D + rotación de muñeca.
        - 0 manos en pinch: Levitación libre.
        """
        pinching_hands = [h for h in detected_hands if h.is_pinching]

        # CASO A: ESCALADO Y ROTACIÓN BIMANUAL (2 manos agarrando a la vez)
        if len(pinching_hands) == 2:
            h1, h2 = pinching_hands[0], pinching_hands[1]
            vec_between = h2.pinch_pos_3d - h1.pinch_pos_3d
            current_dist = float(np.linalg.norm(vec_between))
            current_angle = math.atan2(vec_between[1], vec_between[0])

            if self.interaction_mode != "ESCALADO_BIMANUAL":
                self.interaction_mode = "ESCALADO_BIMANUAL"
                self.bimanual_initial_dist = max(current_dist, 0.05)
                self.bimanual_initial_scale = self.hologram.scale
                self.bimanual_initial_angle = current_angle
                self.bimanual_base_roll = self.hologram.rotation[2]

            # Escalar proporcionalmente al estiramiento entre ambas manos
            scale_factor = current_dist / self.bimanual_initial_dist
            self.hologram.target_scale = self.bimanual_initial_scale * scale_factor

            # Rotar en el eje Z según la inclinación entre ambas manos
            delta_angle = current_angle - self.bimanual_initial_angle
            self.hologram.target_rotation[2] = self.bimanual_base_roll + delta_angle

            # Mover el centro del holograma al punto medio entre las dos manos
            self.hologram.target_position = (h1.pinch_pos_3d + h2.pinch_pos_3d) * 0.5
            self.single_grab_offset = None

        # CASO B: AGARRE Y ROTACIÓN 3D CON UNA SOLA MANO
        elif len(pinching_hands) == 1:
            hand = pinching_hands[0]
            dist_to_hologram = float(np.linalg.norm(hand.pinch_pos_3d - self.hologram.position))
            grab_radius = 0.85 * self.hologram.scale

            if self.interaction_mode == "AGARRE_1_MANO" or dist_to_hologram < grab_radius:
                if self.interaction_mode != "AGARRE_1_MANO":
                    self.interaction_mode = "AGARRE_1_MANO"
                    self.single_grab_offset = self.hologram.position - hand.pinch_pos_3d
                    self.initial_hand_angles = hand.palm_angles.copy()
                    self.initial_obj_angles = self.hologram.rotation.copy()

                # Trasladar y rotar siguiendo exactamente la muñeca del usuario
                self.hologram.target_position = hand.pinch_pos_3d + self.single_grab_offset
                angle_delta = hand.palm_angles - self.initial_hand_angles
                self.hologram.target_rotation = self.initial_obj_angles + angle_delta
            else:
                self.interaction_mode = "PINCH_FUERA_DE_RANGO"
                self.single_grab_offset = None

        # CASO C: SIN AGARRE (Levitación suave en el espacio)
        else:
            self.interaction_mode = "LIBRE"
            self.single_grab_offset = None
            self.bimanual_initial_dist = None
            # Rotación orbital suave en reposo
            self.hologram.target_rotation[1] += 0.012

        self.hologram.update_physics()
        self.broadcast_telemetry()

    def draw_hologram_3d(self, frame):
        """Renderiza el holograma 3D con ordenamiento Z-buffer y brillo neón."""
        h, w, _ = frame.shape
        R = rotation_matrix_3d(*self.hologram.rotation)

        # Transformar vértices del cubo exterior
        transformed_cube = (self.hologram.cube_vertices * self.hologram.scale) @ R.T + self.hologram.position
        pts_2d, depths = self.project_3d_to_2d(transformed_cube, w, h)

        # Color dinámico según el estado de interacción
        if self.interaction_mode == "ESCALADO_BIMANUAL":
            primary_color = (255, 110, 255)  # Magenta cuántico (Escala 2 manos)
        elif self.interaction_mode == "AGARRE_1_MANO":
            primary_color = (120, 255, 100)  # Verde neón (Agarre 1 mano)
        else:
            primary_color = (255, 220, 0)    # Cian holográfico (Reposo)

        # Ordenar caras de atrás hacia adelante (Algoritmo del Pintor / Z-Sorting)
        face_depths = []
        for idx, face in enumerate(self.hologram.cube_faces):
            avg_z = sum(depths[v] for v in face) / 4.0
            face_depths.append((avg_z, idx))
        face_depths.sort(reverse=True, key=lambda x: x[0])

        overlay = frame.copy()
        for _, face_idx in face_depths:
            face = self.hologram.cube_faces[face_idx]
            poly = np.array([pts_2d[v] for v in face], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (primary_color[0] // 3, primary_color[1] // 3, primary_color[2] // 3))
            cv2.polylines(frame, [poly], isClosed=True, color=primary_color, thickness=2, lineType=cv2.LINE_AA)

        # Aplicar transparencia cristalina a las caras del holograma
        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

        # Renderizar el Núcleo de Energía Interior (rotación inversa)
        R_core = rotation_matrix_3d(-self.hologram.rotation[0] * 1.5, -self.hologram.rotation[1] * 2.0, self.hologram.rotation[2])
        transformed_core = (self.hologram.core_vertices * self.hologram.scale) @ R_core.T + self.hologram.position
        core_2d, _ = self.project_3d_to_2d(transformed_core, w, h)

        for e1, e2 in self.hologram.core_edges:
            cv2.line(frame, core_2d[e1], core_2d[e2], (255, 255, 255), 1, cv2.LINE_AA)

        # Dibujar nodos en los vértices 3D
        for pt in pts_2d:
            cv2.circle(frame, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 8, primary_color, 1, cv2.LINE_AA)

    def draw_hand_hud(self, frame, detected_hands):
        """Dibuja los vectores de las manos y el rayo tensor cuando hay escalado bimanual."""
        pinching_pts = []
        for hand in detected_hands:
            # Conexiones óseas vectoriales
            for connection in self.mp_hands.HAND_CONNECTIONS:
                p1 = hand.landmarks_2d[connection[0]]
                p2 = hand.landmarks_2d[connection[1]]
                cv2.line(frame, p1, p2, (180, 180, 180), 1, cv2.LINE_AA)

            thumb_2d = hand.landmarks_2d[4]
            index_2d = hand.landmarks_2d[8]
            mid_2d = ((thumb_2d[0] + index_2d[0]) // 2, (thumb_2d[1] + index_2d[1]) // 2)

            if hand.is_pinching:
                cv2.circle(frame, mid_2d, 14, (0, 255, 128), 2, cv2.LINE_AA)
                cv2.circle(frame, mid_2d, 5, (255, 255, 255), -1, cv2.LINE_AA)
                pinching_pts.append(mid_2d)
            else:
                cv2.line(frame, thumb_2d, index_2d, (0, 200, 255), 1, cv2.LINE_AA)

        # Si ambas manos están escalando, dibujar el puente de energía entre ellas
        if len(pinching_pts) == 2:
            cv2.line(frame, pinching_pts[0], pinching_pts[1], (255, 100, 255), 2, cv2.LINE_AA)

    def broadcast_telemetry(self):
        """Envía el estado 3D por UDP (puerto 5055) para conectarlo con otros motores."""
        payload = {
            "mode": self.interaction_mode,
            "vr_blend": round(self.vr_blend, 2),
            "position": self.hologram.position.tolist(),
            "rotation_euler": self.hologram.rotation.tolist(),
            "scale": round(self.hologram.scale, 3)
        }
        try:
            self.sock.sendto(json.dumps(payload).encode("utf-8"), self.udp_target)
        except OSError:
            pass

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("=================================================================")
        print(" NEXUS-XR: MOTOR DE HOLOGRAMAS INTERACTIVOS INICIADO")
        print(" - 1 Mano (Juntar Pulgar + Índice): Agarrar, mover y rotar 3D")
        print(" - 2 Manos (Ambas en Pinch): Estirar para escalar y girar")
        print(" - Tecla 'm': Alternar entre Realidad Aumentada, Mixta y Virtual")
        print(" - Tecla 'r': Resetear posición del holograma | 'q': Salir")
        print("=================================================================")

        prev_time = time.time()
        while cap.isOpened():
            success, raw_frame = cap.read()
            if not success:
                break

            frame = cv2.flip(raw_frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            # 1. Extraer vectores de todas las manos visibles
            detected_hands = []
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    detected_hands.append(analyze_hand_vectors(hand_lms, w, h))

            # 2. Actualizar físicas 3D del holograma según gestos
            self.update_spatial_logic(detected_hands)

            # 3. Renderizar fusión AR / VR y el Holograma 3D
            scene = self.render_vr_environment(frame)
            self.draw_hologram_3d(scene)
            self.draw_hand_hud(scene, detected_hands)

            # 4. Panel de Telemetría Espacial en Pantalla
            curr_time = time.time()
            fps = int(1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            cv2.rectangle(scene, (15, 15), (490, 135), (20, 20, 20), -1)
            cv2.rectangle(scene, (15, 15), (490, 135), (255, 200, 0), 1)
            cv2.putText(scene, f"NEXUS-XR ENGINE | FPS: {fps} | MODO: {self.interaction_mode}",
                        (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
            cv2.putText(scene, f"ESCALA: {self.hologram.scale:.2f}x | MEZCLA AR/VR: {int(self.vr_blend*100)}% VR",
                        (28, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            cv2.putText(scene, "[1 Mano Pinch]=Mover/Rotar  [2 Manos Pinch]=Escalar  [M]=AR/VR",
                        (28, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)

            cv2.imshow("NexusXR - Interaccion Holografica 3D", scene)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                # Ciclar entre AR pura (0%), Realidad Mixta (50%) y VR pura (100%)
                self.vr_blend = (self.vr_blend + 0.5) if self.vr_blend < 0.9 else 0.0
            elif key == ord('r'):
                self.hologram = SpatialHologram3D()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = NexusSpatialInteractionEngine()
    engine.run()
