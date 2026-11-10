import cv2
import mediapipe as mp
import numpy as np

class SpatialHandVectorizer:
    """
    Extrae 21 puntos 3D de cada mano y los transforma en vectores matemáticos
    normalizados para interactuar con objetos de Realidad Aumentada y Virtual.
    """
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract_vectors(self, landmarks):
        """Convierte los 21 puntos de la mano en una matriz de vectores 3D (x, y, z)."""
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        # El punto 0 es la muñeca (origen del sistema de coordenadas de la mano)
        wrist_origin = coords[0]
        # Vectores relativos a la muñeca
        relative_vectors = coords - wrist_origin
        # Normalización para que funcione sin importar qué tan lejos esté la mano de la cámara
        norm = np.linalg.norm(relative_vectors, axis=1, keepdims=True) + 1e-8
        unit_vectors = relative_vectors / norm
        return coords, unit_vectors

    def detect_spatial_interaction(self, coords):
        """
        Calcula la magnitud del vector entre la punta del pulgar (índice 4)
        y la punta del índice (índice 8) para agarrar objetos virtuales.
        """
        thumb_tip = coords[4]
        index_tip = coords[8]
        pinch_vector = index_tip - thumb_tip
        distance = np.linalg.norm(pinch_vector)

        is_grabbing = distance < 0.045
        return {
            "is_grabbing": bool(is_grabbing),
            "pinch_distance": float(distance),
            "interaction_point_3d": ((thumb_tip + index_tip) / 2.0).tolist()
        }

    def run_realtime_perception(self):
        cap = cv2.VideoCapture(0)
        print("[MOTOR XR] Percepción vectorial de manos iniciada. Presiona 'q' para salir.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar la malla vectorial sobre la mano real (AR)
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                    coords, unit_vectors = self.extract_vectors(hand_landmarks)
                    interaction = self.detect_spatial_interaction(coords)

                    # Si el usuario junta los dedos, proyectar un nodo interactivo
                    h, w, _ = frame.shape
                    ix, iy = int(interaction["interaction_point_3d"][0] * w), int(interaction["interaction_point_3d"][1] * h)

                    if interaction["is_grabbing"]:
                        cv2.circle(frame, (ix, iy), 18, (0, 255, 128), -1)
                        cv2.putText(frame, "AGARRE XR ACTIVO", (ix + 25, iy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)

            cv2.imshow("NexusXR - Motor de Percepcion Vectorial", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = SpatialHandVectorizer()
    tracker.run_realtime_perception()
