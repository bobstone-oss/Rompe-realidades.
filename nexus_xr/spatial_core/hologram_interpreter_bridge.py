import cv2
import mediapipe as mp
import numpy as np
import math
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional

# Intento de carga opcional para reconocimiento de voz en vivo
try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False


# =====================================================================
# 1. GEOMETRÍA Y MOTOR MULTI-HOLOGRAMA 3D
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
    "AZUL": (255, 120, 40),
    "BLANCO": (255, 255, 255),
}


class HologramEntity3D:
    """Representa un objeto 3D independiente dentro del grafo de escena."""
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
                [0.5, 0.45, 0.5], [-0.5, 0.45, 0.5],
                [0.0, -0.55, 0.0]  # Ápice superior
            ], dtype=np.float32)
            edges = [(0,1), (1,2), (2,3), (3,0), (0,4), (1,4), (2,4), (3,4)]
            faces = [(0,1,4), (1,2,4), (2,3,4), (3,0,4)]
            return verts, edges, faces

        elif shape == "OCTAEDRO" or shape == "DIAMANTE":
            verts = np.array([
                [0.55, 0, 0], [-0.55, 0, 0],
                [0, 0.65, 0], [0, -0.65, 0],
                [0, 0, 0.55], [0, 0, -0.55]
            ], dtype=np.float32)
            edges = [(0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,4), (4,3), (3,5), (5,2)]
            faces = [(0,2,4), (0,4,3), (1,2,4), (1,4,3)]
            return verts, edges, faces

        elif shape == "ESFERA":
            # Aproximación geodésica por anillos meridianos
            verts = []
            edges = []
            rings, sectors = 5, 10
            for r in range(rings):
                phi = math.pi * (r + 1) / (rings + 1) - (math.pi / 2)
                for s in range(sectors):
                    theta = 2.0 * math.pi * s / sectors
                    x = 0.52 * math.cos(phi) * math.cos(theta)
                    y = 0.52 * math.sin(phi)
                    z = 0.52 * math.cos(phi) * math.sin(theta)
                    verts.append([x, y, z])
                    curr = r * sectors + s
                    next_s = r * sectors + ((s + 1) % sectors)
                    edges.append((curr, next_s))
                    if r > 0:
                        edges.append((curr, (r - 1) * sectors + s))
            return np.array(verts, dtype=np.float32), edges, []

        else:  # CUBO por defecto
            verts = np.array([
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
            ], dtype=np.float32) * 0.45
            edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
            faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6)]
            return verts, edges, faces

    def tick(self, smoothing: float = 0.22):
        self.position += (self.target_position - self.position) * smoothing
        self.scale += (self.target_scale - self.scale) * smoothing
        self.scale = float(np.clip(self.scale, 0.2, 4.0))
        self.rotation += self.spin_speed


# =====================================================================
# 2. INTÉRPRETE EN VIVO DE NEXUS-LANG (CON METAPROGRAMACIÓN Y VOZ)
# =====================================================================

class LiveNexusInterpreter:
    """
    Compila y ejecuta código NexusLang o frases dictadas por voz directamente
    sobre los objetos 3D de la escena holográfica.
    """
    def __init__(self, scene_objects: Dict[str, HologramEntity3D]):
        self.scene = scene_objects
        self.macros: Dict[str, List[str]] = {}
        self.vr_blend_ref = [0.25]  # Lista mutable compartida con el motor
        self.execution_log: List[str] = [
            "SISTEMA NEXUS-LANG ACTIVO.",
            "Escribe abajo o en la terminal para mutar hologramas."
        ]

    def log(self, msg: str):
        print(f"[NEXUS-LANG] {msg}")
        self.execution_log.append(msg)
        if len(self.execution_log) > 6:
            self.execution_log.pop(0)

    def _parse_vec3(self, text: str) -> Tuple[float, float, float]:
        clean = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        parts = [float(x) for x in clean.replace(",", " ").split() if x]
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
        return 0.0, 0.0, 0.0

    def normalize_natural_voice(self, text: str) -> str:
        """
        Convierte comandos hablados en español natural (ej. 'crea una esfera roja')
        en instrucciones formales de NexusLang.
        """
        t = text.upper().strip()
        for old, new in [("Á", "A"), ("É", "E"), ("Í", "I"), ("Ó", "O"), ("Ú", "U")]:
            t = t.replace(old, new)

        # Si el usuario ya escribió un comando formal, devolverlo directo
        formal_starts = ("CREAR", "ESCALAR", "MOVER", "COLOR", "VELOCIDAD", "ANCLAR_MANO", "SOLTAR_MANO", "BORRAR", "MACRO", "REALIDAD")
        if any(t.startswith(k) for k in formal_starts):
            return t

        # Traducción inteligente de voz natural -> NexusLang
        if "CREA" in t or "NUEVO" in t or "HOLOGRAMA" in t:
            shape = "CUBO"
            for s in ["ESFERA", "PIRAMIDE", "OCTAEDRO", "DIAMANTE", "CUBO"]:
                if s in t:
                    shape = s
                    break
            color = "CIAN"
            for c in COLOR_PALETTE.keys():
                if c in t:
                    color = c
                    break
            new_id = f"holo{len(self.scene) + 1}"
            rx = round(np.random.uniform(-0.8, 0.8), 2)
            ry = round(np.random.uniform(-0.4, 0.4), 2)
            return f"CREAR {new_id} {shape} [{rx},{ry},0.0] ; COLOR {new_id} {color}"

        if "AGRANDA" in t or "GIGANTE" in t or "MAS GRANDE" in t:
            target = list(self.scene.keys())[-1] if self.scene else "cubo1"
            return f"ESCALAR {target} 2.0"

        if "PEQUENO" in t or "ENCOGE" in t or "REDUCE" in t:
            target = list(self.scene.keys())[-1] if self.scene else "cubo1"
            return f"ESCALAR {target} 0.5"

        if "MANO" in t or "SIGUEME" in t or "AGARRA" in t:
            target = list(self.scene.keys())[-1] if self.scene else "cubo1"
            return f"ANCLAR_MANO {target}"

        return t

    def execute(self, raw_command: str):
        if not raw_command.strip():
            return

        # 1. Soportar múltiples comandos separados por ';'
        if ";" in raw_command and not raw_command.strip().upper().startswith("MACRO"):
            for sub in raw_command.split(";"):
                self.execute(sub)
            return

        normalized = self.normalize_natural_voice(raw_command)
        if ";" in normalized and not normalized.startswith("MACRO"):
            for sub in normalized.split(";"):
                self.execute(sub)
            return

        tokens = normalized.split()
        if not tokens:
            return
        cmd = tokens[0].upper()
        args = tokens[1:]

        try:
            # --- METAPROGRAMACIÓN: DEFINICIÓN DE NUEVOS COMANDOS EN VIVO ---
            if cmd == "MACRO":
                # Ejemplo: MACRO MODO_COMBATE = COLOR cubo1 ROJO ; ESCALAR cubo1 2.2
                rest = normalized[5:].strip()
                macro_name, macro_body = rest.split("=", 1)
                m_name = macro_name.strip().upper()
                self.macros[m_name] = [c.strip() for c in macro_body.split(";") if c.strip()]
                self.log(f"METAPROGRAMACIÓN: Nuevo comando '{m_name}' compilado.")
                return

            # Si el comando es una Macro inventada por el usuario, ejecutar su árbol
            if cmd in self.macros:
                self.log(f"Ejecutando Macro '{cmd}'...")
                for sub_cmd in self.macros[cmd]:
                    self.execute(sub_cmd)
                return

            # --- COMANDOS ESPACIALES DEL MOTOR 3D ---
            if cmd == "CREAR":
                name = args[0].lower()
                shape = args[1].upper() if len(args) > 1 else "CUBO"
                pos = self._parse_vec3(" ".join(args[2:])) if len(args) > 2 else (0.0, 0.0, 0.0)
                self.scene[name] = HologramEntity3D(name, shape, pos)
                self.log(f"Creado '{name}' ({shape}) en {pos}")

            elif cmd == "ESCALAR":
                name = args[0].lower()
                val = float(args[1])
                if name == "TODOS":
                    for obj in self.scene.values(): obj.target_scale = val
                elif name in self.scene:
                    self.scene[name].target_scale = val
                self.log(f"Escala de '{name}' -> {val}x")

            elif cmd == "MOVER":
                name = args[0].lower()
                pos = self._parse_vec3(" ".join(args[1:]))
                if name in self.scene:
                    self.scene[name].target_position = np.array(pos, dtype=np.float32)
                    self.scene[name].bound_to_hand = False
                    self.log(f"Moviendo '{name}' a {pos}")

            elif cmd == "COLOR":
                name = args[0].lower()
                col_name = args[1].upper()
                rgb = COLOR_PALETTE.get(col_name, (255, 255, 255))
                if name == "TODOS":
                    for obj in self.scene.values(): obj.color = rgb
                elif name in self.scene:
                    self.scene[name].color = rgb
                self.log(f"Color de '{name}' cambiado a {col_name}")

            elif cmd == "VELOCIDAD":
                name = args[0].lower()
                mult = float(args[1])
                if name in self.scene:
                    self.scene[name].spin_speed = np.array([0.01, 0.025, 0.008], dtype=np.float32) * mult
                    self.log(f"Velocidad orbital de '{name}' -> {mult}x")

            elif cmd == "ANCLAR_MANO":
                name = args[0].lower()
                if name in self.scene:
                    self.scene[name].bound_to_hand = True
                    self.log(f"Holograma '{name}' vinculado a los vectores de tu mano.")

            elif cmd == "SOLTAR_MANO":
                name = args[0].lower()
                if name in self.scene:
                    self.scene[name].bound_to_hand = False
                    self.log(f"Holograma '{name}' liberado en el espacio.")

            elif cmd == "BORRAR":
                name = args[0].lower()
                if name == "TODOS":
                    self.scene.clear()
                    self.log("Todos los hologramas eliminados.")
                elif name in self.scene:
                    del self.scene[name]
                    self.log(f"Holograma '{name}' eliminado.")

            elif cmd == "REALIDAD":
                pct = float(args[0])
                self.vr_blend_ref[0] = float(np.clip(pct / 100.0, 0.0, 1.0))
                self.log(f"Fusión AR/VR ajustada al {int(pct)}% VR.")

            else:
                self.log(f"Comando no reconocido: '{cmd}'")

        except Exception as e:
            self.log(f"Error de sintaxis en '{raw_command}': {e}")


# =====================================================================
# 3. MOTOR VISUAL 3D CON TERMINAL HOLOGRÁFICA Y MANOS EN VIVO
# =====================================================================

class NexusLiveProgrammableEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Escena inicial con un cubo central
        self.scene: Dict[str, HologramEntity3D] = {
            "cubo1": HologramEntity3D("cubo1", "CUBO", (0.0, 0.0, 0.0), "CIAN")
        }
        self.interpreter = LiveNexusInterpreter(self.scene)
        self.command_queue: queue.Queue = queue.Queue()

        # Barra de código interactiva dentro de la ventana de video
        self.onscreen_input_buffer = ""
        self.running = True

    def _terminal_repl_worker(self):
        """Hilo en segundo plano para escribir código en la terminal del sistema sin pausar el 3D."""
        while self.running:
            try:
                line = input("NexusLang >> ")
                if line.strip():
                    self.command_queue.put(line.strip())
            except EOFError:
                break

    def trigger_voice_dictation(self):
        """Escucha un comando por micrófono en un hilo paralelo sin congelar los FPS."""
        if not VOICE_AVAILABLE:
            self.interpreter.log("Instala 'pip install SpeechRecognition pyaudio' para usar voz.")
            return

        def _listen_job():
            recognizer = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    self.interpreter.log("ESCUCHANDO MICRÓFONO... Habla ahora.")
                    recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = recognizer.listen(source, timeout=4, phrase_time_limit=5)
                spoken_text = recognizer.recognize_google(audio, language="es-ES")
                self.interpreter.log(f"Voz detectada: '{spoken_text}'")
                self.command_queue.put(spoken_text)
            except Exception as e:
                self.interpreter.log(f"Dictado finalizado sin orden clara ({type(e).__name__}).")

        threading.Thread(target=_listen_job, daemon=True).start()

    def project_3d(self, points_3d, w, h, focal=640.0, cam_dist=3.6):
        pts_2d = []
        for pt in points_3d:
            z = max(cam_dist + pt[2], 0.35)
            px = int((pt[0] * focal) / z + (w * 0.5))
            py = int((pt[1] * focal) / z + (h * 0.5))
            pts_2d.append((px, py))
        return pts_2d

    def render_scene(self, frame, hand_pinch_pos_3d: Optional[np.ndarray]):
        h, w, _ = frame.shape
        vr_blend = self.interpreter.vr_blend_ref[0]

        # 1. Mezcla AR / VR
        if vr_blend > 0.02:
            vr_bg = np.zeros_like(frame)
            horizon = int(h * 0.50)
            for i in range(-10, 11):
                cv2.line(vr_bg, (int(w*0.5 + i*115), h), (int(w*0.5 + i*18), horizon), (55, 30, 15), 1)
            for d in range(1, 9):
                y = int(horizon + (d**1.75) * 6.5)
                if y < h:
                    cv2.line(vr_bg, (0, y), (w, y), (65, 35, 15), 1)
            frame = cv2.addWeighted(frame, 1.0 - vr_blend, vr_bg, vr_blend, 0)

        # 2. Actualizar y dibujar todos los hologramas del diccionario
        overlay = frame.copy()
        for name, obj in list(self.scene.items()):
            if obj.bound_to_hand and hand_pinch_pos_3d is not None:
                obj.target_position = hand_pinch_pos_3d.copy()
            elif hand_pinch_pos_3d is not None:
                # Si la mano hace pinch cerca del objeto, también permite moverlo
                if np.linalg.norm(hand_pinch_pos_3d - obj.position) < (0.65 * obj.scale):
                    obj.target_position = hand_pinch_pos_3d.copy()

            obj.tick()
            R = rotation_matrix_3d(*obj.rotation)
            world_verts = (obj.vertices * obj.scale) @ R.T + obj.position
            pts_2d = self.project_3d(world_verts, w, h)

            # Caras translúcidas
            for face in obj.faces:
                poly = np.array([pts_2d[idx] for idx in face], dtype=np.int32)
                c = obj.color
                cv2.fillPoly(overlay, [poly], (c[0]//4, c[1]//4, c[2]//4))

            # Aristas brillantes
            for e1, e2 in obj.edges:
                cv2.line(frame, pts_2d[e1], pts_2d[e2], obj.color, 2, cv2.LINE_AA)

            # Etiqueta flotante con el nombre de la variable en NexusLang
            center_2d = self.project_3d([obj.position], w, h)[0]
            cv2.putText(frame, f"<{obj.name}:{obj.shape}>", (center_2d[0] - 45, center_2d[1] - int(55 * obj.scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
        return frame

    def draw_holographic_terminal_hud(self, frame):
        h, w, _ = frame.shape

        # Panel superior: Registro de ejecución del intérprete
        cv2.rectangle(frame, (15, 15), (680, 175), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (680, 175), (0, 220, 255), 1)
        cv2.putText(frame, f"CONSOLA HOLOGRAFICA NEXUS-LANG | Objetos 3D activos: {len(self.scene)}",
                    (28, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 2)

        for idx, log_line in enumerate(self.interpreter.execution_log[-5:]):
            cv2.putText(frame, f"> {log_line}", (28, 64 + idx * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (215, 230, 245), 1)

        # Barra inferior: Entrada de código en vivo directamente sobre el video
        cv2.rectangle(frame, (15, h - 68), (w - 15, h - 15), (12, 14, 20), -1)
        cv2.rectangle(frame, (15, h - 68), (w - 15, h - 15), (0, 255, 140), 1)

        cursor = "_" if int(time.time() * 2.5) % 2 == 0 else " "
        prompt_txt = f"CODIGO EN VIVO >> {self.onscreen_input_buffer}{cursor}"
        cv2.putText(frame, prompt_txt, (28, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 140), 2)

        help_txt = "Ejemplos: CREAR e1 ESFERA [-0.6,0,0]  |  COLOR cubo1 MAGENTA  |  ESCALAR cubo1 1.8  |  [ENTER]=Ejecutar  [F2]=Voz  [ESC]=Salir"
        cv2.putText(frame, help_txt, (28, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 190, 210), 1)

    def run(self):
        # Iniciar hilo de lectura de consola estándar
        threading.Thread(target=self._terminal_repl_worker, daemon=True).start()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("==========================================================================")
        print(" NEXUS-XR: INTÉRPRETE EN VIVO + MOTOR HOLOGRÁFICO 3D")
        print(" Puedes escribir comandos directamente en la ventana de video o aquí:")
        print("   1. CREAR satelite1 ESFERA [-0.7, 0.2, 0.0]")
        print("   2. CREAR p1 PIRAMIDE [0.7, -0.1, 0.0]")
        print("   3. COLOR satelite1 MAGENTA")
        print("   4. ANCLAR_MANO p1")
        print("   5. MACRO ALERTA = COLOR TODOS ROJO ; ESCALAR TODOS 1.7")
        print("   6. ALERTA")
        print("==========================================================================")

        while cap.isOpened() and self.running:
            success, raw = cap.read()
            if not success:
                break

            frame = cv2.flip(raw, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            # 1. Procesar comandos en cola (desde terminal, teclado en pantalla o voz)
            while not self.command_queue.empty():
                cmd_text = self.command_queue.get_nowait()
                self.interpreter.execute(cmd_text)

            # 2. Extraer punto de interacción 3D de la mano
            hand_pinch_pos_3d = None
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                mid = (pts[4] + pts[8]) * 0.5
                pinch_dist = float(np.linalg.norm(pts[8] - pts[4]))

                # Coordenadas 3D de la mano en el espacio del motor
                hand_3d = np.array([(mid[0] - 0.5) * 3.4, (mid[1] - 0.5) * 2.0, mid[2] * 1.4], dtype=np.float32)
                if pinch_dist < 0.06 or any(o.bound_to_hand for o in self.scene.values()):
                    hand_pinch_pos_3d = hand_3d
                    px, py = int(mid[0] * w), int(mid[1] * h)
                    cv2.circle(frame, (px, py), 12, (0, 255, 140), 2, cv2.LINE_AA)

            # 3. Renderizar hologramas 3D y consola HUD
            scene_frame = self.render_scene(frame, hand_pinch_pos_3d)
            self.draw_holographic_terminal_hud(scene_frame)

            cv2.imshow("NexusXR - Interprete Holografico en Vivo", scene_frame)

            # 4. Captura de escritura en vivo dentro de la ventana de video
            key = cv2.waitKey(1)
            if key == -1:
                continue
            elif key == 27:  # Tecla ESC para salir
                self.running = False
                break
            elif key in (13, 10):  # Tecla ENTER: ejecutar el comando escrito en pantalla
                if self.onscreen_input_buffer.strip():
                    self.command_queue.put(self.onscreen_input_buffer.strip())
                    self.onscreen_input_buffer = ""
            elif key in (8, 127):  # Tecla BACKSPACE: borrar letra
                self.onscreen_input_buffer = self.onscreen_input_buffer[:-1]
            elif key == 191 or key == 0:  # Tecla especial o F2 según sistema operativo
                self.trigger_voice_dictation()
            elif 32 <= (key & 0xFF) <= 126:
                char = chr(key & 0xFF)
                self.onscreen_input_buffer += char

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = NexusLiveProgrammableEngine()
    engine.run()
