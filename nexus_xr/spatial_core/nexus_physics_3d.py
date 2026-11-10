import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
import threading
import queue
from typing import Dict, List, Tuple, Optional


# =====================================================================
# 1. MATEMÁTICA 3D Y EFECTOS DE ONDA DE CHOQUE
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
    "VERDE": (90, 255, 90),
    "MAGENTA": (255, 80, 255),
    "ORO": (0, 215, 255),
    "ROJO": (60, 60, 255),
    "AZUL": (255, 140, 40),
    "BLANCO": (255, 255, 255),
}


class CollisionShockwave:
    """Efecto visual de anillo expansivo cuando dos cuerpos 3D chocan."""
    def __init__(self, pos_3d: np.ndarray, color=(255, 255, 255)):
        self.pos_3d = pos_3d.copy()
        self.radius_px = 6
        self.alpha_life = 1.0
        self.color = color

    def update(self) -> bool:
        self.radius_px += 5
        self.alpha_life -= 0.12
        return self.alpha_life > 0.0


# =====================================================================
# 2. CUERPO RÍGIDO HOLOGRÁFICO 3D (MASA, INERCIA, REBOTE Y CARGA MAGNÉTICA)
# =====================================================================

class RigidBodyHologram3D:
    def __init__(self, name: str, shape: str = "CUBO", pos=(0.0, 0.0, 0.0), color_name="CIAN"):
        self.name = name
        self.shape = shape.upper()
        self.position = np.array(pos, dtype=np.float32)
        self.velocity = np.array([
            random.uniform(-0.02, 0.02),
            random.uniform(-0.02, 0.02),
            random.uniform(-0.01, 0.01)
        ], dtype=np.float32)

        self.rotation = np.array([0.2, 0.4, 0.1], dtype=np.float32)
        self.angular_vel = np.array([0.01, 0.02, 0.008], dtype=np.float32)

        self.scale = 0.85
        self.base_radius = 0.36  # Radio físico de colisión esférica base
        self.mass = 1.0
        self.restitution = 0.88  # Coeficiente de elasticidad (rebote)

        # Carga magnética: +1.0 (POSITIVO/NORTE), -1.0 (NEGATIVO/SUR), 0.0 (NEUTRO)
        self.magnetic_charge = 0.0

        self.color = COLOR_PALETTE.get(color_name.upper(), (255, 230, 0))
        self.is_grabbed = False
        self.vertices, self.edges, self.faces = self._build_mesh(self.shape)

    @property
    def collider_radius(self) -> float:
        return self.base_radius * self.scale

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

        else:  # CUBO
            verts = np.array([
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
            ], dtype=np.float32) * 0.36
            edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
            faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6)]
            return verts, edges, faces


# =====================================================================
# 3. COLISIONADORES CINEMÁTICOS DE LAS MANOS
# =====================================================================

class HandColliderNode:
    """Esfera sólida anclada a puntos clave de la mano para golpear o agarrar objetos."""
    def __init__(self, pos_3d: np.ndarray, vel_3d: np.ndarray, radius: float = 0.22, is_pinching: bool = False):
        self.pos_3d = pos_3d
        self.vel_3d = vel_3d
        self.radius = radius
        self.is_pinching = is_pinching


# =====================================================================
# 4. MOTOR DE FÍSICAS 3D (COLISIONES ELÁSTICAS, IMANES Y GRAVEDAD)
# =====================================================================

class SpatialPhysicsSolver3D:
    def __init__(self):
        # Límites de la cámara / cuarto holográfico 3D (X, Y, Z)
        self.bounds_x = 1.65
        self.bounds_y = 0.95
        self.bounds_z = 0.85

        self.gravity = 0.0018       # Aceleración gravitatoria hacia abajo (+Y en cámara)
        self.air_damping = 0.992    # Fricción aerodinámica suave
        self.magnetic_k = 0.0022    # Constante de fuerza magnética de Coulomb
        self.hand_magnet_charge = 0.0  # Si != 0, tus propias manos actúan como un imán

        self.shockwaves: List[CollisionShockwave] = []
        self.magnetic_links: List[Tuple[np.ndarray, np.ndarray, bool]] = []

    def step_simulation(self, bodies: List[RigidBodyHologram3D], hand_colliders: List[HandColliderNode]):
        self.magnetic_links.clear()
        n = len(bodies)

        # 1. INTERACCIONES MAGNÉTICAS (LEY DE COULOMB 3D ENTRE HOLOGRAMAS)
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = bodies[i], bodies[j]
                if b1.magnetic_charge != 0.0 and b2.magnetic_charge != 0.0:
                    delta = b2.position - b1.position
                    dist_sq = float(np.dot(delta, delta)) + 0.04
                    dist = math.sqrt(dist_sq)
                    unit_dir = delta / dist

                    # Cargas opuestas (< 0) se atraen; cargas iguales (> 0) se repelen
                    charge_prod = b1.magnetic_charge * b2.magnetic_charge
                    force_mag = (self.magnetic_k * charge_prod) / dist_sq
                    force_vec = unit_dir * force_mag

                    if not b1.is_grabbed:
                        b1.velocity -= force_vec / b1.mass
                    if not b2.is_grabbed:
                        b2.velocity += force_vec / b2.mass

                    is_attracting = (charge_prod < 0)
                    self.magnetic_links.append((b1.position.copy(), b2.position.copy(), is_attracting))

        # 2. INTEGRACIÓN DE FUERZAS, GRAVEDAD Y REBOTE CONTRA PAREDES 3D
        for b in bodies:
            if not b.is_grabbed:
                b.velocity[1] += self.gravity
                b.velocity *= self.air_damping
                # Límite de velocidad máxima para estabilidad numérica
                speed = float(np.linalg.norm(b.velocity))
                if speed > 0.22:
                    b.velocity = (b.velocity / speed) * 0.22

                b.position += b.velocity

            b.rotation += b.angular_vel
            b.angular_vel *= 0.985 + 0.01 * min(float(np.linalg.norm(b.velocity)) * 10.0, 1.0)

            # Rebote elástico contra los 6 muros de contención del espacio AR/VR
            r = b.collider_radius
            if b.position[0] - r < -self.bounds_x:
                b.position[0] = -self.bounds_x + r
                b.velocity[0] = abs(b.velocity[0]) * b.restitution
                b.angular_vel[1] += 0.04
            elif b.position[0] + r > self.bounds_x:
                b.position[0] = self.bounds_x - r
                b.velocity[0] = -abs(b.velocity[0]) * b.restitution
                b.angular_vel[1] -= 0.04

            if b.position[1] - r < -self.bounds_y:
                b.position[1] = -self.bounds_y + r
                b.velocity[1] = abs(b.velocity[1]) * b.restitution
            elif b.position[1] + r > self.bounds_y:
                b.position[1] = self.bounds_y - r
                b.velocity[1] = -abs(b.velocity[1]) * b.restitution
                # Fricción contra el suelo
                b.velocity[0] *= 0.96

            if b.position[2] - r < -self.bounds_z:
                b.position[2] = -self.bounds_z + r
                b.velocity[2] = abs(b.velocity[2]) * b.restitution
            elif b.position[2] + r > self.bounds_z:
                b.position[2] = self.bounds_z - r
                b.velocity[2] = -abs(b.velocity[2]) * b.restitution

        # 3. COLISIONES ELÁSTICAS HOLOGRAMA CONTRA HOLOGRAMA (IMPULSO 3D)
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = bodies[i], bodies[j]
                delta = b2.position - b1.position
                dist = float(np.linalg.norm(delta))
                min_dist = b1.collider_radius + b2.collider_radius

                if dist < min_dist and dist > 1e-6:
                    normal = delta / dist
                    overlap = min_dist - dist

                    # Corrección posicional para evitar que un holograma atraviese a otro
                    inv_m1 = 0.0 if b1.is_grabbed else (1.0 / b1.mass)
                    inv_m2 = 0.0 if b2.is_grabbed else (1.0 / b2.mass)
                    inv_mass_sum = inv_m1 + inv_m2
                    if inv_mass_sum > 0:
                        b1.position -= normal * (overlap * (inv_m1 / inv_mass_sum))
                        b2.position += normal * (overlap * (inv_m2 / inv_mass_sum))

                    # Velocidad relativa a lo largo de la normal de impacto
                    rel_vel = b2.velocity - b1.velocity
                    vel_along_normal = float(np.dot(rel_vel, normal))

                    if vel_along_normal < 0:
                        e = min(b1.restitution, b2.restitution)
                        j_impulse = -(1.0 + e) * vel_along_normal / max(inv_mass_sum, 1e-5)
                        impulse_vec = normal * j_impulse

                        if not b1.is_grabbed:
                            b1.velocity -= impulse_vec * inv_m1
                            b1.angular_vel += np.cross(normal, b1.velocity) * 0.6
                        if not b2.is_grabbed:
                            b2.velocity += impulse_vec * inv_m2
                            b2.angular_vel -= np.cross(normal, b2.velocity) * 0.6

                        # Generar onda de choque en el punto exacto del impacto
                        contact_pt = b1.position + normal * b1.collider_radius
                        self.shockwaves.append(CollisionShockwave(contact_pt, (255, 255, 120)))

        # 4. COLISIÓN FÍSICA, GOLPEO Y AGARRE CONTRA LAS MANOS REALES
        for b in bodies:
            b.is_grabbed = False

        for hc in hand_colliders:
            for b in bodies:
                delta = b.position - hc.pos_3d
                dist = float(np.linalg.norm(delta))
                combined_r = b.collider_radius + hc.radius

                # Si la mano tiene imán activado
                if self.hand_magnet_charge != 0.0 and dist > 0.1:
                    u_dir = delta / dist
                    mag_f = (self.magnetic_k * 1.8 * self.hand_magnet_charge) / (dist * dist + 0.05)
                    b.velocity += u_dir * mag_f
                    self.magnetic_links.append((hc.pos_3d.copy(), b.position.copy(), self.hand_magnet_charge < 0))

                # Si el usuario hace Pinch dentro del radio: agarrar y cargar con inercia
                if hc.is_pinching and dist < (combined_r * 1.15):
                    b.is_grabbed = True
                    b.position = hc.pos_3d.copy()
                    # Al soltar el pinch, el holograma saldrá lanzado con la velocidad de la mano
                    b.velocity = hc.vel_3d * 0.85
                    continue

                # Colisión elástica Mano -> Holograma (Golpear con la palma o el dedo)
                if dist < combined_r and dist > 1e-6:
                    normal = delta / dist
                    overlap = combined_r - dist
                    b.position += normal * overlap

                    rel_v = b.velocity - hc.vel_3d
                    v_norm = float(np.dot(rel_v, normal))
                    if v_norm < 0:
                        bounce = (1.0 + b.restitution) * (-v_norm)
                        b.velocity += normal * bounce + hc.vel_3d * 0.45
                        b.angular_vel += np.array([
                            random.uniform(-0.08, 0.08),
                            random.uniform(-0.08, 0.08),
                            random.uniform(-0.08, 0.08)
                        ], dtype=np.float32)
                        self.shockwaves.append(CollisionShockwave(b.position - normal * b.collider_radius, (0, 255, 160)))

        # Actualizar vida de las ondas de choque
        self.shockwaves = [sw for sw in self.shockwaves if sw.update()]


# =====================================================================
# 5. INTÉRPRETE DE COMANDOS FÍSICOS NEXUS-LANG
# =====================================================================

class PhysicsCommandInterpreter:
    def __init__(self, scene: Dict[str, RigidBodyHologram3D], solver: SpatialPhysicsSolver3D):
        self.scene = scene
        self.solver = solver
        self.logs: List[str] = [
            "MOTOR DE FISICAS 3D Y COLISIONES ACTIVO.",
            "Golpea los hologramas con tus manos o usa IMAN / GRAVEDAD."
        ]

    def log(self, msg: str):
        print(f"[PHYSICS-3D] {msg}")
        self.logs.append(msg)
        if len(self.logs) > 4:
            self.logs.pop(0)

    def _parse_vec3(self, text: str) -> Tuple[float, float, float]:
        clean = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        parts = [float(x) for x in clean.replace(",", " ").split() if x]
        return (parts[0], parts[1], parts[2]) if len(parts) >= 3 else (0.0, 0.0, 0.0)

    def execute(self, raw: str):
        if not raw.strip():
            return
        if ";" in raw:
            for sub in raw.split(";"):
                self.execute(sub)
            return

        tokens = raw.strip().split()
        cmd = tokens[0].upper()
        args = tokens[1:]

        try:
            if cmd == "CREAR":
                name = args[0].lower()
                shape = args[1].upper() if len(args) > 1 else "CUBO"
                pos = self._parse_vec3(" ".join(args[2:])) if len(args) > 2 else (
                    round(random.uniform(-0.9, 0.9), 2), round(random.uniform(-0.5, 0.2), 2), 0.0
                )
                self.scene[name] = RigidBodyHologram3D(name, shape, pos)
                self.log(f"Cuerpo rígido '{name}' ({shape}) creado en {pos}")

            elif cmd == "GRAVEDAD":
                val = float(args[0])
                self.solver.gravity = val * 0.0025
                self.log(f"Gravedad 3D ajustada a {val}G")

            elif cmd == "REBOTE":
                target = args[0].lower()
                val = float(np.clip(float(args[1]), 0.1, 1.25))
                if target == "todos":
                    for b in self.scene.values(): b.restitution = val
                elif target in self.scene:
                    self.scene[target].restitution = val
                self.log(f"Elasticidad (rebote) de '{target}' -> {val}")

            elif cmd == "MASA":
                target = args[0].lower()
                val = max(0.1, float(args[1]))
                if target in self.scene:
                    self.scene[target].mass = val
                    self.log(f"Masa de '{target}' -> {val} kg")

            elif cmd == "IMAN":
                target = args[0].lower()
                polarity = args[1].upper()
                charge = 1.0 if polarity in ("POSITIVO", "NORTE", "+", "REPELER") else (-1.0 if polarity in ("NEGATIVO", "SUR", "-", "ATRAER") else 0.0)
                if target == "mano":
                    self.solver.hand_magnet_charge = charge
                    self.log(f"Guante Magnético en Mano -> {polarity}")
                elif target == "todos":
                    for b in self.scene.values(): b.magnetic_charge = charge
                elif target in self.scene:
                    self.scene[target].magnetic_charge = charge
                    self.log(f"Carga magnética de '{target}' -> {polarity} ({charge:+.0f})")

            elif cmd == "LANZAR":
                target = args[0].lower()
                vel = self._parse_vec3(" ".join(args[1:]))
                if target in self.scene:
                    self.scene[target].velocity = np.array(vel, dtype=np.float32)
                    self.log(f"Impulso aplicado a '{target}': {vel}")

            elif cmd == "EXPLOSION":
                for b in self.scene.values():
                    direction = b.position + np.array([random.uniform(-0.1, 0.1), -0.2, random.uniform(-0.1, 0.1)], dtype=np.float32)
                    norm = float(np.linalg.norm(direction)) + 1e-5
                    b.velocity = (direction / norm) * 0.14
                self.log("¡Onda expansiva 3D disparada!")

            elif cmd == "COLOR":
                target = args[0].lower()
                col = COLOR_PALETTE.get(args[1].upper(), (255, 255, 255))
                if target == "todos":
                    for b in self.scene.values(): b.color = col
                elif target in self.scene:
                    self.scene[target].color = col

            elif cmd == "ESCALAR":
                target = args[0].lower()
                sc = float(np.clip(float(args[1]), 0.35, 2.2))
                if target == "todos":
                    for b in self.scene.values(): b.scale = sc
                elif target in self.scene:
                    self.scene[target].scale = sc

            elif cmd == "BORRAR":
                if args[0].lower() == "todos":
                    self.scene.clear()
                elif args[0].lower() in self.scene:
                    del self.scene[args[0].lower()]

        except Exception as e:
            self.log(f"Error en comando '{raw}': {e}")


# =====================================================================
# 6. MOTOR INTERACTIVO PRINCIPAL (RENDERIZADO 3D + COLISIONES DE MANO)
# =====================================================================

class NexusInteractivePhysicsEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.solver = SpatialPhysicsSolver3D()
        # Escena inicial con 3 cuerpos rígidos listos para chocar
        self.scene: Dict[str, RigidBodyHologram3D] = {
            "cubo1": RigidBodyHologram3D("cubo1", "CUBO", (-0.65, -0.2, 0.0), "CIAN"),
            "esfera1": RigidBodyHologram3D("esfera1", "ESFERA", (0.0, -0.3, 0.0), "MAGENTA"),
            "piramide1": RigidBodyHologram3D("piramide1", "PIRAMIDE", (0.65, -0.2, 0.0), "ORO"),
        }
        self.scene["esfera1"].restitution = 0.94

        self.interpreter = PhysicsCommandInterpreter(self.scene, self.solver)
        self.cmd_queue: queue.Queue = queue.Queue()

        # Memoria de posiciones previas de las manos para calcular el vector de velocidad del golpe
        self.prev_hand_centers: Dict[int, np.ndarray] = {}
        self.onscreen_input = ""
        self.running = True

    def _console_worker(self):
        while self.running:
            try:
                line = input("NexusPhysics >> ")
                if line.strip():
                    self.cmd_queue.put(line.strip())
            except EOFError:
                break

    def project_3d(self, pts_3d, w, h, focal=640.0, cam_dist=3.6):
        out = []
        for pt in pts_3d:
            z = max(cam_dist + pt[2], 0.35)
            out.append((int((pt[0] * focal) / z + w * 0.5), int((pt[1] * focal) / z + h * 0.5)))
        return out

    def extract_hand_colliders(self, multi_hand_landmarks, frame) -> List[HandColliderNode]:
        h, w, _ = frame.shape
        colliders: List[HandColliderNode] = []
        current_centers: Dict[int, np.ndarray] = {}

        if not multi_hand_landmarks:
            self.prev_hand_centers.clear()
            return colliders

        for idx, hand_lms in enumerate(multi_hand_landmarks):
            self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

            # 1. Nodo de agarre/golpe en los dedos (Punto medio entre Pulgar 4 e Índice 8)
            pinch_mid = (pts[4] + pts[8]) * 0.5
            pinch_dist = float(np.linalg.norm(pts[8] - pts[4]))
            is_pinching = pinch_dist < 0.060

            finger_pos_3d = np.array([
                (pinch_mid[0] - 0.5) * 3.3,
                (pinch_mid[1] - 0.5) * 1.9,
                pinch_mid[2] * 1.2
            ], dtype=np.float32)

            # 2. Nodo sólido de la palma (Landmark 9) para batear/empujar hologramas con mano abierta
            palm_pt = pts[9]
            palm_pos_3d = np.array([
                (palm_pt[0] - 0.5) * 3.3,
                (palm_pt[1] - 0.5) * 1.9,
                palm_pt[2] * 1.2
            ], dtype=np.float32)

            prev_pos = self.prev_hand_centers.get(idx, finger_pos_3d)
            hand_vel_3d = finger_pos_3d - prev_pos
            current_centers[idx] = finger_pos_3d

            colliders.append(HandColliderNode(finger_pos_3d, hand_vel_3d, radius=0.22, is_pinching=is_pinching))
            if not is_pinching:
                colliders.append(HandColliderNode(palm_pos_3d, hand_vel_3d, radius=0.26, is_pinching=False))

            # Dibujar el escudo colisionador en la mano
            fx, fy = int(pinch_mid[0] * w), int(pinch_mid[1] * h)
            shield_col = (0, 255, 140) if is_pinching else (255, 200, 0)
            cv2.circle(frame, (fx, fy), 20 if is_pinching else 28, shield_col, 2, cv2.LINE_AA)

        self.prev_hand_centers = current_centers
        return colliders

    def render_physics_scene(self, frame):
        h, w, _ = frame.shape

        # 1. Dibujar piso de la caja de contención 3D
        floor_corners_3d = [
            [-self.solver.bounds_x, self.solver.bounds_y, -self.solver.bounds_z],
            [ self.solver.bounds_x, self.solver.bounds_y, -self.solver.bounds_z],
            [ self.solver.bounds_x, self.solver.bounds_y,  self.solver.bounds_z],
            [-self.solver.bounds_x, self.solver.bounds_y,  self.solver.bounds_z],
        ]
        floor_2d = self.project_3d(floor_corners_3d, w, h)
        cv2.polylines(frame, [np.array(floor_2d, dtype=np.int32)], True, (90, 70, 40), 1, cv2.LINE_AA)

        # 2. Dibujar líneas de flujo magnético entre hologramas (Verde = Atracción, Rojo = Repulsión)
        for p1_3d, p2_3d, is_attracting in self.solver.magnetic_links:
            l_2d = self.project_3d([p1_3d, p2_3d], w, h)
            mag_color = (80, 255, 120) if is_attracting else (70, 70, 255)
            cv2.line(frame, l_2d[0], l_2d[1], mag_color, 2, cv2.LINE_AA)

        # 3. Dibujar los cuerpos rígidos holográficos
        overlay = frame.copy()
        for name, b in list(self.scene.items()):
            R = rotation_matrix_3d(*b.rotation)
            world_verts = (b.vertices * b.scale) @ R.T + b.position
            pts_2d = self.project_3d(world_verts, w, h)

            for face in b.faces:
                poly = np.array([pts_2d[i] for i in face], dtype=np.int32)
                c = b.color
                cv2.fillPoly(overlay, [poly], (c[0]//4, c[1]//4, c[2]//4))

            for e1, e2 in b.edges:
                cv2.line(frame, pts_2d[e1], pts_2d[e2], b.color, 2, cv2.LINE_AA)

            # Etiqueta con polaridad magnética si está activa
            c2d = self.project_3d([b.position], w, h)[0]
            mag_tag = " [+]" if b.magnetic_charge > 0 else (" [-]" if b.magnetic_charge < 0 else "")
            cv2.putText(frame, f"{b.name}{mag_tag}", (c2d[0] - 35, c2d[1] - int(48 * b.scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

        # 4. Dibujar ondas de choque en puntos de impacto
        for sw in self.solver.shockwaves:
            pt_2d = self.project_3d([sw.pos_3d], w, h)[0]
            cv2.circle(frame, pt_2d, sw.radius_px, sw.color, 2, cv2.LINE_AA)

    def draw_physics_hud(self, frame):
        h, w, _ = frame.shape
        cv2.rectangle(frame, (15, 15), (730, 165), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (730, 165), (0, 230, 255), 1)

        g_status = f"{self.solver.gravity / 0.0025:.1f}G"
        cv2.putText(frame, f"NEXUS 3D PHYSICS & MAGNETISM | Cuerpos: {len(self.scene)} | Gravedad: {g_status}",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 160), 2)

        for idx, line in enumerate(self.interpreter.logs[-3:]):
            cv2.putText(frame, f"> {line}", (26, 65 + idx * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 230, 245), 1)

        cv2.putText(frame, "Atajos: [1] Gravedad ON/OFF  [2] Iman Atractivo  [3] Iman Repulsivo  [4] Explosion  [5] Crear Esfera",
                    (26, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 210, 80), 1)

        # Barra Inferior de Código
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (12, 14, 20), -1)
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (0, 255, 140), 1)
        cursor = "_" if int(time.time() * 2.5) % 2 == 0 else " "
        cv2.putText(frame, f"PHYSICS-CMD >> {self.onscreen_input}{cursor}",
                    (28, h - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 140), 2)
        cv2.putText(frame, "Escribe: IMAN cubo1 POSITIVO ; IMAN esfera1 NEGATIVO   |   GRAVEDAD 1.0   |   REBOTE TODOS 0.98",
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

            # 1. Extraer colisionadores esféricos y velocidades 3D de tus manos
            hand_colliders = self.extract_hand_colliders(results.multi_hand_landmarks, frame)

            # 2. Avanzar la simulación física (Gravedad, Imanes, Choques entre hologramas y manos)
            self.solver.step_simulation(list(self.scene.values()), hand_colliders)

            # 3. Renderizar escena 3D y HUD
            self.render_physics_scene(frame)
            self.draw_physics_hud(frame)

            cv2.imshow("NexusXR - Motor de Fisicas 3D, Colisiones e Imanes", frame)

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
            # Atajos rápidos de físicas cuando no estás escribiendo en la barra
            elif not self.onscreen_input and (key & 0xFF) == ord('1'):
                new_g = 0.0 if self.solver.gravity > 0 else 1.0
                self.cmd_queue.put(f"GRAVEDAD {new_g}")
            elif not self.onscreen_input and (key & 0xFF) == ord('2'):
                # Polaridades opuestas: los hologramas se atraen y chocan entre sí
                names = list(self.scene.keys())
                for idx, n_obj in enumerate(names):
                    pol = "POSITIVO" if idx % 2 == 0 else "NEGATIVO"
                    self.cmd_queue.put(f"IMAN {n_obj} {pol}")
            elif not self.onscreen_input and (key & 0xFF) == ord('3'):
                # Misma polaridad: todos los hologramas se repelen mutuamente
                self.cmd_queue.put("IMAN TODOS POSITIVO")
            elif not self.onscreen_input and (key & 0xFF) == ord('4'):
                self.cmd_queue.put("EXPLOSION")
            elif not self.onscreen_input and (key & 0xFF) == ord('5'):
                new_id = f"b{len(self.scene)+1}"
                self.cmd_queue.put(f"CREAR {new_id} ESFERA")
            elif 32 <= (key & 0xFF) <= 126:
                self.onscreen_input += chr(key & 0xFF)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = NexusInteractivePhysicsEngine()
    engine.run()
