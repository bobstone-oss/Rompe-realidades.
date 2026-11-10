import math
import time
import random
import copy
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =====================================================================
# 1. TIPO VECTORIAL 3D Y NODOS DEL ÁRBOL SINTÁCTICO (AST VIVO)
# =====================================================================

class Vec3:
    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @staticmethod
    def from_numpy(arr: np.ndarray):
        return Vec3(float(arr[0]), float(arr[1]), float(arr[2]))

    def __add__(self, o):
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z) if isinstance(o, Vec3) else Vec3(self.x + o, self.y + o, self.z + o)

    def __sub__(self, o):
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z) if isinstance(o, Vec3) else Vec3(self.x - o, self.y - o, self.z - o)

    def __mul__(self, o):
        return Vec3(self.x * o.x, self.y * o.y, self.z * o.z) if isinstance(o, Vec3) else Vec3(self.x * float(o), self.y * float(o), self.z * float(o))

    def __rmul__(self, o):
        return self.__mul__(o)

    def __repr__(self):
        return f"[{self.x:.2f}, {self.y:.2f}, {self.z:.2f}]"


class ASTNode:
    def eval(self, env: Dict[str, Any]) -> Any:
        raise NotImplementedError()

    def to_code(self, indent: int = 0) -> str:
        raise NotImplementedError()


@dataclass
class NumberNode(ASTNode):
    value: float
    def eval(self, env: Dict[str, Any]) -> float:
        return self.value
    def to_code(self, indent: int = 0) -> str:
        return f"{self.value:.3f}"


@dataclass
class IdentNode(ASTNode):
    name: str
    def eval(self, env: Dict[str, Any]) -> Any:
        return env[self.name]
    def to_code(self, indent: int = 0) -> str:
        return self.name


@dataclass
class BinaryOpNode(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode

    def eval(self, env: Dict[str, Any]) -> Any:
        l_val = self.left.eval(env)
        r_val = self.right.eval(env)
        if self.op == "+": return l_val + r_val
        if self.op == "-": return l_val - r_val
        if self.op == "*": return l_val * r_val
        return l_val

    def to_code(self, indent: int = 0) -> str:
        return f"({self.left.to_code()} {self.op} {self.right.to_code()})"


@dataclass
class AssignNode(ASTNode):
    var_name: str
    expr: ASTNode

    def eval(self, env: Dict[str, Any]) -> Any:
        val = self.expr.eval(env)
        env[self.var_name] = val
        return val

    def to_code(self, indent: int = 0) -> str:
        pad = "  " * indent
        return f"{pad}{self.var_name} = {self.expr.to_code()}"


@dataclass
class ReturnNode(ASTNode):
    expr: ASTNode
    def eval(self, env: Dict[str, Any]) -> Any:
        return self.expr.eval(env)
    def to_code(self, indent: int = 0) -> str:
        pad = "  " * indent
        return f"{pad}RETORNAR {self.expr.to_code()}"


class MutableASTFunction:
    """
    Función de NexusLang representada como un Árbol Sintáctico Abstracto vivo.
    La IA en PyTorch puede inspeccionar y reescribir sus nodos en caliente.
    """
    def __init__(self):
        self.name = "controlador_espacial"
        self.version = 1
        self.kp_gain = 0.18       # Ganancia proporcional en el AST
        self.kd_damping = 0.05    # Factor predictivo de velocidad en el AST
        self.nonlinear_boost = False
        self.body: List[ASTNode] = []
        self.rebuild_ast()

    def rebuild_ast(self):
        """Reconstruye los nodos del AST según las mutaciones decididas por la IA."""
        self.body = [
            # error = mano - holo
            AssignNode("error", BinaryOpNode("-", IdentNode("mano"), IdentNode("holo"))),
            # ganancia_kp = <valor mutado por IA>
            AssignNode("ganancia_kp", NumberNode(self.kp_gain)),
            # amortiguacion_kd = <valor mutado por IA>
            AssignNode("amortiguacion_kd", NumberNode(self.kd_damping)),
            # fuerza_base = error * ganancia_kp
            AssignNode("fuerza_base", BinaryOpNode("*", IdentNode("error"), IdentNode("ganancia_kp"))),
            # prediccion = vel_mano * amortiguacion_kd
            AssignNode("prediccion", BinaryOpNode("*", IdentNode("vel_mano"), IdentNode("amortiguacion_kd"))),
        ]

        if not self.nonlinear_boost:
            # Rama Lineal Estándar: nueva_pos = holo + (fuerza_base + prediccion)
            self.body.append(
                AssignNode("impulso_total", BinaryOpNode("+", IdentNode("fuerza_base"), IdentNode("prediccion")))
            )
        else:
            # Rama Turbo No-Lineal (Inyectada por la IA cuando detecta movimientos rápidos):
            # impulso_total = (fuerza_base * 1.45) + prediccion
            boosted_force = BinaryOpNode("*", IdentNode("fuerza_base"), NumberNode(1.45))
            self.body.append(
                AssignNode("impulso_total", BinaryOpNode("+", boosted_force, IdentNode("prediccion")))
            )

        self.body.append(
            ReturnNode(BinaryOpNode("+", IdentNode("holo"), IdentNode("impulso_total")))
        )

    def execute(self, mano: Vec3, holo: Vec3, vel_mano: Vec3) -> Vec3:
        env = {"mano": mano, "holo": holo, "vel_mano": vel_mano}
        result = holo
        for stmt in self.body:
            res = stmt.eval(env)
            if isinstance(stmt, ReturnNode):
                result = res
        return result

    def get_source_lines(self) -> List[str]:
        lines = [f"FUNCION {self.name}(mano, holo, vel_mano) {{ # [AST v{self.version}]"]
        for stmt in self.body:
            lines.append(stmt.to_code(indent=1))
        lines.append("}")
        return lines


# =====================================================================
# 2. AGENTE DE APRENDIZAJE POR REFUERZO (PYTORCH DQN AST-ARCHITECT)
# =====================================================================

class ASTArchitectPolicyNet(nn.Module):
    """
    Red Neuronal de Aprendizaje por Refuerzo que observa la cinemática de la mano
    y elige cómo mutar el Árbol Sintáctico Abstracto (AST).
    """
    def __init__(self, state_dim: int = 8, action_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuroSymbolicRLAgent:
    ACTIONS = {
        0: "MANTENER_AST_ESTABLE",
        1: "MUTAR_AST: INCREMENTAR_KP (+0.04)",
        2: "MUTAR_AST: REDUCIR_KP (-0.04)",
        3: "MUTAR_AST: INCREMENTAR_PREDICCION_KD (+0.03)",
        4: "MUTAR_AST: REDUCIR_PREDICCION_KD (-0.03)",
        5: "SINTETIZAR_RAMA: CONMUTAR_TURBO_NO_LINEAL"
    }

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ASTArchitectPolicyNet(8, 6).to(self.device)
        self.target_net = ASTArchitectPolicyNet(8, 6).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=2e-3)
        self.criterion = nn.SmoothL1Loss()
        self.memory = deque(maxlen=10000)

        self.gamma = 0.95
        self.epsilon = 0.28  # Exploración inicial para probar mutaciones de código
        self.last_loss = 0.0
        self.last_reward = 0.0
        self.last_action_name = "INICIANDO"
        self.total_mutations = 0

    def select_action(self, state_vec: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 5)
        self.policy_net.eval()
        with torch.no_grad():
            t_state = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(t_state)
            return int(torch.argmax(q_vals).item())

    def apply_ast_mutation(self, action: int, ast_func: MutableASTFunction) -> bool:
        """Modifica quirúrgicamente los nodos del AST según la decisión de la red neuronal."""
        mutated = False
        if action == 1 and ast_func.kp_gain < 0.65:
            ast_func.kp_gain = round(ast_func.kp_gain + 0.04, 3)
            mutated = True
        elif action == 2 and ast_func.kp_gain > 0.06:
            ast_func.kp_gain = round(ast_func.kp_gain - 0.04, 3)
            mutated = True
        elif action == 3 and ast_func.kd_damping < 0.45:
            ast_func.kd_damping = round(ast_func.kd_damping + 0.03, 3)
            mutated = True
        elif action == 4 and ast_func.kd_damping > 0.0:
            ast_func.kd_damping = round(max(0.0, ast_func.kd_damping - 0.03), 3)
            mutated = True
        elif action == 5:
            ast_func.nonlinear_boost = not ast_func.nonlinear_boost
            mutated = True

        if mutated:
            ast_func.version += 1
            ast_func.rebuild_ast()
            self.total_mutations += 1

        self.last_action_name = self.ACTIONS[action]
        return mutated

    def learn_step(self, state, action, reward, next_state, batch_size: int = 24):
        self.last_reward = reward
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) < batch_size:
            return

        self.policy_net.train()
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        s_t = torch.FloatTensor(np.array(states)).to(self.device)
        a_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns_t = torch.FloatTensor(np.array(next_states)).to(self.device)

        curr_q = self.policy_net(s_t).gather(1, a_t)
        with torch.no_grad():
            next_q = self.target_net(ns_t).max(1, keepdim=True)[0]
            target_q = r_t + self.gamma * next_q

        loss = self.criterion(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_loss = float(loss.item())
        self.epsilon = max(0.05, self.epsilon * 0.996)

        if self.total_mutations % 15 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# =====================================================================
# 3. MOTOR VISUAL 3D CON TELEMETRÍA DE MUTACIÓN DE CÓDIGO EN VIVO
# =====================================================================

def rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


class NeuroSymbolicXREngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.ast_function = MutableASTFunction()
        self.rl_agent = NeuroSymbolicRLAgent()

        # Estado espacial del holograma y cinemática de la mano
        self.holo_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_holo_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.holo_rot = np.array([0.3, 0.5, 0.1], dtype=np.float32)

        self.prev_hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_hand_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_state: Optional[np.ndarray] = None
        self.prev_action: int = 0
        self.frame_counter = 0

        # Geometría del Cubo Holográfico
        self.cube_verts = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ], dtype=np.float32) * 0.48
        self.cube_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        self.cube_faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6)]

    def build_kinematic_state(self, hand_pos: np.ndarray, hand_vel: np.ndarray, hand_acc: np.ndarray, holo_jerk: float) -> np.ndarray:
        tracking_error = float(np.linalg.norm(hand_pos - self.holo_pos))
        speed = float(np.linalg.norm(hand_vel))
        accel = float(np.linalg.norm(hand_acc))
        return np.array([
            tracking_error,
            speed,
            accel,
            holo_jerk,
            self.ast_function.kp_gain,
            self.ast_function.kd_damping,
            1.0 if self.ast_function.nonlinear_boost else 0.0,
            float(hand_pos[2])
        ], dtype=np.float32)

    def compute_rl_reward(self, tracking_error: float, holo_jerk: float, hand_speed: float) -> float:
        """
        Recompensa alta cuando el AST logra que el holograma esté pegado a la mano (bajo error)
        y penaliza si el código mutado causa temblores u oscilaciones (alto jerk).
        """
        precision_reward = math.exp(-3.5 * tracking_error) * 2.0
        stability_penalty = 0.8 * holo_jerk
        # Premiar el uso de la rama turbo cuando la mano se mueve muy rápido
        turbo_bonus = 0.35 if (hand_speed > 0.08 and self.ast_function.nonlinear_boost) else 0.0
        return float(precision_reward - stability_penalty + turbo_bonus)

    def project_3d(self, pts_3d, w, h, focal=640.0, cam_dist=3.5):
        out = []
        for pt in pts_3d:
            z = max(cam_dist + pt[2], 0.35)
            out.append((int((pt[0] * focal) / z + w * 0.5), int((pt[1] * focal) / z + h * 0.5)))
        return out

    def draw_hologram_and_vectors(self, frame, hand_pos_3d: Optional[np.ndarray]):
        h, w, _ = frame.shape
        self.holo_rot += np.array([0.01, 0.02, 0.005], dtype=np.float32)
        R = rotation_matrix_3d(*self.holo_rot)
        world_verts = self.cube_verts @ R.T + self.holo_pos
        pts_2d = self.project_3d(world_verts, w, h)

        # Color del holograma cambia según si la IA activó la rama Turbo No-Lineal en el AST
        color = (255, 90, 255) if self.ast_function.nonlinear_boost else (255, 230, 0)

        overlay = frame.copy()
        for face in self.cube_faces:
            poly = np.array([pts_2d[i] for i in face], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (color[0]//4, color[1]//4, color[2]//4))
        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

        for e1, e2 in self.cube_edges:
            cv2.line(frame, pts_2d[e1], pts_2d[e2], color, 2, cv2.LINE_AA)

        # Dibujar vector de error entre la mano y el holograma
        if hand_pos_3d is not None:
            holo_2d = self.project_3d([self.holo_pos], w, h)[0]
            hand_2d = self.project_3d([hand_pos_3d], w, h)[0]
            cv2.line(frame, hand_2d, holo_2d, (0, 255, 140), 2, cv2.LINE_AA)
            cv2.circle(frame, hand_2d, 8, (0, 255, 140), -1, cv2.LINE_AA)

    def draw_neuro_ast_hud(self, frame, speed: float, error: float):
        h, w, _ = frame.shape

        # Panel Izquierdo: Telemetría de Aprendizaje por Refuerzo (PyTorch DQN)
        cv2.rectangle(frame, (15, 15), (560, 175), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (560, 175), (0, 255, 150), 1)
        cv2.putText(frame, "AGENTE RL PYTORCH (ARQUITECTO DE AST)", (28, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 2)
        cv2.putText(frame, f"Decision IA: {self.rl_agent.last_action_name}", (28, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 220, 90), 1)
        cv2.putText(frame, f"Recompensa RL: {self.rl_agent.last_reward:+.2f} | Loss DQN: {self.rl_agent.last_loss:.4f} | Mutaciones: {self.rl_agent.total_mutations}",
                    (28, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        cv2.putText(frame, f"Velocidad Mano: {speed:.3f} | Error Seguimiento: {error:.3f}",
                    (28, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 230, 255), 1)
        cv2.putText(frame, "Mueve la mano rapido o lento y observa como la IA reescribe el codigo ->",
                    (28, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 255, 180), 1)

        # Panel Derecho: Inspector en Vivo del Código AST Auto-Reescrito
        cv2.rectangle(frame, (w - 600, 15), (w - 15, 275), (14, 14, 22), -1)
        border_col = (255, 90, 255) if self.ast_function.nonlinear_boost else (255, 200, 0)
        cv2.rectangle(frame, (w - 600, 15), (w - 15, 275), border_col, 1)

        cv2.putText(frame, f"CODIGO AST EN VIVO (Auto-Modificado v{self.ast_function.version})",
                    (w - 585, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, border_col, 2)

        source_lines = self.ast_function.get_source_lines()
        for idx, code_line in enumerate(source_lines):
            line_col = (120, 255, 140) if ("ganancia_kp" in code_line or "impulso_total" in code_line) else (230, 230, 240)
            cv2.putText(frame, code_line, (w - 585, 70 + idx * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, line_col, 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("=========================================================================")
        print(" NEXUS NEURO-SYMBOLIC ENGINE: RL + AST AUTO-MUTANTE")
        print(" - Mueve tu mano frente a la cámara.")
        print(" - La red neuronal en PyTorch medirá tu velocidad, aceleración y error,")
        print("   y reescribirá en vivo la función 'controlador_espacial' del AST.")
        print(" - Presiona 'q' o ESC para salir.")
        print("=========================================================================")

        while cap.isOpened():
            success, raw = cap.read()
            if not success:
                break

            frame = cv2.flip(raw, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            self.frame_counter += 1

            hand_pos_3d = None
            speed = 0.0
            tracking_error = float(np.linalg.norm(self.prev_hand_pos - self.holo_pos))

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                # Coordenada 3D del centro de interacción (punto medio entre pulgar e índice)
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                mid = (pts[4] + pts[8]) * 0.5
                hand_pos_3d = np.array([(mid[0] - 0.5) * 3.2, (mid[1] - 0.5) * 2.0, mid[2] * 1.4], dtype=np.float32)

                # Cinemática instantánea de la mano
                hand_vel = hand_pos_3d - self.prev_hand_pos
                hand_acc = hand_vel - self.prev_hand_vel
                speed = float(np.linalg.norm(hand_vel))

                # 1. EJECUTAR EL AST ACTUAL PARA MOVER EL HOLOGRAMA
                new_holo_vec = self.ast_function.execute(
                    Vec3.from_numpy(hand_pos_3d),
                    Vec3.from_numpy(self.holo_pos),
                    Vec3.from_numpy(hand_vel)
                ).to_numpy()

                holo_vel = new_holo_vec - self.holo_pos
                holo_jerk = float(np.linalg.norm(holo_vel - self.prev_holo_vel))
                self.holo_pos = new_holo_vec
                self.prev_holo_vel = holo_vel

                # 2. CICLO DE APRENDIZAJE POR REFUERZO Y MUTACIÓN DEL AST (Cada 6 fotogramas)
                tracking_error = float(np.linalg.norm(hand_pos_3d - self.holo_pos))
                curr_state = self.build_kinematic_state(hand_pos_3d, hand_vel, hand_acc, holo_jerk)

                if self.frame_counter % 6 == 0:
                    if self.prev_state is not None:
                        reward = self.compute_rl_reward(tracking_error, holo_jerk, speed)
                        self.rl_agent.learn_step(self.prev_state, self.prev_action, reward, curr_state)

                    action = self.rl_agent.select_action(curr_state)
                    self.rl_agent.apply_ast_mutation(action, self.ast_function)
                    self.prev_state = curr_state
                    self.prev_action = action

                self.prev_hand_pos = hand_pos_3d.copy()
                self.prev_hand_vel = hand_vel.copy()

            self.draw_hologram_and_vectors(frame, hand_pos_3d)
            self.draw_neuro_ast_hud(frame, speed, tracking_error)

            cv2.imshow("NexusXR - IA Reescribiendo el AST en Tiempo Real", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = NeuroSymbolicXREngine()
    engine.run()
