import math
import time
import random
from enum import IntEnum
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =====================================================================
# 1. DEFINICIÓN DE OPCODES Y EMPAQUETADO BINARIO DE 32 BITS
# =====================================================================

class VMOp(IntEnum):
    LOAD_CONST  = 0x01  # Sube const_pool[arg1] al tope de la pila
    LOAD_LOCAL  = 0x02  # Sube locals[arg1] al tope de la pila
    STORE_LOCAL = 0x03  # Guarda el tope de la pila en locals[arg1]
    BINARY_ADD  = 0x04  # Suma los 2 vectores superiores de la pila
    BINARY_SUB  = 0x05  # Resta los 2 vectores superiores de la pila
    BINARY_MUL  = 0x06  # Multiplica los 2 vectores superiores de la pila
    # Superinstrucciones fusionadas por el optimizador JIT (Peephole)
    ADD_STORE   = 0x08  # Suma los 2 topes y guarda directo en locals[arg1]
    SUB_STORE   = 0x09  # Resta los 2 topes y guarda directo en locals[arg1]
    MUL_STORE   = 0x0A  # Multiplica los 2 topes y guarda directo en locals[arg1]
    RETURN_VAL  = 0x0F  # Finaliza y devuelve el vector 3D en el tope de la pila


def pack_word32(op: VMOp, arg1: int = 0, arg2: int = 0) -> int:
    """Codifica [OpCode (8 bits) | Arg1 (12 bits) | Arg2 (12 bits)] en un uint32."""
    return ((int(op) & 0xFF) << 24) | ((arg1 & 0xFFF) << 12) | (arg2 & 0xFFF)


def unpack_word32(word: int) -> Tuple[VMOp, int, int]:
    op = VMOp((word >> 24) & 0xFF)
    arg1 = (word >> 12) & 0xFFF
    arg2 = word & 0xFFF
    return op, arg1, arg2


# =====================================================================
# 2. ÁRBOL SINTÁCTICO ABSTRACTO (AST) MUTABLE POR LA IA
# =====================================================================

class ASTNode:
    def to_source(self) -> str:
        raise NotImplementedError()


@dataclass
class NumberNode(ASTNode):
    value: float
    def to_source(self) -> str:
        return f"{self.value:.3f}"


@dataclass
class IdentNode(ASTNode):
    name: str
    def to_source(self) -> str:
        return self.name


@dataclass
class BinaryOpNode(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode
    def to_source(self) -> str:
        return f"({self.left.to_source()} {self.op} {self.right.to_source()})"


@dataclass
class AssignNode(ASTNode):
    var_name: str
    expr: ASTNode
    def to_source(self) -> str:
        return f"  {self.var_name} = {self.expr.to_source()}"


@dataclass
class ReturnNode(ASTNode):
    expr: ASTNode
    def to_source(self) -> str:
        return f"  RETORNAR {self.expr.to_source()}"


class MutableASTFunction:
    """
    Representa la función 'controlador_espacial(mano, holo, vel_mano)' en forma de AST.
    Cada vez que el agente RL modifica sus parámetros o ramas, reconstruye sus nodos
    para que el Compilador JIT genere un nuevo bloque de Bytecode de 32 bits.
    """
    def __init__(self):
        self.name = "controlador_espacial"
        self.params = ["mano", "holo", "vel_mano"]
        self.version = 1

        self.kp_gain = 0.18
        self.kd_damping = 0.06
        self.nonlinear_turbo = False
        self.body: List[ASTNode] = []
        self.rebuild_ast()

    def rebuild_ast(self):
        self.body = [
            AssignNode("error", BinaryOpNode("-", IdentNode("mano"), IdentNode("holo"))),
            AssignNode("kp", NumberNode(self.kp_gain)),
            AssignNode("kd", NumberNode(self.kd_damping)),
            AssignNode("fuerza_p", BinaryOpNode("*", IdentNode("error"), IdentNode("kp"))),
            AssignNode("fuerza_d", BinaryOpNode("*", IdentNode("vel_mano"), IdentNode("kd"))),
        ]

        if not self.nonlinear_turbo:
            # Rama Lineal Estándar
            self.body.append(
                AssignNode("impulso", BinaryOpNode("+", IdentNode("fuerza_p"), IdentNode("fuerza_d")))
            )
        else:
            # Rama Turbo No-Lineal inyectada en el AST por la IA cuando hay alta velocidad
            boosted_p = BinaryOpNode("*", IdentNode("fuerza_p"), NumberNode(1.55))
            self.body.append(
                AssignNode("impulso", BinaryOpNode("+", boosted_p, IdentNode("fuerza_d")))
            )

        self.body.append(
            ReturnNode(BinaryOpNode("+", IdentNode("holo"), IdentNode("impulso")))
        )

    def get_source_lines(self) -> List[str]:
        lines = [f"FUNCION {self.name}(mano, holo, vel_mano) {{ # [AST v{self.version}]"]
        for stmt in self.body:
            lines.append(stmt.to_source())
        lines.append("}")
        return lines


# =====================================================================
# 3. COMPILADOR JIT DE AST -> BYTECODE DE 32 BITS + STACK VM
# =====================================================================

class CompiledBytecodeProgram:
    def __init__(self, name: str, param_count: int, version: int):
        self.name = name
        self.param_count = param_count
        self.version = version

        self.raw_ops: List[Tuple[VMOp, int, int]] = []
        self.packed_words: np.ndarray = np.zeros(0, dtype=np.uint32)

        self.constants_list: List[np.ndarray] = []
        self.const_pool: np.ndarray = np.zeros((1, 3), dtype=np.float64)

        self.local_names: List[str] = []
        self.name_to_slot: Dict[str, int] = {}

    def resolve_local(self, name: str) -> int:
        if name not in self.name_to_slot:
            idx = len(self.local_names)
            self.name_to_slot[name] = idx
            self.local_names.append(name)
        return self.name_to_slot[name]

    def add_scalar_constant(self, val: float) -> int:
        vec = np.array([val, val, val], dtype=np.float64)
        for i, existing in enumerate(self.constants_list):
            if np.allclose(existing, vec):
                return i
        idx = len(self.constants_list)
        self.constants_list.append(vec)
        return idx

    def emit(self, op: VMOp, arg1: int = 0, arg2: int = 0):
        self.raw_ops.append((op, arg1, arg2))

    def optimize_and_pack(self):
        """
        Optimizador Peephole: Fusiona pares (BINARY_OP + STORE_LOCAL) en una sola
        superinstrucción (ADD_STORE, SUB_STORE, MUL_STORE) y genera el array uint32.
        """
        fuse_table = {
            VMOp.BINARY_ADD: VMOp.ADD_STORE,
            VMOp.BINARY_SUB: VMOp.SUB_STORE,
            VMOp.BINARY_MUL: VMOp.MUL_STORE,
        }
        fused: List[Tuple[VMOp, int, int]] = []
        i = 0
        n = len(self.raw_ops)

        while i < n:
            op, a1, a2 = self.raw_ops[i]
            if i + 1 < n and op in fuse_table and self.raw_ops[i + 1][0] == VMOp.STORE_LOCAL:
                dest_slot = self.raw_ops[i + 1][1]
                fused.append((fuse_table[op], dest_slot, 0))
                i += 2
            else:
                fused.append((op, a1, a2))
                i += 1

        self.raw_ops = fused
        self.packed_words = np.array(
            [pack_word32(op, a1, a2) for op, a1, a2 in self.raw_ops],
            dtype=np.uint32
        )
        if self.constants_list:
            self.const_pool = np.vstack(self.constants_list).astype(np.float64)

    def disassemble_short(self) -> List[str]:
        lines = [f"=== BYTECODE 32-BIT (v{self.version} | {len(self.packed_words)} words) ==="]
        for ip, w in enumerate(self.packed_words):
            op, a1, _ = unpack_word32(int(w))
            hex_s = f"0x{int(w):08X}"
            if op == VMOp.LOAD_CONST:
                info = f"const[{a1}] ({self.const_pool[a1][0]:.2f})"
            elif op in (VMOp.LOAD_LOCAL, VMOp.STORE_LOCAL, VMOp.ADD_STORE, VMOp.SUB_STORE, VMOp.MUL_STORE):
                info = f"${self.local_names[a1]} (slot {a1})"
            else:
                info = ""
            lines.append(f"{ip:02d}: {hex_s}  {op.name:<11} {info}")
        return lines


class JITBytecodeCompiler:
    """Compila instantáneamente cualquier versión mutada del AST a Bytecode binario."""
    @staticmethod
    def compile_ast(ast_fn: MutableASTFunction) -> CompiledBytecodeProgram:
        prog = CompiledBytecodeProgram(ast_fn.name, len(ast_fn.params), ast_fn.version)

        for p in ast_fn.params:
            prog.resolve_local(p)

        for stmt in ast_fn.body:
            if isinstance(stmt, AssignNode):
                JITBytecodeCompiler._compile_expr(stmt.expr, prog)
                slot = prog.resolve_local(stmt.var_name)
                prog.emit(VMOp.STORE_LOCAL, slot)
            elif isinstance(stmt, ReturnNode):
                JITBytecodeCompiler._compile_expr(stmt.expr, prog)
                prog.emit(VMOp.RETURN_VAL)

        prog.optimize_and_pack()
        return prog

    @staticmethod
    def _compile_expr(expr: ASTNode, prog: CompiledBytecodeProgram):
        if isinstance(expr, NumberNode):
            c_idx = prog.add_scalar_constant(expr.value)
            prog.emit(VMOp.LOAD_CONST, c_idx)
        elif isinstance(expr, IdentNode):
            slot = prog.resolve_local(expr.name)
            prog.emit(VMOp.LOAD_LOCAL, slot)
        elif isinstance(expr, BinaryOpNode):
            JITBytecodeCompiler._compile_expr(expr.left, prog)
            JITBytecodeCompiler._compile_expr(expr.right, prog)
            if expr.op == "+": prog.emit(VMOp.BINARY_ADD)
            elif expr.op == "-": prog.emit(VMOp.BINARY_SUB)
            elif expr.op == "*": prog.emit(VMOp.BINARY_MUL)


class FastStackVM:
    """
    Máquina Virtual de Pila Vectorizada con registros estáticos pre-alojados.
    Ejecuta el Bytecode mutado por la IA en microsegundos.
    """
    def __init__(self):
        self.stack = np.zeros((32, 3), dtype=np.float64)
        self.locals = np.zeros((16, 3), dtype=np.float64)
        self.last_exec_us = 0.0
        self.jit_compile_us = 0.0
        self.total_vm_calls = 0

    def execute(self, prog: CompiledBytecodeProgram, mano: np.ndarray, holo: np.ndarray, vel_mano: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()

        stack = self.stack
        loc = self.locals
        consts = prog.const_pool
        code = prog.packed_words

        # Inyectar los 3 vectores de entrada directamente en los registros 0, 1 y 2
        loc[0] = mano
        loc[1] = holo
        loc[2] = vel_mano

        sp = 0
        for word in code:
            w = int(word)
            op = (w >> 24) & 0xFF
            a1 = (w >> 12) & 0xFFF

            if op == VMOp.LOAD_CONST:
                stack[sp] = consts[a1]
                sp += 1
            elif op == VMOp.LOAD_LOCAL:
                stack[sp] = loc[a1]
                sp += 1
            elif op == VMOp.STORE_LOCAL:
                sp -= 1
                loc[a1] = stack[sp]
            elif op == VMOp.BINARY_ADD:
                sp -= 1
                stack[sp - 1] = stack[sp - 1] + stack[sp]
            elif op == VMOp.BINARY_SUB:
                sp -= 1
                stack[sp - 1] = stack[sp - 1] - stack[sp]
            elif op == VMOp.BINARY_MUL:
                sp -= 1
                stack[sp - 1] = stack[sp - 1] * stack[sp]
            elif op == VMOp.ADD_STORE:
                sp -= 2
                loc[a1] = stack[sp] + stack[sp + 1]
            elif op == VMOp.SUB_STORE:
                sp -= 2
                loc[a1] = stack[sp] - stack[sp + 1]
            elif op == VMOp.MUL_STORE:
                sp -= 2
                loc[a1] = stack[sp] * stack[sp + 1]
            elif op == VMOp.RETURN_VAL:
                res = stack[sp - 1].copy()
                self.last_exec_us = (time.perf_counter() - t0) * 1_000_000.0
                self.total_vm_calls += 1
                return res.astype(np.float32)

        self.last_exec_us = (time.perf_counter() - t0) * 1_000_000.0
        return stack[0].copy().astype(np.float32)


# =====================================================================
# 4. AGENTE DE APRENDIZAJE POR REFUERZO (PYTORCH DQN AST-MUTATOR)
# =====================================================================

class ASTArchitectNet(nn.Module):
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


class NeuroJITAgent:
    ACTIONS = {
        0: "MANTENER_BYTECODE_ACTUAL",
        1: "MUTAR_AST -> SUBIR_KP (+0.04) & RECOMPILAR_JIT",
        2: "MUTAR_AST -> BAJAR_KP (-0.04) & RECOMPILAR_JIT",
        3: "MUTAR_AST -> SUBIR_KD (+0.03) & RECOMPILAR_JIT",
        4: "MUTAR_AST -> BAJAR_KD (-0.03) & RECOMPILAR_JIT",
        5: "SINTETIZAR_RAMA -> TURBO_NO_LINEAL & RECOMPILAR_JIT"
    }

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ASTArchitectNet(8, 6).to(self.device)
        self.target_net = ASTArchitectNet(8, 6).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=2e-3)
        self.criterion = nn.SmoothL1Loss()
        self.memory = deque(maxlen=10000)

        self.gamma = 0.95
        self.epsilon = 0.28
        self.last_loss = 0.0
        self.last_reward = 0.0
        self.last_action_desc = "INICIANDO_JIT"
        self.recompilations = 0

    def select_action(self, state_vec: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 5)
        self.policy_net.eval()
        with torch.no_grad():
            t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            return int(torch.argmax(self.policy_net(t)).item())

    def mutate_ast_and_recompile(
        self,
        action: int,
        ast_fn: MutableASTFunction,
        vm: FastStackVM
    ) -> Optional[CompiledBytecodeProgram]:
        """
        Si la acción elegida por PyTorch modifica el AST, dispara inmediatamente
        el compilador JIT y devuelve el nuevo binario de 32 bits.
        """
        mutated = False
        if action == 1 and ast_fn.kp_gain < 0.65:
            ast_fn.kp_gain = round(ast_fn.kp_gain + 0.04, 3); mutated = True
        elif action == 2 and ast_fn.kp_gain > 0.06:
            ast_fn.kp_gain = round(ast_fn.kp_gain - 0.04, 3); mutated = True
        elif action == 3 and ast_fn.kd_damping < 0.45:
            ast_fn.kd_damping = round(ast_fn.kd_damping + 0.03, 3); mutated = True
        elif action == 4 and ast_fn.kd_damping > 0.0:
            ast_fn.kd_damping = round(max(0.0, ast_fn.kd_damping - 0.03), 3); mutated = True
        elif action == 5:
            ast_fn.nonlinear_turbo = not ast_fn.nonlinear_turbo; mutated = True

        self.last_action_desc = self.ACTIONS[action]

        if mutated:
            t_jit = time.perf_counter()
            ast_fn.version += 1
            ast_fn.rebuild_ast()
            new_bytecode = JITBytecodeCompiler.compile_ast(ast_fn)
            vm.jit_compile_us = (time.perf_counter() - t_jit) * 1_000_000.0
            self.recompilations += 1
            return new_bytecode
        return None

    def learn(self, state, action, reward, next_state, batch_size: int = 24):
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
        if self.recompilations % 15 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# =====================================================================
# 5. MOTOR HOLOGRÁFICO 3D EJECUTANDO EL BYTECODE AUTO-MUTADO
# =====================================================================

def rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


class NexusNeuroJITEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        # 1. Crear el AST vivo, compilar el primer bloque Bytecode y arrancar la Stack VM
        self.ast_fn = MutableASTFunction()
        self.vm = FastStackVM()
        self.active_bytecode: CompiledBytecodeProgram = JITBytecodeCompiler.compile_ast(self.ast_fn)
        self.rl_agent = NeuroJITAgent()

        # Estado espacial 3D
        self.holo_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_holo_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.holo_rot = np.array([0.3, 0.5, 0.1], dtype=np.float32)

        self.prev_hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_hand_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_state: Optional[np.ndarray] = None
        self.prev_action: int = 0
        self.frame_count = 0

        self.cube_verts = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ], dtype=np.float32) * 0.48
        self.cube_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        self.cube_faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6)]

    def build_state(self, hand_pos: np.ndarray, hand_vel: np.ndarray, hand_acc: np.ndarray, jerk: float) -> np.ndarray:
        err = float(np.linalg.norm(hand_pos - self.holo_pos))
        return np.array([
            err,
            float(np.linalg.norm(hand_vel)),
            float(np.linalg.norm(hand_acc)),
            jerk,
            self.ast_fn.kp_gain,
            self.ast_fn.kd_damping,
            1.0 if self.ast_fn.nonlinear_turbo else 0.0,
            float(hand_pos[2])
        ], dtype=np.float32)

    def compute_reward(self, err: float, jerk: float, speed: float) -> float:
        precision = math.exp(-3.5 * err) * 2.0
        stability_pen = 0.8 * jerk
        turbo_bonus = 0.35 if (speed > 0.08 and self.ast_fn.nonlinear_turbo) else 0.0
        return float(precision - stability_pen + turbo_bonus)

    def project_3d(self, pts_3d, w, h, focal=640.0, cam_dist=3.5):
        out = []
        for pt in pts_3d:
            z = max(cam_dist + pt[2], 0.35)
            out.append((int((pt[0] * focal) / z + w * 0.5), int((pt[1] * focal) / z + h * 0.5)))
        return out

    def draw_hologram(self, frame, hand_pos_3d: Optional[np.ndarray]):
        h, w, _ = frame.shape
        self.holo_rot += np.array([0.01, 0.02, 0.005], dtype=np.float32)
        R = rotation_matrix_3d(*self.holo_rot)
        world_verts = self.cube_verts @ R.T + self.holo_pos
        pts_2d = self.project_3d(world_verts, w, h)

        color = (255, 85, 255) if self.ast_fn.nonlinear_turbo else (255, 230, 0)
        overlay = frame.copy()
        for face in self.cube_faces:
            poly = np.array([pts_2d[i] for i in face], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (color[0]//4, color[1]//4, color[2]//4))
        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

        for e1, e2 in self.cube_edges:
            cv2.line(frame, pts_2d[e1], pts_2d[e2], color, 2, cv2.LINE_AA)

        if hand_pos_3d is not None:
            h2d = self.project_3d([self.holo_pos], w, h)[0]
            m2d = self.project_3d([hand_pos_3d], w, h)[0]
            cv2.line(frame, m2d, h2d, (0, 255, 140), 2, cv2.LINE_AA)
            cv2.circle(frame, m2d, 8, (0, 255, 140), -1, cv2.LINE_AA)

    def draw_neuro_jit_hud(self, frame):
        h, w, _ = frame.shape

        # Panel Izquierdo: Telemetría del Agente RL + Tiempos de Stack VM en microsegundos
        cv2.rectangle(frame, (15, 15), (615, 175), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (615, 175), (0, 255, 150), 1)
        cv2.putText(frame, f"NEURO-JIT STACK VM | Ejecucion VM: {self.vm.last_exec_us:.1f} us | JIT Compile: {self.vm.jit_compile_us:.1f} us",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 150), 2)
        cv2.putText(frame, f"Accion IA: {self.rl_agent.last_action_desc}",
                    (26, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 90), 1)
        cv2.putText(frame, f"Recompensa RL: {self.rl_agent.last_reward:+.2f} | Loss DQN: {self.rl_agent.last_loss:.4f} | Recompilaciones JIT: {self.rl_agent.recompilations}",
                    (26, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 225, 235), 1)
        cv2.putText(frame, f"Palabras Binarias Activas: {len(self.active_bytecode.packed_words)} uint32 | Llamadas VM: {self.vm.total_vm_calls}",
                    (26, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 220, 255), 1)
        cv2.putText(frame, "Mueve tu mano rapido o lento: la IA muta el AST y recompila el Bytecode en vivo ->",
                    (26, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (140, 255, 180), 1)

        # Panel Derecho Superior: Árbol AST Mutado por PyTorch
        ast_lines = self.ast_fn.get_source_lines()
        cv2.rectangle(frame, (w - 570, 15), (w - 15, 225), (14, 14, 22), -1)
        border_c = (255, 85, 255) if self.ast_fn.nonlinear_turbo else (255, 200, 0)
        cv2.rectangle(frame, (w - 570, 15), (w - 15, 225), border_c, 1)
        cv2.putText(frame, f"1. AST SIMBOLICO MUTADO POR IA (v{self.ast_fn.version})",
                    (w - 555, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.46, border_c, 2)
        for i, l_str in enumerate(ast_lines):
            cv2.putText(frame, l_str, (w - 555, 62 + i * 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (225, 235, 245), 1)

        # Panel Derecho Inferior: Instrucciones Binarias de 32 Bits en la Stack VM
        asm_lines = self.active_bytecode.disassemble_short()
        box_top = 240
        box_h = 30 + len(asm_lines) * 19
        cv2.rectangle(frame, (w - 570, box_top), (w - 15, box_top + box_h), (12, 16, 20), -1)
        cv2.rectangle(frame, (w - 570, box_top), (w - 15, box_top + box_h), (0, 230, 255), 1)
        for i, asm in enumerate(asm_lines):
            col = (0, 255, 180) if i == 0 else ((120, 255, 140) if "_STORE" in asm else (200, 215, 230))
            cv2.putText(frame, asm, (w - 555, box_top + 24 + i * 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39, col, 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("==========================================================================")
        print(" NEXUS NEURO-JIT ENGINE: PYTORCH RL -> AST -> 32-BIT BYTECODE -> STACK VM")
        print(" - Mueve tu mano frente a la cámara.")
        print(" - Cada vez que la IA decide mutar el AST, el compilador JIT genera un")
        print("   nuevo bloque binario de 32 bits y lo inyecta en la Máquina Virtual.")
        print("==========================================================================")

        while cap.isOpened():
            success, raw = cap.read()
            if not success:
                break

            frame = cv2.flip(raw, 1)
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.frame_count += 1

            hand_pos_3d = None

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                mid = (pts[4] + pts[8]) * 0.5
                hand_pos_3d = np.array([(mid[0] - 0.5) * 3.2, (mid[1] - 0.5) * 2.0, mid[2] * 1.4], dtype=np.float32)

                hand_vel = hand_pos_3d - self.prev_hand_pos
                hand_acc = hand_vel - self.prev_hand_vel
                speed = float(np.linalg.norm(hand_vel))

                # 1. EJECUTAR EL BYTECODE DE 32 BITS EN LA MÁQUINA VIRTUAL DE PILA
                new_holo_pos = self.vm.execute(
                    self.active_bytecode,
                    hand_pos_3d.astype(np.float64),
                    self.holo_pos.astype(np.float64),
                    hand_vel.astype(np.float64)
                )

                holo_vel = new_holo_pos - self.holo_pos
                holo_jerk = float(np.linalg.norm(holo_vel - self.prev_holo_vel))
                self.holo_pos = new_holo_pos
                self.prev_holo_vel = holo_vel

                # 2. CICLO DE APRENDIZAJE POR REFUERZO Y RECOMPILACIÓN JIT (Cada 6 frames)
                err = float(np.linalg.norm(hand_pos_3d - self.holo_pos))
                curr_state = self.build_state(hand_pos_3d, hand_vel, hand_acc, holo_jerk)

                if self.frame_count % 6 == 0:
                    if self.prev_state is not None:
                        reward = self.compute_reward(err, holo_jerk, speed)
                        self.rl_agent.learn(self.prev_state, self.prev_action, reward, curr_state)

                    action = self.rl_agent.select_action(curr_state)
                    maybe_new_bytecode = self.rl_agent.mutate_ast_and_recompile(action, self.ast_fn, self.vm)
                    if maybe_new_bytecode is not None:
                        # Reemplazo atómico en caliente del binario en la Stack VM
                        self.active_bytecode = maybe_new_bytecode

                    self.prev_state = curr_state
                    self.prev_action = action

                self.prev_hand_pos = hand_pos_3d.copy()
                self.prev_hand_vel = hand_vel.copy()

            self.draw_hologram(frame, hand_pos_3d)
            self.draw_neuro_jit_hud(frame)
            cv2.imshow("NexusXR - IA Mutando AST a Bytecode en Stack VM", frame)

            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = NexusNeuroJITEngine()
    engine.run()
