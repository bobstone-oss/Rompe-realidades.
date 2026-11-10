import os
import math
import time
import copy
import json
import random
from enum import IntEnum
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =====================================================================
# 1. ISA DE 32 BITS Y EMPAQUETADO BINARIO PARA LA STACK VM
# =====================================================================

class VMOp(IntEnum):
    LOAD_CONST  = 0x01
    LOAD_LOCAL  = 0x02
    STORE_LOCAL = 0x03
    BINARY_ADD  = 0x04
    BINARY_SUB  = 0x05
    BINARY_MUL  = 0x06
    ADD_STORE   = 0x08
    SUB_STORE   = 0x09
    MUL_STORE   = 0x0A
    RETURN_VAL  = 0x0F


def pack_word32(op: VMOp, arg1: int = 0, arg2: int = 0) -> int:
    return ((int(op) & 0xFF) << 24) | ((arg1 & 0xFFF) << 12) | (arg2 & 0xFFF)


# =====================================================================
# 2. NODOS DEL ÁRBOL SINTÁCTICO (AST) CON INSPECCIÓN RECURSIVA
# =====================================================================

class ASTNode:
    def to_source(self) -> str:
        raise NotImplementedError()

    def depth(self) -> int:
        return 1

    def node_count(self) -> int:
        return 1


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

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def node_count(self) -> int:
        return 1 + self.left.node_count() + self.right.node_count()


@dataclass
class AssignNode(ASTNode):
    var_name: str
    expr: ASTNode

    def to_source(self) -> str:
        return f"  {self.var_name} = {self.expr.to_source()}"

    def node_count(self) -> int:
        return 1 + self.expr.node_count()


@dataclass
class ReturnNode(ASTNode):
    expr: ASTNode

    def to_source(self) -> str:
        return f"  RETORNAR {self.expr.to_source()}"

    def node_count(self) -> int:
        return 1 + self.expr.node_count()


# =====================================================================
# 3. INDIVIDUO AST Y OPERADORES GENÉTICOS (KOZA SUBTREE CROSSOVER)
# =====================================================================

TERMINAL_VARS = ["error", "vel_mano", "vel_holo"]
OPERATORS = ["+", "-", "*"]


class EvolvableASTFunction:
    """
    Representa una función AST evolutiva donde el árbol de cálculo de fuerza
    (`force_expr_tree`) tiene topología libre y puede cruzarse en cualquier nodo.
    """
    def __init__(
        self,
        ind_id: int,
        generation: int = 1,
        force_tree: Optional[ASTNode] = None,
        parent_info: str = "Semilla Inicial"
    ):
        self.ind_id = ind_id
        self.generation = generation
        self.parent_info = parent_info
        self.params = ["mano", "holo", "vel_mano", "vel_holo"]

        self.force_tree: ASTNode = force_tree if force_tree is not None else self.generate_random_tree(max_depth=3)
        self.body: List[ASTNode] = []
        self.rebuild_body()

    @staticmethod
    def generate_random_tree(max_depth: int = 3) -> ASTNode:
        """Genera un árbol matemático sintácticamente válido para física 3D."""
        if max_depth <= 1:
            var = random.choice(TERMINAL_VARS)
            gain = round(random.uniform(0.04, 0.48), 3)
            return BinaryOpNode("*", IdentNode(var), NumberNode(gain))

        choice = random.random()
        if choice < 0.35:
            var = random.choice(TERMINAL_VARS)
            gain = round(random.uniform(0.05, 0.52), 3)
            return BinaryOpNode("*", IdentNode(var), NumberNode(gain))
        elif choice < 0.75:
            # Combinación aditiva/sustractiva de dos sub-ramas vectoriales
            op = random.choice(["+", "-"])
            left = EvolvableASTFunction.generate_random_tree(max_depth - 1)
            right = EvolvableASTFunction.generate_random_tree(max_depth - 1)
            return BinaryOpNode(op, left, right)
        else:
            # Escalamiento de una sub-rama por un factor constante
            sub = EvolvableASTFunction.generate_random_tree(max_depth - 1)
            scale = round(random.uniform(0.5, 1.45), 3)
            return BinaryOpNode("*", sub, NumberNode(scale))

    def rebuild_body(self):
        self.body = [
            AssignNode("error", BinaryOpNode("-", IdentNode("mano"), IdentNode("holo"))),
            AssignNode("impulso_genetico", copy.deepcopy(self.force_tree)),
            ReturnNode(BinaryOpNode("+", IdentNode("holo"), IdentNode("impulso_genetico")))
        ]

    def complexity(self) -> int:
        return sum(s.node_count() for s in self.body)

    def get_source_lines(self) -> List[str]:
        lines = [
            f"# Origen: {self.parent_info}",
            f"FUNCION ast_individuo_{self.ind_id}_gen_{self.generation}(mano, holo, vel_mano, vel_holo) {{"
        ]
        for s in self.body:
            lines.append(s.to_source())
        lines.append("}")
        return lines


class KozaGeneticEngine:
    """Implementa Cruce de Sub-Árboles en Nodos Arbitrarios, Mutación y Poda Algebraica."""

    @staticmethod
    def collect_vector_subtrees(node: ASTNode) -> List[ASTNode]:
        """Recolecta todas las sub-ramas vectoriales válidas dentro de un árbol AST."""
        subtrees = [node]
        if isinstance(node, BinaryOpNode):
            if node.op in ("+", "-"):
                subtrees.extend(KozaGeneticEngine.collect_vector_subtrees(node.left))
                subtrees.extend(KozaGeneticEngine.collect_vector_subtrees(node.right))
            elif node.op == "*" and isinstance(node.left, BinaryOpNode):
                subtrees.extend(KozaGeneticEngine.collect_vector_subtrees(node.left))
        return subtrees

    @staticmethod
    def replace_random_subtree(root: ASTNode, donor: ASTNode, current_depth: int = 0) -> ASTNode:
        """Inserta la rama 'donor' del Padre B dentro de un punto de corte del Padre A."""
        if current_depth >= 3 or random.random() < 0.35:
            return copy.deepcopy(donor)

        if isinstance(root, BinaryOpNode) and root.op in ("+", "-"):
            if random.random() < 0.5:
                new_left = KozaGeneticEngine.replace_random_subtree(root.left, donor, current_depth + 1)
                return BinaryOpNode(root.op, new_left, copy.deepcopy(root.right))
            else:
                new_right = KozaGeneticEngine.replace_random_subtree(root.right, donor, current_depth + 1)
                return BinaryOpNode(root.op, copy.deepcopy(root.left), new_right)

        return BinaryOpNode("+", copy.deepcopy(root), BinaryOpNode("*", copy.deepcopy(donor), NumberNode(0.35)))

    @classmethod
    def crossover(
        cls,
        parent_1: EvolvableASTFunction,
        parent_2: EvolvableASTFunction,
        child_id: int,
        generation: int
    ) -> EvolvableASTFunction:
        donor_candidates = cls.collect_vector_subtrees(parent_2.force_tree)
        donor_branch = random.choice(donor_candidates)

        crossed_tree = cls.replace_random_subtree(parent_1.force_tree, donor_branch)

        # Control de profundidad máxima (Anti-Bloat): si supera profundidad 5, compactar
        if crossed_tree.depth() > 5:
            crossed_tree = BinaryOpNode(
                "+",
                BinaryOpNode("*", copy.deepcopy(parent_1.force_tree), NumberNode(0.6)),
                BinaryOpNode("*", copy.deepcopy(donor_branch), NumberNode(0.4))
            )

        origin = f"Cruce(AST#{parent_1.ind_id} x AST#{parent_2.ind_id})"
        return EvolvableASTFunction(child_id, generation, crossed_tree, parent_info=origin)

    @classmethod
    def mutate(cls, indiv: EvolvableASTFunction, rate: float = 0.40):
        def _mutate_recursive(node: ASTNode, depth: int) -> ASTNode:
            if isinstance(node, NumberNode):
                if random.random() < rate:
                    perturbed = float(np.clip(node.value + random.gauss(0.0, 0.07), 0.01, 1.35))
                    return NumberNode(round(perturbed, 3))
                return node

            if isinstance(node, IdentNode):
                if node.name in TERMINAL_VARS and random.random() < (rate * 0.35):
                    return IdentNode(random.choice(TERMINAL_VARS))
                return node

            if isinstance(node, BinaryOpNode):
                # Posibilidad de mutar una sub-rama entera por un sub-árbol fresco
                if depth >= 2 and random.random() < 0.15:
                    return EvolvableASTFunction.generate_random_tree(max_depth=1)

                new_op = node.op
                if node.op in ("+", "-") and random.random() < 0.20:
                    new_op = "-" if node.op == "+" else "+"
                return BinaryOpNode(
                    new_op,
                    _mutate_recursive(node.left, depth + 1),
                    _mutate_recursive(node.right, depth + 1)
                )
            return node

        indiv.force_tree = _mutate_recursive(indiv.force_tree, 0)
        indiv.rebuild_body()


# =====================================================================
# 4. COMPILADOR JIT DE 32 BITS Y STACK VM DE EJECUCIÓN EN LOTE
# =====================================================================

class BytecodeChunk32:
    def __init__(self, ind_id: int, generation: int):
        self.ind_id = ind_id
        self.generation = generation
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

    def add_const(self, val: float) -> int:
        vec = np.array([val, val, val], dtype=np.float64)
        for i, ex in enumerate(self.constants_list):
            if np.allclose(ex, vec):
                return i
        idx = len(self.constants_list)
        self.constants_list.append(vec)
        return idx

    def emit(self, op: VMOp, a1: int = 0, a2: int = 0):
        self.raw_ops.append((op, a1, a2))

    def finalize_peephole(self):
        fuse = {VMOp.BINARY_ADD: VMOp.ADD_STORE, VMOp.BINARY_SUB: VMOp.SUB_STORE, VMOp.BINARY_MUL: VMOp.MUL_STORE}
        opt: List[Tuple[VMOp, int, int]] = []
        i, n = 0, len(self.raw_ops)
        while i < n:
            op, a1, a2 = self.raw_ops[i]
            if i + 1 < n and op in fuse and self.raw_ops[i + 1][0] == VMOp.STORE_LOCAL:
                opt.append((fuse[op], self.raw_ops[i + 1][1], 0))
                i += 2
            else:
                opt.append((op, a1, a2))
                i += 1
        self.raw_ops = opt
        self.packed_words = np.array([pack_word32(op, a1, a2) for op, a1, a2 in self.raw_ops], dtype=np.uint32)
        if self.constants_list:
            self.const_pool = np.vstack(self.constants_list).astype(np.float64)


class GeneticJITCompiler:
    @staticmethod
    def compile(indiv: EvolvableASTFunction) -> BytecodeChunk32:
        chunk = BytecodeChunk32(indiv.ind_id, indiv.generation)
        for p in indiv.params:
            chunk.resolve_local(p)
        for stmt in indiv.body:
            if isinstance(stmt, AssignNode):
                GeneticJITCompiler._emit_expr(stmt.expr, chunk)
                chunk.emit(VMOp.STORE_LOCAL, chunk.resolve_local(stmt.var_name))
            elif isinstance(stmt, ReturnNode):
                GeneticJITCompiler._emit_expr(stmt.expr, chunk)
                chunk.emit(VMOp.RETURN_VAL)
        chunk.finalize_peephole()
        return chunk

    @staticmethod
    def _emit_expr(expr: ASTNode, chunk: BytecodeChunk32):
        if isinstance(expr, NumberNode):
            chunk.emit(VMOp.LOAD_CONST, chunk.add_const(expr.value))
        elif isinstance(expr, IdentNode):
            chunk.emit(VMOp.LOAD_LOCAL, chunk.resolve_local(expr.name))
        elif isinstance(expr, BinaryOpNode):
            GeneticJITCompiler._emit_expr(expr.left, chunk)
            GeneticJITCompiler._emit_expr(expr.right, chunk)
            if expr.op == "+": chunk.emit(VMOp.BINARY_ADD)
            elif expr.op == "-": chunk.emit(VMOp.BINARY_SUB)
            elif expr.op == "*": chunk.emit(VMOp.BINARY_MUL)


class ParallelGeneticStackVM:
    def __init__(self):
        self.stack = np.zeros((48, 3), dtype=np.float64)
        self.locals = np.zeros((16, 3), dtype=np.float64)
        self.last_batch_time_us = 0.0

    def execute_chunk(
        self,
        chunk: BytecodeChunk32,
        mano: np.ndarray,
        holo: np.ndarray,
        vel_mano: np.ndarray,
        vel_holo: np.ndarray
    ) -> np.ndarray:
        stack = self.stack
        loc = self.locals
        consts = chunk.const_pool
        code = chunk.packed_words

        loc[0] = mano
        loc[1] = holo
        loc[2] = vel_mano
        loc[3] = vel_holo
        sp = 0

        for word in code:
            w = int(word)
            op = (w >> 24) & 0xFF
            a1 = (w >> 12) & 0xFFF

            if op == VMOp.LOAD_CONST:
                stack[sp] = consts[a1]; sp += 1
            elif op == VMOp.LOAD_LOCAL:
                stack[sp] = loc[a1]; sp += 1
            elif op == VMOp.STORE_LOCAL:
                sp -= 1; loc[a1] = stack[sp]
            elif op == VMOp.BINARY_ADD:
                sp -= 1; stack[sp - 1] = stack[sp - 1] + stack[sp]
            elif op == VMOp.BINARY_SUB:
                sp -= 1; stack[sp - 1] = stack[sp - 1] - stack[sp]
            elif op == VMOp.BINARY_MUL:
                sp -= 1; stack[sp - 1] = stack[sp - 1] * stack[sp]
            elif op == VMOp.ADD_STORE:
                sp -= 2; loc[a1] = stack[sp] + stack[sp + 1]
            elif op == VMOp.SUB_STORE:
                sp -= 2; loc[a1] = stack[sp] - stack[sp + 1]
            elif op == VMOp.MUL_STORE:
                sp -= 2; loc[a1] = stack[sp] * stack[sp + 1]
            elif op == VMOp.RETURN_VAL:
                out = stack[sp - 1].copy()
                if not np.all(np.isfinite(out)):
                    return holo.astype(np.float32)
                return np.clip(out, -2.4, 2.4).astype(np.float32)

        return holo.astype(np.float32)


# =====================================================================
# 5. CRÍTICO DE APTITUD EN PYTORCH + SALÓN DE LA FAMA (.NXS EXPORT)
# =====================================================================

class PyTorchEvolutionCritic(nn.Module):
    def __init__(self, state_dim: int = 6):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        ).to(self.device)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=2e-3)
        self.criterion = nn.SmoothL1Loss()
        self.last_loss = 0.0

    def score_population(
        self,
        errors: np.ndarray,
        jerks: np.ndarray,
        vel_Alignment: np.ndarray,
        hand_speed: float,
        op_counts: np.ndarray
    ) -> np.ndarray:
        n = len(errors)
        speeds = np.full(n, hand_speed, dtype=np.float32)
        norm_ops = op_counts / 30.0

        features = np.stack([errors, jerks, vel_Alignment, speeds, norm_ops, errors ** 2], axis=1).astype(np.float32)
        feat_t = torch.from_numpy(features).to(self.device)

        # Función de recompensa física + parsimonia (premia código limpio y seguimiento sin vibración)
        raw_reward = (
            np.exp(-3.6 * errors) * 2.6
            + (0.35 * vel_Alignment)
            - (1.15 * jerks)
            - (0.012 * op_counts)
        ).astype(np.float32)
        target_t = torch.from_numpy(raw_reward).unsqueeze(1).to(self.device)

        self.net.train()
        pred_v = self.net(feat_t)
        loss = self.criterion(pred_v, target_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_loss = float(loss.item())

        with torch.no_grad():
            combined = 0.72 * target_t.squeeze(1) + 0.28 * pred_v.squeeze(1)
            return combined.cpu().numpy()


class GeneticHallOfFame:
    """Guarda automáticamente en disco ('.nxs') el mejor código AST descubierto por la evolución."""
    def __init__(self, out_file: str = "campeon_genetico_ast.nxs"):
        self.out_file = out_file
        self.all_time_best_fitness = -999.0
        self.best_generation = 0

    def check_and_export(self, champ: EvolvableASTFunction, chunk: BytecodeChunk32, avg_fitness: float) -> bool:
        if avg_fitness > self.all_time_best_fitness and avg_fitness > 0.5:
            self.all_time_best_fitness = avg_fitness
            self.best_generation = champ.generation

            record_doc = {
                "header": {
                    "magic": "NEXUS_GENETIC_CHAMPION_NXS",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "generation": champ.generation,
                    "individual_id": champ.ind_id,
                    "pytorch_fitness_record": round(float(avg_fitness), 4),
                    "bytecode_word_count": len(chunk.packed_words),
                },
                "ast_source_code": champ.get_source_lines(),
                "bytecode_hex_words": [f"0x{int(w):08X}" for w in chunk.packed_words],
            }
            with open(self.out_file, "w", encoding="utf-8") as f:
                json.dump(record_doc, f, indent=2, ensure_ascii=False)
            return True
        return False


# =====================================================================
# 6. SIMULADOR VISUAL 3D CON LOS 10 INDIVIDUOS COMPITIENDO EN VIVO
# =====================================================================

def rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


class NexusKozaGeneticArena:
    POP_SIZE = 10
    EPOCH_LENGTH = 35

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.vm = ParallelGeneticStackVM()
        self.critic = PyTorchEvolutionCritic()
        self.hall_of_fame = GeneticHallOfFame()
        self.generation = 1

        # Inicialización Ramped Half-and-Half (profundidades 1, 2 y 3)
        self.population: List[EvolvableASTFunction] = [
            EvolvableASTFunction(
                ind_id=i,
                generation=1,
                force_tree=EvolvableASTFunction.generate_random_tree(max_depth=(i % 3) + 1)
            )
            for i in range(self.POP_SIZE)
        ]
        self.chunks: List[BytecodeChunk32] = [GeneticJITCompiler.compile(ind) for ind in self.population]

        self.positions = np.zeros((self.POP_SIZE, 3), dtype=np.float32)
        self.velocities = np.zeros((self.POP_SIZE, 3), dtype=np.float32)
        self.prev_velocities = np.zeros((self.POP_SIZE, 3), dtype=np.float32)
        self.fitness_accum = np.zeros(self.POP_SIZE, dtype=np.float32)

        self.alpha_idx = 0
        self.beta_idx = 1
        self.epoch_frames = 0
        self.evolution_status = "Poblacion Gen 1 activa (10 Arboles AST de topologia libre)."

        self.prev_hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.holo_rot = np.array([0.3, 0.5, 0.1], dtype=np.float32)

        self.cube_verts = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ], dtype=np.float32) * 0.46
        self.cube_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        self.cube_faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6)]

    def step_generation(self, hand_pos: np.ndarray, hand_vel: np.ndarray):
        t0 = time.perf_counter()
        errors = np.zeros(self.POP_SIZE, dtype=np.float32)
        jerks = np.zeros(self.POP_SIZE, dtype=np.float32)
        alignments = np.zeros(self.POP_SIZE, dtype=np.float32)
        op_counts = np.array([len(c.packed_words) for c in self.chunks], dtype=np.float32)

        hand_pos_64 = hand_pos.astype(np.float64)
        hand_vel_64 = hand_vel.astype(np.float64)
        hand_speed = float(np.linalg.norm(hand_vel))

        for i in range(self.POP_SIZE):
            cur_p = self.positions[i]
            cur_v = self.velocities[i]

            next_p = self.vm.execute_chunk(
                self.chunks[i],
                hand_pos_64,
                cur_p.astype(np.float64),
                hand_vel_64,
                cur_v.astype(np.float64)
            )
            new_v = next_p - cur_p
            jerk = float(np.linalg.norm(new_v - self.prev_velocities[i]))
            err = float(np.linalg.norm(hand_pos - next_p))

            # Alineación coseno entre el movimiento de la mano y la respuesta del AST
            if hand_speed > 0.005 and np.linalg.norm(new_v) > 0.002:
                align = float(np.dot(hand_vel, new_v) / (hand_speed * np.linalg.norm(new_v) + 1e-6))
            else:
                align = 0.0

            self.positions[i] = next_p
            self.prev_velocities[i] = cur_v
            self.velocities[i] = new_v

            errors[i] = err
            jerks[i] = jerk
            alignments[i] = align

        self.vm.last_batch_time_us = (time.perf_counter() - t0) * 1_000_000.0

        # Evaluar los 10 árboles AST en PyTorch
        rewards = self.critic.score_population(errors, jerks, alignments, hand_speed, op_counts)
        self.fitness_accum += rewards

        order = np.argsort(self.fitness_accum)[::-1]
        self.alpha_idx = int(order[0])
        self.beta_idx = int(order[1])

        self.epoch_frames += 1
        if self.epoch_frames >= self.EPOCH_LENGTH:
            self.breed_new_generation()

    def breed_new_generation(self):
        order = np.argsort(self.fitness_accum)[::-1]
        p1_idx, p2_idx = int(order[0]), int(order[1])

        parent_1 = copy.deepcopy(self.population[p1_idx])
        parent_2 = copy.deepcopy(self.population[p2_idx])
        best_avg_fit = float(self.fitness_accum[p1_idx]) / max(1, self.epoch_frames)

        # Verificar si el campeón actual rompió el récord histórico para exportarlo a '.nxs'
        is_new_record = self.hall_of_fame.check_and_export(parent_1, self.chunks[p1_idx], best_avg_fit)

        self.generation += 1
        next_pop: List[EvolvableASTFunction] = []

        # 1. Elitismo: Los 2 mejores padres pasan intactos a la nueva generación
        parent_1.ind_id = 0
        parent_1.generation = self.generation
        parent_1.parent_info = f"Elite Alfa (Gen {self.generation-1} #{p1_idx})"
        parent_1.rebuild_body()
        next_pop.append(parent_1)

        parent_2.ind_id = 1
        parent_2.generation = self.generation
        parent_2.parent_info = f"Elite Beta (Gen {self.generation-1} #{p2_idx})"
        parent_2.rebuild_body()
        next_pop.append(parent_2)

        # 2. Cruce de Sub-Árboles (Koza Crossover) entre Padre 1 y Padre 2 para hijos 2..8
        for child_id in range(2, self.POP_SIZE - 1):
            if child_id % 2 == 0:
                child = KozaGeneticEngine.crossover(parent_1, parent_2, child_id, self.generation)
            else:
                child = KozaGeneticEngine.crossover(parent_2, parent_1, child_id, self.generation)
            KozaGeneticEngine.mutate(child, rate=0.42)
            next_pop.append(child)

        # 3. Explorador mutante nuevo en el slot 9
        fresh = EvolvableASTFunction(self.POP_SIZE - 1, self.generation, parent_info="Explorador Aleatorio")
        next_pop.append(fresh)

        # 4. Compilar los 10 nuevos árboles AST a Bytecode de 32 bits
        self.population = next_pop
        self.chunks = [GeneticJITCompiler.compile(ind) for ind in self.population]

        anchor = self.positions[p1_idx].copy()
        for i in range(self.POP_SIZE):
            self.positions[i] = anchor.copy()
            self.velocities[i] = 0.0
            self.prev_velocities[i] = 0.0

        rec_tag = " [NUEVO RECORD .NXS GUARDADO!]" if is_new_record else ""
        self.evolution_status = f"Gen {self.generation}: Cruzados #{p1_idx} x #{p2_idx} (Fit: {best_avg_fit:+.2f}){rec_tag}"
        print(f"[KOZA-GP] {self.evolution_status}")

        self.fitness_accum.fill(0.0)
        self.epoch_frames = 0
        self.alpha_idx = 0
        self.beta_idx = 1

    def project_3d(self, pts_3d, w, h, focal=640.0, cam_dist=3.5):
        out = []
        for pt in pts_3d:
            z = max(cam_dist + pt[2], 0.35)
            out.append((int((pt[0] * focal) / z + w * 0.5), int((pt[1] * focal) / z + h * 0.5)))
        return out

    def render_arena(self, frame, hand_pos_3d: Optional[np.ndarray]):
        h, w, _ = frame.shape

        # Dibujar los 9 competidores secundarios en el espacio 3D
        for i in range(self.POP_SIZE):
            if i == self.alpha_idx:
                continue
            pt_2d = self.project_3d([self.positions[i]], w, h)[0]
            is_beta = (i == self.beta_idx)
            col = (255, 110, 255) if is_beta else (175, 155, 85)
            cv2.circle(frame, pt_2d, 12 if is_beta else 7, col, 2, cv2.LINE_AA)
            cv2.putText(frame, f"#{i}", (pt_2d[0] + 9, pt_2d[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

        # Dibujar el Cubo Holográfico gobernado por el Individuo ALFA
        alpha_pos = self.positions[self.alpha_idx]
        self.holo_rot += np.array([0.012, 0.022, 0.006], dtype=np.float32)
        R = rotation_matrix_3d(*self.holo_rot)
        verts = self.cube_verts @ R.T + alpha_pos
        pts_2d = self.project_3d(verts, w, h)

        overlay = frame.copy()
        for face in self.cube_faces:
            poly = np.array([pts_2d[idx] for idx in face], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (0, 80, 50))
        cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)

        for e1, e2 in self.cube_edges:
            cv2.line(frame, pts_2d[e1], pts_2d[e2], (0, 255, 160), 2, cv2.LINE_AA)

        c2d = self.project_3d([alpha_pos], w, h)[0]
        cv2.putText(frame, f"ALFA AST #{self.alpha_idx} (Gen {self.generation})",
                    (c2d[0] - 68, c2d[1] - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 160), 2)

        if hand_pos_3d is not None:
            m2d = self.project_3d([hand_pos_3d], w, h)[0]
            cv2.line(frame, m2d, c2d, (0, 255, 160), 1, cv2.LINE_AA)
            cv2.circle(frame, m2d, 7, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_hud(self, frame):
        h, w, _ = frame.shape

        # Panel Izquierdo: Telemetría Genética y Récord Histórico
        cv2.rectangle(frame, (15, 15), (635, 175), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (635, 175), (0, 255, 160), 1)
        cv2.putText(frame, f"KOZA GENETIC PROGRAMMING | GEN #{self.generation} (Frame {self.epoch_frames}/{self.EPOCH_LENGTH})",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 160), 2)
        cv2.putText(frame, f"10 Bytecodes en Stack VM: {self.vm.last_batch_time_us:.1f} us | Record Hall of Fame: {self.hall_of_fame.all_time_best_fitness:+.2f}",
                    (26, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 235, 255), 1)
        cv2.putText(frame, f"Estado: {self.evolution_status}",
                    (26, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 215, 80), 1)

        bar_w = int((self.epoch_frames / self.EPOCH_LENGTH) * 585)
        cv2.rectangle(frame, (26, 108), (611, 126), (35, 40, 48), -1)
        cv2.rectangle(frame, (26, 108), (26 + bar_w, 126), (255, 95, 255), -1)
        cv2.putText(frame, "Teclas: [E] Cruzar Padres Ahora  |  [R] Reiniciar Gen 1  |  Campeon auto-guardado en '.nxs'",
                    (26, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 240, 200), 1)

        # Panel Derecho Superior: Tabla de Clasificación de los 10 Árboles AST
        cv2.rectangle(frame, (w - 585, 15), (w - 15, 275), (14, 14, 22), -1)
        cv2.rectangle(frame, (w - 585, 15), (w - 15, 275), (255, 180, 0), 1)
        cv2.putText(frame, "RANKING PYTORCH DE LOS 10 ARBOLES AST EN COMPETENCIA",
                    (w - 570, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 210, 0), 2)

        order = np.argsort(self.fitness_accum)[::-1]
        div = max(1, self.epoch_frames)
        for rank, idx in enumerate(order):
            fit = float(self.fitness_accum[idx]) / div
            ops = len(self.chunks[idx].packed_words)
            expr_str = self.population[idx].force_tree.to_source()
            if len(expr_str) > 30:
                expr_str = expr_str[:27] + "..."

            if rank == 0:
                tag, col = "[PADRE 1 ALFA]", (0, 255, 150)
            elif rank == 1:
                tag, col = "[PADRE 2 BETA]", (255, 120, 255)
            else:
                tag, col = f"  Puesto #{rank+1:<2d}  ", (205, 210, 220)

            cv2.putText(frame, f"{tag} AST#{idx} | Fit:{fit:+.2f} | {ops:2d} ops | {expr_str}",
                        (w - 570, 62 + rank * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.37, col, 1)

        # Panel Derecho Inferior: Código Fuente del Individuo ALFA
        alpha_lines = self.population[self.alpha_idx].get_source_lines()
        top_y = 290
        box_h = 30 + len(alpha_lines) * 19
        cv2.rectangle(frame, (w - 585, top_y), (w - 15, top_y + box_h), (12, 16, 20), -1)
        cv2.rectangle(frame, (w - 585, top_y), (w - 15, top_y + box_h), (0, 230, 255), 1)
        for i, line in enumerate(alpha_lines):
            col = (255, 215, 80) if i == 0 else ((120, 255, 150) if "impulso_genetico" in line else (225, 235, 245))
            cv2.putText(frame, line[:68], (w - 570, top_y + 24 + i * 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, col, 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            success, raw = cap.read()
            if not success:
                break

            frame = cv2.flip(raw, 1)
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            hand_pos_3d = None
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                mid = (pts[4] + pts[8]) * 0.5
                hand_pos_3d = np.array([(mid[0] - 0.5) * 3.2, (mid[1] - 0.5) * 2.0, mid[2] * 1.4], dtype=np.float32)

                hand_vel = hand_pos_3d - self.prev_hand_pos
                self.prev_hand_pos = hand_pos_3d.copy()

                self.step_generation(hand_pos_3d, hand_vel)

            self.render_arena(frame, hand_pos_3d)
            self.draw_hud(frame)
            cv2.imshow("NexusXR - Koza Genetic Programming Arena (10 ASTs)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('e'):
                self.breed_new_generation()
            elif key == ord('r'):
                self.generation = 1
                self.population = [
                    EvolvableASTFunction(i, 1, EvolvableASTFunction.generate_random_tree((i % 3) + 1))
                    for i in range(self.POP_SIZE)
                ]
                self.chunks = [GeneticJITCompiler.compile(ind) for ind in self.population]
                self.fitness_accum.fill(0.0)
                self.epoch_frames = 0

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    arena = NexusKozaGeneticArena()
    arena.run()
