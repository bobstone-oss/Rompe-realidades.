import math
import time
import copy
import random
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =====================================================================
# 1. CONJUNTO DE INSTRUCCIONES BINARIAS (32-BIT BYTECODE ISA)
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
# 2. ÁRBOL SINTÁCTICO ABSTRACTO (AST) PARA PROGRAMACIÓN GENÉTICA
# =====================================================================

class ASTNode:
    def to_source(self) -> str:
        raise NotImplementedError()

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


class GeneticASTChromosome:
    """
    Representa un individuo de la población genética:
    Una función AST completa con 3 cromosomas estructurales (sub-árboles intercambiables):
      - gene_prop_tree : Sub-árbol de fuerza proporcional (depende de 'error')
      - gene_damp_tree : Sub-árbol de amortiguación/predicción (depende de 'vel_mano')
      - gene_nl_tree   : Sub-árbol de corrección no lineal o impulso extra
    """
    def __init__(
        self,
        ind_id: int,
        generation: int = 1,
        gene_prop: Optional[ASTNode] = None,
        gene_damp: Optional[ASTNode] = None,
        gene_nl: Optional[ASTNode] = None
    ):
        self.ind_id = ind_id
        self.generation = generation
        self.params = ["mano", "holo", "vel_mano"]

        # Inicialización aleatoria diversa si no viene de un cruce genético
        self.gene_prop = gene_prop if gene_prop else self._random_prop_subtree()
        self.gene_damp = gene_damp if gene_damp else self._random_damp_subtree()
        self.gene_nl = gene_nl if gene_nl else self._random_nl_subtree()

        self.body: List[ASTNode] = []
        self.rebuild_full_ast()

    @staticmethod
    def _random_prop_subtree() -> ASTNode:
        kp = round(random.uniform(0.05, 0.55), 3)
        base = BinaryOpNode("*", IdentNode("error"), NumberNode(kp))
        if random.random() < 0.4:
            boost = round(random.uniform(0.8, 1.6), 2)
            return BinaryOpNode("*", base, NumberNode(boost))
        return base

    @staticmethod
    def _random_damp_subtree() -> ASTNode:
        kd = round(random.uniform(0.0, 0.45), 3)
        return BinaryOpNode("*", IdentNode("vel_mano"), NumberNode(kd))

    @staticmethod
    def _random_nl_subtree() -> ASTNode:
        choice = random.choice(["ZERO", "ERROR_BOOST", "VEL_DIFF"])
        if choice == "ERROR_BOOST":
            k2 = round(random.uniform(0.02, 0.18), 3)
            return BinaryOpNode("*", IdentNode("error"), NumberNode(k2))
        elif choice == "VEL_DIFF":
            k3 = round(random.uniform(0.02, 0.15), 3)
            return BinaryOpNode("*", BinaryOpNode("-", IdentNode("vel_mano"), IdentNode("error")), NumberNode(k3))
        else:
            return BinaryOpNode("*", IdentNode("error"), NumberNode(0.0))

    def rebuild_full_ast(self):
        self.body = [
            AssignNode("error", BinaryOpNode("-", IdentNode("mano"), IdentNode("holo"))),
            AssignNode("rama_p", copy.deepcopy(self.gene_prop)),
            AssignNode("rama_d", copy.deepcopy(self.gene_damp)),
            AssignNode("rama_nl", copy.deepcopy(self.gene_nl)),
            AssignNode("impulso", BinaryOpNode("+", BinaryOpNode("+", IdentNode("rama_p"), IdentNode("rama_d")), IdentNode("rama_nl"))),
            ReturnNode(BinaryOpNode("+", IdentNode("holo"), IdentNode("impulso")))
        ]

    def total_complexity(self) -> int:
        return sum(stmt.node_count() for stmt in self.body)

    def get_source_lines(self) -> List[str]:
        lines = [f"FUNCION individuo_{self.ind_id}_gen_{self.generation}(mano, holo, vel_mano) {{"]
        for s in self.body:
            lines.append(s.to_source())
        lines.append("}")
        return lines


# =====================================================================
# 3. OPERADORES DE PROGRAMACIÓN GENÉTICA (CRUCE DE SUB-ÁRBOLES Y MUTACIÓN)
# =====================================================================

class ASTGeneticOperators:
    @staticmethod
    def crossover(parent_a: GeneticASTChromosome, parent_b: GeneticASTChromosome, new_id: int, new_gen: int) -> GeneticASTChromosome:
        """
        Cruce Genético de Árboles (Subtree Crossover):
        Combina las ramas matemáticas de los dos mejores padres (Padre A y Padre B)
        o crea una combinación lineal simbiótica de sus sub-árboles.
        """
        # Seleccionar o fusionar el gen proporcional
        if random.random() < 0.5:
            child_prop = copy.deepcopy(parent_a.gene_prop if random.random() < 0.6 else parent_b.gene_prop)
        else:
            # Fusión algebraica de las ramas de ambos padres: (rama_A * 0.5) + (rama_B * 0.5)
            child_prop = BinaryOpNode(
                "+",
                BinaryOpNode("*", copy.deepcopy(parent_a.gene_prop), NumberNode(0.5)),
                BinaryOpNode("*", copy.deepcopy(parent_b.gene_prop), NumberNode(0.5))
            )

        child_damp = copy.deepcopy(parent_a.gene_damp if random.random() < 0.5 else parent_b.gene_damp)
        child_nl = copy.deepcopy(parent_b.gene_nl if random.random() < 0.5 else parent_a.gene_nl)

        # Podar si el sub-árbol creció demasiado (Control de Bloat Genético)
        if child_prop.node_count() > 9:
            child_prop = copy.deepcopy(parent_a.gene_prop)

        child = GeneticASTChromosome(new_id, new_gen, child_prop, child_damp, child_nl)
        return child

    @staticmethod
    def mutate(chrom: GeneticASTChromosome, mutation_rate: float = 0.45):
        """
        Mutación Quirúrgica de Nodos:
        Recorre los sub-árboles del individuo y perturba constantes numéricas (NumberNode),
        cambia operadores (+/-) o reemplaza una rama entera.
        """
        def _mutate_node(node: ASTNode) -> ASTNode:
            if isinstance(node, NumberNode):
                if random.random() < mutation_rate:
                    delta = random.gauss(0.0, 0.06)
                    new_val = float(np.clip(node.value + delta, 0.0, 1.6))
                    return NumberNode(round(new_val, 3))
                return node
            elif isinstance(node, BinaryOpNode):
                new_left = _mutate_node(node.left)
                new_right = _mutate_node(node.right)
                new_op = node.op
                if node.op in ("+", "-") and random.random() < 0.15:
                    new_op = "-" if node.op == "+" else "+"
                return BinaryOpNode(new_op, new_left, new_right)
            return node

        chrom.gene_prop = _mutate_node(chrom.gene_prop)
        chrom.gene_damp = _mutate_node(chrom.gene_damp)
        if random.random() < 0.25:
            chrom.gene_nl = GeneticASTChromosome._random_nl_subtree()
        else:
            chrom.gene_nl = _mutate_node(chrom.gene_nl)

        chrom.rebuild_full_ast()


# =====================================================================
# 4. COMPILADOR JIT A BYTECODE DE 32 BITS Y STACK VM PARALELA
# =====================================================================

class CompiledBytecodeProgram:
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

    def pack(self):
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


class JITCompiler:
    @staticmethod
    def compile(chrom: GeneticASTChromosome) -> CompiledBytecodeProgram:
        prog = CompiledBytecodeProgram(chrom.ind_id, chrom.generation)
        for p in chrom.params:
            prog.resolve_local(p)
        for stmt in chrom.body:
            if isinstance(stmt, AssignNode):
                JITCompiler._compile_expr(stmt.expr, prog)
                prog.emit(VMOp.STORE_LOCAL, prog.resolve_local(stmt.var_name))
            elif isinstance(stmt, ReturnNode):
                JITCompiler._compile_expr(stmt.expr, prog)
                prog.emit(VMOp.RETURN_VAL)
        prog.pack()
        return prog

    @staticmethod
    def _compile_expr(expr: ASTNode, prog: CompiledBytecodeProgram):
        if isinstance(expr, NumberNode):
            prog.emit(VMOp.LOAD_CONST, prog.add_const(expr.value))
        elif isinstance(expr, IdentNode):
            prog.emit(VMOp.LOAD_LOCAL, prog.resolve_local(expr.name))
        elif isinstance(expr, BinaryOpNode):
            JITCompiler._compile_expr(expr.left, prog)
            JITCompiler._compile_expr(expr.right, prog)
            if expr.op == "+": prog.emit(VMOp.BINARY_ADD)
            elif expr.op == "-": prog.emit(VMOp.BINARY_SUB)
            elif expr.op == "*": prog.emit(VMOp.BINARY_MUL)


class PopulationStackVM:
    """Ejecuta los 10 bloques de Bytecode de la población en un mismo fotograma."""
    def __init__(self):
        self.stack = np.zeros((32, 3), dtype=np.float64)
        self.locals = np.zeros((16, 3), dtype=np.float64)
        self.last_batch_exec_us = 0.0

    def run_single(self, prog: CompiledBytecodeProgram, mano: np.ndarray, holo: np.ndarray, vel_mano: np.ndarray) -> np.ndarray:
        stack = self.stack
        loc = self.locals
        consts = prog.const_pool
        code = prog.packed_words

        loc[0] = mano
        loc[1] = holo
        loc[2] = vel_mano
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
                return np.clip(stack[sp - 1].copy(), -2.4, 2.4).astype(np.float32)

        return holo.copy()


# =====================================================================
# 5. CRÍTICO DE RECOMPENSA EN PYTORCH (NEURAL VALUE / FITNESS CRITIC)
# =====================================================================

class PyTorchFitnessCritic(nn.Module):
    """
    Red Neuronal en PyTorch que evalúa en paralelo el lote de 10 funciones AST:
    Aprende a predecir el Valor Esperado (V(s)) de cada trayectoria cinemática
    y combina la recompensa física instantánea con su estimación neuronal.
    """
    def __init__(self, state_dim: int = 6):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        ).to(self.device)
        self.optimizer = optim.AdamW(self.value_net.parameters(), lr=2e-3)
        self.loss_fn = nn.SmoothL1Loss()
        self.last_critic_loss = 0.0

    def evaluate_population_batch(
        self,
        errors: np.ndarray,       # (10,)
        jerks: np.ndarray,        # (10,)
        overshoots: np.ndarray,   # (10,)
        hand_speed: float,
        complexities: np.ndarray  # (10,)
    ) -> np.ndarray:
        """
        Evalúa simultáneamente a los 10 candidatos en PyTorch y entrena al Crítico.
        Devuelve el vector de recompensas (10,) para este fotograma.
        """
        pop_size = len(errors)
        speeds = np.full(pop_size, hand_speed, dtype=np.float32)
        norm_comp = complexities / 25.0

        # Tensor de estado por individuo: [error, jerk, overshoot, hand_speed, complexity, error^2]
        state_mat = np.stack([errors, jerks, overshoots, speeds, norm_comp, errors ** 2], axis=1).astype(np.float32)
        state_t = torch.from_numpy(state_mat).to(self.device)

        # Recompensa física objetivo (Ground-Truth RL Objective):
        # Alta precisión exponencial - penalización por vibración (jerk) - penalización por exceso de código
        target_rewards_np = (
            np.exp(-3.8 * errors) * 2.5
            - (1.1 * jerks)
            - (0.6 * overshoots)
            - (0.015 * complexities)
        ).astype(np.float32)
        target_t = torch.from_numpy(target_rewards_np).unsqueeze(1).to(self.device)

        # Paso de aprendizaje en vivo del Crítico en PyTorch
        self.value_net.train()
        pred_values = self.value_net(state_t)
        loss = self.loss_fn(pred_values, target_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_critic_loss = float(loss.item())

        # Aptitud híbrida: 70% recompensa física directa + 30% potencial estimado por la red PyTorch
        with torch.no_grad():
            blended_fitness = 0.70 * target_t.squeeze(1) + 0.30 * pred_values.squeeze(1)
            return blended_fitness.cpu().numpy()


# =====================================================================
# 6. ARENA DE EVOLUCIÓN GENÉTICA EN TIEMPO REAL (10 INDIVIDUOS EN PARALELO)
# =====================================================================

def rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


class NexusGeneticASTArena:
    POP_SIZE = 10
    EPOCH_FRAMES = 35  # Cada 35 fotogramas de movimiento se cruzan los 2 mejores árboles AST

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.vm = PopulationStackVM()
        self.critic = PyTorchFitnessCritic()
        self.generation = 1

        # 1. Crear población inicial de 10 funciones AST diversas y compilarlas a Bytecode
        self.population: List[GeneticASTChromosome] = [
            GeneticASTChromosome(ind_id=i, generation=1) for i in range(self.POP_SIZE)
        ]
        self.bytecodes: List[CompiledBytecodeProgram] = [
            JITCompiler.compile(chrom) for chrom in self.population
        ]

        # Estado espacial paralelo de los 10 individuos: posiciones, velocidades y aptitud acumulada
        self.ind_positions = np.zeros((self.POP_SIZE, 3), dtype=np.float32)
        self.ind_prev_vels = np.zeros((self.POP_SIZE, 3), dtype=np.float32)
        self.cumulative_fitness = np.zeros(self.POP_SIZE, dtype=np.float32)

        self.alpha_idx = 0  # Índice del campeón actual (Mejor AST de la población)
        self.second_idx = 1
        self.last_crossover_log = "Generacion 1 inicializada (10 ASTs compilados a Bytecode)."

        self.prev_hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.active_frames_in_epoch = 0
        self.holo_rot = np.array([0.3, 0.5, 0.1], dtype=np.float32)

        self.cube_verts = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ], dtype=np.float32) * 0.46
        self.cube_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        self.cube_faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6)]

    def step_parallel_population(self, hand_pos: np.ndarray, hand_vel: np.ndarray):
        """
        Ejecuta los 10 programas de Bytecode en la Stack VM y evalúa su recompensa en PyTorch.
        """
        t0 = time.perf_counter()
        errors = np.zeros(self.POP_SIZE, dtype=np.float32)
        jerks = np.zeros(self.POP_SIZE, dtype=np.float32)
        overshoots = np.zeros(self.POP_SIZE, dtype=np.float32)
        complexities = np.array([c.total_complexity() for c in self.population], dtype=np.float32)

        hand_pos_64 = hand_pos.astype(np.float64)
        hand_vel_64 = hand_vel.astype(np.float64)

        for i in range(self.POP_SIZE):
            cur_pos = self.ind_positions[i]
            new_pos = self.vm.run_single(
                self.bytecodes[i],
                hand_pos_64,
                cur_pos.astype(np.float64),
                hand_vel_64
            )
            vel = new_pos - cur_pos
            jerk = float(np.linalg.norm(vel - self.ind_prev_vels[i]))
            err = float(np.linalg.norm(hand_pos - new_pos))

            # Detectar sobreimpulso (si el holograma se pasa de largo respecto a la dirección de la mano)
            overshoot = max(0.0, float(np.linalg.norm(vel)) - float(np.linalg.norm(hand_vel)) * 1.6)

            self.ind_positions[i] = new_pos
            self.ind_prev_vels[i] = vel
            errors[i] = err
            jerks[i] = jerk
            overshoots[i] = overshoot

        self.vm.last_batch_exec_us = (time.perf_counter() - t0) * 1_000_000.0

        # Evaluar lote completo de 10 individuos en el Crítico de PyTorch
        hand_speed = float(np.linalg.norm(hand_vel))
        step_rewards = self.critic.evaluate_population_batch(errors, jerks, overshoots, hand_speed, complexities)
        self.cumulative_fitness += step_rewards

        # Actualizar en vivo quiénes son los 2 mejores individuos
        ranked_indices = np.argsort(self.cumulative_fitness)[::-1]
        self.alpha_idx = int(ranked_indices[0])
        self.second_idx = int(ranked_indices[1])

        self.active_frames_in_epoch += 1
        if self.active_frames_in_epoch >= self.EPOCH_FRAMES:
            self.evolve_next_generation()

    def evolve_next_generation(self):
        """
        Selecciona los 2 mejores árboles AST según la recompensa de PyTorch,
        los conserva por elitismo, cruza sus sub-árboles para generar 8 hijos nuevos,
        aplica mutación genética y recompila los 10 a Bytecode de 32 bits.
        """
        ranked = np.argsort(self.cumulative_fitness)[::-1]
        best_1_idx = int(ranked[0])
        best_2_idx = int(ranked[1])

        parent_alpha = copy.deepcopy(self.population[best_1_idx])
        parent_beta = copy.deepcopy(self.population[best_2_idx])
        best_score = float(self.cumulative_fitness[best_1_idx]) / max(1, self.active_frames_in_epoch)

        self.generation += 1
        new_pop: List[GeneticASTChromosome] = []

        # 1. Elitismo: Los 2 mejores pasan intactos como Individuo 0 e Individuo 1
        parent_alpha.ind_id = 0
        parent_alpha.generation = self.generation
        parent_alpha.rebuild_full_ast()
        new_pop.append(parent_alpha)

        parent_beta.ind_id = 1
        parent_beta.generation = self.generation
        parent_beta.rebuild_full_ast()
        new_pop.append(parent_beta)

        # 2. Cruce Genético (Subtree Crossover) de los 2 mejores + Mutación para los puestos 2..8
        for child_id in range(2, self.POP_SIZE - 1):
            child = ASTGeneticOperators.crossover(parent_alpha, parent_beta, child_id, self.generation)
            ASTGeneticOperators.mutate(child, mutation_rate=0.45)
            new_pop.append(child)

        # 3. Inmigrante aleatorio en el puesto 9 para mantener diversidad genética
        new_pop.append(GeneticASTChromosome(ind_id=self.POP_SIZE - 1, generation=self.generation))

        # 4. Recompilar toda la población a Bytecode de 32 bits en caliente
        self.population = new_pop
        self.bytecodes = [JITCompiler.compile(c) for c in self.population]

        # Sincronizar posiciones de los nuevos hijos con la del campeón Alfa para evitar saltos bruscos
        alpha_pos = self.ind_positions[best_1_idx].copy()
        for i in range(self.POP_SIZE):
            self.ind_positions[i] = alpha_pos.copy()
            self.ind_prev_vels[i] = 0.0

        self.last_crossover_log = (
            f"Gen {self.generation}: Cruzados AST #{best_1_idx} y AST #{best_2_idx} "
            f"(Fitness Alfa: {best_score:+.2f})"
        )
        print(f"[EVOLUCIÓN GENÉTICA] {self.last_crossover_log}")

        self.cumulative_fitness.fill(0.0)
        self.active_frames_in_epoch = 0
        self.alpha_idx = 0
        self.second_idx = 1

    def project_3d(self, pts_3d, w, h, focal=640.0, cam_dist=3.5):
        out = []
        for pt in pts_3d:
            z = max(cam_dist + pt[2], 0.35)
            out.append((int((pt[0] * focal) / z + w * 0.5), int((pt[1] * focal) / z + h * 0.5)))
        return out

    def draw_population_holograms(self, frame, hand_pos_3d: Optional[np.ndarray]):
        h, w, _ = frame.shape

        # 1. Dibujar los 9 candidatos competidores como esferas espectrales numeradas
        for i in range(self.POP_SIZE):
            if i == self.alpha_idx:
                continue
            pt_2d = self.project_3d([self.ind_positions[i]], w, h)[0]
            is_second = (i == self.second_idx)
            col = (255, 120, 255) if is_second else (160, 140, 80)
            radius = 12 if is_second else 7
            cv2.circle(frame, pt_2d, radius, col, 2, cv2.LINE_AA)
            cv2.putText(frame, f"#{i}", (pt_2d[0] + 9, pt_2d[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

        # 2. Dibujar el Cubo Holográfico Principal gobernado por el Individuo ALFA (#1 en PyTorch)
        alpha_pos = self.ind_positions[self.alpha_idx]
        self.holo_rot += np.array([0.012, 0.022, 0.006], dtype=np.float32)
        R = rotation_matrix_3d(*self.holo_rot)
        world_verts = self.cube_verts @ R.T + alpha_pos
        pts_2d = self.project_3d(world_verts, w, h)

        color_alpha = (0, 255, 160)
        overlay = frame.copy()
        for face in self.cube_faces:
            poly = np.array([pts_2d[idx] for idx in face], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (0, 75, 45))
        cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)

        for e1, e2 in self.cube_edges:
            cv2.line(frame, pts_2d[e1], pts_2d[e2], color_alpha, 2, cv2.LINE_AA)

        c2d = self.project_3d([alpha_pos], w, h)[0]
        cv2.putText(frame, f"ALFA AST #{self.alpha_idx} (Gen {self.generation})",
                    (c2d[0] - 68, c2d[1] - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 160), 2)

        if hand_pos_3d is not None:
            m2d = self.project_3d([hand_pos_3d], w, h)[0]
            cv2.line(frame, m2d, c2d, (0, 255, 160), 1, cv2.LINE_AA)
            cv2.circle(frame, m2d, 7, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_genetic_arena_hud(self, frame):
        h, w, _ = frame.shape

        # Panel Izquierdo: Estado de la Generación, VM Paralela y Crítico PyTorch
        cv2.rectangle(frame, (15, 15), (630, 175), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (630, 175), (0, 255, 160), 1)
        cv2.putText(frame, f"PROGRAMACION GENETICA AST | GENERACION #{self.generation} (Epoca: {self.active_frames_in_epoch}/{self.EPOCH_FRAMES})",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 160), 2)
        cv2.putText(frame, f"Tiempo VM (10 Bytecodes en paralelo): {self.vm.last_batch_exec_us:.1f} us | Loss Critico PyTorch: {self.critic.last_critic_loss:.4f}",
                    (26, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 230, 255), 1)
        cv2.putText(frame, f"Ultimo Cruce: {self.last_crossover_log}",
                    (26, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 215, 80), 1)

        # Barra de progreso para el siguiente Cruce Genético (Crossover)
        bar_w = int((self.active_frames_in_epoch / self.EPOCH_FRAMES) * 580)
        cv2.rectangle(frame, (26, 108), (606, 126), (35, 40, 48), -1)
        cv2.rectangle(frame, (26, 108), (26 + bar_w, 126), (255, 100, 255), -1)
        cv2.putText(frame, "Tecla [E]: Forzar Cruce Genetico Ahora  |  [R]: Reiniciar Poblacion  |  [Q]: Salir",
                    (26, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (170, 240, 200), 1)

        # Panel Derecho Superior: Tabla de Clasificación en Vivo de las 10 Funciones AST
        cv2.rectangle(frame, (w - 580, 15), (w - 15, 275), (14, 14, 22), -1)
        cv2.rectangle(frame, (w - 580, 15), (w - 15, 275), (255, 180, 0), 1)
        cv2.putText(frame, "RANKING EN VIVO DE LOS 10 INDIVIDUOS AST (Recompensa PyTorch)",
                    (w - 566, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 210, 0), 2)

        ranked = np.argsort(self.cumulative_fitness)[::-1]
        frames_div = max(1, self.active_frames_in_epoch)
        for rank_pos, ind_i in enumerate(ranked):
            avg_fit = float(self.cumulative_fitness[ind_i]) / frames_div
            ops_len = len(self.bytecodes[ind_i].packed_words)
            gene_p_str = self.population[ind_i].gene_prop.to_source()
            if len(gene_p_str) > 28:
                gene_p_str = gene_p_str[:25] + "..."

            if rank_pos == 0:
                tag = "[PADRE 1 - ALFA]"
                col = (0, 255, 150)
            elif rank_pos == 1:
                tag = "[PADRE 2 - BETA]"
                col = (255, 120, 255)
            else:
                tag = f"  Candidato #{rank_pos+1:<2d} "
                col = (205, 210, 220)

            row_txt = f"{tag} AST #{ind_i} | Fit: {avg_fit:+.2f} | {ops_len} ops | P: {gene_p_str}"
            cv2.putText(frame, row_txt, (w - 566, 62 + rank_pos * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, col, 1)

        # Panel Derecho Inferior: Código Fuente del Individuo ALFA Ganador
        alpha_chrom = self.population[self.alpha_idx]
        src_lines = alpha_chrom.get_source_lines()
        top_y = 290
        box_h = 32 + len(src_lines) * 19
        cv2.rectangle(frame, (w - 580, top_y), (w - 15, top_y + box_h), (12, 16, 20), -1)
        cv2.rectangle(frame, (w - 580, top_y), (w - 15, top_y + box_h), (0, 230, 255), 1)
        cv2.putText(frame, f"CODIGO AST DEL CAMPEON ACTUAL (AST #{self.alpha_idx} - {len(self.bytecodes[self.alpha_idx].packed_words)} Words 32-Bit)",
                    (w - 566, top_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1)

        for i, code_l in enumerate(src_lines):
            c = (120, 255, 150) if ("rama_" in code_l or "RETORNAR" in code_l) else (225, 235, 245)
            cv2.putText(frame, code_l, (w - 566, top_y + 44 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("==========================================================================")
        print(" NEXUS GENETIC AST ARENA: 10 FUNCIONES COMPITIENDO EN PARALELO")
        print(" - Mueve tu mano: los 10 árboles AST (compilados a Bytecode) intentarán")
        print("   seguir tu mano al mismo tiempo.")
        print(" - El Crítico en PyTorch puntúa a los 10 en cada fotograma y cada 35")
        print("   fotogramas cruza automáticamente los 2 mejores (PADRE 1 y PADRE 2).")
        print("==========================================================================")

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

                # Ejecutar los 10 Bytecodes en paralelo y evaluarlos en PyTorch
                self.step_parallel_population(hand_pos_3d, hand_vel)

            self.draw_population_holograms(frame, hand_pos_3d)
            self.draw_genetic_arena_hud(frame)
            cv2.imshow("NexusXR - Programacion Genetica de 10 ASTs en Paralelo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('e'):
                # Forzar cruce genético inmediato
                self.evolve_next_generation()
            elif key == ord('r'):
                # Reiniciar población desde Gen 1
                self.generation = 1
                self.population = [GeneticASTChromosome(ind_id=i, generation=1) for i in range(self.POP_SIZE)]
                self.bytecodes = [JITCompiler.compile(c) for c in self.population]
                self.cumulative_fitness.fill(0.0)
                self.active_frames_in_epoch = 0

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    arena = NexusGeneticASTArena()
    arena.run()
