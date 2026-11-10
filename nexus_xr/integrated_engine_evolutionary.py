import os
import glob
import json
import time
import math
import copy
import random
import hashlib
from enum import IntEnum
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

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


def unpack_word32(word: int) -> Tuple[VMOp, int, int]:
    return VMOp((word >> 24) & 0xFF), (word >> 12) & 0xFFF, word & 0xFFF


# =====================================================================
# 2. NODOS DEL ÁRBOL SINTÁCTICO (AST) CON SERIALIZACIÓN JSON
# =====================================================================

class ASTNode:
    def to_source(self) -> str:
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ASTNode":
        t = d["type"]
        if t == "Number":
            return NumberNode(float(d["value"]))
        if t == "Ident":
            return IdentNode(str(d["name"]))
        if t == "BinaryOp":
            return BinaryOpNode(str(d["op"]), ASTNode.from_dict(d["left"]), ASTNode.from_dict(d["right"]))
        raise ValueError(f"Tipo de nodo AST desconocido: {t}")

    def depth(self) -> int:
        return 1

    def node_count(self) -> int:
        return 1


@dataclass
class NumberNode(ASTNode):
    value: float
    def to_source(self) -> str:
        return f"{self.value:.3f}"
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Number", "value": round(float(self.value), 4)}


@dataclass
class IdentNode(ASTNode):
    name: str
    def to_source(self) -> str:
        return self.name
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Ident", "name": self.name}


@dataclass
class BinaryOpNode(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode

    def to_source(self) -> str:
        return f"({self.left.to_source()} {self.op} {self.right.to_source()})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "BinaryOp",
            "op": self.op,
            "left": self.left.to_dict(),
            "right": self.right.to_dict()
        }

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
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Assign", "var": self.var_name, "expr": self.expr.to_dict()}
    def node_count(self) -> int:
        return 1 + self.expr.node_count()


@dataclass
class ReturnNode(ASTNode):
    expr: ASTNode
    def to_source(self) -> str:
        return f"  RETORNAR {self.expr.to_source()}"
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Return", "expr": self.expr.to_dict()}
    def node_count(self) -> int:
        return 1 + self.expr.node_count()


# =====================================================================
# 3. FUNCIÓN AST EVOLUTIVA Y OPTIMIZADOR SIMBÓLICO (CONSTANT FOLDING)
# =====================================================================

TERMINAL_VARS = ["error", "vel_mano", "vel_holo"]


class EvolvableASTFunction:
    def __init__(
        self,
        ind_id: int,
        generation: int = 1,
        force_tree: Optional[ASTNode] = None,
        origin_tag: str = "Semilla Inicial"
    ):
        self.ind_id = ind_id
        self.generation = generation
        self.origin_tag = origin_tag
        self.params = ["mano", "holo", "vel_mano", "vel_holo"]

        raw_tree = force_tree if force_tree is not None else self.generate_random_tree(max_depth=3)
        self.force_tree: ASTNode = self.optimize_algebraic_tree(raw_tree)
        self.body: List[ASTNode] = []
        self.rebuild_body()

    @staticmethod
    def generate_random_tree(max_depth: int = 3) -> ASTNode:
        if max_depth <= 1 or random.random() < 0.35:
            var = random.choice(TERMINAL_VARS)
            gain = round(random.uniform(0.05, 0.50), 3)
            return BinaryOpNode("*", IdentNode(var), NumberNode(gain))
        elif random.random() < 0.75:
            op = random.choice(["+", "-"])
            return BinaryOpNode(
                op,
                EvolvableASTFunction.generate_random_tree(max_depth - 1),
                EvolvableASTFunction.generate_random_tree(max_depth - 1)
            )
        else:
            sub = EvolvableASTFunction.generate_random_tree(max_depth - 1)
            scale = round(random.uniform(0.55, 1.40), 3)
            return BinaryOpNode("*", sub, NumberNode(scale))

    @classmethod
    def optimize_algebraic_tree(cls, node: ASTNode) -> ASTNode:
        """
        Simplificación Simbólica Automática:
        - Plegado de constantes: (A * 0.5) * 2.0 -> A * 1.0 -> A
        - Eliminación de ramas nulas: X * 0.0 -> 0.0
        """
        if not isinstance(node, BinaryOpNode):
            return node

        left = cls.optimize_algebraic_tree(node.left)
        right = cls.optimize_algebraic_tree(node.right)

        # Plegado de dos números constantes
        if isinstance(left, NumberNode) and isinstance(right, NumberNode):
            if node.op == "+": return NumberNode(round(left.value + right.value, 4))
            if node.op == "-": return NumberNode(round(left.value - right.value, 4))
            if node.op == "*": return NumberNode(round(left.value * right.value, 4))

        # Asociatividad de escalares: (SubExpr * c1) * c2 -> SubExpr * (c1 * c2)
        if node.op == "*" and isinstance(right, NumberNode):
            if abs(right.value - 1.0) < 1e-4:
                return left
            if isinstance(left, BinaryOpNode) and left.op == "*" and isinstance(left.right, NumberNode):
                combined_k = round(left.right.value * right.value, 3)
                return BinaryOpNode("*", left.left, NumberNode(combined_k))

        # Identidad de resta consigo mismo: (X - X) -> X * 0.001
        if node.op == "-" and left.to_source() == right.to_source():
            return BinaryOpNode("*", IdentNode("error"), NumberNode(0.01))

        return BinaryOpNode(node.op, left, right)

    def rebuild_body(self):
        self.force_tree = self.optimize_algebraic_tree(self.force_tree)
        self.body = [
            AssignNode("error", BinaryOpNode("-", IdentNode("mano"), IdentNode("holo"))),
            AssignNode("impulso_ia", copy.deepcopy(self.force_tree)),
            ReturnNode(BinaryOpNode("+", IdentNode("holo"), IdentNode("impulso_ia")))
        ]

    def structural_hash(self) -> str:
        """Huella SHA-256 única de la fórmula matemática descubierta."""
        formula = self.force_tree.to_source()
        return hashlib.sha256(formula.encode("utf-8")).hexdigest()[:12]

    def get_source_code(self, fn_custom_name: Optional[str] = None) -> str:
        fn_name = fn_custom_name or f"controlador_ia_gen_{self.generation}_id_{self.ind_id}"
        lines = [
            f"# Origen Evolutivo: {self.origin_tag} | Hash AST: {self.structural_hash()}",
            f"FUNCION {fn_name}(mano, holo, vel_mano, vel_holo) {{"
        ]
        for s in self.body:
            lines.append(s.to_source())
        lines.append("}")
        return "\n".join(lines)


# =====================================================================
# 4. COMPILADOR JIT DE 32 BITS Y STACK VM
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

    def disassemble_lines(self) -> List[str]:
        out = []
        for ip, w in enumerate(self.packed_words):
            op, a1, _ = unpack_word32(int(w))
            if op == VMOp.LOAD_CONST:
                extra = f"const[{a1}]={self.const_pool[a1][0]:.3f}"
            elif op in (VMOp.LOAD_LOCAL, VMOp.STORE_LOCAL, VMOp.ADD_STORE, VMOp.SUB_STORE, VMOp.MUL_STORE):
                extra = f"${self.local_names[a1]}"
            else:
                extra = ""
            out.append(f"{ip:02d}: 0x{int(w):08X} {op.name:<11} {extra}")
        return out


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


class ParallelStackVM:
    def __init__(self):
        self.stack = np.zeros((48, 3), dtype=np.float64)
        self.locals = np.zeros((16, 3), dtype=np.float64)
        self.last_batch_us = 0.0

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

        loc[0] = mano; loc[1] = holo; loc[2] = vel_mano; loc[3] = vel_holo
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
# 5. EXPORTADOR AUTOMÁTICO DE DESCUBRIMIENTOS AST A ARCHIVOS '.NXS'
# =====================================================================

class AutonomousASTExporter:
    """
    Monitorea todas las funciones AST evaluadas por PyTorch durante la sesión.
    Cuando una función rompe el récord de Fitness o descubre una simplificación
    Pareto-óptima, la exporta automáticamente a archivos '.nxs'.
    """
    def __init__(self, export_dir: str = "nxs_discoveries"):
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

        self.master_catalog_path = "hall_of_fame_catalog.nxs"
        self.best_script_path = "best_discovered_physics.nxs"

        self.best_fitness_ever = -999.0
        self.best_op_count_at_record = 999
        self.total_auto_saves = 0

        # Lista de los Top-5 descubrimientos únicos (ordenados por Fitness descendente)
        self.top_discoveries: List[Dict[str, Any]] = []
        self.seen_hashes = set()
        self.export_logs: List[str] = [
            f"Exportador Automatico activo en carpeta './{self.export_dir}/'",
            "Esperando primer descubrimiento de la IA..."
        ]

    def log(self, msg: str):
        print(f"[AUTO-EXPORTER .NXS] {msg}")
        self.export_logs.append(msg)
        if len(self.export_logs) > 4:
            self.export_logs.pop(0)

    def evaluate_and_auto_export(
        self,
        candidate: EvolvableASTFunction,
        chunk: BytecodeChunk32,
        avg_fitness: float,
        avg_tracking_error: float
    ) -> bool:
        """
        Verifica si el candidato merece ser exportado automáticamente a disco.
        Criterio 1: Nuevo récord absoluto de recompensa en PyTorch.
        Criterio 2: Optimización de Pareto (misma recompensa dentro del 98%, pero menos instrucciones Bytecode).
        """
        if avg_fitness < 0.45:
            return False

        op_len = len(chunk.packed_words)
        ast_hash = candidate.structural_hash()

        is_strict_record = avg_fitness > (self.best_fitness_ever + 0.015)
        is_pareto_shorter = (
            avg_fitness >= (self.best_fitness_ever * 0.98)
            and op_len < self.best_op_count_at_record
            and ast_hash not in self.seen_hashes
        )

        if not (is_strict_record or is_pareto_shorter):
            return False

        reason = "NUEVO_RECORD_FITNESS" if is_strict_record else "OPTIMIZACION_PARETO_BYTECODE"
        if avg_fitness > self.best_fitness_ever:
            self.best_fitness_ever = avg_fitness
            self.best_op_count_at_record = op_len

        self.seen_hashes.add(ast_hash)
        self.total_auto_saves += 1
        timestamp_str = datetime.now().strftime("%H%M%S")

        # 1. Construir paquete estructurado del descubrimiento (.nxs)
        discovery_entry = {
            "rank_id": self.total_auto_saves,
            "discovery_reason": reason,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "generation": candidate.generation,
            "individual_id": candidate.ind_id,
            "structural_sha256": ast_hash,
            "metrics": {
                "pytorch_fitness": round(float(avg_fitness), 4),
                "avg_tracking_error_m": round(float(avg_tracking_error), 4),
                "bytecode_32bit_words": op_len,
                "ast_tree_depth": candidate.force_tree.depth(),
            },
            "nexus_lang_source": candidate.get_source_code().splitlines(),
            "force_tree_serialized": candidate.force_tree.to_dict(),
            "bytecode_hex": [f"0x{int(w):08X}" for w in chunk.packed_words],
            "bytecode_disassembly": chunk.disassemble_lines(),
        }

        # 2. Guardar archivo '.nxs' individual en la carpeta de descubrimientos
        indiv_filename = os.path.join(
            self.export_dir,
            f"ast_gen_{candidate.generation:03d}_fit_{avg_fitness:.2f}_{ast_hash[:6]}.nxs"
        )
        self._atomic_write_json(indiv_filename, {
            "header": {"magic": "NEXUS_AUTO_DISCOVERY_NXS_V1", "sha256": ast_hash},
            "discovery": discovery_entry
        })

        # 3. Actualizar el archivo de código fuente puro ('best_discovered_physics.nxs')
        if is_strict_record:
            script_header = (
                f"# =====================================================================\n"
                f"# FUNCION AST DESCUBIERTA Y OPTIMIZADA POR IA (AUTO-EXPORTADA)\n"
                f"# Generación: {candidate.generation} | Fitness PyTorch: {avg_fitness:+.4f} | Bytecode: {op_len} ops\n"
                f"# Fecha: {discovery_entry['timestamp']} | Razón: {reason}\n"
                f"# =====================================================================\n\n"
            )
            with open(self.best_script_path, "w", encoding="utf-8") as sf:
                sf.write(script_header + candidate.get_source_code("calcular_fisica_ia_campeona") + "\n")

        # 4. Actualizar el Catálogo Maestro Top-5 ('hall_of_fame_catalog.nxs')
        self.top_discoveries.append(discovery_entry)
        self.top_discoveries.sort(key=lambda x: x["metrics"]["pytorch_fitness"], reverse=True)
        self.top_discoveries = self.top_discoveries[:5]

        self._atomic_write_json(self.master_catalog_path, {
            "header": {
                "magic": "NEXUS_HALL_OF_FAME_CATALOG_V1",
                "total_discoveries_session": self.total_auto_saves,
                "best_fitness_record": round(float(self.best_fitness_ever), 4),
            },
            "top_5_ast_functions": self.top_discoveries
        })

        self.log(f"[{reason}] Guardado '{os.path.basename(indiv_filename)}' (Fit:{avg_fitness:+.2f}, {op_len} ops)")
        return True

    @staticmethod
    def _atomic_write_json(filepath: str, data: Dict[str, Any]):
        tmp = filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, filepath)

    def load_best_ast_from_disk(self) -> Optional[EvolvableASTFunction]:
        """Permite re-inyectar el mejor árbol AST guardado en disco de vuelta a la simulación."""
        if not os.path.exists(self.master_catalog_path):
            self.log("Aún no existe 'hall_of_fame_catalog.nxs' en disco.")
            return None
        try:
            with open(self.master_catalog_path, "r", encoding="utf-8") as f:
                cat = json.load(f)
            top_list = cat.get("top_5_ast_functions", [])
            if not top_list:
                return None
            best_entry = top_list[0]
            restored_tree = ASTNode.from_dict(best_entry["force_tree_serialized"])
            fn = EvolvableASTFunction(
                ind_id=0,
                generation=best_entry["generation"],
                force_tree=restored_tree,
                origin_tag=f"Recargado de .NXS ({best_entry['structural_sha256'][:6]})"
            )
            self.log(f"Inyectado campeón histórico desde '{self.master_catalog_path}' (Fit:{best_entry['metrics']['pytorch_fitness']}).")
            return fn
        except Exception as e:
            self.log(f"Error cargando catálogo .nxs: {e}")
            return None


# =====================================================================
# 6. OPERADORES GENÉTICOS Y CRÍTICO PYTORCH
# =====================================================================

class GeneticOperators:
    @staticmethod
    def collect_subtrees(node: ASTNode) -> List[ASTNode]:
        res = [node]
        if isinstance(node, BinaryOpNode):
            if node.op in ("+", "-"):
                res.extend(GeneticOperators.collect_subtrees(node.left))
                res.extend(GeneticOperators.collect_subtrees(node.right))
        return res

    @staticmethod
    def inject_subtree(root: ASTNode, donor: ASTNode, depth: int = 0) -> ASTNode:
        if depth >= 3 or random.random() < 0.35:
            return copy.deepcopy(donor)
        if isinstance(root, BinaryOpNode) and root.op in ("+", "-"):
            if random.random() < 0.5:
                return BinaryOpNode(root.op, GeneticOperators.inject_subtree(root.left, donor, depth + 1), copy.deepcopy(root.right))
            else:
                return BinaryOpNode(root.op, copy.deepcopy(root.left), GeneticOperators.inject_subtree(root.right, donor, depth + 1))
        return BinaryOpNode("+", copy.deepcopy(root), BinaryOpNode("*", copy.deepcopy(donor), NumberNode(0.35)))

    @classmethod
    def crossover_and_mutate(cls, p1: EvolvableASTFunction, p2: EvolvableASTFunction, child_id: int, gen: int) -> EvolvableASTFunction:
        donor = random.choice(cls.collect_subtrees(p2.force_tree))
        merged = cls.inject_subtree(p1.force_tree, donor)
        if merged.depth() > 5:
            merged = BinaryOpNode("+", BinaryOpNode("*", copy.deepcopy(p1.force_tree), NumberNode(0.6)), BinaryOpNode("*", copy.deepcopy(donor), NumberNode(0.4)))

        def _mutate(n: ASTNode) -> ASTNode:
            if isinstance(n, NumberNode) and random.random() < 0.45:
                return NumberNode(round(float(np.clip(n.value + random.gauss(0.0, 0.06), 0.02, 1.35)), 3))
            if isinstance(n, IdentNode) and n.name in TERMINAL_VARS and random.random() < 0.15:
                return IdentNode(random.choice(TERMINAL_VARS))
            if isinstance(n, BinaryOpNode):
                op = ("-" if n.op == "+" else "+") if (n.op in ("+", "-") and random.random() < 0.15) else n.op
                return BinaryOpNode(op, _mutate(n.left), _mutate(n.right))
            return n

        mutated_tree = _mutate(merged)
        return EvolvableASTFunction(child_id, gen, mutated_tree, origin_tag=f"Cruce(#{p1.ind_id}x#{p2.ind_id})")


class PyTorchFitnessCritic(nn.Module):
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

    def evaluate_batch(self, errors: np.ndarray, jerks: np.ndarray, aligns: np.ndarray, speed: float, ops: np.ndarray) -> np.ndarray:
        n = len(errors)
        feats = np.stack([errors, jerks, aligns, np.full(n, speed), ops / 30.0, errors ** 2], axis=1).astype(np.float32)
        feat_t = torch.from_numpy(feats).to(self.device)

        target_np = (np.exp(-3.6 * errors) * 2.6 + 0.35 * aligns - 1.15 * jerks - 0.015 * ops).astype(np.float32)
        target_t = torch.from_numpy(target_np).unsqueeze(1).to(self.device)

        self.net.train()
        pred = self.net(feat_t)
        loss = self.criterion(pred, target_t)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

        with torch.no_grad():
            return (0.72 * target_t.squeeze(1) + 0.28 * pred.squeeze(1)).cpu().numpy()


# =====================================================================
# 7. MOTOR PRINCIPAL CON AUTO-EXPORTACIÓN '.NXS' EN TIEMPO REAL
# =====================================================================

def rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


class NexusAutoExportDiscoveryEngine:
    POP_SIZE = 10
    EPOCH_FRAMES = 32

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.vm = ParallelStackVM()
        self.critic = PyTorchFitnessCritic()
        self.exporter = AutonomousASTExporter()
        self.generation = 1

        self.population: List[EvolvableASTFunction] = [
            EvolvableASTFunction(i, 1, EvolvableASTFunction.generate_random_tree((i % 3) + 1))
            for i in range(self.POP_SIZE)
        ]
        self.chunks: List[BytecodeChunk32] = [GeneticJITCompiler.compile(ind) for ind in self.population]

        self.positions = np.zeros((self.POP_SIZE, 3), dtype=np.float32)
        self.velocities = np.zeros((self.POP_SIZE, 3), dtype=np.float32)
        self.prev_vels = np.zeros((self.POP_SIZE, 3), dtype=np.float32)
        self.fitness_accum = np.zeros(self.POP_SIZE, dtype=np.float32)
        self.error_accum = np.zeros(self.POP_SIZE, dtype=np.float32)

        self.alpha_idx = 0
        self.epoch_frames = 0
        self.prev_hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.holo_rot = np.array([0.3, 0.5, 0.1], dtype=np.float32)

        self.cube_verts = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ], dtype=np.float32) * 0.46
        self.cube_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        self.cube_faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6)]

    def step_simulation(self, hand_pos: np.ndarray, hand_vel: np.ndarray):
        t0 = time.perf_counter()
        errors = np.zeros(self.POP_SIZE, dtype=np.float32)
        jerks = np.zeros(self.POP_SIZE, dtype=np.float32)
        aligns = np.zeros(self.POP_SIZE, dtype=np.float32)
        ops = np.array([len(c.packed_words) for c in self.chunks], dtype=np.float32)

        hp_64 = hand_pos.astype(np.float64)
        hv_64 = hand_vel.astype(np.float64)
        speed = float(np.linalg.norm(hand_vel))

        for i in range(self.POP_SIZE):
            cur_p = self.positions[i]
            cur_v = self.velocities[i]
            next_p = self.vm.execute_chunk(self.chunks[i], hp_64, cur_p.astype(np.float64), hv_64, cur_v.astype(np.float64))

            new_v = next_p - cur_p
            jerk = float(np.linalg.norm(new_v - self.prev_vels[i]))
            err = float(np.linalg.norm(hand_pos - next_p))
            align = float(np.dot(hand_vel, new_v) / (speed * np.linalg.norm(new_v) + 1e-6)) if speed > 0.005 else 0.0

            self.positions[i] = next_p
            self.prev_vels[i] = cur_v
            self.velocities[i] = new_v
            errors[i] = err; jerks[i] = jerk; aligns[i] = align

        self.vm.last_batch_us = (time.perf_counter() - t0) * 1_000_000.0
        self.fitness_accum += self.critic.evaluate_batch(errors, jerks, aligns, speed, ops)
        self.error_accum += errors

        order = np.argsort(self.fitness_accum)[::-1]
        self.alpha_idx = int(order[0])

        self.epoch_frames += 1
        if self.epoch_frames >= self.EPOCH_FRAMES:
            self.evolve_and_auto_export()

    def evolve_and_auto_export(self):
        order = np.argsort(self.fitness_accum)[::-1]
        p1_idx, p2_idx = int(order[0]), int(order[1])

        champ = copy.deepcopy(self.population[p1_idx])
        runner_up = copy.deepcopy(self.population[p2_idx])

        avg_fit = float(self.fitness_accum[p1_idx]) / max(1, self.epoch_frames)
        avg_err = float(self.error_accum[p1_idx]) / max(1, self.epoch_frames)

        # 1. INVOCAR AL EXPORTADOR AUTOMÁTICO '.NXS' SI EL CAMPEÓN DESCUBRIÓ UN MEJOR AST
        self.exporter.evaluate_and_auto_export(champ, self.chunks[p1_idx], avg_fit, avg_err)

        # 2. Cruzar los dos mejores y mutar para la siguiente generación
        self.generation += 1
        champ.ind_id = 0; champ.generation = self.generation; champ.rebuild_body()
        runner_up.ind_id = 1; runner_up.generation = self.generation; runner_up.rebuild_body()

        new_pop = [champ, runner_up]
        for cid in range(2, self.POP_SIZE - 1):
            new_pop.append(GeneticOperators.crossover_and_mutate(champ, runner_up, cid, self.generation))
        new_pop.append(EvolvableASTFunction(self.POP_SIZE - 1, self.generation, origin_tag="Explorador Fresco"))

        self.population = new_pop
        self.chunks = [GeneticJITCompiler.compile(ind) for ind in self.population]

        anchor = self.positions[p1_idx].copy()
        for i in range(self.POP_SIZE):
            self.positions[i] = anchor.copy()
            self.velocities[i] = 0.0; self.prev_vels[i] = 0.0

        self.fitness_accum.fill(0.0)
        self.error_accum.fill(0.0)
        self.epoch_frames = 0
        self.alpha_idx = 0

    def project_3d(self, pts_3d, w, h, focal=640.0, cam_dist=3.5):
        out = []
        for pt in pts_3d:
            z = max(cam_dist + pt[2], 0.35)
            out.append((int((pt[0] * focal) / z + w * 0.5), int((pt[1] * focal) / z + h * 0.5)))
        return out

    def draw_scene_and_hud(self, frame, hand_pos_3d: Optional[np.ndarray]):
        h, w, _ = frame.shape

        for i in range(self.POP_SIZE):
            if i == self.alpha_idx: continue
            p2d = self.project_3d([self.positions[i]], w, h)[0]
            cv2.circle(frame, p2d, 7, (170, 150, 80), 1, cv2.LINE_AA)

        alpha_pos = self.positions[self.alpha_idx]
        self.holo_rot += np.array([0.012, 0.022, 0.006], dtype=np.float32)
        R = rotation_matrix_3d(*self.holo_rot)
        pts_2d = self.project_3d(self.cube_verts @ R.T + alpha_pos, w, h)

        overlay = frame.copy()
        for face in self.cube_faces:
            cv2.fillPoly(overlay, [np.array([pts_2d[idx] for idx in face], dtype=np.int32)], (0, 80, 50))
        cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)
        for e1, e2 in self.cube_edges:
            cv2.line(frame, pts_2d[e1], pts_2d[e2], (0, 255, 160), 2, cv2.LINE_AA)

        # Panel Izquierdo: Registro en Vivo del Exportador Automático '.NXS'
        cv2.rectangle(frame, (15, 15), (650, 185), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (650, 185), (0, 255, 160), 1)
        cv2.putText(frame, f"AUTO-EXPORTADOR DE ASTs (.NXS) | Gen #{self.generation} | Guardados: {self.exporter.total_auto_saves}",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 160), 2)
        cv2.putText(frame, f"Record Fitness PyTorch: {self.exporter.best_fitness_ever:+.3f} | Optimo Bytecode: {self.exporter.best_op_count_at_record} ops",
                    (26, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (255, 215, 80), 1)

        for idx, log_l in enumerate(self.exporter.export_logs[-4:]):
            cv2.putText(frame, f"> {log_l}", (26, 88 + idx * 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (215, 230, 245), 1)

        cv2.putText(frame, "Teclas: [S] Forzar Exportacion .nxs  |  [L] Cargar Mejor AST de Disco  |  [Q] Salir",
                    (26, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 240, 255), 1)

        # Panel Derecho: Top-5 Salón de la Fama Guardado en 'hall_of_fame_catalog.nxs'
        cv2.rectangle(frame, (w - 585, 15), (w - 15, 295), (14, 14, 22), -1)
        cv2.rectangle(frame, (w - 585, 15), (w - 15, 295), (255, 180, 0), 1)
        cv2.putText(frame, "CATALOGO '.NXS' TOP-5 DESCUBRIMIENTOS DE LA IA",
                    (w - 570, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 210, 0), 2)

        for idx, item in enumerate(self.exporter.top_discoveries[:5]):
            m = item["metrics"]
            sha = item["structural_sha256"][:6]
            line_s = f"#{idx+1} [Gen {item['generation']:02d} | SHA:{sha}] Fit:{m['pytorch_fitness']:+.2f} | {m['bytecode_32bit_words']} ops | Err:{m['avg_tracking_error_m']:.3f}"
            cv2.putText(frame, line_s, (w - 570, 64 + idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 255, 150) if idx == 0 else (210, 215, 225), 1)

        # Mostrar el código fuente de la función Alfa actual debajo del catálogo
        src_lines = self.population[self.alpha_idx].get_source_code().splitlines()
        for i, code_l in enumerate(src_lines[:5]):
            cv2.putText(frame, code_l[:68], (w - 570, 185 + i * 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (160, 230, 255), 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            success, raw = cap.read()
            if not success: break

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
                self.step_simulation(hand_pos_3d, hand_vel)

            self.draw_scene_and_hud(frame, hand_pos_3d)
            cv2.imshow("NexusXR - Auto-Exportador de Descubrimientos AST (.nxs)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                # Forzar guardado manual inmediato del campeón actual
                champ = self.population[self.alpha_idx]
                self.exporter.best_fitness_ever = -999.0
                self.exporter.evaluate_and_auto_export(champ, self.chunks[self.alpha_idx], 1.50, 0.05)
            elif key == ord('l'):
                # Cargar el mejor AST desde disco e inyectarlo como Individuo 0
                loaded_fn = self.exporter.load_best_ast_from_disk()
                if loaded_fn is not None:
                    self.population[0] = loaded_fn
                    self.chunks[0] = GeneticJITCompiler.compile(loaded_fn)
                    self.alpha_idx = 0

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = NexusAutoExportDiscoveryEngine()
    engine.run()
