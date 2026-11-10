import cv2
import mediapipe as mp
import numpy as np
import math
import time
import copy
import queue
import threading
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


# =====================================================================
# 1. PRIMITIVA MATEMÁTICA SIMD-LIKE: VECTOR ESPACIAL 3D (Vec3)
# =====================================================================

class Vec3:
    __slots__ = ("x", "y", "z")  # Optimización de memoria a nivel de C-struct en Python

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def to_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @staticmethod
    def from_np(arr: np.ndarray) -> "Vec3":
        return Vec3(float(arr[0]), float(arr[1]), float(arr[2]))

    def __add__(self, o):
        if isinstance(o, Vec3):
            return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)
        v = float(o)
        return Vec3(self.x + v, self.y + v, self.z + v)

    def __radd__(self, o):
        return self.__add__(o)

    def __sub__(self, o):
        if isinstance(o, Vec3):
            return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)
        v = float(o)
        return Vec3(self.x - v, self.y - v, self.z - v)

    def __mul__(self, o):
        if isinstance(o, Vec3):
            return Vec3(self.x * o.x, self.y * o.y, self.z * o.z)
        v = float(o)
        return Vec3(self.x * v, self.y * v, self.z * v)

    def __rmul__(self, o):
        return self.__mul__(o)

    def __truediv__(self, o):
        v = float(o) if not isinstance(o, Vec3) else 1.0
        v = v if abs(v) > 1e-6 else 1e-6
        return Vec3(self.x / v, self.y / v, self.z / v)

    def __repr__(self):
        return f"[{self.x:.2f}, {self.y:.2f}, {self.z:.2f}]"


# =====================================================================
# 2. DEFINICIÓN DE OPCODES Y MÁQUINA VIRTUAL DE PILA (STACK VM)
# =====================================================================

class OpCode(IntEnum):
    LOAD_CONST  = 1   # Empuja constants[arg] a la pila
    LOAD_LOCAL  = 2   # Empuja locals[arg] a la pila
    STORE_LOCAL = 3   # Saca el tope de la pila y lo guarda en locals[arg]
    MAKE_VEC3   = 4   # Saca Z, Y, X y empuja Vec3(X, Y, Z)
    BINARY_ADD  = 5   # Saca B, A -> empuja A + B
    BINARY_SUB  = 6   # Saca B, A -> empuja A - B
    BINARY_MUL  = 7   # Saca B, A -> empuja A * B
    BINARY_DIV  = 8   # Saca B, A -> empuja A / B
    RETURN_VAL  = 9   # Retorna el tope de la pila


class CompiledBytecodeChunk:
    """
    Contiene el código binario compilado de una función AST:
    - instructions: lista plana de tuplas (OpCode, operando_entero)
    - constants: tabla de constantes numéricas o Vec3 inmutables
    - var_names: mapeo de índice de slot -> nombre de variable
    """
    def __init__(self, name: str, param_count: int, version: int = 1):
        self.name = name
        self.param_count = param_count
        self.version = version
        self.instructions: List[Tuple[OpCode, int]] = []
        self.constants: List[Any] = []
        self.var_names: List[str] = []
        self.name_to_slot: Dict[str, int] = {}

    def resolve_local_slot(self, name: str) -> int:
        if name not in self.name_to_slot:
            idx = len(self.var_names)
            self.name_to_slot[name] = idx
            self.var_names.append(name)
        return self.name_to_slot[name]

    def add_constant(self, val: Any) -> int:
        for i, existing in enumerate(self.constants):
            if type(existing) is type(val) and existing == val:
                return i
        idx = len(self.constants)
        self.constants.append(val)
        return idx

    def emit(self, opcode: OpCode, operand: int = 0):
        self.instructions.append((opcode, operand))

    def disassemble(self) -> List[str]:
        """Genera la vista en lenguaje ensamblador del Bytecode actual."""
        lines = [f"=== BYTECODE [{self.name} v{self.version}] ({len(self.instructions)} ops) ==="]
        for ip, (op, arg) in enumerate(self.instructions):
            op_name = op.name.ljust(12)
            if op == OpCode.LOAD_CONST:
                comment = f"({self.constants[arg]})"
                lines.append(f"{ip:02d}: {op_name} {arg:<2d} {comment}")
            elif op in (OpCode.LOAD_LOCAL, OpCode.STORE_LOCAL):
                comment = f"(${self.var_names[arg]})"
                lines.append(f"{ip:02d}: {op_name} {arg:<2d} {comment}")
            else:
                lines.append(f"{ip:02d}: {op_name}")
        return lines


class NexusStackVM:
    """
    Máquina Virtual de Pila de alta velocidad.
    Ejecuta bloques CompiledBytecodeChunk sin usar diccionarios ni recursión.
    """
    def __init__(self):
        self.last_exec_time_us: float = 0.0
        self.total_instructions_executed: int = 0

    def execute_chunk(self, chunk: CompiledBytecodeChunk, positional_args: List[Any]) -> Any:
        t0 = time.perf_counter()

        # Pre-alojar registros locales contiguos en memoria
        locals_reg: List[Any] = [None] * len(chunk.var_names)
        for i in range(chunk.param_count):
            locals_reg[i] = positional_args[i]

        stack: List[Any] = []
        push = stack.append
        pop = stack.pop
        consts = chunk.constants
        code = chunk.instructions

        # Ciclo de despacho de instrucciones (Instruction Dispatch Loop)
        for op, arg in code:
            if op == OpCode.LOAD_CONST:
                push(consts[arg])
            elif op == OpCode.LOAD_LOCAL:
                push(locals_reg[arg])
            elif op == OpCode.STORE_LOCAL:
                locals_reg[arg] = pop()
            elif op == OpCode.BINARY_ADD:
                b = pop(); a = pop()
                push(a + b)
            elif op == OpCode.BINARY_SUB:
                b = pop(); a = pop()
                push(a - b)
            elif op == OpCode.BINARY_MUL:
                b = pop(); a = pop()
                push(a * b)
            elif op == OpCode.BINARY_DIV:
                b = pop(); a = pop()
                push(a / b)
            elif op == OpCode.MAKE_VEC3:
                z = pop(); y = pop(); x = pop()
                push(Vec3(x, y, z))
            elif op == OpCode.RETURN_VAL:
                res = pop()
                self.last_exec_time_us = (time.perf_counter() - t0) * 1_000_000.0
                self.total_instructions_executed += len(code)
                return res

        self.last_exec_time_us = (time.perf_counter() - t0) * 1_000_000.0
        return stack[-1] if stack else None


# =====================================================================
# 3. LEXER, PARSER, AST HOMOICÓNICO Y COMPILADOR DE BYTECODE
# =====================================================================

@dataclass
class Token:
    type: str
    value: Any
    line: int

class NexusLexer:
    KEYWORDS = {
        "FUNCION": "FUNCION",
        "RETORNAR": "RETORNAR",
        "OPTIMIZAR_AST": "OPTIMIZAR_AST",
        "MUTAR_FUNCION": "MUTAR_FUNCION",
        "SINTETIZAR_FUNCION": "SINTETIZAR_FUNCION",
    }

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1

    def _peek(self) -> str:
        return self.source[self.pos] if self.pos < len(self.source) else "\0"

    def _advance(self) -> str:
        ch = self._peek()
        self.pos += 1
        if ch == "\n": self.line += 1
        return ch

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while self.pos < len(self.source):
            ch = self._peek()
            if ch in " \t\r\n":
                self._advance(); continue
            if ch == "#":
                while self._peek() not in ("\n", "\0"): self._advance()
                continue
            if ch.isdigit() or (ch == "-" and self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit()):
                num = self._advance()
                while self._peek().isdigit() or self._peek() == ".": num += self._advance()
                tokens.append(Token("NUMBER", float(num), self.line)); continue
            if ch.isalpha() or ch == "_":
                ident = ""
                while self._peek().isalnum() or self._peek() == "_": ident += self._advance()
                tokens.append(Token(self.KEYWORDS.get(ident.upper(), "IDENT"), ident, self.line)); continue
            if self.source[self.pos:self.pos + 2] == "->":
                self._advance(); self._advance()
                tokens.append(Token("ARROW", "->", self.line)); continue

            single = {"+": "OP", "-": "OP", "*": "OP", "/": "OP", "=": "ASSIGN",
                      "(": "LPAREN", ")": "RPAREN", "{": "LBRACE", "}": "RBRACE",
                      "[": "LBRACKET", "]": "RBRACKET", ",": "COMMA"}
            if ch in single:
                tokens.append(Token(single[ch], self._advance(), self.line)); continue
            raise SyntaxError(f"Carácter no válido '{ch}' en línea {self.line}")
        tokens.append(Token("EOF", None, self.line))
        return tokens


class ASTNode:
    def to_source(self, indent: int = 0) -> str:
        raise NotImplementedError()

@dataclass
class NumberNode(ASTNode):
    value: float
    def to_source(self, indent: int = 0) -> str:
        return f"{int(self.value)}" if self.value.is_integer() else f"{self.value:.3f}"

@dataclass
class IdentNode(ASTNode):
    name: str
    def to_source(self, indent: int = 0) -> str:
        return self.name

@dataclass
class Vector3Node(ASTNode):
    x: ASTNode; y: ASTNode; z: ASTNode
    def to_source(self, indent: int = 0) -> str:
        return f"[{self.x.to_source()}, {self.y.to_source()}, {self.z.to_source()}]"

@dataclass
class BinaryOpNode(ASTNode):
    op: str; left: ASTNode; right: ASTNode
    def to_source(self, indent: int = 0) -> str:
        return f"({self.left.to_source()} {self.op} {self.right.to_source()})"

@dataclass
class AssignNode(ASTNode):
    name: str; expr: ASTNode
    def to_source(self, indent: int = 0) -> str:
        return f"{'  '*indent}{self.name} = {self.expr.to_source()}"

@dataclass
class ReturnNode(ASTNode):
    expr: ASTNode
    def to_source(self, indent: int = 0) -> str:
        return f"{'  '*indent}RETORNAR {self.expr.to_source()}"

@dataclass
class FunctionDeclNode(ASTNode):
    name: str; params: List[str]; body: List[ASTNode]; version: int = 1
    def to_source(self, indent: int = 0) -> str:
        lines = [f"FUNCION {self.name}({', '.join(self.params)}) {{ # [AST v{self.version}]"]
        for s in self.body: lines.append(s.to_source(indent + 1))
        lines.append("}")
        return "\n".join(lines)

@dataclass
class OptimizeAstNode(ASTNode):
    target_func: str
    def to_source(self, indent: int = 0) -> str: return f"OPTIMIZAR_AST({self.target_func})"

@dataclass
class MutateFunctionNode(ASTNode):
    target_func: str; target_var: str; new_expr: ASTNode
    def to_source(self, indent: int = 0) -> str:
        return f"MUTAR_FUNCION({self.target_func}, {self.target_var} -> {self.new_expr.to_source()})"

@dataclass
class SynthesizeFunctionNode(ASTNode):
    new_name: str; func_a: str; func_b: str; op: str
    def to_source(self, indent: int = 0) -> str:
        return f"SINTETIZAR_FUNCION({self.new_name} = {self.func_a} {self.op} {self.func_b})"


class NexusParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens; self.pos = 0

    def _curr(self) -> Token: return self.tokens[self.pos]

    def _eat(self, t_type: str) -> Token:
        tok = self._curr()
        if tok.type != t_type:
            raise SyntaxError(f"Se esperaba {t_type}, llegó {tok.type}")
        self.pos += 1
        return tok

    def parse(self) -> List[ASTNode]:
        stmts = []
        while self._curr().type != "EOF": stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self) -> ASTNode:
        t = self._curr().type
        if t == "FUNCION":
            self._eat("FUNCION"); name = self._eat("IDENT").value; self._eat("LPAREN")
            params = []
            if self._curr().type != "RPAREN":
                params.append(self._eat("IDENT").value)
                while self._curr().type == "COMMA":
                    self._eat("COMMA"); params.append(self._eat("IDENT").value)
            self._eat("RPAREN"); self._eat("LBRACE")
            body = []
            while self._curr().type != "RBRACE": body.append(self.parse_stmt())
            self._eat("RBRACE")
            return FunctionDeclNode(name, params, body)
        if t == "RETORNAR":
            self._eat("RETORNAR"); return ReturnNode(self.parse_expr())
        if t == "OPTIMIZAR_AST":
            self._eat("OPTIMIZAR_AST"); self._eat("LPAREN")
            fn = self._eat("IDENT").value; self._eat("RPAREN")
            return OptimizeAstNode(fn)
        if t == "MUTAR_FUNCION":
            self._eat("MUTAR_FUNCION"); self._eat("LPAREN")
            fn = self._eat("IDENT").value; self._eat("COMMA")
            var = self._eat("IDENT").value; self._eat("ARROW")
            expr = self.parse_expr(); self._eat("RPAREN")
            return MutateFunctionNode(fn, var, expr)
        if t == "SINTETIZAR_FUNCION":
            self._eat("SINTETIZAR_FUNCION"); self._eat("LPAREN")
            new_n = self._eat("IDENT").value; self._eat("ASSIGN")
            fa = self._eat("IDENT").value; op = self._eat("OP").value; fb = self._eat("IDENT").value
            self._eat("RPAREN")
            return SynthesizeFunctionNode(new_n, fa, fb, op)
        if t == "IDENT" and self.tokens[self.pos + 1].type == "ASSIGN":
            name = self._eat("IDENT").value; self._eat("ASSIGN")
            return AssignNode(name, self.parse_expr())
        return self.parse_expr()

    def parse_expr(self) -> ASTNode:
        node = self.parse_term()
        while self._curr().type == "OP" and self._curr().value in ("+", "-"):
            op = self._eat("OP").value
            node = BinaryOpNode(op, node, self.parse_term())
        return node

    def parse_term(self) -> ASTNode:
        node = self.parse_primary()
        while self._curr().type == "OP" and self._curr().value in ("*", "/"):
            op = self._eat("OP").value
            node = BinaryOpNode(op, node, self.parse_primary())
        return node

    def parse_primary(self) -> ASTNode:
        tok = self._curr()
        if tok.type == "NUMBER": self._eat("NUMBER"); return NumberNode(tok.value)
        if tok.type == "IDENT": self._eat("IDENT"); return IdentNode(tok.value)
        if tok.type == "LBRACKET":
            self._eat("LBRACKET")
            x = self.parse_expr(); self._eat("COMMA")
            y = self.parse_expr(); self._eat("COMMA")
            z = self.parse_expr(); self._eat("RBRACKET")
            return Vector3Node(x, y, z)
        if tok.type == "LPAREN":
            self._eat("LPAREN"); e = self.parse_expr(); self._eat("RPAREN")
            return e
        raise SyntaxError(f"Token inesperado: {tok}")


class BytecodeCompiler:
    """
    Traduce un árbol AST (FunctionDeclNode) a un bloque lineal CompiledBytecodeChunk.
    Si un Vector3Node tiene componentes constantes, lo pre-construye en el Constant Pool.
    """
    @staticmethod
    def compile_function(fn_node: FunctionDeclNode) -> CompiledBytecodeChunk:
        chunk = CompiledBytecodeChunk(fn_node.name, len(fn_node.params), fn_node.version)

        # 1. Reservar los primeros slots para los parámetros de entrada
        for param in fn_node.params:
            chunk.resolve_local_slot(param)

        # 2. Emitir Bytecode para cada sentencia del cuerpo del AST
        for stmt in fn_node.body:
            BytecodeCompiler._compile_stmt(stmt, chunk)

        return chunk

    @staticmethod
    def _compile_stmt(stmt: ASTNode, chunk: CompiledBytecodeChunk):
        if isinstance(stmt, AssignNode):
            BytecodeCompiler._compile_expr(stmt.expr, chunk)
            slot = chunk.resolve_local_slot(stmt.name)
            chunk.emit(OpCode.STORE_LOCAL, slot)
        elif isinstance(stmt, ReturnNode):
            BytecodeCompiler._compile_expr(stmt.expr, chunk)
            chunk.emit(OpCode.RETURN_VAL, 0)

    @staticmethod
    def _compile_expr(expr: ASTNode, chunk: CompiledBytecodeChunk):
        if isinstance(expr, NumberNode):
            c_idx = chunk.add_constant(expr.value)
            chunk.emit(OpCode.LOAD_CONST, c_idx)

        elif isinstance(expr, IdentNode):
            slot = chunk.resolve_local_slot(expr.name)
            chunk.emit(OpCode.LOAD_LOCAL, slot)

        elif isinstance(expr, Vector3Node):
            # Optimización: Si los 3 componentes son constantes numéricas, guardar Vec3 directo en Constant Pool
            if isinstance(expr.x, NumberNode) and isinstance(expr.y, NumberNode) and isinstance(expr.z, NumberNode):
                vec_const = Vec3(expr.x.value, expr.y.value, expr.z.value)
                c_idx = chunk.add_constant(vec_const)
                chunk.emit(OpCode.LOAD_CONST, c_idx)
            else:
                BytecodeCompiler._compile_expr(expr.x, chunk)
                BytecodeCompiler._compile_expr(expr.y, chunk)
                BytecodeCompiler._compile_expr(expr.z, chunk)
                chunk.emit(OpCode.MAKE_VEC3, 0)

        elif isinstance(expr, BinaryOpNode):
            BytecodeCompiler._compile_expr(expr.left, chunk)
            BytecodeCompiler._compile_expr(expr.right, chunk)
            if expr.op == "+": chunk.emit(OpCode.BINARY_ADD, 0)
            elif expr.op == "-": chunk.emit(OpCode.BINARY_SUB, 0)
            elif expr.op == "*": chunk.emit(OpCode.BINARY_MUL, 0)
            elif expr.op == "/": chunk.emit(OpCode.BINARY_DIV, 0)


# =====================================================================
# 4. GESTOR JIT: AST MUTABLE -> RECOMPILACIÓN INSTANTÁNEA A BYTECODE
# =====================================================================

class HomoiconicBytecodeRuntime:
    def __init__(self):
        self.ast_table: Dict[str, FunctionDeclNode] = {}
        self.bytecode_table: Dict[str, CompiledBytecodeChunk] = {}
        self.vm = NexusStackVM()
        self.logs: List[str] = [
            "COMPILADOR BYTECODE + STACK VM ACTIVOS.",
            "Cada mutacion del AST se recompila a instrucciones binarias."
        ]

    def log(self, msg: str):
        print(f"[JIT-VM] {msg}")
        self.logs.append(msg)
        if len(self.logs) > 5:
            self.logs.pop(0)

    def _fold_constants(self, node: ASTNode) -> ASTNode:
        if isinstance(node, BinaryOpNode):
            l = self._fold_constants(node.left)
            r = self._fold_constants(node.right)
            if isinstance(l, NumberNode) and isinstance(r, NumberNode):
                if node.op == "+": return NumberNode(l.value + r.value)
                if node.op == "-": return NumberNode(l.value - r.value)
                if node.op == "*": return NumberNode(l.value * r.value)
                if node.op == "/" and abs(r.value) > 1e-6: return NumberNode(l.value / r.value)
            if node.op == "*" and isinstance(r, NumberNode) and r.value == 1.0: return l
            if node.op == "+" and isinstance(r, NumberNode) and r.value == 0.0: return l
            return BinaryOpNode(node.op, l, r)
        if isinstance(node, AssignNode):
            return AssignNode(node.name, self._fold_constants(node.expr))
        if isinstance(node, ReturnNode):
            return ReturnNode(self._fold_constants(node.expr))
        if isinstance(node, Vector3Node):
            return Vector3Node(self._fold_constants(node.x), self._fold_constants(node.y), self._fold_constants(node.z))
        return node

    def _recompile_to_bytecode(self, func_name: str):
        """Paso clave: Toma el AST mutado, lo compila a Bytecode y reemplaza el chunk en la VM."""
        fn_ast = self.ast_table[func_name]
        chunk = BytecodeCompiler.compile_function(fn_ast)
        self.bytecode_table[func_name] = chunk
        self.log(f"JIT: '{func_name}' v{fn_ast.version} -> {len(chunk.instructions)} OpCodes compilados.")

    def execute_source(self, source_code: str):
        try:
            tokens = NexusLexer(source_code).tokenize()
            nodes = NexusParser(tokens).parse()
            for node in nodes:
                self._process_node(node)
        except Exception as e:
            self.log(f"Error de compilación: {e}")

    def _process_node(self, node: ASTNode):
        if isinstance(node, FunctionDeclNode):
            self.ast_table[node.name] = copy.deepcopy(node)
            self._recompile_to_bytecode(node.name)

        elif isinstance(node, OptimizeAstNode):
            if node.target_func in self.ast_table:
                fn = self.ast_table[node.target_func]
                fn.body = [self._fold_constants(s) for s in fn.body]
                fn.version += 1
                self._recompile_to_bytecode(fn.name)

        elif isinstance(node, MutateFunctionNode):
            if node.target_func in self.ast_table:
                fn = self.ast_table[node.target_func]
                replaced = False
                new_body = []
                for stmt in fn.body:
                    if isinstance(stmt, AssignNode) and stmt.name == node.target_var:
                        new_body.append(AssignNode(node.target_var, copy.deepcopy(node.new_expr)))
                        replaced = True
                    else:
                        new_body.append(stmt)
                if not replaced:
                    new_body.insert(0, AssignNode(node.target_var, copy.deepcopy(node.new_expr)))
                fn.body = new_body
                fn.version += 1
                self._recompile_to_bytecode(fn.name)

        elif isinstance(node, SynthesizeFunctionNode):
            if node.func_a in self.ast_table and node.func_b in self.ast_table:
                fa = self.ast_table[node.func_a]
                fb = self.ast_table[node.func_b]
                ret_a = next(s.expr for s in fa.body if isinstance(s, ReturnNode))
                ret_b = next(s.expr for s in fb.body if isinstance(s, ReturnNode))

                combined = [copy.deepcopy(s) for s in fa.body if not isinstance(s, ReturnNode)]
                combined += [copy.deepcopy(s) for s in fb.body if not isinstance(s, ReturnNode)]
                combined.append(ReturnNode(BinaryOpNode(node.op, copy.deepcopy(ret_a), copy.deepcopy(ret_b))))

                old_v = self.ast_table[node.new_name].version if node.new_name in self.ast_table else 0
                self.ast_table[node.new_name] = FunctionDeclNode(
                    name=node.new_name,
                    params=list(dict.fromkeys(fa.params + fb.params)),
                    body=combined,
                    version=old_v + 1
                )
                self._recompile_to_bytecode(node.new_name)

    def run_vm_function(self, func_name: str, positional_args: List[Any]) -> Any:
        chunk = self.bytecode_table[func_name]
        return self.vm.execute_chunk(chunk, positional_args)


# =====================================================================
# 5. MOTOR HOLOGRÁFICO 3D EJECUTADO POR LA STACK VM EN TIEMPO REAL
# =====================================================================

INITIAL_KERNEL_CODE = """
# 1. Función física principal compilada a Bytecode
#    Tiene operaciones constantes ((0.16 * 1) + 0) para probar OPTIMIZAR_AST
FUNCION calcular_fisica_posicion(pos_mano, pos_holo, vel_holo, vel_mano) {
    rigidez_k = (0.16 * 1) + 0
    amortiguacion_d = 0.62
    impulso_mano = 0.14
    gravedad = [0.0, 0.0, 0.0]

    error_espacial = pos_mano - pos_holo
    fuerza_resorte = error_espacial * rigidez_k
    fuerza_inercia = (vel_holo * amortiguacion_d) + (vel_mano * impulso_mano)
    aceleracion_total = (fuerza_resorte + fuerza_inercia) + gravedad
    RETORNAR pos_holo + aceleracion_total
}

# 2. Función de levitación lista para fusionar en Bytecode con la tecla [4]
FUNCION campo_antigravedad(pos_mano, pos_holo, vel_holo, vel_mano) {
    empuje_vertical = [0.0, -0.055, 0.0]
    RETORNAR empuje_vertical
}

# 3. Función de escala bimanual compilada a Bytecode
FUNCION calcular_fisica_escala(escala_objetivo, escala_actual) {
    elasticidad = 0.26
    delta = (escala_objetivo - escala_actual) * elasticidad
    RETORNAR escala_actual + delta
}
"""


def rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


class NexusBytecodeHologramEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        self.runtime = HomoiconicBytecodeRuntime()
        self.runtime.execute_source(INITIAL_KERNEL_CODE)

        self.holo_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.holo_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.holo_rot = np.array([0.35, 0.55, 0.1], dtype=np.float32)
        self.holo_scale = 1.0
        self.target_scale = 1.0

        self.prev_hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.bimanual_base_dist: Optional[float] = None
        self.bimanual_base_scale: float = 1.0

        self.cube_verts = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ], dtype=np.float32) * 0.48
        self.cube_faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6), (0,3,7,4), (1,2,6,5)]
        self.cube_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

        self.cmd_queue: queue.Queue = queue.Queue()
        self.onscreen_input = ""
        self.running = True

    def _console_worker(self):
        while self.running:
            try:
                line = input("NexusVM >> ")
                if line.strip():
                    self.cmd_queue.put(line.strip())
            except EOFError:
                break

    def step_vm_physics(self, target_hand_pos: np.ndarray, hand_vel: np.ndarray):
        """Ejecuta el Bytecode en la Máquina Virtual de Pila para actualizar la física 3D."""
        new_pos_vec3 = self.runtime.run_vm_function(
            "calcular_fisica_posicion",
            [
                Vec3.from_np(target_hand_pos),
                Vec3.from_np(self.holo_pos),
                Vec3.from_np(self.holo_vel),
                Vec3.from_np(hand_vel),
            ]
        )
        new_pos = np.clip(new_pos_vec3.to_np(), -2.2, 2.2)
        self.holo_vel = new_pos - self.holo_pos
        self.holo_pos = new_pos

        new_scale = float(self.runtime.run_vm_function(
            "calcular_fisica_escala",
            [float(self.target_scale), float(self.holo_scale)]
        ))
        self.holo_scale = float(np.clip(new_scale, 0.30, 3.2))

        speed_mag = float(np.linalg.norm(self.holo_vel))
        self.holo_rot += np.array([
            0.01 + self.holo_vel[1] * 0.45,
            0.018 + self.holo_vel[0] * 0.45 + speed_mag * 0.2,
            0.005
        ], dtype=np.float32)

    def project_3d(self, pts_3d, w, h, focal=640.0, cam_dist=3.6):
        out = []
        for pt in pts_3d:
            z = max(cam_dist + pt[2], 0.35)
            out.append((int((pt[0] * focal) / z + w * 0.5), int((pt[1] * focal) / z + h * 0.5)))
        return out

    def draw_hologram_3d(self, frame):
        h, w, _ = frame.shape
        R = rotation_matrix_3d(*self.holo_rot)
        verts = (self.cube_verts * self.holo_scale) @ R.T + self.holo_pos
        pts_2d = self.project_3d(verts, w, h)

        chunk = self.runtime.bytecode_table["calcular_fisica_posicion"]
        palette = [(255, 225, 0), (100, 255, 120), (255, 90, 255), (0, 200, 255), (120, 120, 255)]
        color = palette[(chunk.version - 1) % len(palette)]

        overlay = frame.copy()
        for face in self.cube_faces:
            poly = np.array([pts_2d[i] for i in face], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (color[0]//4, color[1]//4, color[2]//4))
        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

        for e1, e2 in self.cube_edges:
            cv2.line(frame, pts_2d[e1], pts_2d[e2], color, 2, cv2.LINE_AA)

        for pt in pts_2d:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_vm_disassembly_hud(self, frame):
        h, w, _ = frame.shape
        chunk = self.runtime.bytecode_table["calcular_fisica_posicion"]

        # Panel Izquierdo: Telemetría de la Máquina Virtual de Pila (microsegundos)
        cv2.rectangle(frame, (15, 15), (610, 190), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (610, 190), (0, 240, 255), 1)
        cv2.putText(frame, f"STACK VM TELEMETRY | Tiempo: {self.runtime.vm.last_exec_time_us:.1f} microsegundos",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (0, 255, 160), 2)
        cv2.putText(frame, f"OpCodes por Frame: {len(chunk.instructions)} | Constantes Pool: {len(chunk.constants)} | Locals: {len(chunk.var_names)}",
                    (26, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (200, 230, 255), 1)

        for i, log_line in enumerate(self.runtime.logs[-4:]):
            cv2.putText(frame, f"> {log_line}", (26, 88 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 225, 235), 1)

        cv2.putText(frame, "Teclas: [1] Resorte Suave [2] Rigido [3] Plegar Constantes (OPTIMIZAR_AST) [4] Fusionar",
                    (26, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 215, 80), 1)

        # Panel Derecho: Ensamblador Bytecode en Vivo (Disassembler)
        asm_lines = chunk.disassemble()[:24]
        box_h = 30 + len(asm_lines) * 18
        cv2.rectangle(frame, (w - 490, 15), (w - 15, 15 + box_h), (12, 12, 18), -1)
        cv2.rectangle(frame, (w - 490, 15), (w - 15, 15 + box_h), (255, 170, 0), 1)

        for idx, asm_line in enumerate(asm_lines):
            if idx == 0:
                col = (255, 210, 0)
            elif "STORE_LOCAL" in asm_line or "RETURN_VAL" in asm_line:
                col = (110, 255, 140)
            elif "BINARY_" in asm_line:
                col = (140, 210, 255)
            else:
                col = (210, 210, 220)
            cv2.putText(frame, asm_line, (w - 476, 38 + idx * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        # Barra Inferior: Consola JIT en Vivo
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (12, 14, 20), -1)
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (0, 255, 140), 1)
        cursor = "_" if int(time.time() * 2.5) % 2 == 0 else " "
        cv2.putText(frame, f"JIT-VM >> {self.onscreen_input}{cursor}",
                    (28, h - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 140), 2)
        cv2.putText(frame, "Escribe: OPTIMIZAR_AST(calcular_fisica_posicion)  o  MUTAR_FUNCION(calcular_fisica_posicion, rigidez_k -> 0.45)",
                    (28, h - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (175, 195, 215), 1)

    def run(self):
        threading.Thread(target=self._console_worker, daemon=True).start()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        target_hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        while cap.isOpened() and self.running:
            success, raw = cap.read()
            if not success: break

            frame = cv2.flip(raw, 1)
            h, w, _ = frame.shape
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            while not self.cmd_queue.empty():
                self.runtime.execute_source(self.cmd_queue.get_nowait())

            pinching_3d = []
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                    mid = (pts[4] + pts[8]) * 0.5
                    if float(np.linalg.norm(pts[8] - pts[4])) < 0.062:
                        pos_3d = np.array([(mid[0] - 0.5) * 3.2, (mid[1] - 0.5) * 2.0, mid[2] * 1.4], dtype=np.float32)
                        pinching_3d.append(pos_3d)
                        cv2.circle(frame, (int(mid[0] * w), int(mid[1] * h)), 13, (0, 255, 140), 2, cv2.LINE_AA)

            if len(pinching_3d) == 2:
                target_hand_pos = (pinching_3d[0] + pinching_3d[1]) * 0.5
                dist = float(np.linalg.norm(pinching_3d[1] - pinching_3d[0]))
                if self.bimanual_base_dist is None:
                    self.bimanual_base_dist = max(dist, 0.05)
                    self.bimanual_base_scale = self.holo_scale
                self.target_scale = self.bimanual_base_scale * (dist / self.bimanual_base_dist)
            elif len(pinching_3d) == 1:
                target_hand_pos = pinching_3d[0]
                self.bimanual_base_dist = None
            else:
                self.bimanual_base_dist = None

            hand_vel = target_hand_pos - self.prev_hand_pos
            self.prev_hand_pos = target_hand_pos.copy()

            # EJECUTAR EL BYTECODE EN LA MÁQUINA VIRTUAL DE PILA
            self.step_vm_physics(target_hand_pos, hand_vel)

            self.draw_hologram_3d(frame)
            self.draw_vm_disassembly_hud(frame)
            cv2.imshow("NexusXR - Stack VM & Bytecode JIT Engine", frame)

            key = cv2.waitKey(1)
            if key == -1: continue
            elif key == 27:
                self.running = False; break
            elif key in (13, 10):
                if self.onscreen_input.strip():
                    self.cmd_queue.put(self.onscreen_input.strip())
                    self.onscreen_input = ""
            elif key in (8, 127):
                self.onscreen_input = self.onscreen_input[:-1]
            elif not self.onscreen_input and (key & 0xFF) == ord('1'):
                self.cmd_queue.put("MUTAR_FUNCION(calcular_fisica_posicion, rigidez_k -> 0.05)")
                self.cmd_queue.put("MUTAR_FUNCION(calcular_fisica_posicion, amortiguacion_d -> 0.88)")
            elif not self.onscreen_input and (key & 0xFF) == ord('2'):
                self.cmd_queue.put("MUTAR_FUNCION(calcular_fisica_posicion, rigidez_k -> 0.48)")
                self.cmd_queue.put("MUTAR_FUNCION(calcular_fisica_posicion, amortiguacion_d -> 0.20)")
            elif not self.onscreen_input and (key & 0xFF) == ord('3'):
                # ¡Observa cómo el número de instrucciones Bytecode se reduce al plegar constantes!
                self.cmd_queue.put("OPTIMIZAR_AST(calcular_fisica_posicion)")
            elif not self.onscreen_input and (key & 0xFF) == ord('4'):
                self.cmd_queue.put("SINTETIZAR_FUNCION(calcular_fisica_posicion = calcular_fisica_posicion + campo_antigravedad)")
            elif 32 <= (key & 0xFF) <= 126:
                self.onscreen_input += chr(key & 0xFF)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = NexusBytecodeHologramEngine()
    engine.run()
