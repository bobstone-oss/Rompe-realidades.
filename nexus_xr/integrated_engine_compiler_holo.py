import cv2
import mediapipe as mp
import numpy as np
import math
import time
import copy
import queue
import threading
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


# =====================================================================
# 1. TIPO DE DATO NATIVO DEL COMPILADOR: VECTOR ESPACIAL 3D (Vec3)
# =====================================================================

class Vec3:
    """Primitiva matemática de NexusLang para operar con coordenadas y fuerzas 3D."""
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
        return Vec3(self.x + float(o), self.y + float(o), self.z + float(o))

    def __radd__(self, o):
        return self.__add__(o)

    def __sub__(self, o):
        if isinstance(o, Vec3):
            return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)
        return Vec3(self.x - float(o), self.y - float(o), self.z - float(o))

    def __mul__(self, o):
        if isinstance(o, Vec3):
            return Vec3(self.x * o.x, self.y * o.y, self.z * o.z)
        val = float(o)
        return Vec3(self.x * val, self.y * val, self.z * val)

    def __rmul__(self, o):
        return self.__mul__(o)

    def __truediv__(self, o):
        val = float(o) if not isinstance(o, Vec3) else 1.0
        val = val if abs(val) > 1e-6 else 1e-6
        return Vec3(self.x / val, self.y / val, self.z / val)

    def __repr__(self):
        return f"[{self.x:.2f}, {self.y:.2f}, {self.z:.2f}]"


# =====================================================================
# 2. LEXER, PARSER Y AST HOMOICÓNICO DE NEXUS-LANG
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
        if ch == "\n":
            self.line += 1
        return ch

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while self.pos < len(self.source):
            ch = self._peek()
            if ch in " \t\r\n":
                self._advance()
                continue
            if ch == "#":
                while self._peek() not in ("\n", "\0"):
                    self._advance()
                continue

            if ch.isdigit() or (ch == "-" and self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit()):
                num = self._advance()
                while self._peek().isdigit() or self._peek() == ".":
                    num += self._advance()
                tokens.append(Token("NUMBER", float(num), self.line))
                continue

            if ch.isalpha() or ch == "_":
                ident = ""
                while self._peek().isalnum() or self._peek() == "_":
                    ident += self._advance()
                tokens.append(Token(self.KEYWORDS.get(ident.upper(), "IDENT"), ident, self.line))
                continue

            if self.source[self.pos:self.pos + 2] == "->":
                self._advance(); self._advance()
                tokens.append(Token("ARROW", "->", self.line))
                continue

            single = {
                "+": "OP", "-": "OP", "*": "OP", "/": "OP",
                "=": "ASSIGN", "(": "LPAREN", ")": "RPAREN",
                "{": "LBRACE", "}": "RBRACE", "[": "LBRACKET", "]": "RBRACKET",
                ",": "COMMA"
            }
            if ch in single:
                tokens.append(Token(single[ch], self._advance(), self.line))
                continue

            raise SyntaxError(f"Carácter inválido '{ch}' en línea {self.line}")

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
    x: ASTNode
    y: ASTNode
    z: ASTNode
    def to_source(self, indent: int = 0) -> str:
        return f"[{self.x.to_source()}, {self.y.to_source()}, {self.z.to_source()}]"

@dataclass
class BinaryOpNode(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode
    def to_source(self, indent: int = 0) -> str:
        return f"({self.left.to_source()} {self.op} {self.right.to_source()})"

@dataclass
class AssignNode(ASTNode):
    name: str
    expr: ASTNode
    def to_source(self, indent: int = 0) -> str:
        return f"{'  '*indent}{self.name} = {self.expr.to_source()}"

@dataclass
class ReturnNode(ASTNode):
    expr: ASTNode
    def to_source(self, indent: int = 0) -> str:
        return f"{'  '*indent}RETORNAR {self.expr.to_source()}"

@dataclass
class FunctionDeclNode(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]
    version: int = 1

    def to_source(self, indent: int = 0) -> str:
        params_s = ", ".join(self.params)
        lines = [f"FUNCION {self.name}({params_s}) {{ # [v{self.version}]"]
        for s in self.body:
            lines.append(s.to_source(indent + 1))
        lines.append("}")
        return "\n".join(lines)

@dataclass
class OptimizeAstNode(ASTNode):
    target_func: str
    def to_source(self, indent: int = 0) -> str:
        return f"OPTIMIZAR_AST({self.target_func})"

@dataclass
class MutateFunctionNode(ASTNode):
    target_func: str
    target_var: str
    new_expr: ASTNode
    def to_source(self, indent: int = 0) -> str:
        return f"MUTAR_FUNCION({self.target_func}, {self.target_var} -> {self.new_expr.to_source()})"

@dataclass
class SynthesizeFunctionNode(ASTNode):
    new_name: str
    func_a: str
    func_b: str
    op: str
    def to_source(self, indent: int = 0) -> str:
        return f"SINTETIZAR_FUNCION({self.new_name} = {self.func_a} {self.op} {self.func_b})"


class NexusParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def _curr(self) -> Token:
        return self.tokens[self.pos]

    def _eat(self, t_type: str) -> Token:
        tok = self._curr()
        if tok.type != t_type:
            raise SyntaxError(f"Se esperaba {t_type}, llegó {tok.type} ('{tok.value}')")
        self.pos += 1
        return tok

    def parse(self) -> List[ASTNode]:
        stmts = []
        while self._curr().type != "EOF":
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self) -> ASTNode:
        t = self._curr().type
        if t == "FUNCION":
            self._eat("FUNCION")
            name = self._eat("IDENT").value
            self._eat("LPAREN")
            params = []
            if self._curr().type != "RPAREN":
                params.append(self._eat("IDENT").value)
                while self._curr().type == "COMMA":
                    self._eat("COMMA")
                    params.append(self._eat("IDENT").value)
            self._eat("RPAREN")
            self._eat("LBRACE")
            body = []
            while self._curr().type != "RBRACE":
                body.append(self.parse_stmt())
            self._eat("RBRACE")
            return FunctionDeclNode(name, params, body)

        if t == "RETORNAR":
            self._eat("RETORNAR")
            return ReturnNode(self.parse_expr())

        if t == "OPTIMIZAR_AST":
            self._eat("OPTIMIZAR_AST"); self._eat("LPAREN")
            fn = self._eat("IDENT").value
            self._eat("RPAREN")
            return OptimizeAstNode(fn)

        if t == "MUTAR_FUNCION":
            self._eat("MUTAR_FUNCION"); self._eat("LPAREN")
            fn = self._eat("IDENT").value
            self._eat("COMMA")
            var = self._eat("IDENT").value
            self._eat("ARROW")
            expr = self.parse_expr()
            self._eat("RPAREN")
            return MutateFunctionNode(fn, var, expr)

        if t == "SINTETIZAR_FUNCION":
            self._eat("SINTETIZAR_FUNCION"); self._eat("LPAREN")
            new_n = self._eat("IDENT").value
            self._eat("ASSIGN")
            fa = self._eat("IDENT").value
            op = self._eat("OP").value
            fb = self._eat("IDENT").value
            self._eat("RPAREN")
            return SynthesizeFunctionNode(new_n, fa, fb, op)

        if t == "IDENT" and self.tokens[self.pos + 1].type == "ASSIGN":
            name = self._eat("IDENT").value
            self._eat("ASSIGN")
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
        if tok.type == "NUMBER":
            self._eat("NUMBER")
            return NumberNode(tok.value)
        if tok.type == "IDENT":
            self._eat("IDENT")
            return IdentNode(tok.value)
        if tok.type == "LBRACKET":
            self._eat("LBRACKET")
            x = self.parse_expr(); self._eat("COMMA")
            y = self.parse_expr(); self._eat("COMMA")
            z = self.parse_expr(); self._eat("RBRACKET")
            return Vector3Node(x, y, z)
        if tok.type == "LPAREN":
            self._eat("LPAREN")
            e = self.parse_expr()
            self._eat("RPAREN")
            return e
        raise SyntaxError(f"Token inesperado: {tok}")


# =====================================================================
# 3. RUNTIME HOMOICÓNICO CON EVALUACIÓN FÍSICA EN TIEMPO REAL
# =====================================================================

class HomoiconicPhysicsRuntime:
    def __init__(self):
        self.functions: Dict[str, FunctionDeclNode] = {}
        self.logs: List[str] = [
            "COMPILADOR NEXUS-LANG INTEGRADO AL MOTOR 3D.",
            "Las funciones AST gobiernan la fisica del cubo a 60 FPS."
        ]

    def log(self, text: str):
        print(f"[AST-PHYSICS] {text}")
        self.logs.append(text)
        if len(self.logs) > 5:
            self.logs.pop(0)

    def compile_and_run(self, code_str: str):
        try:
            tokens = NexusLexer(code_str).tokenize()
            ast_nodes = NexusParser(tokens).parse()
            for node in ast_nodes:
                self.eval_top_level(node)
        except Exception as e:
            self.log(f"Error de compilación: {e}")

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

    def eval_top_level(self, node: ASTNode):
        if isinstance(node, FunctionDeclNode):
            self.functions[node.name] = copy.deepcopy(node)
            self.log(f"Función AST '{node.name}' compilada (v{node.version}).")

        elif isinstance(node, OptimizeAstNode):
            if node.target_func in self.functions:
                fn = self.functions[node.target_func]
                fn.body = [self._fold_constants(s) for s in fn.body]
                fn.version += 1
                self.log(f"AST de '{fn.name}' optimizado -> v{fn.version}")

        elif isinstance(node, MutateFunctionNode):
            if node.target_func in self.functions:
                fn = self.functions[node.target_func]
                replaced = False
                new_body = []
                for stmt in fn.body:
                    if isinstance(stmt, AssignNode) and stmt.name == node.target_var:
                        new_body.append(AssignNode(node.target_var, copy.deepcopy(node.new_expr)))
                        replaced = True
                    else:
                        new_body.append(stmt)
                if not replaced:
                    # Si la variable no existía, inyectarla al principio del AST
                    new_body.insert(0, AssignNode(node.target_var, copy.deepcopy(node.new_expr)))
                fn.body = new_body
                fn.version += 1
                self.log(f"AST Mutado: '{fn.name}.{node.target_var}' -> {node.new_expr.to_source()} (v{fn.version})")

        elif isinstance(node, SynthesizeFunctionNode):
            if node.func_a in self.functions and node.func_b in self.functions:
                fa = self.functions[node.func_a]
                fb = self.functions[node.func_b]
                ret_a = next(s.expr for s in fa.body if isinstance(s, ReturnNode))
                ret_b = next(s.expr for s in fb.body if isinstance(s, ReturnNode))

                combined_body = [copy.deepcopy(s) for s in fa.body if not isinstance(s, ReturnNode)]
                combined_body += [copy.deepcopy(s) for s in fb.body if not isinstance(s, ReturnNode)]
                combined_body.append(ReturnNode(BinaryOpNode(node.op, copy.deepcopy(ret_a), copy.deepcopy(ret_b))))

                old_ver = self.functions[node.new_name].version if node.new_name in self.functions else 0
                self.functions[node.new_name] = FunctionDeclNode(
                    name=node.new_name,
                    params=list(dict.fromkeys(fa.params + fb.params)),
                    body=combined_body,
                    version=old_ver + 1
                )
                self.log(f"Síntesis AST: '{node.new_name}' = '{fa.name}' {node.op} '{fb.name}' (v{old_ver+1})")

    def call_ast_function(self, func_name: str, args_dict: Dict[str, Any]) -> Any:
        fn = self.functions[func_name]
        local_env = dict(args_dict)
        for stmt in fn.body:
            if isinstance(stmt, AssignNode):
                local_env[stmt.name] = self._eval_expr(stmt.expr, local_env)
            elif isinstance(stmt, ReturnNode):
                return self._eval_expr(stmt.expr, local_env)
        return None

    def _eval_expr(self, expr: ASTNode, env: Dict[str, Any]) -> Any:
        if isinstance(expr, NumberNode):
            return expr.value
        if isinstance(expr, IdentNode):
            return env.get(expr.name, 0.0)
        if isinstance(expr, Vector3Node):
            return Vec3(self._eval_expr(expr.x, env), self._eval_expr(expr.y, env), self._eval_expr(expr.z, env))
        if isinstance(expr, BinaryOpNode):
            l = self._eval_expr(expr.left, env)
            r = self._eval_expr(expr.right, env)
            if expr.op == "+": return l + r
            if expr.op == "-": return l - r
            if expr.op == "*": return l * r
            if expr.op == "/": return l / r
        return 0.0


# =====================================================================
# 4. MOTOR HOLOGRÁFICO 3D GOBERNADO POR FUNCIONES AST EN VIVO
# =====================================================================

DEFAULT_PHYSICS_SCRIPT = """
# 1. Ley física de Resorte-Amortiguador para posición 3D del holograma
FUNCION calcular_fisica_posicion(pos_mano, pos_holo, vel_holo, vel_mano) {
    rigidez_k = (0.14 * 1) + 0
    amortiguacion_d = 0.58
    impulso_mano = 0.12
    gravedad = [0.0, 0.0, 0.0]

    error_espacial = pos_mano - pos_holo
    fuerza_resorte = error_espacial * rigidez_k
    fuerza_inercia = (vel_holo * amortiguacion_d) + (vel_mano * impulso_mano)
    aceleracion_total = (fuerza_resorte + fuerza_inercia) + gravedad
    RETORNAR pos_holo + aceleracion_total
}

# 2. Función secundaria de levitación/vórtice lista para sintetizarse en vivo
FUNCION campo_antigravedad(pos_mano, pos_holo, vel_holo, vel_mano) {
    empuje_solar = [0.0, -0.06, 0.0]
    RETORNAR empuje_solar
}

# 3. Ley física para la elasticidad de la escala bimanual
FUNCION calcular_fisica_escala(escala_objetivo, escala_actual) {
    elasticidad = 0.25
    delta_escala = (escala_objetivo - escala_actual) * elasticidad
    RETORNAR escala_actual + delta_escala
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


class NexusASTHologramEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_draw = mp.solutions.drawing_utils

        # Inicializar el compilador homoicónico y compilar las leyes físicas base
        self.runtime = HomoiconicPhysicsRuntime()
        self.runtime.compile_and_run(DEFAULT_PHYSICS_SCRIPT)

        # Estado cinemático del Cubo Holográfico 3D
        self.holo_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.holo_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.holo_rot = np.array([0.35, 0.55, 0.1], dtype=np.float32)
        self.holo_scale = 1.0
        self.target_scale = 1.0

        self.prev_hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.bimanual_base_dist: Optional[float] = None
        self.bimanual_base_scale: float = 1.0

        # Geometría del Cubo y Octaedro interior
        self.cube_verts = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
        ], dtype=np.float32) * 0.48
        self.cube_faces = [(0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6), (0,3,7,4), (1,2,6,5)]
        self.cube_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

        self.cmd_queue: queue.Queue = queue.Queue()
        self.onscreen_input = ""
        self.running = True

    def _console_repl_thread(self):
        while self.running:
            try:
                line = input("NexusAST >> ")
                if line.strip():
                    self.cmd_queue.put(line.strip())
            except EOFError:
                break

    def step_ast_physics(self, target_hand_pos: np.ndarray, hand_vel: np.ndarray):
        """
        Evalúa los árboles AST de NexusLang en tiempo real para calcular
        la siguiente posición, velocidad y escala física del holograma.
        """
        new_pos_vec3 = self.runtime.call_ast_function(
            "calcular_fisica_posicion",
            {
                "pos_mano": Vec3.from_np(target_hand_pos),
                "pos_holo": Vec3.from_np(self.holo_pos),
                "vel_holo": Vec3.from_np(self.holo_vel),
                "vel_mano": Vec3.from_np(hand_vel),
            }
        )
        new_pos = np.clip(new_pos_vec3.to_np(), -2.2, 2.2)
        self.holo_vel = new_pos - self.holo_pos
        self.holo_pos = new_pos

        # Evaluar el AST de escala
        new_scale = float(self.runtime.call_ast_function(
            "calcular_fisica_escala",
            {
                "escala_objetivo": float(self.target_scale),
                "escala_actual": float(self.holo_scale),
            }
        ))
        self.holo_scale = float(np.clip(new_scale, 0.30, 3.2))

        # La velocidad lineal generada por el AST alimenta el giro inercial 3D
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

        fn_ver = self.runtime.functions["calcular_fisica_posicion"].version
        # El color del holograma evoluciona cada vez que el AST se muta en caliente
        palette = [(255, 225, 0), (100, 255, 120), (255, 90, 255), (0, 200, 255), (120, 120, 255)]
        color = palette[(fn_ver - 1) % len(palette)]

        overlay = frame.copy()
        for face in self.cube_faces:
            poly = np.array([pts_2d[i] for i in face], dtype=np.int32)
            cv2.fillPoly(overlay, [poly], (color[0]//4, color[1]//4, color[2]//4))
        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

        for e1, e2 in self.cube_edges:
            cv2.line(frame, pts_2d[e1], pts_2d[e2], color, 2, cv2.LINE_AA)

        for pt in pts_2d:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_ast_inspector_hud(self, frame):
        h, w, _ = frame.shape
        fn_pos = self.runtime.functions["calcular_fisica_posicion"]

        # Panel Izquierdo: Logs de Compilación y Atajos de Mutación
        cv2.rectangle(frame, (15, 15), (600, 185), (16, 18, 24), -1)
        cv2.rectangle(frame, (15, 15), (600, 185), (0, 230, 255), 1)
        cv2.putText(frame, f"MOTOR FISICO AST EN VIVO | Vel: {np.linalg.norm(self.holo_vel):.3f} | Escala: {self.holo_scale:.2f}x",
                    (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 2)

        for i, log_line in enumerate(self.runtime.logs[-4:]):
            cv2.putText(frame, f"> {log_line}", (26, 64 + i * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.41, (215, 230, 245), 1)

        cv2.putText(frame, "Atajos: [1] Resorte Suave [2] Resorte Hiper-Rigido [3] Gravedad Lunar [4] Sintetizar Antigravedad",
                    (26, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 210, 90), 1)

        # Panel Derecho: Código Fuente reconstruido en vivo desde el Árbol AST
        source_lines = fn_pos.to_source().splitlines()
        box_h = 45 + len(source_lines) * 21
        cv2.rectangle(frame, (w - 645, 15), (w - 15, 15 + box_h), (14, 14, 22), -1)
        cv2.rectangle(frame, (w - 645, 15), (w - 15, 15 + box_h), (255, 180, 0), 1)
        cv2.putText(frame, f"ARBOL AST ACTIVO: {fn_pos.name} (Version v{fn_pos.version})",
                    (w - 630, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 210, 0), 2)

        for idx, code_line in enumerate(source_lines):
            highlight = any(k in code_line for k in ("rigidez_k", "amortiguacion_d", "gravedad", "RETORNAR"))
            col = (120, 255, 140) if highlight else (225, 225, 235)
            cv2.putText(frame, code_line, (w - 630, 62 + idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

        # Barra Inferior: Consola de Metaprogramación en Vivo
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (12, 14, 20), -1)
        cv2.rectangle(frame, (15, h - 66), (w - 15, h - 15), (0, 255, 140), 1)
        cursor = "_" if int(time.time() * 2.5) % 2 == 0 else " "
        cv2.putText(frame, f"NEXUS-AST >> {self.onscreen_input}{cursor}",
                    (28, h - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 140), 2)
        cv2.putText(frame, "Ejemplo: MUTAR_FUNCION(calcular_fisica_posicion, rigidez_k -> 0.45)  |  OPTIMIZAR_AST(calcular_fisica_posicion)",
                    (28, h - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (175, 195, 215), 1)

    def run(self):
        threading.Thread(target=self._console_repl_thread, daemon=True).start()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("==========================================================================")
        print(" NEXUS-XR: COMPILADOR HOMOICÓNICO CONTROLANDO FÍSICAS 3D EN VIVO")
        print(" - Junta Pulgar e Índice (1 mano) para atraer el cubo según el AST actual.")
        print(" - Junta Pulgar e Índice (2 manos) para escalar con la función AST elástica.")
        print(" - Presiona las teclas rápidas [1], [2], [3], [4] o escribe cualquier")
        print("   comando MUTAR_FUNCION / SINTETIZAR_FUNCION en la barra inferior.")
        print("==========================================================================")

        target_hand_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        while cap.isOpened() and self.running:
            success, raw = cap.read()
            if not success:
                break

            frame = cv2.flip(raw, 1)
            h, w, _ = frame.shape
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            while not self.cmd_queue.empty():
                self.runtime.compile_and_run(self.cmd_queue.get_nowait())

            pinching_positions_3d = []
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                    mid = (pts[4] + pts[8]) * 0.5
                    pinch_dist = float(np.linalg.norm(pts[8] - pts[4]))

                    pos_3d = np.array([(mid[0] - 0.5) * 3.2, (mid[1] - 0.5) * 2.0, mid[2] * 1.4], dtype=np.float32)
                    if pinch_dist < 0.062:
                        pinching_positions_3d.append(pos_3d)
                        px, py = int(mid[0] * w), int(mid[1] * h)
                        cv2.circle(frame, (px, py), 13, (0, 255, 140), 2, cv2.LINE_AA)

            # Actualizar vectores de entrada para las funciones AST
            if len(pinching_positions_3d) == 2:
                target_hand_pos = (pinching_positions_3d[0] + pinching_positions_3d[1]) * 0.5
                dist = float(np.linalg.norm(pinching_positions_3d[1] - pinching_positions_3d[0]))
                if self.bimanual_base_dist is None:
                    self.bimanual_base_dist = max(dist, 0.05)
                    self.bimanual_base_scale = self.holo_scale
                self.target_scale = self.bimanual_base_scale * (dist / self.bimanual_base_dist)
            elif len(pinching_positions_3d) == 1:
                target_hand_pos = pinching_positions_3d[0]
                self.bimanual_base_dist = None
            else:
                self.bimanual_base_dist = None

            hand_vel = target_hand_pos - self.prev_hand_pos
            self.prev_hand_pos = target_hand_pos.copy()

            # EJECUTAR LAS FUNCIONES DEL ÁRBOL SINTÁCTICO (AST) PARA CALCULAR LA FÍSICA DEL CUBO
            self.step_ast_physics(target_hand_pos, hand_vel)

            # Renderizar Cubo 3D e Inspector de AST
            self.draw_hologram_3d(frame)
            self.draw_ast_inspector_hud(frame)

            cv2.imshow("NexusXR - Compilador AST Controlando Fisicas 3D", frame)

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
            # Atajos instantáneos de metaprogramación cuando la barra de texto está vacía
            elif not self.onscreen_input and (key & 0xFF) == ord('1'):
                self.cmd_queue.put("MUTAR_FUNCION(calcular_fisica_posicion, rigidez_k -> 0.05)")
                self.cmd_queue.put("MUTAR_FUNCION(calcular_fisica_posicion, amortiguacion_d -> 0.88)")
            elif not self.onscreen_input and (key & 0xFF) == ord('2'):
                self.cmd_queue.put("MUTAR_FUNCION(calcular_fisica_posicion, rigidez_k -> 0.48)")
                self.cmd_queue.put("MUTAR_FUNCION(calcular_fisica_posicion, amortiguacion_d -> 0.20)")
            elif not self.onscreen_input and (key & 0xFF) == ord('3'):
                self.cmd_queue.put("MUTAR_FUNCION(calcular_fisica_posicion, gravedad -> [0.0, 0.045, 0.0])")
            elif not self.onscreen_input and (key & 0xFF) == ord('4'):
                self.cmd_queue.put("SINTETIZAR_FUNCION(calcular_fisica_posicion = calcular_fisica_posicion + campo_antigravedad)")
            elif 32 <= (key & 0xFF) <= 126:
                self.onscreen_input += chr(key & 0xFF)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = NexusASTHologramEngine()
    engine.run()
