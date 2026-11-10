import math
import copy
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

# =====================================================================
# 1. TIPO DE DATO NATIVO ESPACIAL: VECTOR 3D
# =====================================================================

class Vec3:
    """Tipo de dato primitivo del lenguaje para coordenadas AR/VR y manos."""
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        return Vec3(self.x + other, self.y + other, self.z + other)

    def __sub__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        return Vec3(self.x - other, self.y - other, self.z - other)

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * float(other), self.y * float(other), self.z * float(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __repr__(self):
        return f"[{self.x:.3f}, {self.y:.3f}, {self.z:.3f}]"


# =====================================================================
# 2. ANALIZADOR LÉXICO (LEXER)
# =====================================================================

@dataclass
class Token:
    type: str
    value: Any
    line: int
    col: int

class NexusLexer:
    """Transforma código fuente de NexusLang en una corriente de Tokens."""
    KEYWORDS = {
        "FUNCION": "FUNCION",
        "RETORNAR": "RETORNAR",
        "SI": "SI",
        "SINO": "SINO",
        "IMPRIMIR": "IMPRIMIR",
        "INSPECCIONAR_AST": "INSPECCIONAR_AST",
        "OPTIMIZAR_AST": "OPTIMIZAR_AST",
        "MUTAR_FUNCION": "MUTAR_FUNCION",
        "SINTETIZAR_FUNCION": "SINTETIZAR_FUNCION",
    }

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1

    def _peek(self) -> str:
        return self.source[self.pos] if self.pos < len(self.source) else "\0"

    def _advance(self) -> str:
        ch = self._peek()
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while self.pos < len(self.source):
            ch = self._peek()

            # Ignorar espacios y tabulaciones
            if ch in " \t\r\n":
                self._advance()
                continue

            # Ignorar comentarios con '#'
            if ch == "#":
                while self._peek() not in ("\n", "\0"):
                    self._advance()
                continue

            start_line, start_col = self.line, self.col

            # Números enteros y flotantes
            if ch.isdigit() or (ch == "-" and self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit()):
                num_str = ""
                if ch == "-":
                    num_str += self._advance()
                while self._peek().isdigit() or self._peek() == ".":
                    num_str += self._advance()
                tokens.append(Token("NUMBER", float(num_str), start_line, start_col))
                continue

            # Identificadores y Palabras Clave
            if ch.isalpha() or ch == "_":
                ident = ""
                while self._peek().isalnum() or self._peek() == "_":
                    ident += self._advance()
                tok_type = self.KEYWORDS.get(ident.upper(), "IDENT")
                tokens.append(Token(tok_type, ident, start_line, start_col))
                continue

            # Cadenas de texto entre comillas
            if ch == '"':
                self._advance()
                text_val = ""
                while self._peek() not in ('"', "\0"):
                    text_val += self._advance()
                self._advance()  # Cerrar comillas
                tokens.append(Token("STRING", text_val, start_line, start_col))
                continue

            # Operadores dobles (==, !=, >=, <=, ->)
            two_char = self.source[self.pos:self.pos + 2]
            if two_char in ("==", "!=", ">=", "<=", "->"):
                self._advance()
                self._advance()
                tokens.append(Token("OP", two_char, start_line, start_col))
                continue

            # Símbolos y operadores simples
            single_map = {
                "+": "OP", "-": "OP", "*": "OP", "/": "OP", ">": "OP", "<": "OP",
                "=": "ASSIGN", "(": "LPAREN", ")": "RPAREN",
                "{": "LBRACE", "}": "RBRACE", "[": "LBRACKET", "]": "RBRACKET",
                ",": "COMMA", ";": "SEMICOLON"
            }
            if ch in single_map:
                tokens.append(Token(single_map[ch], self._advance(), start_line, start_col))
                continue

            raise SyntaxError(f"[Error Léxico] Carácter inesperado '{ch}' en línea {start_line}, col {start_col}")

        tokens.append(Token("EOF", None, self.line, self.col))
        return tokens


# =====================================================================
# 3. ÁRBOL SINTÁCTICO ABSTRACTO (AST HOMOICÓNICO Y AUTO-DESCRIPTIVO)
# =====================================================================

class ASTNode:
    """
    Clase base para todos los nodos del AST.
    Permite serializarse de vuelta a código fuente (`to_source`) y dibujar su árbol (`pretty_tree`).
    """
    def to_source(self, indent: int = 0) -> str:
        raise NotImplementedError()

    def pretty_tree(self, prefix: str = "") -> str:
        raise NotImplementedError()

@dataclass
class NumberNode(ASTNode):
    value: float
    def to_source(self, indent: int = 0) -> str:
        return str(int(self.value)) if self.value.is_integer() else str(self.value)
    def pretty_tree(self, prefix: str = "") -> str:
        return f"{prefix}Number({self.to_source()})"

@dataclass
class StringNode(ASTNode):
    value: str
    def to_source(self, indent: int = 0) -> str:
        return f'"{self.value}"'
    def pretty_tree(self, prefix: str = "") -> str:
        return f'{prefix}String("{self.value}")'

@dataclass
class IdentifierNode(ASTNode):
    name: str
    def to_source(self, indent: int = 0) -> str:
        return self.name
    def pretty_tree(self, prefix: str = "") -> str:
        return f"{prefix}Ident({self.name})"

@dataclass
class Vector3Node(ASTNode):
    x: ASTNode
    y: ASTNode
    z: ASTNode
    def to_source(self, indent: int = 0) -> str:
        return f"[{self.x.to_source()}, {self.y.to_source()}, {self.z.to_source()}]"
    def pretty_tree(self, prefix: str = "") -> str:
        return (f"{prefix}Vector3D\n"
                f"{self.x.pretty_tree(prefix + '  ├── X: ')}\n"
                f"{self.y.pretty_tree(prefix + '  ├── Y: ')}\n"
                f"{self.z.pretty_tree(prefix + '  └── Z: ')}")

@dataclass
class BinaryOpNode(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode
    def to_source(self, indent: int = 0) -> str:
        return f"({self.left.to_source()} {self.op} {self.right.to_source()})"
    def pretty_tree(self, prefix: str = "") -> str:
        return (f"{prefix}BinaryOp('{self.op}')\n"
                f"{self.left.pretty_tree(prefix + '  ├── ')}\n"
                f"{self.right.pretty_tree(prefix + '  └── ')}")

@dataclass
class AssignNode(ASTNode):
    name: str
    expr: ASTNode
    def to_source(self, indent: int = 0) -> str:
        pad = "    " * indent
        return f"{pad}{self.name} = {self.expr.to_source()}"
    def pretty_tree(self, prefix: str = "") -> str:
        return f"{prefix}Assign({self.name})\n{self.expr.pretty_tree(prefix + '  └── ')}"

@dataclass
class ReturnNode(ASTNode):
    expr: ASTNode
    def to_source(self, indent: int = 0) -> str:
        pad = "    " * indent
        return f"{pad}RETORNAR {self.expr.to_source()}"
    def pretty_tree(self, prefix: str = "") -> str:
        return f"{prefix}Return\n{self.expr.pretty_tree(prefix + '  └── ')}"

@dataclass
class PrintNode(ASTNode):
    expr: ASTNode
    def to_source(self, indent: int = 0) -> str:
        pad = "    " * indent
        return f"{pad}IMPRIMIR({self.expr.to_source()})"
    def pretty_tree(self, prefix: str = "") -> str:
        return f"{prefix}Print\n{self.expr.pretty_tree(prefix + '  └── ')}"

@dataclass
class CallNode(ASTNode):
    func_name: str
    args: List[ASTNode]
    def to_source(self, indent: int = 0) -> str:
        pad = "    " * indent
        args_str = ", ".join(a.to_source() for a in self.args)
        return f"{pad}{self.func_name}({args_str})"
    def pretty_tree(self, prefix: str = "") -> str:
        lines = [f"{prefix}Call({self.func_name})"]
        for a in self.args:
            lines.append(a.pretty_tree(prefix + "  └── "))
        return "\n".join(lines)

@dataclass
class FunctionDeclNode(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]
    version: int = 1

    def to_source(self, indent: int = 0) -> str:
        pad = "    " * indent
        params_str = ", ".join(self.params)
        body_lines = "\n".join(stmt.to_source(indent + 1) for stmt in self.body)
        return f"{pad}FUNCION {self.name}({params_str}) {{ # [Versión AST v{self.version}]\n{body_lines}\n{pad}}}"

    def pretty_tree(self, prefix: str = "") -> str:
        lines = [f"{prefix}FunctionDecl('{self.name}', params={self.params}, v{self.version})"]
        for i, stmt in enumerate(self.body):
            branch = "  └── " if i == len(self.body) - 1 else "  ├── "
            lines.append(stmt.pretty_tree(prefix + branch))
        return "\n".join(lines)

# --- Nodos Especiales de Metaprogramación ---

@dataclass
class InspectAstNode(ASTNode):
    target_func: str
    def to_source(self, indent: int = 0) -> str:
        return f"{'    '*indent}INSPECCIONAR_AST({self.target_func})"
    def pretty_tree(self, prefix: str = "") -> str:
        return f"{prefix}MetaInspect({self.target_func})"

@dataclass
class OptimizeAstNode(ASTNode):
    target_func: str
    def to_source(self, indent: int = 0) -> str:
        return f"{'    '*indent}OPTIMIZAR_AST({self.target_func})"
    def pretty_tree(self, prefix: str = "") -> str:
        return f"{prefix}MetaOptimize({self.target_func})"

@dataclass
class MutateFunctionNode(ASTNode):
    """
    Instrucción de Metaprogramación:
    Busca en el AST de `target_func` todas las operaciones `old_op` o variables
    y las reescribe dinámicamente o inyecta una nueva expresión en el retorno.
    """
    target_func: str
    target_var: str
    new_expr: ASTNode

    def to_source(self, indent: int = 0) -> str:
        return f"{'    '*indent}MUTAR_FUNCION({self.target_func}, {self.target_var} -> {self.new_expr.to_source()})"
    def pretty_tree(self, prefix: str = "") -> str:
        return f"{prefix}MetaMutate({self.target_func}: {self.target_var})\n{self.new_expr.pretty_tree(prefix + '  └── ')}"

@dataclass
class SynthesizeFunctionNode(ASTNode):
    """Crea una función nueva combinando el AST de dos funciones existentes."""
    new_name: str
    func_a: str
    func_b: str
    combine_op: str

    def to_source(self, indent: int = 0) -> str:
        return f"{'    '*indent}SINTETIZAR_FUNCION({self.new_name} = {self.func_a} {self.combine_op} {self.func_b})"
    def pretty_tree(self, prefix: str = "") -> str:
        return f"{prefix}MetaSynthesize({self.new_name} = {self.func_a} {self.combine_op} {self.func_b})"


# =====================================================================
# 4. ANALIZADOR SINTÁCTICO (PARSER RECURSIVO DESCENDENTE)
# =====================================================================

class NexusParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _eat(self, expected_type: str, expected_val: Any = None) -> Token:
        tok = self._current()
        if tok.type != expected_type or (expected_val is not None and tok.value != expected_val):
            raise SyntaxError(
                f"[Error Sintáctico] Se esperaba {expected_type} ({expected_val}), "
                f"pero se encontró {tok.type} ('{tok.value}') en línea {tok.line}"
            )
        self.pos += 1
        return tok

    def parse_program(self) -> List[ASTNode]:
        statements = []
        while self._current().type != "EOF":
            statements.append(self.parse_statement())
        return statements

    def parse_statement(self) -> ASTNode:
        tok = self._current()

        if tok.type == "FUNCION":
            return self.parse_function_decl()
        elif tok.type == "RETORNAR":
            self._eat("RETORNAR")
            return ReturnNode(self.parse_expression())
        elif tok.type == "IMPRIMIR":
            self._eat("IMPRIMIR")
            self._eat("LPAREN")
            expr = self.parse_expression()
            self._eat("RPAREN")
            return PrintNode(expr)
        elif tok.type == "INSPECCIONAR_AST":
            self._eat("INSPECCIONAR_AST")
            self._eat("LPAREN")
            name = self._eat("IDENT").value
            self._eat("RPAREN")
            return InspectAstNode(name)
        elif tok.type == "OPTIMIZAR_AST":
            self._eat("OPTIMIZAR_AST")
            self._eat("LPAREN")
            name = self._eat("IDENT").value
            self._eat("RPAREN")
            return OptimizeAstNode(name)
        elif tok.type == "MUTAR_FUNCION":
            # Sintaxis: MUTAR_FUNCION(nombre_func, variable_objetivo -> nueva_expresion)
            self._eat("MUTAR_FUNCION")
            self._eat("LPAREN")
            func_name = self._eat("IDENT").value
            self._eat("COMMA")
            target_var = self._eat("IDENT").value
            self._eat("OP", "->")
            new_expr = self.parse_expression()
            self._eat("RPAREN")
            return MutateFunctionNode(func_name, target_var, new_expr)
        elif tok.type == "SINTETIZAR_FUNCION":
            # Sintaxis: SINTETIZAR_FUNCION(nueva_func = funcA + funcB)
            self._eat("SINTETIZAR_FUNCION")
            self._eat("LPAREN")
            new_name = self._eat("IDENT").value
            self._eat("ASSIGN")
            func_a = self._eat("IDENT").value
            op = self._eat("OP").value
            func_b = self._eat("IDENT").value
            self._eat("RPAREN")
            return SynthesizeFunctionNode(new_name, func_a, func_b, op)
        elif tok.type == "IDENT":
            # Puede ser asignación (x = ...) o llamada a función (mi_func(...))
            next_tok = self.tokens[self.pos + 1]
            if next_tok.type == "ASSIGN":
                name = self._eat("IDENT").value
                self._eat("ASSIGN")
                return AssignNode(name, self.parse_expression())
            else:
                return self.parse_expression()

        raise SyntaxError(f"[Error Sintáctico] Instrucción no válida '{tok.value}' en línea {tok.line}")

    def parse_function_decl(self) -> FunctionDeclNode:
        self._eat("FUNCION")
        name = self._eat("IDENT").value
        self._eat("LPAREN")
        params = []
        if self._current().type != "RPAREN":
            params.append(self._eat("IDENT").value)
            while self._current().type == "COMMA":
                self._eat("COMMA")
                params.append(self._eat("IDENT").value)
        self._eat("RPAREN")
        self._eat("LBRACE")
        body = []
        while self._current().type != "RBRACE":
            body.append(self.parse_statement())
        self._eat("RBRACE")
        return FunctionDeclNode(name, params, body)

    def parse_expression(self) -> ASTNode:
        return self.parse_additive()

    def parse_additive(self) -> ASTNode:
        node = self.parse_multiplicative()
        while self._current().type == "OP" and self._current().value in ("+", "-"):
            op = self._eat("OP").value
            right = self.parse_multiplicative()
            node = BinaryOpNode(op, node, right)
        return node

    def parse_multiplicative(self) -> ASTNode:
        node = self.parse_primary()
        while self._current().type == "OP" and self._current().value in ("*", "/"):
            op = self._eat("OP").value
            right = self.parse_primary()
            node = BinaryOpNode(op, node, right)
        return node

    def parse_primary(self) -> ASTNode:
        tok = self._current()

        if tok.type == "NUMBER":
            self._eat("NUMBER")
            return NumberNode(tok.value)
        elif tok.type == "STRING":
            self._eat("STRING")
            return StringNode(tok.value)
        elif tok.type == "LBRACKET":
            # Literal de Vector 3D: [x, y, z]
            self._eat("LBRACKET")
            x = self.parse_expression()
            self._eat("COMMA")
            y = self.parse_expression()
            self._eat("COMMA")
            z = self.parse_expression()
            self._eat("RBRACKET")
            return Vector3Node(x, y, z)
        elif tok.type == "LPAREN":
            self._eat("LPAREN")
            expr = self.parse_expression()
            self._eat("RPAREN")
            return expr
        elif tok.type == "IDENT":
            ident = self._eat("IDENT").value
            if self._current().type == "LPAREN":
                self._eat("LPAREN")
                args = []
                if self._current().type != "RPAREN":
                    args.append(self.parse_expression())
                    while self._current().type == "COMMA":
                        self._eat("COMMA")
                        args.append(self.parse_expression())
                self._eat("RPAREN")
                return CallNode(ident, args)
            return IdentifierNode(ident)

        raise SyntaxError(f"[Error Sintáctico] Expresión inesperada '{tok.value}' en línea {tok.line}")


# =====================================================================
# 5. MOTOR DE METAPROGRAMACIÓN Y EJECUCIÓN HOMOICÓNICA
# =====================================================================

class ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value

class HomoiconicRuntime:
    """
    Ejecuta el AST y permite que el programa modifique sus propias funciones,
    simplifique ramas algebraicas y sintetice funciones nuevas en caliente.
    """
    def __init__(self):
        self.global_env: Dict[str, Any] = {}
        self.function_table: Dict[str, FunctionDeclNode] = {}

    def _fold_constants_recursive(self, node: ASTNode) -> ASTNode:
        """
        Optimizador de AST: Recorre el árbol de una función y resuelve
        operaciones matemáticas constantes antes de ejecutarlas.
        """
        if isinstance(node, BinaryOpNode):
            left = self._fold_constants_recursive(node.left)
            right = self._fold_constants_recursive(node.right)

            # Si ambos lados son números constantes, calcular en tiempo de compilación
            if isinstance(left, NumberNode) and isinstance(right, NumberNode):
                if node.op == "+": return NumberNode(left.value + right.value)
                if node.op == "-": return NumberNode(left.value - right.value)
                if node.op == "*": return NumberNode(left.value * right.value)
                if node.op == "/" and right.value != 0: return NumberNode(left.value / right.value)

            # Identidades algebraicas: (x * 1) -> x, (x + 0) -> x
            if node.op == "*" and isinstance(right, NumberNode) and right.value == 1.0:
                return left
            if node.op == "+" and isinstance(right, NumberNode) and right.value == 0.0:
                return left

            return BinaryOpNode(node.op, left, right)

        elif isinstance(node, Vector3Node):
            return Vector3Node(
                self._fold_constants_recursive(node.x),
                self._fold_constants_recursive(node.y),
                self._fold_constants_recursive(node.z)
            )
        elif isinstance(node, AssignNode):
            return AssignNode(node.name, self._fold_constants_recursive(node.expr))
        elif isinstance(node, ReturnNode):
            return ReturnNode(self._fold_constants_recursive(node.expr))
        return node

    def _mutate_variable_in_ast(self, node: ASTNode, target_var: str, replacement_expr: ASTNode) -> ASTNode:
        """
        Reescribe quirúrgicamente cualquier asignación a `target_var` dentro del AST
        o inyecta la nueva fórmula directamente en la función.
        """
        if isinstance(node, AssignNode) and node.name == target_var:
            return AssignNode(target_var, copy.deepcopy(replacement_expr))
        return node

    def execute_program(self, ast_nodes: List[ASTNode]):
        for node in ast_nodes:
            self.eval_node(node, self.global_env)

    def eval_node(self, node: ASTNode, env: Dict[str, Any]) -> Any:
        if isinstance(node, NumberNode):
            return node.value
        elif isinstance(node, StringNode):
            return node.value
        elif isinstance(node, IdentifierNode):
            if node.name in env:
                return env[node.name]
            if node.name in self.global_env:
                return self.global_env[node.name]
            raise NameError(f"Variable '{node.name}' no definida.")

        elif isinstance(node, Vector3Node):
            return Vec3(
                self.eval_node(node.x, env),
                self.eval_node(node.y, env),
                self.eval_node(node.z, env)
            )

        elif isinstance(node, BinaryOpNode):
            l_val = self.eval_node(node.left, env)
            r_val = self.eval_node(node.right, env)
            if node.op == "+": return l_val + r_val
            if node.op == "-": return l_val - r_val
            if node.op == "*": return l_val * r_val
            if node.op == "/": return l_val / r_val

        elif isinstance(node, AssignNode):
            val = self.eval_node(node.expr, env)
            env[node.name] = val
            return val

        elif isinstance(node, PrintNode):
            val = self.eval_node(node.expr, env)
            print(f"[SALIDA MOTOR XR] >> {val}")
            return val

        elif isinstance(node, ReturnNode):
            raise ReturnSignal(self.eval_node(node.expr, env))

        elif isinstance(node, FunctionDeclNode):
            self.function_table[node.name] = copy.deepcopy(node)
            print(f"[COMPILADOR] Función '{node.name}' registrada en memoria AST (v{node.version}).")

        elif isinstance(node, CallNode):
            if node.func_name not in self.function_table:
                raise NameError(f"Función '{node.func_name}' no existe en la tabla AST.")
            fn_ast = self.function_table[node.func_name]
            local_env = {}
            for param_name, arg_node in zip(fn_ast.params, node.args):
                local_env[param_name] = self.eval_node(arg_node, env)

            try:
                for stmt in fn_ast.body:
                    self.eval_node(stmt, local_env)
            except ReturnSignal as ret:
                return ret.value
            return None

        # --- EJECUCIÓN DE METAPROGRAMACIÓN EN VIVO ---

        elif isinstance(node, InspectAstNode):
            fn = self.function_table[node.target_func]
            print("\n" + "="*68)
            print(f" [INSPECCIÓN REFLEXIVA DE AST] Función: '{fn.name}' (Versión v{fn.version})")
            print("-" * 68)
            print(" >> Código Fuente Reconstruido desde el AST:")
            print(fn.to_source())
            print("\n >> Estructura Interna del Árbol Sintáctico:")
            print(fn.pretty_tree())
            print("="*68 + "\n")

        elif isinstance(node, OptimizeAstNode):
            fn = self.function_table[node.target_func]
            fn.body = [self._fold_constants_recursive(stmt) for stmt in fn.body]
            fn.version += 1
            print(f"[METAPROGRAMACIÓN] AST de '{fn.name}' auto-optimizado algebraicamente -> Nueva versión v{fn.version}")

        elif isinstance(node, MutateFunctionNode):
            fn = self.function_table[node.target_func]
            fn.body = [
                self._mutate_variable_in_ast(stmt, node.target_var, node.new_expr)
                for stmt in fn.body
            ]
            fn.version += 1
            print(f"[AUTO-REESCRITURA] La función '{fn.name}' ha reescrito su propia variable '{node.target_var}' -> v{fn.version}")

        elif isinstance(node, SynthesizeFunctionNode):
            fn_a = self.function_table[node.func_a]
            fn_b = self.function_table[node.func_b]

            # Extraer las expresiones de retorno de ambas funciones para fusionarlas
            ret_a = next(s.expr for s in fn_a.body if isinstance(s, ReturnNode))
            ret_b = next(s.expr for s in fn_b.body if isinstance(s, ReturnNode))

            pre_body = [copy.deepcopy(s) for s in fn_a.body if not isinstance(s, ReturnNode)]
            pre_body += [copy.deepcopy(s) for s in fn_b.body if not isinstance(s, ReturnNode)]

            fused_return = ReturnNode(BinaryOpNode(node.combine_op, copy.deepcopy(ret_a), copy.deepcopy(ret_b)))
            pre_body.append(fused_return)

            new_fn = FunctionDeclNode(
                name=node.new_name,
                params=list(dict.fromkeys(fn_a.params + fn_b.params)),
                body=pre_body,
                version=1
            )
            self.function_table[node.new_name] = new_fn
            print(f"[SÍNTESIS GENÉTICA] Creada nueva función '{node.new_name}' fusionando '{node.func_a}' y '{node.func_b}'!")


# =====================================================================
# 6. DEMOSTRACIÓN EN VIVO DE AUTO-REESCRITURA ESPACIAL
# =====================================================================

if __name__ == "__main__":
    codigo_nexus = """
    # 1. Definimos una función espacial para mover un holograma con la mano
    #    Incluye operaciones redundantes a propósito (2 * 3 * 1) para probar el auto-optimizador
    FUNCION calcular_impulso_mano(vector_mano) {
        ganancia = (2 * 3) + 0
        desplazamiento = vector_mano * ganancia
        RETORNAR desplazamiento
    }

    # 2. Definimos una segunda función que genera un campo de levitación vertical
    FUNCION levitacion_magnetica(vector_mano) {
        empuje_vertical = [0, 1.5, 0]
        RETORNAR empuje_vertical
    }

    # 3. Inspeccionamos el AST inicial de la función antes de tocarla
    INSPECCIONAR_AST(calcular_impulso_mano)

    # 4. Probamos la función original con un vector de mano 3D: [1.0, 2.0, -0.5]
    IMPRIMIR("Resultado con AST Original (v1):")
    pos_inicial = calcular_impulso_mano([1.0, 2.0, -0.5])
    IMPRIMIR(pos_inicial)

    # 5. ¡AUTO-OPTIMIZACIÓN! El lenguaje simplifica (2 * 3) + 0 -> 6 en el árbol AST
    OPTIMIZAR_AST(calcular_impulso_mano)
    INSPECCIONAR_AST(calcular_impulso_mano)

    # 6. ¡AUTO-REESCRITURA EN CALIENTE!
    #    El motor detecta que el usuario necesita precisión milimétrica y reescribe
    #    internamente la variable 'ganancia' de la función sin detener el programa:
    MUTAR_FUNCION(calcular_impulso_mano, ganancia -> 25.0)
    INSPECCIONAR_AST(calcular_impulso_mano)

    IMPRIMIR("Resultado tras Auto-Reescritura en caliente (v3):")
    pos_mutada = calcular_impulso_mano([1.0, 2.0, -0.5])
    IMPRIMIR(pos_mutada)

    # 7. ¡SÍNTESIS DE NUEVO CÓDIGO!
    #    El lenguaje fusiona el AST de 'calcular_impulso_mano' y 'levitacion_magnetica'
    #    para inventar una tercera función llamada 'telequinesis_hibrida'
    SINTETIZAR_FUNCION(telequinesis_hibrida = calcular_impulso_mano + levitacion_magnetica)
    INSPECCIONAR_AST(telequinesis_hibrida)

    IMPRIMIR("Resultado ejecutando la función recién inventada por el motor:")
    pos_final = telequinesis_hibrida([1.0, 2.0, -0.5])
    IMPRIMIR(pos_final)
    """

    # Ejecutar pipeline completo: Lexer -> Parser -> AST -> Metaprogramación
    lexer = NexusLexer(codigo_nexus)
    tokens = lexer.tokenize()

    parser = NexusParser(tokens)
    ast_program = parser.parse_program()

    runtime = HomoiconicRuntime()
    runtime.execute_program(ast_program)
