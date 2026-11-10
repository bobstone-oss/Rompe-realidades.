import struct
import time
import copy
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

# Intentar importar el módulo base del compilador
try:
    from .nexus_lang_ast import (
        NexusLexer, NexusParser, ASTNode, NumberNode, IdentifierNode,
        Vector3Node, BinaryOpNode, AssignNode, ReturnNode, FunctionDeclNode,
        OptimizeAstNode, MutateFunctionNode, SynthesizeFunctionNode,
        HomoiconicRuntime, Vec3
    )
except (ImportError, ValueError):
    from nexus_lang_ast import (
        NexusLexer, NexusParser, ASTNode, NumberNode, IdentifierNode,
        Vector3Node, BinaryOpNode, AssignNode, ReturnNode, FunctionDeclNode,
        OptimizeAstNode, MutateFunctionNode, SynthesizeFunctionNode,
        HomoiconicRuntime, Vec3
    )


# =====================================================================
# 1. ARQUITECTURA DE CONJUNTO DE INSTRUCCIONES (ISA DE 32 BITS)
# =====================================================================

class VMOp(IntEnum):
    LOAD_CONST  = 0x01
    LOAD_LOCAL  = 0x02
    STORE_LOCAL = 0x03
    ADD         = 0x04
    SUB         = 0x05
    MUL         = 0x06
    DIV         = 0x07
    # Superinstrucciones fusionadas (Peephole Optimization)
    ADD_STORE   = 0x08
    SUB_STORE   = 0x09
    MUL_STORE   = 0x0A
    DIV_STORE   = 0x0B
    MAKE_VEC3   = 0x0E
    RET         = 0x0F


def pack_instruction(op: VMOp, arg1: int = 0, arg2: int = 0) -> int:
    """Empaqueta OpCode (8 bits) + Arg1 (12 bits) + Arg2 (12 bits) en un entero de 32 bits."""
    return ((int(op) & 0xFF) << 24) | ((arg1 & 0xFFF) << 12) | (arg2 & 0xFFF)


def unpack_instruction(word: int) -> Tuple[VMOp, int, int]:
    """Desempaqueta una palabra de 32 bits en (OpCode, Arg1, Arg2)."""
    op = VMOp((word >> 24) & 0xFF)
    arg1 = (word >> 12) & 0xFFF
    arg2 = word & 0xFFF
    return op, arg1, arg2


# =====================================================================
# 2. BLOQUE BINARIO COMPILADO Y SERIALIZADOR (.nxbc)
# =====================================================================

class PackedBytecodeModule:
    """
    Representa una función AST compilada a código binario de 32 bits.
    Todas las constantes y variables se almacenan como vectores float64 de 3 componentes
    (los escalares se promueven a [v, v, v] para ejecución SIMD uniforme).
    """
    MAGIC_HEADER = b"NXBC"

    def __init__(self, name: str, param_count: int, version: int = 1):
        self.name = name
        self.param_count = param_count
        self.version = version
        self.raw_instructions: List[Tuple[VMOp, int, int]] = []
        self.packed_code: np.ndarray = np.zeros(0, dtype=np.uint32)

        self.constants_list: List[np.ndarray] = []
        self.const_pool: np.ndarray = np.zeros((0, 3), dtype=np.float64)

        self.local_names: List[str] = []
        self.name_to_slot: Dict[str, int] = {}

    def resolve_local(self, var_name: str) -> int:
        if var_name not in self.name_to_slot:
            slot = len(self.local_names)
            self.name_to_slot[var_name] = slot
            self.local_names.append(var_name)
        return self.name_to_slot[var_name]

    def add_const_vec(self, vec3_arr: np.ndarray) -> int:
        for idx, existing in enumerate(self.constants_list):
            if np.allclose(existing, vec3_arr):
                return idx
        idx = len(self.constants_list)
        self.constants_list.append(vec3_arr.astype(np.float64))
        return idx

    def emit(self, op: VMOp, arg1: int = 0, arg2: int = 0):
        self.raw_instructions.append((op, arg1, arg2))

    def finalize_and_optimize(self):
        """
         Fase de Optimización de Mirilla (Peephole Optimizer):
        Fusiona pares (OP_ARITMETICA + STORE_LOCAL) en una sola superinstrucción
        y empaqueta todo en arreglos contiguos de C/NumPy.
        """
        fuse_map = {
            VMOp.ADD: VMOp.ADD_STORE,
            VMOp.SUB: VMOp.SUB_STORE,
            VMOp.MUL: VMOp.MUL_STORE,
            VMOp.DIV: VMOp.DIV_STORE,
        }
        optimized: List[Tuple[VMOp, int, int]] = []
        i = 0
        n = len(self.raw_instructions)

        while i < n:
            curr_op, a1, a2 = self.raw_instructions[i]
            if i + 1 < n and curr_op in fuse_map and self.raw_instructions[i + 1][0] == VMOp.STORE_LOCAL:
                target_slot = self.raw_instructions[i + 1][1]
                optimized.append((fuse_map[curr_op], target_slot, 0))
                i += 2  # Saltar el STORE_LOCAL absorbido
            else:
                optimized.append((curr_op, a1, a2))
                i += 1

        self.raw_instructions = optimized
        self.packed_code = np.array(
            [pack_instruction(op, a1, a2) for op, a1, a2 in self.raw_instructions],
            dtype=np.uint32
        )
        if self.constants_list:
            self.const_pool = np.vstack(self.constants_list).astype(np.float64)
        else:
            self.const_pool = np.zeros((1, 3), dtype=np.float64)

    def disassemble(self) -> str:
        lines = [
            f"=== DISASSEMBLY BINARIO 32-BIT: '{self.name}' (v{self.version}) ===",
            f"Registros Locales ({len(self.local_names)}): { {i: k for i, k in enumerate(self.local_names)} }",
            "-" * 66
        ]
        for ip, word in enumerate(self.packed_code):
            op, a1, a2 = unpack_instruction(int(word))
            hex_word = f"0x{int(word):08X}"
            if op == VMOp.LOAD_CONST:
                c_val = self.const_pool[a1]
                desc = f"const[{a1}] = [{c_val[0]:.2f}, {c_val[1]:.2f}, {c_val[2]:.2f}]"
            elif op in (VMOp.LOAD_LOCAL, VMOp.STORE_LOCAL):
                desc = f"local[{a1}] (${self.local_names[a1]})"
            elif op in (VMOp.ADD_STORE, VMOp.SUB_STORE, VMOp.MUL_STORE, VMOp.DIV_STORE):
                desc = f"-> FUSIONADO directo a local[{a1}] (${self.local_names[a1]})"
            else:
                desc = ""
            lines.append(f"  {ip:02d} | {hex_word} | {op.name:<12} | {desc}")
        lines.append("=" * 66)
        return "\n".join(lines)

    def save_binary_file(self, filepath: str):
        """Guarda el Bytecode empaquetado y el Constant Pool en un archivo binario '.nxbc'."""
        with open(filepath, "wb") as f:
            name_bytes = self.name.encode("utf-8")
            f.write(self.MAGIC_HEADER)
            f.write(struct.pack("<HHIHH", len(name_bytes), self.param_count, self.version,
                                len(self.local_names), len(self.constants_list)))
            f.write(name_bytes)
            for vec in self.constants_list:
                f.write(struct.pack("<ddd", float(vec[0]), float(vec[1]), float(vec[2])))
            f.write(struct.pack("<I", len(self.packed_code)))
            f.write(self.packed_code.tobytes())


# =====================================================================
# 3. COMPILADOR DE AST -> BYTECODE EMPAQUETADO
# =====================================================================

class PackedBytecodeCompiler:
    @staticmethod
    def compile(fn_node: FunctionDeclNode) -> PackedBytecodeModule:
        mod = PackedBytecodeModule(fn_node.name, len(fn_node.params), fn_node.version)

        for p in fn_node.params:
            mod.resolve_local(p)

        for stmt in fn_node.body:
            if isinstance(stmt, AssignNode):
                PackedBytecodeCompiler._compile_expr(stmt.expr, mod)
                slot = mod.resolve_local(stmt.name)
                mod.emit(VMOp.STORE_LOCAL, slot)
            elif isinstance(stmt, ReturnNode):
                PackedBytecodeCompiler._compile_expr(stmt.expr, mod)
                mod.emit(VMOp.RET)

        mod.finalize_and_optimize()
        return mod

    @staticmethod
    def _compile_expr(expr: ASTNode, mod: PackedBytecodeModule):
        if isinstance(expr, NumberNode):
            v = float(expr.value)
            c_idx = mod.add_const_vec(np.array([v, v, v], dtype=np.float64))
            mod.emit(VMOp.LOAD_CONST, c_idx)

        elif isinstance(expr, IdentifierNode):
            slot = mod.resolve_local(expr.name)
            mod.emit(VMOp.LOAD_LOCAL, slot)

        elif isinstance(expr, Vector3Node):
            if isinstance(expr.x, NumberNode) and isinstance(expr.y, NumberNode) and isinstance(expr.z, NumberNode):
                c_idx = mod.add_const_vec(np.array([expr.x.value, expr.y.value, expr.z.value], dtype=np.float64))
                mod.emit(VMOp.LOAD_CONST, c_idx)
            else:
                PackedBytecodeCompiler._compile_expr(expr.x, mod)
                PackedBytecodeCompiler._compile_expr(expr.y, mod)
                PackedBytecodeCompiler._compile_expr(expr.z, mod)
                mod.emit(VMOp.MAKE_VEC3)

        elif isinstance(expr, BinaryOpNode):
            PackedBytecodeCompiler._compile_expr(expr.left, mod)
            PackedBytecodeCompiler._compile_expr(expr.right, mod)
            op_map = {"+": VMOp.ADD, "-": VMOp.SUB, "*": VMOp.MUL, "/": VMOp.DIV}
            mod.emit(op_map[expr.op])


# =====================================================================
# 4. MÁQUINA VIRTUAL DE PILA VECTORIZADA (FAST REGISTER-STACK VM)
# =====================================================================

class FastVectorStackVM:
    """
    Ejecuta bloques PackedBytecodeModule sobre matrices pre-alojadas en C/NumPy,
    evitando crear objetos nuevos durante el ciclo de ejecución.
    """
    def __init__(self, max_stack: int = 32, max_locals: int = 32):
        self.stack = np.zeros((max_stack, 3), dtype=np.float64)
        self.locals = np.zeros((max_locals, 3), dtype=np.float64)

    def execute(self, mod: PackedBytecodeModule, args: List[np.ndarray]) -> np.ndarray:
        stack = self.stack
        loc = self.locals
        consts = mod.const_pool
        code = mod.packed_code

        # Cargar argumentos de entrada en los primeros slots locales
        for i in range(mod.param_count):
            loc[i] = args[i]

        sp = 0  # Puntero de Pila (Stack Pointer)

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
            elif op == VMOp.ADD:
                sp -= 1
                stack[sp - 1] = stack[sp - 1] + stack[sp]
            elif op == VMOp.SUB:
                sp -= 1
                stack[sp - 1] = stack[sp - 1] - stack[sp]
            elif op == VMOp.MUL:
                sp -= 1
                stack[sp - 1] = stack[sp - 1] * stack[sp]
            elif op == VMOp.DIV:
                sp -= 1
                stack[sp - 1] = stack[sp - 1] / stack[sp]
            # --- Ejecución de Superinstrucciones Fusionadas ---
            elif op == VMOp.ADD_STORE:
                sp -= 2
                loc[a1] = stack[sp] + stack[sp + 1]
            elif op == VMOp.SUB_STORE:
                sp -= 2
                loc[a1] = stack[sp] - stack[sp + 1]
            elif op == VMOp.MUL_STORE:
                sp -= 2
                loc[a1] = stack[sp] * stack[sp + 1]
            elif op == VMOp.DIV_STORE:
                sp -= 2
                loc[a1] = stack[sp] / stack[sp + 1]
            elif op == VMOp.MAKE_VEC3:
                sp -= 3
                stack[sp] = (stack[sp][0], stack[sp + 1][0], stack[sp + 2][0])
                sp += 1
            elif op == VMOp.RET:
                return stack[sp - 1].copy()

        return stack[0].copy()


# =====================================================================
# 5. DEMOSTRACIÓN Y BENCHMARK: AST INTERPRETER VS. PACKED STACK VM
# =====================================================================

if __name__ == "__main__":
    source_program = """
    FUNCION calcular_impulso_mano(vector_mano) {
        ganancia = (2 * 3) + 0
        amortiguacion = 0.85
        desplazamiento = (vector_mano * ganancia) * amortiguacion
        RETORNAR desplazamiento
    }

    FUNCION levitacion_magnetica(vector_mano) {
        empuje_vertical = [0.0, 1.5, 0.0]
        RETORNAR empuje_vertical
    }

    # 1. Optimizamos constantes, mutamos la ganancia y sintetizamos la función híbrida
    OPTIMIZAR_AST(calcular_impulso_mano)
    MUTAR_FUNCION(calcular_impulso_mano, ganancia -> 25.0)
    SINTETIZAR_FUNCION(telequinesis_hibrida = calcular_impulso_mano + levitacion_magnetica)
    """

    # Paso 1: Ejecutar las mutaciones del AST usando el compilador homoicónico
    tokens = NexusLexer(source_program).tokenize()
    ast_program = NexusParser(tokens).parse()
    ast_runtime = HomoiconicRuntime()
    ast_runtime.execute_program(ast_program)

    # Paso 2: Compilar la función sintetizada ('telequinesis_hibrida') a Bytecode de 32 bits
    synthesized_ast = ast_runtime.function_table["telequinesis_hibrida"]
    bytecode_mod = PackedBytecodeCompiler.compile(synthesized_ast)

    print("\n" + bytecode_mod.disassemble())

    # Paso 3: Guardar el binario compilado en disco (.nxbc)
    bytecode_mod.save_binary_file("telequinesis_hibrida.nxbc")
    print("[SERIALIZADOR] Binario guardado exitosamente en 'telequinesis_hibrida.nxbc'\n")

    # Paso 4: Benchmark de 50,000 ejecuciones (Evaluador AST vs. Stack VM de 32 bits)
    vm = FastVectorStackVM()
    input_vec_np = np.array([1.0, 2.0, -0.5], dtype=np.float64)
    iterations = 50_000

    # A) Medir tiempo del intérprete AST tradicional
    call_ast_node = NexusParser(NexusLexer("telequinesis_hibrida([1.0, 2.0, -0.5])").tokenize()).parse()[0]
    t0 = time.perf_counter()
    for _ in range(iterations):
        res_ast = ast_runtime.eval_node(call_ast_node, {})
    ast_time_ms = (time.perf_counter() - t0) * 1000.0

    # B) Medir tiempo de la Stack VM con palabras de 32 bits y superinstrucciones
    t1 = time.perf_counter()
    for _ in range(iterations):
        res_vm = vm.execute(bytecode_mod, [input_vec_np])
    vm_time_ms = (time.perf_counter() - t1) * 1000.0

    print("==================================================================")
    print(f" RESULTADO AST : {res_ast}")
    print(f" RESULTADO VM  : [{res_vm[0]:.3f}, {res_vm[1]:.3f}, {res_vm[2]:.3f}]")
    print("------------------------------------------------------------------")
    print(f" Tiempo 50,000 llamadas (Intérprete AST) : {ast_time_ms:.2f} ms")
    print(f" Tiempo 50,000 llamadas (32-Bit Stack VM): {vm_time_ms:.2f} ms")
    print(f" ACELERACIÓN REAL (Speedup)              : {ast_time_ms / max(vm_time_ms, 1e-6):.2f}x más rápido")
    print("==================================================================")
