"""
Pruebas unitarias para el Runtime Políglota y Máquinas Virtuales de NexusLang
"""
import unittest
import numpy as np
from nexus_xr.polyglot_runtime import (
    NexusUniversalLanguage,
    Vec3,
    NexusLexer,
    NexusParser,
    HomoiconicRuntime,
    FunctionDeclNode,
    FastVectorStackVM,
    PackedBytecodeCompiler,
    FastStackVM,
    JITBytecodeCompiler,
)
from nexus_xr.polyglot_runtime.nexus_lang_ast import ReturnSignal


class TestNexusUniversalLanguage(unittest.TestCase):
    def test_meta_language_execution(self):
        interp = NexusUniversalLanguage()
        script = """
        MODO_REALIDAD HIBRIDO_CUANTICO_AR_VR
        CREAR_HOLOGRAMA OrbePrimario ESFERA_PLASMA 0.0 1.0 -0.5
        VINCULAR_MANO OrbePrimario DEDO_INDICE_PUNTA
        APRENDER_COMANDO ACTIVAR_DEFENSA MODO_REALIDAD VR_TOTAL ; HARDWARE_PIN 12 ENCENDIDO
        ACTIVAR_DEFENSA
        """
        interp.execute_script(script)
        self.assertEqual(interp.reality_mode, "VR_TOTAL")
        self.assertIn("OrbePrimario", interp.world_objects)
        self.assertIn("ACTIVAR_DEFENSA", interp.learned_macros)


class TestNexusASTAndMetaprogramming(unittest.TestCase):
    def test_lexer_and_parser(self):
        source = """
        FUNCION prueba_calculo(pos) {
            factor = 2 + 3
            resultado = pos * factor
            RETORNAR resultado
        }
        """
        lexer = NexusLexer(source)
        tokens = lexer.tokenize()
        self.assertTrue(len(tokens) > 5)

        parser = NexusParser(tokens)
        ast = parser.parse_function_decl()
        self.assertIsInstance(ast, FunctionDeclNode)
        self.assertEqual(ast.name, "prueba_calculo")
        self.assertEqual(ast.params, ["pos"])

    def test_ast_execution_and_optimization(self):
        source = """
        FUNCION calcular_impulso_mano(vector_mano) {
            ganancia = (2 * 3) + 0
            desplazamiento = vector_mano * ganancia
            RETORNAR desplazamiento
        }
        OPTIMIZAR_AST(calcular_impulso_mano)
        MUTAR_FUNCION(calcular_impulso_mano, ganancia -> 10.0)
        """
        runtime = HomoiconicRuntime()
        lexer = NexusLexer(source)
        parser = NexusParser(lexer.tokenize())
        program_nodes = parser.parse_program()
        runtime.execute_program(program_nodes)

        func_decl = runtime.function_table["calcular_impulso_mano"]
        env = {"vector_mano": Vec3(1.0, 2.0, -0.5)}
        res = None
        try:
            for stmt in func_decl.body:
                runtime.eval_node(stmt, env)
        except ReturnSignal as ret:
            res = ret.value

        self.assertIsNotNone(res)
        self.assertAlmostEqual(res.x, 10.0)
        self.assertAlmostEqual(res.y, 20.0)
        self.assertAlmostEqual(res.z, -5.0)


class TestNexusVM(unittest.TestCase):
    def test_packed_bytecode_compiler_and_vm(self):
        source = """
        FUNCION calcular_fisica(pos) {
            impulso = pos * 2.5
            RETORNAR impulso
        }
        """
        lexer = NexusLexer(source)
        parser = NexusParser(lexer.tokenize())
        func_ast = parser.parse_function_decl()

        compiler = PackedBytecodeCompiler()
        module = compiler.compile(func_ast)
        self.assertTrue(len(module.packed_code) > 0)

        vm = FastVectorStackVM()
        arg = [np.array([1.0, 2.0, 3.0], dtype=np.float64)]
        result = vm.execute(module, arg)
        self.assertAlmostEqual(result[0], 2.5)
        self.assertAlmostEqual(result[1], 5.0)
        self.assertAlmostEqual(result[2], 7.5)


if __name__ == "__main__":
    unittest.main()
