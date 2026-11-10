"""
Capa 02: Polyglot Runtime y Lenguaje Universal NexusLang
"""
from .nexus_lang_meta import NexusUniversalLanguage
from .nexus_lang_ast import (
    Vec3, Token, NexusLexer, NexusParser,
    HomoiconicRuntime, FunctionDeclNode, ASTNode
)
from .nexus_vm_packed import FastVectorStackVM, PackedBytecodeCompiler
from .nexus_vm_jit import FastStackVM, JITBytecodeCompiler, NeuroJITAgent

__all__ = [
    "NexusUniversalLanguage",
    "Vec3",
    "Token",
    "NexusLexer",
    "NexusParser",
    "HomoiconicRuntime",
    "FunctionDeclNode",
    "ASTNode",
    "FastVectorStackVM",
    "PackedBytecodeCompiler",
    "FastStackVM",
    "JITBytecodeCompiler",
    "NeuroJITAgent",
]
