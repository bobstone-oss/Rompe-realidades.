import os
import time
import struct
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importar la arquitectura desde tu módulo anterior (o definirla aquí para autonomía total)
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TemporalGestureTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 84,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        embed_dim: int = 128,
        seq_len: int = 32
    ):
        super().__init__()
        self.frame_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=seq_len + 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,  # Dropout en 0.0 para inferencia determinista en ONNX
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h = self.frame_proj(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls_tokens, h], dim=1)
        h = self.pos_encoder(h)
        attended = self.transformer(h)
        cls_out = attended[:, 0, :]
        embedding = self.head(cls_out)
        return F.normalize(embedding, p=2, dim=-1)


def export_transformer_and_vault(
    vault_pt_path: str = "nexus_temporal_transformer_vault.pt",
    onnx_out_path: str = "nexus_transformer.onnx",
    prototypes_bin_path: str = "nexus_prototypes.bin"
):
    device = torch.device("cpu")
    model = TemporalGestureTransformer().to(device)
    gestures_dict = {}

    # 1. Cargar pesos y prototipos si existe una bóveda entrenada; si no, crear prototipos de prueba
    if os.path.exists(vault_pt_path):
        ckpt = torch.load(vault_pt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        gestures_dict = ckpt.get("gestures", {})
        print(f"[EXPORTADOR] Bóveda cargada desde '{vault_pt_path}' con {len(gestures_dict)} gestos.")
    else:
        print("[EXPORTADOR] No se encontró bóveda previa. Generando modelo base y 3 prototipos de demostración...")
        for demo_name in ("TRAZO_CIRCULO", "TRAZO_TRIANGULO", "SIMBOLO_Z"):
            vec = F.normalize(torch.randn(128), p=2, dim=-1)
            gestures_dict[demo_name] = {"proto": vec}

    model.eval()

    # 2. Crear tensor de entrada de ejemplo: (Batch=1, SeqLen=32, Features=84)
    dummy_input = torch.randn(1, 32, 84, dtype=torch.float32, device=device)

    # 3. Exportar a ONNX con plegado de constantes y Opset 17
    torch.onnx.export(
        model,
        dummy_input,
        onnx_out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["hand_sequence_84d"],
        output_names=["hypersphere_embedding_128d"],
        dynamic_axes={
            "hand_sequence_84d": {0: "batch_size"},
            "hypersphere_embedding_128d": {0: "batch_size"}
        }
    )
    print(f"[ONNX] Grafo exportado exitosamente a '{onnx_out_path}'.")

    # 4. Serializar los prototipos hipersféricos en un formato binario plano para Rust
    # Formato binario:
    # [Magic: 4 bytes 'NXPR'] [NumGestos: uint32]
    # Por cada gesto: [NameLen: uint16] [NameUTF8: bytes] [128 floats IEEE-754 (512 bytes)]
    with open(prototypes_bin_path, "wb") as f:
        f.write(b"NXPR")
        f.write(struct.pack("<I", len(gestures_dict)))
        for g_name, g_data in gestures_dict.items():
            proto_tensor = g_data["proto"] if "proto" in g_data else g_data["prototype"]
            proto_np = proto_tensor.detach().cpu().numpy().astype(np.float32).flatten()
            name_bytes = g_name.encode("utf-8")

            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(proto_np.tobytes())

    print(f"[BINARIO RUST] {len(gestures_dict)} prototipos empaquetados en '{prototypes_bin_path}'.")

    # 5. Verificación numérica y Benchmark de Latencia (< 1 ms) con ONNX Runtime
    try:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        ort_session = ort.InferenceSession(onnx_out_path, sess_options, providers=["CPUExecutionProvider"])
        input_np = dummy_input.numpy()

        # Comparar salida exacta entre PyTorch y ONNX
        with torch.no_grad():
            pt_out = model(dummy_input).numpy()
        ort_out = ort_session.run(None, {"hand_sequence_84d": input_np})[0]
        max_diff = float(np.max(np.abs(pt_out - ort_out)))

        # Calentamiento y medición de 1,000 inferencias
        for _ in range(50):
            ort_session.run(None, {"hand_sequence_84d": input_np})

        iterations = 1000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ort_session.run(None, {"hand_sequence_84d": input_np})
        avg_onnx_ms = ((time.perf_counter() - t0) * 1000.0) / iterations

        t1 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                model(dummy_input)
        avg_pt_ms = ((time.perf_counter() - t1) * 1000.0) / iterations

        print("\n==================================================================")
        print(f" VERIFICACIÓN DE PRECISIÓN NUMÉRICA (Diff Máx): {max_diff:.7f}")
        print(f" Latencia Promedio PyTorch      : {avg_pt_ms:.3f} ms / inferencia")
        print(f" Latencia Promedio ONNX Runtime : {avg_onnx_ms:.3f} ms / inferencia")
        print(f" ACELERACIÓN ONNX               : {avg_pt_ms / max(avg_onnx_ms, 1e-6):.2f}x más rápido")
        print("==================================================================")

    except ImportError:
        print("[AVISO] Instala 'pip install onnxruntime' para ejecutar el benchmark local.")


if __name__ == "__main__":
    export_transformer_and_vault()
