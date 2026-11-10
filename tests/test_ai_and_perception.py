"""
Pruebas unitarias para Percepción Vectorial y Cerebro IA (PyTorch, Transformer, Genética)
"""
import unittest
import numpy as np
import torch
import os
from nexus_xr.ai_brain import (
    HypersphericalGestureNet,
    SelfLearningGestureVault,
    CanonicalHandGeometry,
    TemporalGestureTransformer,
    ContinuousLearningNetwork,
    AdaptiveXREngineBrain,
    NexusGeneticASTArena,
    KozaGeneticEngine,
    EvolvableASTFunction,
)
from nexus_xr.perception import SpatialHandVectorizer


class TestAIBrainPyTorch(unittest.TestCase):
    def test_hyperspherical_gesture_net(self):
        net = HypersphericalGestureNet()
        # Entrada: vector canónico de mano de 78 dimensiones (63 coords + 15 distancias)
        dummy_hand = torch.randn(1, 78)
        embedding = net(dummy_hand)
        # Debe producir un vector en la hipersfera unitaria (norma = 1.0)
        norm = torch.norm(embedding, dim=1).item()
        self.assertAlmostEqual(norm, 1.0, places=4)
        self.assertEqual(embedding.shape[1], 128)

    def test_gesture_vault_few_shot_learning(self):
        vault_file = "test_vault.pt"
        if os.path.exists(vault_file):
            os.remove(vault_file)

        vault = SelfLearningGestureVault(vault_path=vault_file)

        # Generar 21 puntos 3D simulados para la mano
        mock_raw_points = np.zeros((21, 3), dtype=np.float32)
        mock_raw_points[0] = [0.0, 0.0, 0.0] # Muñeca
        mock_raw_points[9] = [0.0, 0.1, 0.0] # Nudillo medio
        mock_raw_points[5] = [-0.04, 0.08, 0.0]
        mock_raw_points[17] = [0.04, 0.07, 0.0]
        for i in range(1, 21):
            if i not in [5, 9, 17]:
                mock_raw_points[i] = [0.01 * i, 0.02 * i, 0.005 * i]

        canonical_vec = CanonicalHandGeometry.extract_invariant_features(mock_raw_points)
        self.assertEqual(len(canonical_vec), 78)

        # Aprender un gesto en vivo sin entrenamiento previo
        vault.learn_gesture_sample("PINZAR_HOLOGRAFICO", canonical_vec)
        self.assertIn("PINZAR_HOLOGRAFICO", vault.gestures)

        # Identificar la misma postura
        match_name, score, latency = vault.recognize(canonical_vec)
        self.assertEqual(match_name, "PINZAR_HOLOGRAFICO")
        self.assertTrue(score > 0.8)

        # Limpiar
        if os.path.exists(vault_file):
            os.remove(vault_file)

    def test_temporal_transformer(self):
        transformer = TemporalGestureTransformer()
        # Secuencia espacio-temporal: (batch=1, seq_len=32, feature_dim=84)
        dummy_seq = torch.randn(1, 32, 84)
        emb = transformer(dummy_seq)
        norm = torch.norm(emb, dim=1).item()
        self.assertAlmostEqual(norm, 1.0, places=4)
        self.assertEqual(emb.shape[1], 128)


class TestGeneticProgramming(unittest.TestCase):
    def test_koza_subtree_crossover(self):
        # Crear dos funciones AST individuales
        parent1 = EvolvableASTFunction(ind_id=0, generation=1)
        parent2 = EvolvableASTFunction(ind_id=1, generation=1)

        # Ejecutar cruce de sub-árboles tipo Koza
        child = KozaGeneticEngine.crossover(parent1, parent2, child_id=2, generation=2)
        self.assertIsNotNone(child)
        self.assertEqual(child.ind_id, 2)
        self.assertTrue(len(child.body) > 0)


if __name__ == "__main__":
    unittest.main()
