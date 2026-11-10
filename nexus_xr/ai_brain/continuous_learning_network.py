import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

class ContinuousLearningNetwork(nn.Module):
    """
    Red Neuronal que recibe los 21 vectores 3D de la mano (63 dimensiones)
    más el contexto espacial, y aprende nuevas acciones dinámicamente.
    """
    def __init__(self, input_dim=63, hidden_dim=256, action_space=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_space)
        )

    def forward(self, state_vectors):
        return self.network(state_vectors)

class AdaptiveXREngineBrain:
    def __init__(self, state_dim=63, action_dim=16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ContinuousLearningNetwork(state_dim, 256, action_dim).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.criterion = nn.SmoothL1Loss()

        # Memoria episódica: guarda las últimas 50,000 interacciones para aprender de ellas
        self.memory = deque(maxlen=50000)
        self.gamma = 0.97  # Factor de descuento de aprendizaje futuro

    def perceive_and_act(self, hand_unit_vectors: np.ndarray, exploration_rate=0.1):
        """Transforma los vectores de la mano en una decisión del motor."""
        flat_state = hand_unit_vectors.flatten()
        if random.random() < exploration_rate:
            return random.randint(0, 15)  # Explora nuevas respuestas

        state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values).item())

    def learn_from_interaction(self, state, action, reward, next_state, done, batch_size=32):
        """
        Se ejecuta automáticamente después de cada acción del usuario.
        Ajusta los pesos sinápticos del motor en tiempo real.
        """
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Predicción actual vs recompensa real observada
        current_q = self.model(states_t).gather(1, actions_t)
        with torch.no_grad():
            max_next_q = self.model(next_states_t).max(1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
