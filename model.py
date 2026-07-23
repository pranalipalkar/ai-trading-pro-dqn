import numpy as np


class NumpyDQN:
    """Lightweight DQN stand-in (no TensorFlow) for Streamlit Cloud compatibility."""

    def __init__(self, state_shape, n_actions, seed=42):
        rng = np.random.default_rng(seed)
        input_dim = state_shape if isinstance(state_shape, int) else int(state_shape[0])

        self.w1 = rng.normal(0, 0.1, size=(input_dim, 64))
        self.b1 = np.zeros(64)
        self.w2 = rng.normal(0, 0.1, size=(64, 32))
        self.b2 = np.zeros(32)
        self.w3 = rng.normal(0, 0.1, size=(32, n_actions))
        self.b3 = np.zeros(n_actions)

    def predict(self, state_input, verbose=0):
        x = np.asarray(state_input, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        h1 = np.maximum(0, x @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        q_values = h2 @ self.w3 + self.b3
        return q_values


def build_dqn_model(state_shape, n_actions):
    return NumpyDQN(state_shape=state_shape, n_actions=n_actions)
