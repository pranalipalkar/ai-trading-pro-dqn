import tensorflow as tf
from tensorflow.keras import layers

def build_dqn_model(state_shape, n_actions):
    model = tf.keras.Sequential([
        layers.Dense(64, input_dim=state_shape, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_actions, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model