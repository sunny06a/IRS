import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions as tfp

class DQNAgent:
    def __init__(self, state_size, action_size, num_atoms=51):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model(num_atoms)
        self.target_model = self._build_model(num_atoms)
        self.num_atoms = num_atoms

    def _build_model(self, num_atoms):
        input_A = Input(shape=self.state_size[0], name="InputA")
        input_B = Input(shape=self.state_size[1], name="InputB")

        # Shared layers for both input streams
        shared_layer = Dense(128, activation='swish')

        x_A = Flatten()(shared_layer(input_A))
        x_B = Flatten()(shared_layer(input_B))

        # Merge both streams
        merged = Concatenate()([x_A, x_B])

        # Dueling network architecture
        advantage = Dense(self.action_size, activation='linear')(merged)
        value = Dense(1, activation='linear')(merged)
        output = Subtract()([advantage, tf.reduce_mean(advantage, axis=-1, keepdims=True)])

        # Distributional RL layer
        distribution = tfp.layers.IndependentBernoulli(self.num_atoms, name="distribution")(output)
        support = np.linspace(-1, 1, self.num_atoms)

        model = Model(inputs=[input_A, input_B], outputs=[distribution, support])
        model.compile(loss=self._rainbow_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def _rainbow_loss(self, y_true, y_pred):
    # Unpack y_pred into distribution and support
        distribution, support = y_pred

    # Unpack y_true into target distribution and target support
        target_distribution, target_support = y_true

    # Compute the standard Q-learning loss
        delta_z = (self.num_atoms - 1) / (self.num_atoms * (support[1] - support[0]))
        target_support_tiled = tf.tile(target_support[:, None], [1, self.num_atoms])

        Tz = target_distribution * tf.clip_by_value(tf.math.log(target_support_tiled), tf.float32.min, tf.float32.max)
        Tz = tf.reduce_sum(Tz, axis=0)
        Tz = tf.clip_by_value(Tz, tf.float32.min, tf.float32.max)

        bellman_errors = Tz + self.gamma * tf.math.log(tf.reduce_sum(tf.exp(target_distribution / delta_z), axis=0))
        bellman_errors = bellman_errors - tf.reduce_sum(distribution * tf.math.log(tf.reduce_sum(tf.exp(y_pred / delta_z), axis=0)), axis=0)

    # Compute the distributional loss
        distributional_loss = tf.reduce_sum(bellman_errors)

    # Compute the entropy loss
        entropy_loss = -tf.reduce_sum(distribution * tf.math.log(distribution), axis=0)

    # Compute the quantile huber loss
        tau = tf.linspace(0.0, 1.0 - 1.0 / self.num_atoms, self.num_atoms)
        tau = tf.tile(tau[None, :], [self.num_atoms, 1])

        u = bellman_errors / delta_z
        quantile_huber_loss = tf.abs(tau - tf.stop_gradient(tf.cast(u < 1, tf.float32))) * tf.clip_by_value(tf.abs(u), 0, 1)

    # Sum all loss components
        total_loss = distributional_loss + 0.001 * entropy_loss + 0.5 * tf.reduce_sum(quantile_huber_loss)

    return total_loss

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            act_type = 'random'
            return random.randrange(self.action_size), act_type
        act_prob, _ = self.model.predict(state, verbose=0)
        act_type = 'RL'
        return np.argmax(act_prob), act_type

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        minibatch[0] = self.memory[len(self.memory) - 1]

        state_batch_0, state_batch_1, target_batch = [], [], []
        for state, action, reward, next_state in minibatch:
            ActionIndex = np.argmax(self.model.predict(next_state, verbose=0)[0])
            target = reward + self.gamma * self.target_model.predict(next_state, verbose=0)[0][ActionIndex]
            target_f = self.model.predict(state, verbose=0)

            target_distribution, _ = self.model.predict(state, verbose=0)
            target_distribution = target_distribution[0]

            target_distribution[action] = target

            state_batch_0.append(state[0])
            state_batch_1.append(state[1])
            target_batch.append(target_distribution)

        InputA_data = np.reshape(np.array(state_batch_0), (batch_size, -1))
        InputB_data = np.reshape(np.array(state_batch_1), (batch_size, -1))
        target_batch = np.array(target_batch)

        target = [target_batch, self.z_support]  # z_support is the support of the distribution

        loss = self._rainbow_loss(target, self.model.predict([InputA_data, InputB_data], verbose=0))

        self.model.fit({"InputA": InputA_data, "InputB": InputB_data}, {"Outputs": target_batch}, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
