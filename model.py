import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class TrainModel:
    def __init__(self, model_type, num_layers, width, batch_size, learning_rate, num_actions, input_dim, output_dim,load_model=False,model_path=''):
        self._model_type = model_type
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_actions = num_actions
        if load_model == True:
          self._model = self._load_my_model(model_path)
        elif model_type == 4:
            self._policymodel, self._valuemodel = self._build_model(num_layers,width)
        else:
            self._model = self._build_model(num_layers, width)


    def _build_model(self, num_layers, width):
        if self._model_type == 1 or self._model_type == 2:
            inputs = keras.Input(shape=(self._input_dim,))
            x = layers.Dense(width, activation='relu')(inputs)
            for _ in range(num_layers):
                x = layers.Dense(width, activation='relu')(x)
            outputs = layers.Dense(self._output_dim, activation='linear')(x)

            model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
            model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
            return model

        elif self._model_type == 3:
            inputs = keras.Input(shape=(self._input_dim,))
            x = layers.Dense(width, activation='relu')(inputs)
            for _ in range(num_layers):
                x = layers.Dense(width, activation='relu')(x)

            value = layers.Dense(1,activation='linear')(x)    
            advantage = layers.Dense(self._output_dim,activation='linear')(x)
            advantage = layers.Lambda(lambda a: a - tf.reduce_mean(tf.expand_dims(a, -1), axis=1))(advantage)
            # advantage = keras.layers.Subtract()([advantage, layers.Average()([advantage])])
            results = keras.layers.Add()([value, advantage])
            outputs = keras.layers.Dense(self._output_dim)(results)

            model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
            model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
            return model
        
        elif self._model_type == 4:
            existing_value_model = self._load_my_model('trained_model_DDNN.h5')
            inputs = keras.Input(shape=(self._input_dim,))

            policy_layer = layers.Dense(width, activation='relu')(inputs)
            for i in range(num_layers):
                policy_layer = layers.Dense(width, activation='relu')(policy_layer)
            policy_layer = layers.Dense(self._num_actions, activation='softmax')(policy_layer)
            policy_model = keras.Model(inputs=inputs, outputs=policy_layer, name='policy_model')
            policy_model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))

            value_layer = layers.Dense(width, activation='relu')(inputs)
            for j in range(num_layers+1):
                value_layer = layers.Dense(width, activation='relu')(value_layer)
            value_layer = layers.Dense(1, activation='linear')(value_layer)
            value_model = keras.Model(inputs=inputs, outputs=value_layer, name='value_model')
            value_model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))

            for j in range(num_layers):
              value_model.layers[j].set_weights(existing_value_model.layers[j].get_weights())
            

            return policy_model,value_model


    def _load_my_model(self, model_file_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")

        

    def predict_one(self, state):
        state = np.reshape(state, [1, self._input_dim])#return a list of action value of state
        if self._model_type == 4: return self._policymodel.predict(state)#for PPO
        return self._model.predict(state)#only for DQN, DDQN, DuelingDQN

    def predict_one_value(self, state):#only for PPO
        state = np.reshape(state, [1, self._input_dim])#return a list of action value of state
        return self._valuemodel.predict(state)


    def predict_batch(self, states):
        if self._model_type == 4: return self._policymodel.predict(states)#for PPO
        return self._model.predict(states)#only for DQN, DDQN, DuelingDQN


    def train_batch(self, states, q_sa):#only for DQN, DDQN, DuelingDQN
        self._model.fit(states, q_sa, epochs=1, verbose=0)
        

    def train_policy_batch(self, states, actions):#for PPO
        self._policymodel.fit(states, actions, epochs=1, verbose=0)

    def train_value_batch(self, states, value):#for PPO
        self._valuemodel.fit(states, value, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 and a model architecture as png
        """
        if self._model_type == 4:
            self._policymodel.save(os.path.join(path, 'trained_policy_model.h5'))
            self._valuemodel.save(os.path.join(path, 'trained_value_model.h5'))
            plot_model(self._policymodel, to_file=os.path.join(path, 'policy_model_structure.png'), show_shapes=True, show_layer_names=True)
            plot_model(self._valuemodel, to_file=os.path.join(path, 'value_model_structure.png'), show_shapes=True, show_layer_names=True)
        else:
            self._model.save(os.path.join(path, 'trained_model.h5'))
            plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim