from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":

    model_paths = ['trained_model_DNN.h5','trained_model_DDNN.h5','trained_model_DuelingDNN.h5']
    for model_path in model_paths:
      config = import_test_configuration(config_file='testing_settings.ini')
      sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
      plot_path = ''

      Model = TestModel(
          input_dim=config['num_states'],
          model_path=model_path
      )

      TrafficGen = TrafficGenerator(
          config['max_steps'], 
          config['n_cars_generated']
      )

      Visualization = Visualization(
          plot_path, 
          dpi=96
      )
          
      Simulation = Simulation(
          Model,
          TrafficGen,
          sumo_cmd,
          config['max_steps'],
          config['green_duration'],
          config['yellow_duration'],
          config['num_states'],
          config['num_actions']
      )

      print('\n----- Test episode')
      simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
      print('Simulation time:', simulation_time, 's')

      print("----- Testing info saved at:", plot_path)

      copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

      Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
      Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue lenght (vehicles)')
