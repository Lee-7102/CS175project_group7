import traci
import numpy as np
import random
import timeit
import os
from utils import import_train_configuration, set_train_path

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, model_type, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs, update_target_epochs):
        self._Model = Model
        self._target_Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._model_type = model_type
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        self._update_target_epochs = update_target_epochs


    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """

        config = import_train_configuration(config_file='training_settings.ini')
        path = set_train_path(config['models_path_name'])
        
        start_time = timeit.default_timer()

        # generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._best_reward = -float('inf')
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:
            current_state = self._get_state()

            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        if self._sum_neg_reward > self._best_reward:
          self._Model.save_model(path)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            if self._model_type == 2 or self._model_type == 3:
                if _ % self._update_target_epochs == 0:
                    self._update_target_q_network() # update target q network only for DDQN and duelingDQN
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds


    def _collect_waiting_times(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state, epsilon): 
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state


    def _set_yellow_phase(self, old_action):
        yellow_phase_code = old_action * 2 + 1 
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)


    def _get_queue_length(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located 
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state

    def _update_target_q_network(self):
        self._target_Model._model.set_weights(self._Model._model.get_weights())


    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states 
            next_states = np.array([val[3] for val in batch])  # extract next states 

            if self._model_type == 4:
                act_prob = self._Model.predict_batch(states)
                next_act_prob = self._Model.predict_batch(next_states)

                # setup training arrays
                x_state = np.zeros((len(batch), self._num_states))#states
                y_action = np.zeros((len(batch), self._num_actions))
                y_value = np.zeros(len(batch))
                grad=0.001

                for i, b in enumerate(batch):
                    state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                    target_value = self._Model.predict_one_value(next_states[i])
                    advantage = reward + self._gamma * target_value - self._Model.predict_one_value(state)
                    old_policy_prob = act_prob[action]
                    new_policy_prob = next_act_prob[action]
                    ratio = new_policy_prob / old_policy_prob
                    surrogate_loss = np.minimum(ratio * advantage, np.clip(ratio, 1 - 0.1, 1 + 0.1) * advantage)

                    x_state[i] = state
                    y_action[i] = action+grad*surrogate_loss
                    y_value[i] = reward + self._gamma * target_value

                self._Model.train_policy_batch(x_state, y_action)

                self._Model.train_value_batch(x_state, y_value)

            else:
                # prediction
                q_s_a = self._Model.predict_batch(states)  # predict Q(state)
                q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state)
                if self._model_type == 2 or self._model_type == 3:
                    q_t = self._target_Model.predict_batch(next_states)

                # setup training arrays
                x = np.zeros((len(batch), self._num_states))
                y = np.zeros((len(batch), self._num_actions))

                for i, b in enumerate(batch):
                    state, action, reward, _ = b[0], b[1], b[2], b[3] 
                    current_q = q_s_a[i]  
                    if self._model_type == 1:#update according to current q
                        current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action) 
                    elif self._model_type == 2 or self._model_type == 3:#update according to targer q 
                        max_act4next = np.argmax(q_t[i])
                        selected_q_next = q_s_a_d[i][max_act4next]
                        current_q[action] = reward + self._gamma * selected_q_next  # update Q(state, action) 
                    x[i] = state
                    y[i] = current_q 

                self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

