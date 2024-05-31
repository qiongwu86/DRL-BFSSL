import time
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'
import Environment
from RL_train1 import SAC_Trainer
from RL_train1 import ReplayBuffer
from RL_train2 import Memory
import pickle
BS_width = 500/2
up_lanes = [i for i in [BS_width+3.5/2, BS_width+3.5+3.5/2]]
down_lanes = [i for i in [BS_width-3.5/2-3.5, BS_width-3.5/2]]
left_lanes = [i for i in [BS_width+3.5/2, BS_width+3.5+3.5/2]]
right_lanes = [i for i in [BS_width-3.5-3.5/2, BS_width-3.5/2]]
print(up_lanes)
print(down_lanes)
print(left_lanes)
print(right_lanes)

width = 200/2
height = 200/2

BS_position = [BS_width, BS_width]
max_power = 200
min_power = 5
max_f = 4e8
min_f = 5e7
m = 0.023
n_veh = 4

batch_size = 64
memory_size = 1000000

n_step_per_episode = 100
n_episode_test = 1000
n_interference_vehicle = 0

n_input = 2 * n_veh  # 8
n_output = 8
# --------------------------------------------------------------
label_sac = 'model/SAC_model_{}_{}'.format(n_episode_test, n_interference_vehicle)

replay_buffer_size = 1e6
replay_buffer = ReplayBuffer(replay_buffer_size)
hidden_dim = 512
AUTO_ENTROPY = True
DETERMINISTIC = False

memory = Memory()

#--------------model--------------
RL_SAC = SAC_Trainer(replay_buffer, n_input, n_output, hidden_dim=hidden_dim, action_range=action_range)


def save_results(name, i, reward_73, calculate_73):

    current_time = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{name}_73_{n_episode_test}_{current_time}_{n_interference_vehicle}"

    folder_path = os.path.join('log', folder_name)

    os.makedirs(folder_path)


    dir_str_reward = f'{name}_reward_73_10_{i + 1}.pkl'
    file_path_reward = os.path.join(folder_path, dir_str_reward)

    os.makedirs(os.path.dirname(file_path_reward), exist_ok=True)
    with open(file_path_reward, 'wb') as f:
        pickle.dump(reward_73, f)

    dir_str_calculate = f'{name}_calculate_73_10_{i + 1}.pkl'
    file_path_calculate = os.path.join(folder_path, dir_str_calculate)

    os.makedirs(os.path.dirname(file_path_calculate), exist_ok=True)
    with open(file_path_calculate, 'wb') as f:
        pickle.dump(calculate_73, f)


def sac_train(lambda_1, lambda_2):
    print("\nRestoring the sac model...")

    Sum_reward_list = []
    Sum_calculate_list = []
    Trans_successful_list = []
    Vehicle_positions_x0 = []
    Vehicle_positions_y0 = []
    Vehicle_positions_x1 = []
    Vehicle_positions_y1 = []
    Vehicle_positions_x2 = []
    Vehicle_positions_y2 = []
    Vehicle_positions_x3 = []
    Vehicle_positions_y3 = []


    for i_episode in range(n_episode_test):
        print('------ Episode', i_episode, '------')

        env.new_random_game()
        Vehicle_positions_x0.append(env.vehicles[0].start_position[0])
        Vehicle_positions_y0.append(env.vehicles[0].start_position[1])
        Vehicle_positions_x1.append(env.vehicles[1].start_position[0])
        Vehicle_positions_y1.append(env.vehicles[1].start_position[1])
        Vehicle_positions_x2.append(env.vehicles[2].start_position[0])
        Vehicle_positions_y2.append(env.vehicles[2].start_position[1])
        Vehicle_positions_x3.append(env.vehicles[3].start_position[0])
        Vehicle_positions_y3.append(env.vehicles[3].start_position[1])

        state_old_all = []

        state = env.get_state()
        state_old_all.append(state)

        Sum_reward_per_episode = []
        Sum_calculate_per_episode = []
        Sum_tran_suc_persent_per_episode = []

        for i_step in range(n_step_per_episode):
            env.renew_position()
            env.renew_position_interference()

            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_veh,2], dtype=np.float64)

            action = RL_SAC.policy_net.get_action(np.asarray(state_old_all).flatten(), deterministic=DETERMINISTIC)
            action = np.clip(action, -0.999, 0.999)
            action_all.append(action)

            for i in range(n_veh):
                action_all_training[i, 0] = ((action[0 + i * 2] + 1) / 2) * (max_power-min_power)+min_power
                action_all_training[i, 1] = ((action[1 + i * 2] + 1) / 2) * (max_f-min_f)+min_f

            action_pf = action_all_training.copy()

            h_i_dB, vel_v = env.overall_channel()

            tran_success, c_total, reward, comp_n_list = env.act_for_training(action_pf, h_i_dB, vel_v, lambda_1, lambda_2)

            reward = -100 * np.log(reward)

            Sum_reward_per_episode.append(np.sum(reward))
            Sum_calculate_per_episode.append(np.round(np.sum(comp_n_list)))
            Sum_tran_suc_persent_per_episode.append(tran_success/c_total)


            state_new = env.get_state()
            state_new_all.append((state_new))

            replay_buffer.push(np.asarray(state_old_all).flatten(), np.asarray(action_all).flatten(),
                           reward, np.asarray(state_new_all).flatten(), 0)

            if len(replay_buffer) > 256:
                for i in range(1):
                    _ = RL_SAC.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*n_output)

            state_old_all = state_new_all


        Sum_reward_list.append((np.mean(Sum_reward_per_episode)))
        Sum_calculate_list.append(np.round(np.mean(Sum_calculate_per_episode)))
        Trans_successful_list.append(np.mean(Sum_tran_suc_persent_per_episode))

        print('Sum_reward_per_episode:', round(np.average(Sum_reward_per_episode), 2))
        print('Sum_calculate_per_episode:', round(np.average(Sum_calculate_per_episode)))
        print('Trans model successful:', round(np.average(Sum_tran_suc_persent_per_episode), 2))
        print('lambda_1:{}, lambda_2:{}'.format(lambda_1, lambda_2))

    return Sum_reward_list, Sum_calculate_list

if __name__ == "__main__":


    for i in range(1):
        name = 'SAC'
        env = Environment.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_interference_vehicle,
                                   BS_width)
        env.new_random_game()
        SAC_reward_73, SAC_calculate_73 = sac_train(0.7, 0.3)

        save_results(name, i, SAC_reward_73, SAC_calculate_73)
