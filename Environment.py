from __future__ import division
import numpy as np
import random
import math
from beta_allocation import BetaAllocation

t_trans_max =5
BS_position = [250, 250]

class V2Ichannels:
    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 10


    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - BS_position[0])
        d2 = abs(position_A[1] - BS_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)


class Vehicle:
    """Vehicle simulator: include all the information for a Vehicle"""
    def __init__(self, start_position, start_direction, velocity):
        self.start_position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []

class Environ:
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, n_interference_vehicle, BS_width):

        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.BS_width = BS_width
        self.height = height

        self.h_bs = 25
        self.h_ms = 1.5
        self.fc = 2
        self.T_n = 0.5
        self.sig2_dB = -114
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 11


        self.n_Veh = n_veh
        self.V2Ichannels = V2Ichannels()
        self.vehicles = []
        self.vehicles_interference = []

        self.V2I_Shadowing = []
        self.delta_distance = []
        self.V2I_pathloss = []

        self.V2I_h = []
        self.vel_v = []
        self.V2I_channels_abs = []
        self.V2I_channels_abs_interference = []
        self.n_interference_vehicle = n_interference_vehicle


        self.beta_all = BetaAllocation()


    def renew_position(self):
        for i in range(len(self.vehicles)):
            dirta_distance = self.T_n * self.vehicles[i].velocity
            if self.vehicles[i].direction == 'd':
                end_position = [self.vehicles[i].start_position[0], self.vehicles[i].start_position[1]-dirta_distance]
                self.vehicles[i].start_position = end_position

            elif self.vehicles[i].direction == 'u':
                end_position = [self.vehicles[i].start_position[0], self.vehicles[i].start_position[1]+dirta_distance]
                self.vehicles[i].start_position = end_position

            elif self.vehicles[i].direction == 'r':
                end_position = [self.vehicles[i].start_position[0]+dirta_distance, self.vehicles[i].start_position[1]]
                self.vehicles[i].start_position = end_position

            elif self.vehicles[i].direction == 'l':
                end_position = [self.vehicles[i].start_position[0]-dirta_distance, self.vehicles[i].start_position[1]]
                self.vehicles[i].start_position = end_position

    def renew_position_interference(self):
        for i in range(len(self.vehicles_interference)):
            dirta_distance = self.T_n * self.vehicles_interference[i].velocity
            if self.vehicles_interference[i].direction == 'd':
                end_position = [self.vehicles_interference[i].start_position[0], self.vehicles_interference[i].start_position[1]-dirta_distance]
                self.vehicles_interference[i].start_position = end_position

            elif self.vehicles_interference[i].direction == 'u':
                end_position = [self.vehicles_interference[i].start_position[0], self.vehicles_interference[i].start_position[1]+dirta_distance]
                self.vehicles_interference[i].start_position = end_position

            elif self.vehicles_interference[i].direction == 'r':
                end_position = [self.vehicles_interference[i].start_position[0]+dirta_distance, self.vehicles_interference[i].start_position[1]]
                self.vehicles_interference[i].start_position = end_position

            elif self.vehicles_interference[i].direction == 'l':
                end_position = [self.vehicles_interference[i].start_position[0]-dirta_distance, self.vehicles_interference[i].start_position[1]]
                self.vehicles_interference[i].start_position = end_position

    def add_new_vehicles_by_number(self, n):
        string = 'dulr'
        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))

            start_position = [self.down_lanes[ind],self.BS_width+7, 2*self.BS_width-np.random.randint(10, 15)]
            start_direction = 'd'  # velocity: 10 ~ 15 m/s, random
            vel_v=np.random.randint(10, 15)

            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

            start_position = [self.up_lanes[ind], np.random.randint(10, 15)]
            start_direction = 'u'
            vel_v = np.random.randint(10, 15)
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

            start_position = [2*self.BS_width-np.random.randint(10, 15), self.left_lanes[ind]]
            start_direction = 'l'
            vel_v = np.random.randint(10, 15)
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

            start_position = [np.random.randint(10, 15), self.right_lanes[ind]]
            start_direction = 'r'
            vel_v = np.random.randint(10, 15)
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

        for j in range(int(self.n_Veh % 4)):
            ind = np.random.randint(0, len(self.down_lanes))
            str = random.choice(string)
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_direction = str
            vel_v = np.random.randint(10, 15)# velocity: 10 ~ 15 m/s, random
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

    def add_interference_vehicles_by_number(self, n):
        string = 'dulr'
        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))

            start_position = [self.down_lanes[ind], self.BS_width+7, 2*self.BS_width-np.random.randint(10, 15)]
            start_direction = 'd'  # velocity: 10 ~ 15 m/s, random
            vel_v = np.random.randint(10, 15)

            self.vehicles_interference.append(Vehicle(start_position, start_direction, vel_v))

            start_position = [self.up_lanes[ind], np.random.randint(10, 15)]
            start_direction = 'u'
            vel_v = np.random.randint(10, 15)
            self.vehicles_interference.append(Vehicle(start_position, start_direction, vel_v))

            start_position = [2*self.BS_width-np.random.randint(10, 15), self.left_lanes[ind]]
            start_direction = 'l'
            vel_v = np.random.randint(10, 15)
            self.vehicles_interference.append(Vehicle(start_position, start_direction, vel_v))

            start_position = [np.random.randint(10, 15), self.right_lanes[ind]]
            start_direction = 'r'
            vel_v = np.random.randint(10, 15)
            self.vehicles_interference.append(Vehicle(start_position, start_direction, vel_v))

        for j in range(int(self.n_Veh % 4)):
            ind = np.random.randint(0, len(self.down_lanes))
            str = random.choice(string)
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_direction = str  # velocity: 10 ~ 15 m/s, random
            vel_v = np.random.randint(10, 15)  # velocity: 10 ~ 15 m/s, random
            self.vehicles_interference.append(Vehicle(start_position, start_direction, vel_v))

    def overall_channel(self):
        """The combined channel"""
        self.V2I_pathloss = np.zeros((len(self.vehicles)))
        self.V2I_h = np.zeros((len(self.vehicles)))
        self.vel_v = np.zeros((len(self.vehicles)))
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        for i in range(len(self.vehicles)):
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].start_position)
            self.vel_v[i] = self.vehicles[i].velocity
        self.V2I_overall_W = 1/np.abs(1/np.power(10, self.V2I_pathloss / 10))
        self.V2I_channels_abs = 10 * np.log10(self.V2I_overall_W)+self.V2I_Shadowing

        return self.V2I_channels_abs, self.vel_v

    def overall_channel_interference(self):
        """The combined channel"""
        self.V2I_pathloss = np.zeros((len(self.vehicles)))
        self.V2I_pathloss_interference = np.zeros((len(self.vehicles_interference)))
        self.V2I_h_interference = np.zeros((len(self.vehicles_interference)))
        self.vel_v_interference = np.zeros((len(self.vehicles_interference)))
        self.V2I_channels_abs_interference = np.zeros((len(self.vehicles)))
        self.V2I_Shadowing_interference = np.random.normal(0, 8, len(self.vehicles))
        for i in range(len(self.vehicles_interference)):
            self.V2I_pathloss_interference[i] = self.V2Ichannels.get_path_loss(self.vehicles_interference[i].start_position)
            self.vel_v_interference[i] = self.vehicles_interference[i].velocity
        self.V2I_overall_W_interference = 1/np.abs(1/np.power(10, self.V2I_pathloss_interference / 10))
        self.V2I_channels_abs_interference = 10 * np.log10(self.V2I_overall_W_interference)+self.V2I_Shadowing_interference  #dB

        return self.V2I_channels_abs_interference, self.V2I_overall_W_interference, self.vel_v_interference

    def Compute_Performance_Reward_Train(self, action_pf, h_i_dB, h_i_W, vel_v, lambda_1, lambda_2):
        p_selection = action_pf[:, 0].reshape(len(self.vehicles), 1)
        p_selection_interference = [random.uniform(5, 20) for _ in range(4)]

        V2I_Signals = np.zeros(self.n_Veh)
        V2I_Interference = 0

        for i in range(len(self.vehicles)):

            for j in range(int(self.n_interference_vehicle)):

                V2I_Interference += 10**((p_selection_interference[j])/10)

            V2I_Interference = V2I_Interference + self.sig2

            V2I_Signals[i] = 10 ** ((p_selection[i] - self.V2I_channels_abs[i]+ self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

        E_total, tran_success, c_total, reward, comp_n_list = self.beta_all.beta_allocation(action_pf, h_i_dB, vel_v, lambda_1, lambda_2, V2I_Interference)

        return E_total, tran_success, c_total, reward, comp_n_list


    def act_for_training(self, action_pf, h_i_dB, vel_v, lambda_1, lambda_2):
        E_total, tran_success, c_total, reward, comp_n_list = self.Compute_Performance_Reward_Train(action_pf, h_i_dB, vel_v, lambda_1, lambda_2)

        return beta_i_list, E_total, tran_success, c_total, reward, comp_n_list

    def new_random_game(self, n_Veh=0):
        self.vehicles = []
        self.vehicles_interference = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(int(self.n_Veh / 4))
        self.add_interference_vehicles_by_number(int(self.n_Veh / 4))
        self.overall_channel()
        self.overall_channel_interference()

