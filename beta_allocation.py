import math
import numpy as np
import random


class BetaAllocation:
    def __init__(self):
        self.Z = 11.2
        self.B = 2000000
        self.h_n_dB = 20 * math.log10((4 * math.pi * 200 * 915e6) / 3e8)

        self.N_0 = 10 ** (-114 / 10)
        self.T_n = 0.5
        self.data_t = 0.02
        self.k = 1e-27
        self.r_n = 1600
        self.D_n = 1500
        self.q_tao = 0.2
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 11
        self.m = 0.023
        self.x_i = 0.05

    def beta_allocation(self, p_f, h_i_dB, vel_v, lambda_1, lambda_2, V2I_Interference):
        beta_i_list = []
        comp_time_list = []
        comp_n_list = []
        trans_time_list = []

        sum_beta = []
        E_1 = 0
        max_t = 0
        count = 0
        count_c_1 = 0
        c_total = 0
        V2I_Signals_dB = np.zeros(len(vel_v))
        V2I_Signals_W = np.zeros(len(vel_v))

        for p_f_1 in p_f:
            V2I_Signals_dB[count] = p_f_1[0] - h_i_dB[count] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure
            V2I_Signals_W[count] = 10 ** (
                        (p_f_1[0] - h_i_dB[count] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

            A_n = (lambda_1 * p_f_1[0] * self.Z) / self.B / math.log(1 + V2I_Signals_W[count] / V2I_Interference,
                                                                     math.e)
            B_n = lambda_1 * self.k * (self.T_n - self.data_t)
            C_n = lambda_1 * self.k * self.D_n * self.r_n

            E_n = lambda_2 * self.Z / self.B / math.log(1 + V2I_Signals_W[count] / V2I_Interference, math.e)
            F_n = lambda_2 * self.D_n * self.r_n

            tao_i = (3 * B_n * math.pow(p_f_1[1], 4) - 2 * C_n * math.pow(p_f_1[1], 3)) / F_n
            sum_beta.append(math.pow(A_n + tao_i * E_n, 1 / 2))

            count += 1

        for x in range(len(vel_v)):
            beta_i = sum_beta[x] / sum(sum_beta)
            beta_i_list.append(beta_i)

        count = 0
        V2I_Signals_W = np.zeros(len(vel_v))
        total_num_comp=0
        deta=0
        for p_f_2 in p_f:

            V2I_Signals_W[count] = 10 ** (
                        (p_f_2[0] - h_i_dB[count] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

            p_min = 10 * np.log10(-1 * self.m * (V2I_Interference) / math.log(1 - self.q_tao, math.e)) + h_i_dB[
                count] - self.vehAntGain - self.bsAntGain + self.bsNoiseFigure

            temp_p_f_2 = p_f_2[0]

            if p_f_2[0] <= p_min:
                p_f_2[0] = p_min
            deta = p_f_2[0] - temp_p_f_2
            q = 1 - math.exp(-1 * self.m * (V2I_Interference / (10 ** (
                    (p_f_2[0] - h_i_dB[count] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10))))

            random_value = random.random()
            c_total += 1
            if random_value > q:
                count_c_1 += 1

            beta_i = beta_i_list[count]
            A_n = (lambda_1 * p_f_2[0] * self.Z) / self.B / math.log(1 + V2I_Signals_W[count] / V2I_Interference,
                                                                     math.e)
            B_n = lambda_1 * self.k * (self.T_n - self.data_t)
            C_n = lambda_1 * self.k * self.D_n * self.r_n

            E_n = lambda_2 * self.Z / self.B / math.log(1 + V2I_Signals_W[count] / V2I_Interference, math.e)
            F_n = lambda_2 * self.D_n * self.r_n

            E_1 += A_n / beta_i + B_n * math.pow(p_f_2[1], 3) - C_n * math.pow(p_f_2[1], 2)

            c_ui = beta_i * self.B * math.log(1 + V2I_Signals_W[count] / V2I_Interference, math.e)
            trans_time = self.Z / c_ui
            trans_time_list.append(trans_time)
            comp_time = self.D_n * self.r_n / p_f_2[1]
            comp_time_list.append(comp_time)
            comp_n = (self.T_n - self.data_t) / comp_time

            comp_n_list.append(comp_n)
            total_num_comp = sum(comp_n_list)

            count += 1

            temp_t = E_n / beta_i + F_n / p_f_2[1]
            if temp_t > max_t:
                max_t = temp_t

        reward = E_1 + max_t + self.x_i * deta -0.00005*np.round(total_num_comp)

        return count_c_1, c_total, reward, comp_n_list
