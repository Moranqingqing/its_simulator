from os import error
import numpy as np
from gym.spaces import Box

from wolf.world.environments.wolfenv.agents.connectors.reward.reward_connector import \
    RewardConnector

import logging

LOGGER = logging.getLogger(__name__)

class VehRewardConnector(RewardConnector):
    def __init__(self, connectors_ids, kernel=None, **kwargs):
        self.veh_ids = connectors_ids
        reward_space = Box(-np.inf, np.inf, shape=(len(self.veh_ids), ), dtype=np.float32)

        self.ttc_arr = []
        self.f_ttc_arr = []
        self.eff_arr = []
        self.f_eff_arr = []
        self.jerk_arr = []
        self.f_jerk_arr = []
        self.total_reward_arr = []
        self.f_eff_type = 1
        
        self.sim_step = 0.1 # Default be 0.1s (time interval of simulation)
        if 'sim_step' in kwargs:
            self.sim_step = kwargs['sim_step'] 
        
        if 'W1' in kwargs:
            self.W1 = kwargs['W1']
        else:
            self.W1 = [10, 0]

        if 'W2' in kwargs:
            self.W2 = kwargs['W2']
        else:
            self.W2 = [2, 0]

        if 'W3' in kwargs:
            self.W3 = kwargs['W3']       
        else:
            self.W3 = [50, 50]

        if 'W4' in kwargs:
            self.W4 = kwargs['W4']
        else:
            self.W4 = [0, 0]
        
        if 'f_eff_type' in kwargs:
            self.f_eff_type = kwargs['f_eff_type']

        super().__init__(reward_space=reward_space, kernel=kernel)

    def a_compute(self):
        curr_obs = self.agent.observations[-1]
        act_records = self.agent.actions
        curr_act = act_records[-1]
        # last_act = act_records[-2] if len(act_records) > 1 else 0 
        last_act = curr_obs[4]

        # safety
        veh_id = self.veh_ids[0]
        # veh_speed = self._kernel.get_vehicle_speed(veh_id)
        veh_speed = curr_obs[0]
        if veh_speed < 0:
            veh_speed=0
        lead_veh = self._kernel.get_vehicle_leader(veh_id)
        if lead_veh is None:
            lead_veh = self._kernel.get_lane_leaders(veh_id)[0]
            lead_veh_speed = self._kernel.get_vehicle_speed(lead_veh)
            # Get the headway
            distance_headway = self._kernel.get_lane_headways(veh_id)[0]
        else:
            lead_veh_speed = self._kernel.get_vehicle_speed(lead_veh)
            # Get the headway
            distance_headway = self._kernel.get_vehicle_headway(veh_id)

        # rel_speed = veh_speed - lead_veh_speed
        rel_speed = -curr_obs[1]
        distance_headway = curr_obs[2]
        f_ttc=0
        # Calculate time to collision (TTC)
        if rel_speed == 0:
            ttc = 100 # Large value would make f_ttc be zero
        # if distance_headway<5:
        #     ttc = -100
        else:
            ttc = distance_headway / rel_speed

        if ttc >= 0 and ttc <= 4:
            f_ttc = np.log(ttc / 4)
        # =============================================
        # print('ttc',ttc)
        # print('headway',distance_headway)
        # efficiency
        if self.f_eff_type == 1:
            time_headway = np.inf
            if veh_speed <= 1e-10:
                f_eff = 0
            else:
                time_headway = distance_headway / veh_speed
                # f_eff = np.exp(-time_headway) # ttc >= 0
                mu = 0.4226
                sigma = 0.4365
                if np.isnan(np.log(time_headway)):
                    f_eff = 0
                else:
                    f_eff = np.exp(-(np.log(time_headway) - mu)**2/(2*sigma**2)) /\
                        (time_headway*sigma*np.sqrt(2*np.pi))
        elif self.f_eff_type == 2:
            time_headway = np.inf
            if veh_speed <= 1e-10:
                f_eff = 0
            else:
                time_headway = distance_headway / veh_speed
                # f_eff = np.exp(-time_headway) # ttc >= 0
                sigma = 0.4365
                f_eff = efficiency_reward(time_headway, mode=0.8, sigma=sigma)
        elif self.f_eff_type == 3:
            time_headway = np.inf
            if veh_speed <= 1e-10:
                f_eff = 0
            else:
                time_headway = distance_headway / veh_speed
                if time_headway < 0:
                    f_eff = 0
                else:
                    f_eff = np.exp(-time_headway) * 0.659
        else:
            raise NotImplementedError(f"Invalid f_eff_type: {self.f_eff_type}")

        # =============================================
        # comfort
        f_comfort = 0
        jerk = (curr_act - last_act)[0] / self.sim_step
        f_comfort = -jerk**2 / 3600
        # =============================================


        # speed limit
        # speed_limit = curr_obs[-1]
        speed_limit = curr_obs[3]
        # print('headway',distance_headway)
        # Linear Function as penalty
        # f_speed = -np.abs(veh_speed - speed_limit) / speed_limit
        f_speed = -(veh_speed / speed_limit - 1) ** 2
        # =============================================
        # print('fspeed',f_speed)
        if distance_headway <= 200:
            w_1 = self.W1[0]
            w_2 = self.W2[0]
            w_3 = self.W3[0]
            w_4 = self.W4[0]
        else:
            w_1 = self.W1[1]
            w_2 = self.W2[1]
            w_3 = self.W3[1]
            w_4 = self.W4[1]

        # if f_speed<-1000:
        #     f_speed=-1000

        # if w_4 * f_speed < -1000:
        #     f_speed = -5 / w_4


        reward_sum = w_1 * f_ttc + w_2 * f_eff + w_3 * f_comfort + w_4 * f_speed
        
        # FIXME: Remove this debug msg
        # print("Debug msg")
        # print(f"Veh id: {veh_id}")
        # print("Observation:")
        # print('veh speed',veh_speed)
        # print(f"veh_speed: {curr_obs[0]}, rel_speed: {rel_speed}, "+ 
        #       f"distance_headway: {distance_headway}, speed_limit: {curr_obs[3]}")
        # print(f"ttc: {ttc}, time_headway: {time_headway}, lead_vehicle: {lead_veh}")
        # print(f"Action took: {curr_act}")
        # print(f"f_ttc: {w_1*f_ttc}, f_eff: {w_2 * f_eff}, f_comfort: {f_comfort}, f_speed: {w_4 * f_speed}")
        # print(f"Sum of Reward: {reward_sum}")

        # record the result to some metrics place
        # TODO: Implement the info metrics connector
        self.ttc_arr.append(ttc)
        self.f_ttc_arr.append(f_ttc)
        self.eff_arr.append([distance_headway, veh_speed, lead_veh_speed, time_headway])
        self.f_eff_arr.append(f_eff)
        self.jerk_arr.append(jerk)
        self.f_jerk_arr.append(f_comfort)
        self.total_reward_arr.append(reward_sum)


        # calculating the action space
        return np.array([reward_sum])

    def compute(self):
        reward = self.a_compute()
        if reward is None:
            raise ValueError("Reward is None")
        
        if not self._reward_space.contains(reward):
            raise ValueError("Reward {} outside expected value range {}"
                             .format(reward, self._reward_space))

        return reward[0]

class BCMVehRewardConnector(RewardConnector):
    def __init__(self, connectors_ids, kernel=None, **kwargs):
        self.veh_ids = connectors_ids
        reward_space = Box(-np.inf, np.inf, shape=(len(self.veh_ids), ), dtype=np.float32)

        self.ttc_arr = []
        self.f_ttc_arr = []
        self.eff_arr = []
        self.f_eff_arr = []
        self.jerk_arr = []
        self.f_jerk_arr = []
        self.total_reward_arr = []
        
        self.sim_step = 0.1 # Default be 0.1s (time interval of simulation)
        self.f_eff_type = 1
        if 'sim_step' in kwargs:
            self.sim_step = kwargs['sim_step'] 
        
        if 'W1' in kwargs:
            self.W1 = kwargs['W1']
        else:
            self.W1 = [10, 0]

        if 'W2' in kwargs:
            self.W2 = kwargs['W2']
        else:
            self.W2 = [2, 0]

        if 'W3' in kwargs:
            self.W3 = kwargs['W3']       
        else:
            self.W3 = [50, 50]

        if 'W4' in kwargs:
            self.W4 = kwargs['W4']
        else:
            self.W4 = [1, 1]

        if 'f_eff_type' in kwargs:
            self.f_eff_type = kwargs['f_eff_type']

        if 'follow_ttc_flag' in kwargs:
            self.follow_ttc_flag = kwargs['follow_ttc_flag']
        else:
            self.follow_ttc_flag = False

        super().__init__(reward_space=reward_space, kernel=kernel)

    def a_compute(self):
        curr_obs = self.agent.observations[-1]
        act_records = self.agent.actions
        curr_act = act_records[-1]
        # last_act = act_records[-2] if len(act_records) > 1 else 0 
        last_act = curr_obs[6]

        # Expand obs array
        veh_speed, rel_speed,\
        distance_headway, follow_rel_speed,\
        follow_distance_headway, speed_limit, last_act = curr_obs
        rel_speed = -rel_speed
        follow_rel_speed = -follow_rel_speed

        

        # safety
        veh_id = self.veh_ids[0]
        if veh_speed < 0:
            veh_speed=0
        lead_veh = self._kernel.get_vehicle_leader(veh_id)
        if lead_veh is None:
            lead_veh = self._kernel.get_lane_leaders(veh_id)[0]

        lead_veh_speed = veh_speed - rel_speed
        follow_veh_speed = veh_speed - follow_rel_speed

        f_ttc=0
        # Calculate time to collision (TTC)
        if rel_speed == 0:
            ttc = 100 # Large value would make f_ttc be zero
        # if distance_headway<5:
        #     ttc = -100
        else:
            ttc = distance_headway / rel_speed

        if ttc >= 0 and ttc <= 4:
            f_ttc = np.log(ttc / 4)

        f_follow_ttc = 0
        if follow_rel_speed == 0:
            follow_ttc = 100 # Large value would take f_follow_ttc be zero
        else:
            follow_ttc = -follow_distance_headway / follow_rel_speed
        
        if follow_ttc >= 0 and follow_ttc <= 4:
            f_follow_ttc = np.log(follow_ttc / 4)
        # =============================================
        # print('ttc',ttc)
        # print('headway',distance_headway)
        # efficiency
        if self.f_eff_type == 1:
            time_headway = np.inf
            if veh_speed <= 1e-10:
                f_eff = 0
            else:
                time_headway = distance_headway / veh_speed
                # f_eff = np.exp(-time_headway) # ttc >= 0
                mu = 0.4226
                sigma = 0.4365
                if np.isnan(np.log(time_headway)):
                    f_eff = 0
                else:
                    f_eff = np.exp(-(np.log(time_headway) - mu)**2/(2*sigma**2)) /\
                        (time_headway*sigma*np.sqrt(2*np.pi))

            # =============================================
            follow_time_headway = np.inf
            if follow_veh_speed <= 1e-10:
                f_follow_eff = 0
            else:
                follow_time_headway = follow_distance_headway / follow_veh_speed
                mu = 0.4226
                sigma = 0.4365
                if np.isnan(np.log(follow_time_headway)):
                    f_follow_eff = 0
                else:
                    f_follow_eff = np.exp(-(np.log(follow_time_headway) - mu)**2/(2*sigma**2)) /\
                        (follow_time_headway*sigma*np.sqrt(2*np.pi))
        elif self.f_eff_type == 2:
            time_headway = np.inf
            if veh_speed <= 1e-10:
                f_eff = 0
            else:
                time_headway = distance_headway / veh_speed
                # f_eff = np.exp(-time_headway) # ttc >= 0
                sigma = 0.4365
                f_eff = efficiency_reward(time_headway, mode=0.8, sigma=sigma)

            # =============================================
            follow_time_headway = np.inf
            if follow_veh_speed <= 1e-10:
                f_follow_eff = 0
            else:
                follow_time_headway = follow_distance_headway / follow_veh_speed
                sigma = 0.4365
                f_follow_eff = efficiency_reward(follow_time_headway, mode=0.8, sigma=sigma)
        elif self.f_eff_type == 3:
            time_headway = np.inf
            if veh_speed <= 1e-10:
                f_eff = 0
            else:
                time_headway = distance_headway / veh_speed
                if time_headway < 0:
                    f_eff = 0
                else:
                    f_eff = np.exp(-time_headway) * 0.659
            # =============================================
            follow_time_headway = np.inf
            if follow_veh_speed <= 1e-10:
                f_follow_eff = 0
            else:
                follow_time_headway = follow_distance_headway / follow_veh_speed
                if follow_time_headway < 0:
                    f_follow_eff = 0
                else:
                    f_follow_eff = np.exp(-follow_time_headway) * 0.659
        else:
            raise NotImplementedError(f"Invalid f_eff_type: {self.f_eff_type}")
            

        # =============================================
        # comfort
        f_comfort = 0
        jerk = (curr_act - last_act)[0] / self.sim_step
        f_comfort = -jerk**2 / 3600
        # =============================================


        # speed limit
        # speed_limit = curr_obs[-1]
        speed_limit = curr_obs[5]
        # print('headway',distance_headway)
        # Linear Function as penalty
        # f_speed = -np.abs(veh_speed - speed_limit) / speed_limit
        if veh_speed < speed_limit:
            f_speed = -(veh_speed - speed_limit) ** 2/100
        else:
            f_speed = -(veh_speed - speed_limit) ** 2/100
        # =============================================
        # print('fspeed',f_speed)
        if distance_headway <= 200:
            w_1 = self.W1[0]
            w_2 = self.W2[0]
            w_3 = self.W3[0]
            w_4 = self.W4[0]
        else:
            w_1 = self.W1[1]
            w_2 = self.W2[1]
            w_3 = self.W3[1]
            w_4 = self.W4[1]

        if f_speed<-1000:
            f_speed=-1000

        # if w_4 * f_speed < -1000:
        #     f_speed = -5 / w_4

        reward_sum = w_1 * f_ttc + w_2 * f_eff + w_2 * f_follow_eff + w_3 * f_comfort + w_4 * f_speed
        if self.follow_ttc_flag:
            reward_sum += w_1 * f_follow_ttc
        
        # FIXME: Remove this debug msg
        # print("Debug msg")
        # print(f"Veh id: {veh_id}")
        # print("Observation:")
        # print('veh speed',veh_speed)
        # print(f"veh_speed: {curr_obs[0]}, rel_speed: {rel_speed}, "+ 
        #       f"distance_headway: {distance_headway}, speed_limit: {curr_obs[3]}")
        # print(f"ttc: {ttc}, time_headway: {time_headway}, lead_vehicle: {lead_veh}")
        # print(f"Action took: {curr_act}")
        # print(f"f_ttc: {w_1*f_ttc}, f_eff: {w_2 * f_eff}, f_comfort: {f_comfort}, f_speed: {w_4 * f_speed}")
        # print(f"Sum of Reward: {reward_sum}")

        # record the result to some metrics place
        # TODO: Implement the info metrics connector
        self.ttc_arr.append(ttc)
        self.f_ttc_arr.append(f_ttc)
        self.eff_arr.append([distance_headway, veh_speed, lead_veh_speed, time_headway])
        self.f_eff_arr.append(f_eff)
        self.jerk_arr.append(jerk)
        self.f_jerk_arr.append(f_comfort)
        self.total_reward_arr.append(reward_sum)


        # calculating the action space
        return np.array([reward_sum])

    def compute(self):
        reward = self.a_compute()
        if reward is None:
            raise ValueError("Reward is None")
        
        if not self._reward_space.contains(reward):
            raise ValueError("Reward {} outside expected value range {}"
                             .format(reward, self._reward_space))

        return reward[0]


def efficiency_reward(headway, mode, sigma, scale=0.659):
    """Generate log normal distribution as reward function given the mode and sigma

    Arguments:
        mode {float} -- check https://en.wikipedia.org/wiki/Mode_(statistics)
        sigma {float} -- variance

    Keyword Arguments:
        scale {float} -- maximum value of the function (default: {0.659})
    """
    mu = np.log(mode) + sigma**2
    if np.isnan(np.log(headway)):
        r = 0
    else:
        r_max = np.exp(-(np.log(mode) - mu)**2/(2*sigma**2)) /\
            (mode*sigma*np.sqrt(2*np.pi))
        r = np.exp(-(np.log(headway) - mu)**2/(2*sigma**2)) /\
            (headway*sigma*np.sqrt(2*np.pi)) / r_max * scale
    return r