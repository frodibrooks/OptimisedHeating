# -* coding: utf-8 -*-
import os
import numpy as np
import gym.spaces
from epynet import Network

class wds():
    """Gym-like environment for water distribution systems."""
    def __init__(self,
            wds_name        = 'anytown_master',
            speed_increment = .05,
            episode_len     = 10,
            pump_groups     = [['78', '79']],
            total_demand_lo = .3,
            total_demand_hi = 1.1,
            reset_orig_pump_speeds  = False,
            reset_orig_demands      = False,
            seed    = None):

        self.seedNum   = seed
        if self.seedNum:
            np.random.seed(self.seedNum)
        else:
            np.random.seed()

        pathToRoot  = os.path.dirname(os.path.realpath(__file__))
        pathToWDS   = os.path.join(pathToRoot, 'water_networks', wds_name+'.inp')

        self.wds        = Network(pathToWDS)
        self.demandDict = self.build_demand_dict()
        self.pumpGroups = pump_groups
        self.pump_speeds= np.ones(shape=(len(self.pumpGroups)), dtype=np.float32)
        self.pumpEffs   = np.empty(shape=(len(self.pumpGroups)), dtype=np.float32)

        nomHCurvePtsDict, nomECurvePtsDict = self.get_performance_curve_points()
        nomHCurvePoliDict       = self.fit_polinomials(
                                    nomHCurvePtsDict,
                                    degree=2,
                                    encapsulated=True)
        self.nomECurvePoliDict  = self.fit_polinomials(
                                    nomECurvePtsDict,
                                    degree=4,
                                    encapsulated=True)
        self.sumOfDemands       = sum(
                            [demand for demand in self.wds.junctions.basedemand])
        self.demandRandomizer   = self.build_truncnorm_randomizer(
                                    lo=.7, hi=1.3, mu=1.0, sigma=1.0)

        # Theoretical bounds of {head, efficiency}
        peak_heads   = []
        for key in nomHCurvePoliDict.keys():
            max_q       = np.max(nomHCurvePtsDict[key][:,0])
            peak_heads.append(nomHCurvePoliDict[key](max_q))
        peak_effs  = []
        for key in nomHCurvePoliDict.keys():
            max_q       = np.max(nomHCurvePtsDict[key][:,0])
            q_list      = np.linspace(0, max_q, 10)
            head_poli   = nomHCurvePoliDict[key]
            eff_poli    = self.nomECurvePoliDict[key]
            peak_effs.append(eff_poli(max_q))
        self.peakTotEff = np.prod(peak_effs)

        # Reward control
        self.dimensions     = len(self.pumpGroups)
        self.episodeLength  = episode_len
        self.headLimitLo    = 15
        self.headLimitHi    = 120
        self.maxHead        = np.max(peak_heads)
        self.rewScale       = [5,8,3] # eff, head, pump
        self.baseReward     = +1
        self.bumpPenalty    = -1
        self.distanceRange  = .5
        self.wrongMovePenalty   = -1
        self.lazinessPenalty    = -1
        self.maxReward   = +1
        self.minReward   = -1

        # Inner variables
        self.spec           = None
        self.metadata       = None
        self.totalDemandLo  = total_demand_lo
        self.totalDemandHi  = total_demand_hi
        self.speedIncrement = speed_increment
        self.speedLimitLo   = .7
        self.speedLimitHi   = 1.2
        self.validSpeeds   = np.arange(
                                self.speedLimitLo,
                                self.speedLimitHi+.001,
                                self.speedIncrement,
                                dtype=np.float32)
        self.resetOrigPumpSpeeds= reset_orig_pump_speeds
        self.resetOrigDemands   = reset_orig_demands
        # initialization of {observation, steps, done}
        observation = self.reset(training=False)
        self.action_space   = gym.spaces.Discrete(2*self.dimensions+1)
        self.observation_space  = gym.spaces.Box(
                                    low     = -1,
                                    high    = +1,
                                    shape   = (len(self.wds.junctions)+len(self.pumpGroups),),
                                    dtype   = np.float32)

    def step(self, action, training=True):
        """ Reward computed from pump speed changes and penalties based on system behavior."""
        self.steps  += 1
        self.done   = (self.steps == self.episodeLength)
        group_id    = action // 2
        command     = action % 2
        if training:
            if group_id != self.dimensions:
                self.n_siesta       = 0
                first_pump_in_grp   = self.wds.pumps[self.pumpGroups[group_id][0]]
                if command == 0:
                    if first_pump_in_grp.speed < self.speedLimitHi:
                        for pump in self.pumpGroups[group_id]:
                            self.wds.pumps[pump].speed  += self.speedIncrement
                        self.update_pump_speeds()
                        reward  = self.baseReward
                    else:
                        self.n_bump += 1
                        reward  = self.bumpPenalty
                else:
                    if first_pump_in_grp.speed > self.speedLimitLo:
                        for pump in self.pumpGroups[group_id]:
                            self.wds.pumps[pump].speed  -= self.speedIncrement
                        self.update_pump_speeds()
                        reward  = self.baseReward
                    else:
                        self.n_bump += 1
                        reward  = self.bumpPenalty
            else:
                self.n_siesta   += 1
                value   = self.get_state_value()
                if self.n_siesta == 3:
                    self.done   = True
                    reward = self.lazinessPenalty
                else:
                    reward = self.lazinessPenalty
            self.wds.solve()
        else:
            if group_id != self.dimensions:
                self.n_siesta       = 0
                first_pump_in_grp   = self.wds.pumps[self.pumpGroups[group_id][0]]
                if command == 0:
                    if first_pump_in_grp.speed < self.speedLimitHi:
                        for pump in self.pumpGroups[group_id]:
                            self.wds.pumps[pump].speed  += self.speedIncrement
                    else:
                        self.n_bump += 1
                else:
                    if first_pump_in_grp.speed > self.speedLimitLo:
                        for pump in self.pumpGroups[group_id]:
                            self.wds.pumps[pump].speed  -= self.speedIncrement
                    else:
                        self.n_bump += 1
            else:
                self.n_siesta   += 1
                if self.n_siesta == 3:
                    self.done   = True
            self.wds.solve()
            reward  = self.get_state_value()
        observation = self.get_observation()
        return observation, reward, self.done, {}

    def reset(self, training=True):
        if training:
            if self.resetOrigDemands:
                self.restore_original_demands()
            else:
                self.randomize_demands()
            if self.resetOrigPumpSpeeds:
                initial_speed   = 1.
                for pump in self.wds.pumps:
                    pump.speed  = initial_speed
            else:
                for pump_grp in self.pumpGroups:
                    initial_speed   = np.random.choice(self.validSpeeds)
                    for pump in pump_grp:
                        self.wds.pumps[pump].speed  = initial_speed
        else:
            if self.resetOrigPumpSpeeds:
                initial_speed   = 1.
                for pump in self.wds.pumps:
                    pump.speed  = initial_speed
            else:
                for pump_grp in self.pumpGroups:
                    initial_speed   = np.random.choice(self.validSpeeds)
                    for pump in pump_grp:
                        self.wds.pumps[pump].speed  = initial_speed
        self.wds.solve()
        observation = self.get_observation()
        self.done   = False
        self.steps  = 0
        self.n_bump = 0
        self.n_siesta= 0
        return observation

    def get_observation(self):
        """ Returns system observation."""
        pump_speeds   = []
        for pump in self.wds.pumps:
            pump_speeds.append(pump.speed)
        junctions = [ junction.demand for junction in self.wds.junctions]
        pumps     = pump_speeds + junctions
        return np.array(pumps, dtype=np.float32)

    def update_pump_speeds(self):
        """ Update speeds for the pumps """
        for pump, speed in zip(self.wds.pumps.values(), self.pump_speeds):
            pump.speed = speed

    def build_demand_dict(self):
        """ Helper function to build demand dictionary."""
        demand_dict = {}
        for i, junc in enumerate(self.wds.junctions):
            demand_dict[i] = junc.demand
        return demand_dict

    def build_truncnorm_randomizer(self, lo, hi, mu, sigma):
        """ Helper function to build a truncated normal randomizer."""
        return scipy.stats.truncnorm(
            (lo - mu) / sigma, (hi - mu) / sigma, loc=mu, scale=sigma)
