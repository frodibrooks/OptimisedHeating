import os
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import gym.spaces 
from epynet import Network

class wds():
    def __init__(self,
            wds_name                = "Vatnsendi_no_pattern",
            speed_increment         = 0.05,
            episode_len             = 10,
            pump_groups             = [['17', '10', '25', '26','27']],
            total_demand_lo         = 0.7,
            total_demand_hi         = 1.3,
            reset_orig_pump_speeds  = False,
            reset_orig_demands      = False,
            seed                    = None 
             ):
        

        # Þetta er seed fyrir random draslið, gefur manni valkost að fá sömu niðutstöður, ef þú setur sama seed, td 8 færðu það sama
        self.seedNum = seed
        if self.seedNum:
            np.random.seed(self.seedNum)
        else:
            np.random.seed()

        # erum að ná í path á current python file, fara einn tilbaka í direcortyið og leita að water_network
        pathToRoot  = os.path.dirname(os.path.realpath(__file__))
        pathToWDS   = os.path.join(pathToRoot, "water_network", wds_name + ".inp")  

        self.wds            = Network(pathToWDS)        
        self.demandDict     = self.build_demand_dict()
        self.pumpGroups     = pump_groups
        self.pump_speeds    = np.ones(shape = (len(self.pumpGroups)),dtype=np.float32)
        self.pumpEffs       = np.empty(shape=(len(self.pumpGroups)),dtype=np.float32)  

        nomHCurvePtsDict, nomECurvePtsDict = self.get_performance_curve_points()
        nomHCurvePolyDict       = self.fit_polynomials(
                                            nomHCurvePtsDict,
                                            degree = 2,
                                            encapulated = True)

        self.sumOfDemands       = sum(
            [demand for demand in self.wds.junctions.basedemand])
        
        self.demandRandomizer   = self.build_truncnorm_randomizer(
                                        lo=0.7,hi=1.3, mu = 1.0, sigma = 1.0)



        # Theoretical bounds of {head, efficiency}
        peak_heads   = []
        for key in nomHCurvePolyDict.keys():
            max_q       = np.max(nomHCurvePtsDict[key][:,0])
            opti_result = minimize(
                -nomHCurvePolyDict[key], x0=1, bounds=[(0, max_q)])
            peak_heads.append(nomHCurvePolyDict[key](opti_result.x[0]))
        peak_effs  = []
        for key in nomHCurvePolyDict.keys():
            max_q       = np.max(nomHCurvePtsDict[key][:,0])
            q_list      = np.linspace(0, max_q, 10)
            head_poli   = nomHCurvePolyDict[key]
            eff_poli    = self.nomECurvePoliDict[key]
            opti_result = minimize(-eff_poli, x0=1, bounds=[(0, max_q)])
            peak_effs.append(eff_poli(opti_result.x[0]))
        self.peakTotEff = np.prod(peak_effs)

        # Reward control
        self.dimensions     = len(self.pumpGroups)
        self.episodeLength  = episode_len
        self.headLimitLo    = 35
        self.headLimitHi    = 120
        self.maxHead        = np.max(peak_heads)
        self.rewScale       = [5,8,3] # eff, head, pump skoða þetta seinna 
        self.baseReward     = +1
        self.bumpPenalty    = -1
        self.distanceRange  = 0.5
        self.wrongMovePenalty   = -1
        self.lazinessPenalty    = -1
        # ----- ----- ----- ----- -----
        # Tweaking reward
        # ----- ----- ----- ----- -----
        #maxReward   = 5
        # ----- ----- ----- ----- -----
        self.maxReward   = +1
        self.minReward   = -1

        # Inner variables
        self.spec           = None
        self.metadata       = None
        self.totalDemandLo  = total_demand_lo
        self.totalDemandHi  = total_demand_hi
        self.speedIncrement = speed_increment
        self.speedLimitLo   = 0.7
        self.speedLimitHi   = 1.3
        self.validSpeeds   = np.arange(
                                self.speedLimitLo,
                                self.speedLimitHi+.001,
                                self.speedIncrement,
                                dtype=np.float32)
        self.resetOrigPumpSpeeds= reset_orig_pump_speeds
        self.resetOrigDemands   = reset_orig_demands
        self.optimized_speeds   = np.empty(shape=(len(self.pumpGroups)),
                                    dtype=np.float32)
        self.optimized_speeds.fill(np.nan)
        self.optimized_value    = np.nan
        self.previous_distance  = np.nan
        # initialization of {observation, steps, done}
        observation = self.reset(training=False)
        self.action_space   = gym.spaces.Discrete(2*self.dimensions+1)
        self.observation_space  = gym.spaces.Box(
                                    low     = -1,
                                    high    = +1,
                                    shape   = (len(self.wds.junctions)+len(self.pumpGroups),),
                                    dtype   = np.float32)




        def step(self, action, training=True):
            "Reward computed from the euclidean distance between the speed of pumps and the optimized speed"
            self.step       += 1
            self.done       = (self.step == self.self.episodeLength) # becomes true when have stepped the number of episodes
            group_id        = action // 2
            command         = action % 2
            if training: # training duh
                if group_id != self.dimensions:
                    self.n_siesta = 0
                    first_pump_in_grp = self.wds.pumps[self.pumpGroups[group_id][0]]
                    if command == 0:
                        if first_pump_in_grp.speed < self.speedLimitHi:
                            for pump in self.pumpGroups[group_id]:
                                self.wds.pumps[pump].speed += self.speedIncrement
                            self.update_pump_speeds()
                            distance = np.linalg.norm(self.optimized_speeds - self.pump_speeds)
                            if distance < self.previous_distance:  # reward for getting closer to the optimized speed
                                reward = distance * self.baseReward / self.distanceRange / self.maxReward
                            else: 
                                reward = self.wrongMovePenalty
                            self.previous_distance = distance
                        else:
                            self.n_bumb += 1
                            reward = self.bumpPenalty
                    else:
                        if first_pump_in_grp.speed > self.speedLimitLo:
                            for pump in self.pumpGroups[group_id]:
                                self.wds.pumps[pump].speed -= self.speedIncrement
                            self.update_pump_speeds()
                            distance = np.linag.norm(self.optimized_speeds - self.pump_speeds)
                            if distance < self.previous_distance:  # reward for getting closer to the optimized speed
                                reward = distance * self.baseReward / self.distanceRange / self.maxReward
                            else:
                                reward = self.wrongMovePenalty
                            self.previous_distance = distance
                        else:
                            self.n_bumb += 1
                            reward = self.bumpPenalty
                else:
                    self.n_siesta += 1
                    value = self.get_state_value()
                    if self.n_siesta == 3:
                        self.done = True
                        if value/self.optimized_value > 0.98:
                            reward = 5/self.maxReward
                        else:
                            reward = self.lazinessPenalty
                    else:
                        if value/self.optimized_value > 0.98:
                            reward = self.n_siesta * self.baseReward
                        else:
                            reward = self.lazinessPenalty
                self.wds.solve()
            else: # Not training, using 
                if group_id != self.dimensions:
                    self.n_siestas = 0
                    first_pump_in_grp = self.wds.pumps[self.pumpGroups[group_id][0]]
                    if command == 0:
                        if first_pump_in_grp.speed < self.speedLimitHi:
                            for pump in self.pumpGroups[group_id]:
                                self.wds.pumps[pump].speed += self.speedIncrement   

                        else:
                            self.n_bump += 1
                    
                    else:
                        if first_pump_in_grp.speed > self.speedLimitLo:
                            for pump in self.pumpGroups[group_id]:
                                self.wds.pumps[pump].speed -= self.speedIncrement
                        else:
                            self.n_bum











        def build_demand_dict(self):
            pass
            return

        def get_performance_curve_points(self):
            pass
            return

        def fit_polynomials(self):
            pass

        def build_truncnorm_randomizer(self):
            pass
                                    
                                    
                            
