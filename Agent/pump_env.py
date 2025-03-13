import os
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import gym.spaces 
from epynet import Network

class wds():
    def __init__(self,
            wds_name = "Vatnsendi_no_pattern",
            speed_increment = 0.05,
            episode_length = 10,
            pump_groups = [['17', '10', '25', '26','27']],
            total_demand_lo = 0.3,
            total_demand_hi = 1.1,
            reset_orig_pump_speeds = False,
            reset_orig_demands = False,
            seed = None 
             ):
        

        # Þetta er seed fyrir random draslið, gefur manni valkost að fá sömu niðutstöður, ef þú setur sama seed, td 8 færðu það sama
        self.seedNum = seed
        if self.seedNum:
            np.random.seed(self.seedNum)
        else:
            np.random.seed()

        # erum að ná í path á current python file, fara einn tilbaka í direcortyið og leita að water_network
        pathToRoot = os.path.dirname(os.path.realpath(__file__))
        pathToWDS = os.path.join(pathToRoot, "water_network", wds_name + ".inp")  

        self.wds = Network(pathToWDS)                         
                                    
                                    
                            
