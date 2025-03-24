try:
    import argparse
    import os
    import glob
    import yaml
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from stable_baselines.deepq.policies import FeedForwardPolicy
    from stable_baselines import DQN
    from stable_baselines.common.schedules import PiecewiseSchedule
    from pump_env import wds


    print("successfully imported all libraries")
except ImportError as e:
    print("Import Error: ", e)



parser = argparse.ArgumentParser()
parser.add_argument('--params', default='Vatnsendi', help='Name of the YAML file')
parser.add_argument('--seed',default=None, type=int, help='Random seed for the optimization methods')
parser.add_argument('--nproc',default=1, type=int, help='Number of processes to use')
parser.add_argument('--tstsplit',default = 20, type=int, help='Percentage of scences moved from validation to testing')
args = parser.parse_args()


pathToRoot = os.path.dirname(os.path.realpath(__file__))