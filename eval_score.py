import os
import sys
import numpy as np
import json
import time
import warnings
import argparse
import grid2op
from grid2op.dtypes import dt_int
from grid2op.utils import EpisodeStatistics, ScoreL2RPN2023
from lightsim2grid import LightSimBackend
from Agent.PowerGridAgent import PowerGridAgent
warnings.filterwarnings("ignore", category=UserWarning)
bk_cls = LightSimBackend



def make_agent(env,CONFIG):
    agent = PowerGridAgent(env,
                           env.action_space,
                           config=CONFIG,
                          verbose=True)
    return agent

def run_file(env_name, config):
        
    with open(config, "r") as config_file:
        CONFIG = json.load(config_file)
            
    print(f"optimCVXPY with config: {CONFIG}")


    env = grid2op.make(env_name, backend=bk_cls())
    agent =make_agent(env,CONFIG)

    with open(f"{env_name}/config_test.json", "r") as f:
        config = json.load(f)
    
    max_int = np.iinfo(dt_int).max
    env_seeds =  [int(config["episodes_info"][os.path.split(el)[-1]]["seed"]) for el in sorted(env.chronics_handler.real_data.subpaths)]
    print("\n env_seeds:",env_seeds) 
    nb_scenario =len(env.chronics_handler.subpaths)  
    print("")
    print(f"Env evaluation: {env_name}: {nb_scenario}")


        
    my_score = ScoreL2RPN2023(env,
                              nb_scenario=nb_scenario,
                              env_seeds=env_seeds[:nb_scenario],
                              agent_seeds=[0 for _ in range(nb_scenario)],
                              verbose=1,
                              nb_process_stats=64,
                              weight_op_score=0.6,
                              weight_assistant_score=0.25,
                              weight_nres_score=0.15)
                        
    res = my_score.get(agent)
    print("\n Score:", res)
    return res



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run agent with file", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-env", help="file env path")
    parser.add_argument("-config", help="file config path")
    args = parser.parse_args()
    
    t0 = time.time()
    run_file(args.env, args.config)
    t1 = time.time()
    print("\n Time running", t1 - t0)