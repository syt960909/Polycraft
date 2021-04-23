# run this file to see how trained models work.

from PolycraftEnv import PolycraftHGEnv
from wrapper import wrap_func
# from model import QNetwork
# from Agent import Agent
from model_pixel import QNetwork
from Agent_pixel import Agent
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
from collections import namedtuple,deque
import json
from config import Config
from config_ import Config_

# def randomize(domain_file):

#     domain_file = domain_file[3:]
#     with open(domain_file, 'r') as f:
#         setting = json.load(f)
    
#     num = [int(i) for i in np.random.randint(10, 30, 6)]
#     angle = int(np.random.randint(0, 360, 1)[0])
#     setting['features'][0]['pos'] = [num[0], 4, num[1]]
#     setting['features'][0]['lookDir'] = [0, angle, 0]
#     setting['features'][2]['pos'] = [num[2], 4, num[3]]
#     setting['features'][4]['blockList'][0]['blockPos'] = [num[4], 4, num[5]]
#     with open('../polycraft_game/experiments/hgv1_1.json', 'w') as f:
#         f.write(json.dumps(setting))
def randomize(domain_file_name):

    domain_file = random.choice(domain_file_name)

    #domain_file = domain_file[3:]

    #with open("C:/Users/ysun465/Downloads/PAL-master/PolycraftAIGym/test/"+domain_file, 'r') as f:
     #   setting = json.load(f)

    #location = [3, 4, 6, 7, 10, 20, 26]


    #num = [int(i) for i in np.random.choice(location, 2)]
    #print(num)
    #location_ =  [5,10,30,45]
    #[45,5][10,5][30,10][5,10]
    #num_ = random.sample(location_,2)
    #print(num_)
    #angle = int(np.random.choice([0,90, 180, 270], 1))
    #print(angle)
    #setting['features'][0]['pos'] = [num[0], 4, num[1]]
    #setting['features'][0]['lookDir'] = [0, angle, 0]
    #setting['features'][2]['pos'] = [num[0], 4, num[1]]

    #setting['features'][16]['blockList'][0]['blockPos'] = [num_[0], 4, num_[1]]
    #with open("C:/Users/ysun465/Downloads/PAL-master/PolycraftAIGym/test/"+domain_file,'w') as f:
     #   f.write(json.dumps(setting))
    return "C:/Users/ysun465/Downloads/PAL-master/PolycraftAIGym/HUGA_L00_T01_S01_VIRGIN_X1000_U9999_V1" \
           "/" + domain_file

def run(opt,opt_,domain_file_name):

    domain_file = randomize(domain_file_name)

    env = PolycraftHGEnv(opt=opt)
    env = wrap_func(env)

    scores = []
    eps = 0.1
    state = env.reset(domain_file)
    dones = 0
    for i_episode in range(1,201):
        
        domain_file = randomize(domain_file_name)
        print()

        score = 0                                          # initialize the score
        # agent = Agent(state_size = opt.state_size, action_size = opt.action_Size, seed = 0, opt=opt)
        agent = Agent(num_input_chnl = opt.num_input_chnl, action_size = opt.action_Size, seed = 0, opt=opt)
        counter = 0
        while True:
            # action = agent.select_act(state,eps)           # select an action
            aug_state = agent.augment_state(state)
            action = agent.select_act(aug_state,eps)           # select an action
            next_state, reward, done, done_,info = env.step(action)   # get the next state
            loss = agent.step(state,action,reward,next_state,done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            counter += 1
            if done:
                agent = Agent(num_input_chnl=opt.num_input_chnl, action_size=opt.action_Size, seed=0, opt=opt_)
                #counter = 0
                # exit loop if episode finished
                while True:
                    # action = agent.select_act(state,eps)           # select an action
                    aug_state = agent.augment_state(state)
                    action = agent.select_act(aug_state, eps)  # select an action
                    next_state, reward,done, done_, info = env.step(action)  # get the next state
                    loss = agent.step(state, action, reward, next_state, done)
                    score += reward  # update the score
                    state = next_state  # roll over the state to next time step
                    counter += 1
                    if done_ or counter == 1200:  # exit loop if episode finished
                        state = env.reset(domain_file)
                        if done_:
                            dones += 1
                        break
                break
        scores.append(score)
        print('Success times: {}/{}'.format(dones, i_episode))
        #print("Score: {}".format(score))
    print('Avg score:',np.mean(scores))
    print('Success rate: ',dones/200)

if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='single', dest='mode')
    args = parser.parse_args()

    opt = Config(args)
    opt_ = Config_(args)

    domain_file_name = []
    path = "C:/Users/ysun465/Downloads/PAL-master/PolycraftAIGym/test"
    dirs = os.listdir(path)
    for i in dirs:
        if os.path.splitext(i)[1] == ".json":
            domain_file_name.append(i)
    #random.seed(10)
    #domain_file_name = random.sample(domain_file_name, 500)

    run(opt,opt_,domain_file_name)
    