import numpy as np
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
import torch
import json
import os
import pickle
import time
import random

from PolycraftEnv import PolycraftHGEnv
from wrapper import wrap_func
from config import Config

# from model import QNetwork
# from Agent import Agent
from model_pixel import QNetwork
from Agent_pixel import Agent

def randomize(domain_file_name):

    domain_file = random.choice(domain_file_name)

    #domain_file = domain_file[3:]
    with open("C:/Users/ysun465/Downloads/PAL-master/PolycraftAIGym/HUGA_L00_T01_S01_VIRGIN_X1000_U9999_V1/"+domain_file, 'r') as f:
        setting = json.load(f)

    location = [3, 4, 6, 7, 10, 20, 26]


    num = [int(i) for i in np.random.choice(location, 2)]
    #print(num)
    #location_ =  [[45,5],[10,5],[30,10],[5,10],[45,10]]
    #[45,5][10,5][30,10][5,10]
    #num_ = random.sample(location_,1)
    #print(num_[0])
    angle = int(np.random.choice([0,90, 180, 270], 1))
    print(angle)
    setting['features'][0]['pos'] = [num[0], 4, num[1]]
    setting['features'][0]['lookDir'] = [0, angle, 0]
    #setting['features'][2]['pos'] = [num[0], 4, num[1]]

    #setting['features'][16]['blockList'][0]['blockPos'] = [num_[0][0], 4, num_[0][1]]
    with open("C:/Users/ysun465/Downloads/PAL-master/PolycraftAIGym/HUGA_L00_T01_S01_VIRGIN_X1000_U9999_V1/"+domain_file,'w') as f:
        f.write(json.dumps(setting))

    return "C:/Users/ysun465/Downloads/PAL-master/PolycraftAIGym/HUGA_L00_T01_S01_VIRGIN_X1000_U9999_V1/" + domain_file
def dqn_unity(opt,domain_file_name):

    domain_file = randomize(domain_file_name)

    # environment
    env = PolycraftHGEnv(opt=opt)
    env = wrap_func(env)

    # agent
    agent = Agent(num_input_chnl = opt.num_input_chnl, action_size = opt.action_Size, seed = 0, opt=opt)
    
    # parameters
    scores = [] # list of scores from each episode
    losses = [] # list of losses
    score_window = deque(maxlen = 100) # a deque of 100 episode scores to average
    eps = opt.eps_start
    state = env.reset(domain_file)

    # create figure
    plt.figure(figsize=(6,3), dpi=80)
    plt.ion()
    ewma = lambda x, span=100: DataFrame({'x':np.asarray(x)}).x.ewm(span=span).mean().values
    
    # start training
    dones = 0
    success_turning = torch.load('success_turning_2.pth') if os.path.exists('success_turning_2.pth') else []
    t0 = time.clock()
    Count_t = 0
    for i_episode in tqdm(range(1,opt.num_episodes+1)):

        if i_episode % 300 == 0 or dones == 100:
            if dones == 100:
                success_turning.append(i_episode)
            #domain_file=randomize(domain_file_name)
            #dones = 0

        print(i_episode)
        
        score = 0
        aloss = 0
        counter = 0
        t1 = time.clock()
        while True:

            aug_state = agent.augment_state(state)
            action = agent.select_act(aug_state,eps)           # select an action
            next_state, reward, done, done_,info = env.step(action)
            loss = agent.step(state,action,reward,next_state,done)
            if loss:
                #print('Current Loss: {:.4f}'.format(loss.item()))
                aloss += loss.item()
            score += reward
            state = next_state
            counter += 1
            #print('==========================')
      
            #print('Current Step: ', counter)
            #print('Done:',done)
            #print('Dones:',dones)
            #print('==========================')
            
            if (done or (counter == 450)):
            #if done:
                if done:
                    dones += 1
                    #print(counter)

                print('time = ', (counter/(time.clock()-t1)))
                Count_t = Count_t+counter
                domain_file = randomize(domain_file_name)
                state = env.reset(domain_file)

                break
        scores.append(score)
        score_window.append(score)
        print('Success times: {}/{}'.format(dones, i_episode))
        if aloss != 0:
            losses.append(aloss)
        eps = max(opt.eps_end, opt.eps_decay*eps) # decrease epsilon
        """
        if i_episode % 20 == 0:
            #print('\rAverage Score: {:.2f}'.format(np.mean(score_window)))
            # sanity check
            for param in agent.qnetwork_local.parameters():
                print(param[0][0][0])
                break
            torch.save(agent.qnetwork_local.state_dict(), opt.local_model_path)
            torch.save(agent.qnetwork_target.state_dict(), opt.target_model_path)
            torch.save(success_turning, 'success_turning_2.pth')
            with open(opt.buffer_path, "wb") as mf:
                mem_to_save = deque(list(agent.memory.memory)[-50000:], maxlen=100000) #agent.memory.memory
                pickle.dump(mem_to_save, mf, pickle.HIGHEST_PROTOCOL)
        if np.mean(score_window)>=99.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(score_window)))
        """
        """
        # Create figure
        plt.cla()
        plt.subplot(1,2,1)
        if scores != []:
            plt.cla()
            plt.scatter(range(len(scores)), scores, label='rewards', s=5)
            plt.plot(ewma(np.array(scores),span=10), marker='.', label='rewards ewma@1000')
            plt.title("Session rewards"); plt.grid(); plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(losses, label='loss')
        plt.plot(ewma(np.array(losses),span=1000), label='loss ewma@1000')
        plt.title("Training Losses"); plt.grid(); plt.legend()

        plt.pause(0.005)
        plt.savefig('results/score.png')
        #print(success_turning)
        #print(eps)
        """
    
    #plt.ioff()

    print('Aver time = ', (Count_t / (time.clock() - t0)))
    return scores


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='single', dest='mode')
    args = parser.parse_args()

    opt = Config(args)
    domain_file_name = []
    path = "C:/Users/ysun465/Downloads/PAL-master/PolycraftAIGym/train"
    dirs = os.listdir(path)
    for i in dirs:
        if os.path.splitext(i)[1] == ".json":
            domain_file_name.append(i)
    #random.seed(10)
    #aa
    # domain_file_name = random.sample(domain_file_name, 700)
    scores = dqn_unity(opt,domain_file_name)
