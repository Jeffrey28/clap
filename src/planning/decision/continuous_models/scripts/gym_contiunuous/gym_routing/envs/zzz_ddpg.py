from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import socket
import sys
import time
import weakref

import matplotlib.pyplot as plt
import msgpack
import networkx as nx
import numpy as np

import gym
from gym import core, error, spaces, utils
from gym.utils import seeding

# from carla import Location, Rotation, Transform

##########################################

class ZZZCarlaEnv(gym.Env):
    metadata = {'render.modes': []}
    def __init__(self, zzz_client="127.0.0.1", port=2333, recv_buffer=4096, socket_time_out = 1000):
    
        self._restart_motivation = 0
        self.state = []
        self.steps = 1
        self.collision_times = 0

        # Socket
        socket.setdefaulttimeout(socket_time_out) # Set time out
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(socket_time_out) # Set time out
        self.sock.bind((zzz_client, port))
        self.sock.listen()
        self.sock_conn = None
        self.sock_buffer = recv_buffer
        self.sock_conn, addr = self.sock.accept()
        self.sock_conn.settimeout(socket_time_out) # Set time out
        self.rule_based_action = []
        self.ego_s = None
        print("ZZZ connected at {}".format(addr))

        # Set action space
        low_action = np.array([-2.0,-15/3.6]) # di - ROAD_WIDTH, tv - TARGET_SPEED - D_T_S * N_S_SAMPLE
        high_action = np.array([2.0, 15/3.6])  #Should be symmetry for DDPG
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        self.state_dimention = 480

        low  = np.zeros(self.state_dimention)
        high = np.zeros(self.state_dimention)

        for i in range(80):
            # s d vs vd omega flag
            low[i*6] = -100.0
            low[i*6+1] = -100.0
            low[i*6+2] = -15.0
            low[i*6+3] = -7.0
            low[i*6+4] = -5.0
            low[i*6+5] = -1.0
            
            high[i*6] = 100.0
            high[i*6+1] = 100.0
            high[i*6+2] = 15.0
            high[i*6+3] = 7.0
            high[i*6+4] = 5.0
            high[i*6+5] = 200.0
        
        '''self.state_dimention = 16

        low  = np.array([-100,  -100,   -20,  -20,  -100, -100,  -20,   -20,   -100, -100,   -20,  -20, -100,  -100, -20, -20])
        high = np.array([100, 100, 20, 20, 100, 100, 20, 20, 100, 100, 20, 20,100, 100, 20, 20])'''

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.seed()


    def step(self, action, q_value, rule_action, rule_q, kill_threshold = 10):

        # send action to zzz planning module
        
        action = action.astype(float)
        action = action.tolist()
        print("-------------",type(action),action)
        no_state_time = time.time()
        no_state_start_time = time.time()
        no_state_flag = 0
        while True:
            try:
                send_action = action
                send_action.append(q_value)
                send_action.append(rule_q)                

                self.sock_conn.sendall(msgpack.packb(send_action))

                # wait next state
                print("-------------try receiving msg in step")
                received_msg = msgpack.unpackb(self.sock_conn.recv(self.sock_buffer))
                print("-------------received msg in step")
                print("msg length: ", len(received_msg))

                state_dim = self.state_dimention

                self.state = received_msg[0:state_dim]
                collision = received_msg[state_dim]
                leave_current_mmap = received_msg[state_dim+1]
                threshold = received_msg[state_dim+2]
                RLpointx = received_msg[state_dim+3]
                RLpointy = received_msg[state_dim+4]
                input_reward = received_msg[state_dim+5]

                self.rule_based_action = [(RLpointx, RLpointy)]

                # judge if finish
                done = False

                # calculate reward:
                ego_s = self.state[0]
                ego_vs = self.state[2]

                print("RL action: ", RLpointx, RLpointy)
                print("ego_vs: ", ego_vs)
                punish_angle = min(pow(abs(action[0] - RLpointx), 3), 8)
                punish_speed = min(pow(abs(ego_vs - RLpointy), 1.5), 8)
                #reward_speed = min(pow(abs(ego_vs), 2), 8)
                print("punish angle: ", punish_angle)
                print("punish speed: ", punish_speed)
                #print("reward speed: ", reward_speed)
                #reward = 10 * (5 - (abs(action[0] - RLpointx) - abs(action[1] - RLpointy)))# + 0.5 * ego_s
                #reward = 10 * (5 - punish_angle - punish_speed)
                #TODO: change reward.
                # reward 1: he who goes forward should get a reward.
                reward = 5 * (15 - punish_angle - punish_speed)
                
                '''print("ego_s: ", ego_s)

                if self.ego_s is None:
                    self.ego_s = ego_s
                else:
                    if ego_s > self.ego_s:
                        reward = reward + 10*(ego_s - self.ego_s)
                    self.ego_s = ego_s

                print("reward ego_s: ", 10*(ego_s - self.ego_s))'''

                # reward 2: the planned trajectory should be inside the boundary. Calculated in VEG_planner.
                #reward = reward + input_reward
                #print("reward2 = ", input_reward)

                # reward 3: final status: collision, success or restart
                if collision:
                    done = True
                    #reward = -3000#-1500
                    print("+++++++++++++++++++++ received collision")
                
                #TODO: check it
                if leave_current_mmap == 1:
                    done = True
                    #reward = 500#+500
                    print("+++++++++++++++++++++ successfully pass current unit")

                elif leave_current_mmap == 2:
                    done = True
                    print("+++++++++++++++++++++ restart by code")

                print("reward:", reward)
                reward = reward / 500
                
                # self.record_rl_intxt(action, q_value, RLpointx, RLpointy, rule_q, collision, leave_current_mmap, ego_s, threshold)
                no_state_flag = 0
                no_state_start_time = time.time()
                return np.array(self.state), reward, done,  {}, np.array(self.rule_based_action)

            except:
                #print("RL cannot receive an state")
                no_state_time = time.time()
                if no_state_flag == 0:
                    no_state_start_time = time.time()
                    no_state_flag = 1
                else:
                    if no_state_time - no_state_start_time > kill_threshold:
                        print("break because RL have not been able to receive an state for 10s")
                        break
            
    def record_rl_intxt(self, action, q_value, RLpointx, RLpointy, rule_q, collision, leave_current_mmap, ego_s, threshold):
        fw = open("/home/carla/openai_baselines_update/zwt_ddpg/test_data/record_rl.txt", 'a')   
        fw.write(str(action[0]))   
        fw.write(",")   
        fw.write(str(action[1]))
        fw.write(",")   
        fw.write(str(q_value))
        fw.write(",")   
        fw.write(str(RLpointx))
        fw.write(",")   
        fw.write(str(RLpointy))
        fw.write(",")   
        fw.write(str(rule_q))
        fw.write(",")   
        fw.write(str(collision))
        fw.write(",")   
        fw.write(str(leave_current_mmap))
        fw.write(",")   
        fw.write(str(ego_s))   
        fw.write(",")   

        if q_value - rule_q > threshold:
            fw.write("kick in")  
            print("kick in!！！!！!！!！!！!！!") 
        fw.write("\n")
        fw.close()

    def reset(self, **kargs):
       
        # receive state
        # if the received information meets requirements
        while True:
            try:
                action = [2333,2333,0,0]
                print("-------------",type(action),action)

                self.sock_conn.sendall(msgpack.packb(action))
                print("-------------try receiving msg in reset")

                received_msg = msgpack.unpackb(self.sock_conn.recv(self.sock_buffer))
                print("-------------received msg in reset")

                state_dim = self.state_dimention

                self.state = received_msg[0:state_dim]
                collision = received_msg[state_dim]
                leave_current_mmap = received_msg[state_dim+1]
                RLpointx = received_msg[state_dim+3]
                RLpointy = received_msg[state_dim+4]
                self.rule_based_action = [(RLpointx,RLpointy - 12.5/3.6)]

                return np.array(self.state), np.array(self.rule_based_action)

                # if not collision and not leave_current_mmap:
            except:
                print("------------- not received msg in reset")
                collision = 0
                leave_current_mmap = 0
                RLpointx = 0
                RLpointy = 0
                self.rule_based_action = [(RLpointx,RLpointy - 12.5/3.6)]

                return np.array(self.state), np.array(self.rule_based_action)

        return np.array(self.state), np.array(self.rule_based_action)


    def render(self, mode='human'):
        # if mode == 'human':
        #     screen_width = 600
        #     screen_height = 400
        #     #world_width = self.problem.xrange
        #     super(MyEnv, self).render(mode=mode)
        pass

