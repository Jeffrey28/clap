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
        print("ZZZ connected at {}".format(addr))

        # Set action space
        low_action = np.array([-4.0,-15/3.6]) # di - ROAD_WIDTH, tv - TARGET_SPEED - D_T_S * N_S_SAMPLE
        high_action = np.array([4.0, 15/3.6])  #Should be symmetry for DDPG
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        self.state_dimention = 600

        low  = np.zeros(600)
        high = np.zeros(600)   

        for i in range(100):
            # s d vs vd omega flag
            low[i*6] = -100
            low[i*6+1] = -100
            low[i*6+2] = -15
            low[i*6+3] = -7
            low[i*6+4] = -5
            low[i*6+5] = 0
            
            high[i*6] = 100
            high[i*6+1] = 100
            high[i*6+2] = 15
            high[i*6+3] = 7
            high[i*6+4] = 5
            high[i*6+5] = 3

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.seed()


    def step(self, action, q_value, rule_action, rule_q, kill_threshold = 10, trajectory_length = 20):

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
                self.state = received_msg[0:600]
                collision = received_msg[600]
                leave_current_mmap = received_msg[601]
                threshold = received_msg[602]
                RLpointx = received_msg[603]
                RLpointy = received_msg[604]
                self.rule_based_action = [(RLpointx, RLpointy)]

                # calculate reward:
                # Originally means difference to rule-based action. This would finally train RL to the rule based decision.
                reward = 50 - (abs(action[0] - RLpointx) + abs(action[1] - (RLpointy))) #+ 0.5 * ego_s
                #TODO: change reward. Now test the state change
                #reward = 0
              
                # judge if finish
                done = False

                # reward 3: final status: collision, success or restart
                if collision:
                    done = True
                    reward = -1500#-1000
                    print("+++++++++++++++++++++ received collision")
                
                #TODO: check it
                if leave_current_mmap == 1:
                    done = True
                    reward = 500#+500
                    print("+++++++++++++++++++++ successfully pass current unit")

                elif leave_current_mmap == 2:
                    done = True
                    print("+++++++++++++++++++++ restart by code")
                
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
                action = [0,0]
                print("-------------",type(action),action)

                self.sock_conn.sendall(msgpack.packb(action))
                print("-------------try receiving msg in reset")

                received_msg = msgpack.unpackb(self.sock_conn.recv(self.sock_buffer))
                print("-------------received msg in reset")

                self.state = received_msg[0:600]
                collision = received_msg[600]
                leave_current_mmap = received_msg[601]
                RLpointx = received_msg[603]
                RLpointy = received_msg[604]
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

