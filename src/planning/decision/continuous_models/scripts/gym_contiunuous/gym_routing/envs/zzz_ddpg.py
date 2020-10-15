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
        self.low_speed_time = time.time()
        self.high_speed_last_time = time.time()
        self.middle_low_speed_time = time.time()
        self.middle_high_speed_last_time = time.time()
        self.start_time = time.time()
        self.last_time = time.time()

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
        low_action = np.array([-2.0,-7.2/3.6]) # di - ROAD_WIDTH, tv - TARGET_SPEED - D_T_S * N_S_SAMPLE
        high_action = np.array([2.0, 7.2/3.6])  #Should be symmetry for DDPG
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        self.state_dimention = 80

        # jxy0711: try to change input from vector to matrix.
        low  = []
        high = []
        for i in range(80):
            # s d vs vd omega flag
            low.append([-100.0, -100.0, -7.2, -7.0, -5.0, -1.0])
            high.append([100.0, 100.0, 7.2, 7.0, 5.0, 200.0])

        low_array = np.array(low)
        high_array = np.array(high)
        
        '''self.state_dimention = 16

        low  = np.array([-100,  -100,   -20,  -20,  -100, -100,  -20,   -20,   -100, -100,   -20,  -20, -100,  -100, -20, -20])
        high = np.array([100, 100, 20, 20, 100, 100, 20, 20, 100, 100, 20, 20,100, 100, 20, 20])'''

        self.observation_space = spaces.Box(low_array, high_array, dtype=np.float32)
        self.seed()


    def step(self, action, q_value, rule_action, rule_q, kill_threshold = 10):

        # send action to zzz planning module
        
        action = action.astype(float)
        action = action.tolist()
        #print("-------------",type(action),action)
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
                #print("-------------try receiving msg in step")
                received_msg = msgpack.unpackb(self.sock_conn.recv(self.sock_buffer))
                #print("-------------received msg in step")
                #print("msg length: ", len(received_msg))

                state_dim = self.state_dimention

                self.state = received_msg[0:state_dim]
                collision = received_msg[state_dim][0]
                leave_current_mmap = received_msg[state_dim][1]
                threshold = received_msg[state_dim][2]
                ego_x = received_msg[state_dim][3]
                ego_y = received_msg[state_dim][4]
                RLpointx = received_msg[state_dim+1][0]
                RLpointy = received_msg[state_dim+1][1]
                input_reward = received_msg[state_dim+1][2]

                self.rule_based_action = [(RLpointx, RLpointy)]

                # judge if finish
                done = False

                # calculate reward:
                reward = 0
                ego_s = self.state[0][0]
                ego_d = self.state[0][1]
                ego_vs = self.state[0][2]
                ego_vd = self.state[0][3]

                #print("+++ rule action: ", RLpointx, RLpointy)
                #print("+++ our action:  ", action[0], action[1])
                #print("ego_vs: ", ego_vs)

                if RLpointx > 2.0:
                    RLpointx = 2.0
                if RLpointx < -2.0:
                    RLpointx = -2.0

                punish_angle = abs(action[0] - RLpointx) #min(pow(abs(action[0] - 0), 2), 8)
                punish_speed = abs(action[1] - RLpointy)
                #reward_speed = min(pow(abs(ego_vs), 2), 8)
                #print("punish angle: ", punish_angle)
                #print("punish speed: ", punish_speed)
                #print("reward speed: ", 0.5 * reward_speed)
                #reward = 10 * (5 - (abs(action[0] - RLpointx) - abs(action[1] - RLpointy)))# + 0.5 * ego_s
                #reward += 9 * (15 - punish_angle - punish_speed)
                #TODO: change reward.
                # reward 1: he who goes forward should get a reward.
                #reward = 5 * (5 - punish_angle - punish_speed)
                
                #print("ego_s: ", ego_s)

                # reward 2: the planned trajectory should be inside the boundary. Calculated in VEG_planner.
                #jxy0720: add manual braking
                #TODO: Now 5% failure, can be further improved. Now test whether the model can learn this.
                braking_flag = 0
                danger_index = []
                for i in range(12):
                    point = received_msg[i+1]
                    if point[5] == 0:
                        continue #empty
                    point_s = point[0]
                    point_d = point[1]
                    point_vs = point[2]
                    point_vd = point[3]
                    #those getting far from reference lane are neglected
                    if (point_d - ego_d) * point_vd > 0: #jxy0721: it seems that the directions of d and vd are different?
                        danger_index.append(i)
                    elif abs(point_vd) < 0.1: #static
                        danger_index.append(i)

                if len(danger_index) != 0:
                    for i in range(len(danger_index)):
                        point = received_msg[danger_index[i] + 1]
                        point_s = point[0]
                        point_d = point[1]
                        point_vs = point[2]
                        point_vd = point[3]
                        dist = math.sqrt(pow((point_s - ego_s), 2) + pow((point_d - ego_d), 2))
                        relative_speed = [point_vs - ego_vs, point_vd - ego_vd]
                        if ego_s - point_s > 0: #those behind and not quickly approaching
                            if relative_speed[0] > 3 and ego_s - point_s < 5:
                                break #should not brake, but go ahead as soon as possible
                            else:
                                continue
                        if abs(point_vd) < 0.2 and abs(point_vs) < 0.2:
                            if abs(ego_d - point_d) > 1.5:
                                continue

                        ttc_s = -(point_s - ego_s) / relative_speed[0]
                        ttc_d = (point_d - ego_d) / relative_speed[1]

                        if dist < 4 or (0 < ttc_s < 2 and 0 < ttc_d < 2) or (0 < ttc_s < 2 and abs(point_d - ego_d) < 2) or (0 < ttc_d < 2 and abs(point_s - ego_s) < 2):
                            #rl_action[1] = -ACTION_SPACE_SYMMERTY
                            braking_flag = 1
                        elif dist < 8:
                            braking_flag = 2
                        break

                print("braking flag: ", braking_flag)
                if braking_flag == 1:
                    if action[1] > 0:
                        reward += -50
                    else:
                        reward += 100 - 75 * (action[1] + 2.0)
                elif braking_flag == 2:
                    if action[1] > 0:
                        reward += -70
                    elif action[1] > -1.5:
                        reward += 70 - 70 * (action[1] + 2.0)
                    else:
                        reward += 210 - 350 * (action[1] + 2.0)
                else:
                    if action[1] > 0:
                        reward += 1.5

                reward = reward * 2
                
                #print("reward2 = ", reward)

                #low speed reward
                if ego_vs < 0.1 and braking_flag == 0:
                    self.low_speed_flag = 1
                    self.low_speed_time = time.time()
                else:
                    self.low_speed_flag = 0
                    self.high_speed_last_time = time.time()

                if self.low_speed_flag == 1 and self.low_speed_time - self.high_speed_last_time > 3:
                    #print("low speed for more than 3s")
                    if (braking_flag == 0 and action[1] < 0) or self.low_speed_time - self.high_speed_last_time > 8:
                        #jxy0715: if the rule decision is also braking, the punishment will be spared
                        #jxy0724: after 7s, the front vehicle will be removed, if still stop, it will be punished.
                        reward = -2
                        #print("low speed reward: ", reward)

                # reward 3: final status: collision, success or restart
                #print("collision: ", collision)
                if collision:
                    done = True
                    reward = -1500#-1500
                    print("+++++++++++++++++++++ received collision")
                
                #TODO: check it
                elif leave_current_mmap == 1:
                    done = True
                    reward = 500#+500
                    print("+++++++++++++++++++++ successfully pass current unit")

                elif leave_current_mmap == 2:
                    done = True
                    print("+++++++++++++++++++++ restart by code")
                    reward = 0

                print("reward:", reward)
                reward = reward / 1500
                
                self.record_rl_intxt(action, q_value, RLpointx, RLpointy, rule_q, collision, leave_current_mmap, ego_s, threshold)
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
        fw = open("/home/carla/ZZZ/record_rl.txt", 'a')   
        fw.write(str(round(action[0], 2)))   
        fw.write(", ")   
        fw.write(str(round(action[1], 2)))
        fw.write(", ")   
        fw.write(str(round(q_value, 2)))
        fw.write(", ")   
        fw.write(str(round(RLpointx, 2)))
        fw.write(", ")   
        fw.write(str(round(RLpointy, 2)))
        fw.write(", ")   
        fw.write(str(round(rule_q, 2)))
        fw.write(", ")   
        fw.write(str(collision))
        fw.write(", ")   
        fw.write(str(leave_current_mmap))
        fw.write(", ")   
        fw.write(str(round(ego_s, 2)))   
        fw.write(", ")   

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
                #print("-------------",type(action),action)

                self.sock_conn.sendall(msgpack.packb(action))
                #print("-------------try receiving msg in reset")

                received_msg = msgpack.unpackb(self.sock_conn.recv(self.sock_buffer))
                #print("-------------received msg in reset")

                state_dim = self.state_dimention

                self.state = received_msg[0:state_dim]
                collision = received_msg[state_dim][0]
                leave_current_mmap = received_msg[state_dim][1]
                RLpointx = received_msg[state_dim+1][0]
                RLpointy = received_msg[state_dim+1][1]
                self.rule_based_action = [(RLpointx,RLpointy - 12.5/3.6)]

                self.low_speed_time = time.time()
                self.high_speed_last_time = time.time()
                self.low_speed_flag = 0

                self.middle_low_speed_time = time.time()
                self.middle_high_speed_last_time = time.time()
                self.middle_low_speed_flag = 0

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

