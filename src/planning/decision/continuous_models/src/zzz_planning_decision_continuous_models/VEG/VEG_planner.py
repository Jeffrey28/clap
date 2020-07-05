
import socket
import msgpack
import rospy
import math
import numpy as np

from zzz_cognition_msgs.msg import MapState
from zzz_driver_msgs.utils import get_speed
from carla import Location, Rotation, Transform
from zzz_common.geometry import dense_polyline2d, dist_from_point_to_closedpolyline2d
from zzz_common.kinematics import get_frenet_state, get_frenet_state_boundary_point

from zzz_planning_msgs.msg import DecisionTrajectory
from zzz_planning_decision_continuous_models.VEG.Werling_planner_RL import Werling
from zzz_planning_decision_continuous_models.common import rviz_display, convert_ndarray_to_pathmsg, convert_path_to_ndarray
from zzz_planning_decision_continuous_models.predict import predict

# PARAMETERS
THRESHOLD = -100000
OBSTACLES_CONSIDERED = 3
ACTION_SPACE_SYMMERTY = 15/3.6 



class VEG_Planner(object):
    """
    Parameter:
        mode: ZZZ TCP connection mode (client/server)
    """
    def __init__(self, openai_server="127.0.0.1", port=2333, mode="client", recv_buffer=4096):
        self._dynamic_map = None
        self._socket_connected = False
        self._rule_based_trajectory_model_instance = Werling()
        self._buffer_size = recv_buffer
        self._collision_signal = False
        self._collision_times = 0
        self._has_clear_buff = False
        print("has clear buff?", self._has_clear_buff)

        self.reference_path = None
        self.ref_path = None
        self.ref_path_tangets = None
        self.reward_buffer = 0

        self.rivz_element = rviz_display()
        self.kick_in_signal = None

    
        if mode == "client":
            rospy.loginfo("Connecting to RL server...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((openai_server, port))
            self._socket_connected = True
            rospy.loginfo("Connected...")
        else:
            # TODO: Implement server mode to make multiple connection to this node.
            #     In this mode, only rule based action is returned to system
            raise NotImplementedError("Server mode is still wating to be implemented.") 
        
    def initialize(self, dynamic_map):
        try:
            if self.reference_path is None:
                self.reference_path = dynamic_map.jmap.reference_path.map_lane.central_path_points
                ref_path_ori = convert_path_to_ndarray(self.reference_path)
                # Prolong
                self.ref_path = dense_polyline2d(ref_path_ori, 2)
                # the last 2 points:
                point1 = self.ref_path[-2]
                point2 = self.ref_path[-1]
                direction1 = point2[0] - point1[0]
                direction2 = point2[1] - point1[1]
                direction_length = np.sqrt(direction1 * direction1 + direction2 * direction2)
                direction1 = direction1 / direction_length
                direction2 = direction2 / direction_length
                new_x = point2[0] + direction1 * 80 #add 80m
                new_y = point2[1] + direction2 * 80
                new_point = np.array([new_x, new_y])
                self.ref_path = np.row_stack((self.ref_path, new_point))
                self.ref_path_tangets = np.zeros(len(self.ref_path))
            return True
        except:
            print("------> VEG: Initialize fail ")
            return False

    def clear_buff(self, dynamic_map):
        print("has clear buff?", self._has_clear_buff)
        self._rule_based_trajectory_model_instance.clear_buff(dynamic_map)
        self._collision_signal = False
        self.reference_path = None
        self.ref_path = None
        self.ref_path_tangets = None
        rospy.loginfo("start to clear buff!")
        print("has clear buff?", self._has_clear_buff)

        # send done to OPENAI
        if self._has_clear_buff == False:
            ego_x = dynamic_map.ego_state.pose.pose.position.x
            ego_y = dynamic_map.ego_state.pose.pose.position.y
            if math.pow((ego_x+10),2) + math.pow((ego_y-94),2) < 64:  # restart point
                leave_current_mmap = 2
            else:  
                leave_current_mmap = 1

            collision = False
            sent_RL_msg = []
            for i in range(480):
                sent_RL_msg.append(0)
            sent_RL_msg.append(collision)
            sent_RL_msg.append(leave_current_mmap)
            sent_RL_msg.append(THRESHOLD) # threshold
            sent_RL_msg.append(0.0) # Rule_based_point.d
            sent_RL_msg.append(0.0 - ACTION_SPACE_SYMMERTY) # Rule_based_point.vs
            sent_RL_msg.append(0)
            rospy.loginfo("sent msg length, %d", len(sent_RL_msg))
            self.sock.sendall(msgpack.packb(sent_RL_msg))
            rospy.loginfo("sent RL msg succeeded!!!\n\n\n")

            try:
                RLS_action = msgpack.unpackb(self.sock.recv(self._buffer_size))
            except:
                pass
            
            self._has_clear_buff = True
        return None

    def trajectory_update(self, dynamic_map, dynamic_boundary):
        if self.initialize(dynamic_map):
            self._has_clear_buff = False
            
            self._dynamic_map = dynamic_map
            self._dynamic_boundary = dynamic_boundary

            # wrap states
            sent_RL_msg = self.wrap_state_dynamic_boundary()
            print("length???", len(sent_RL_msg))
            print(sent_RL_msg)
            '''sent_RL_msg = []
            for i in range(303):
                sent_RL_msg.append(0.5)'''
            rospy.loginfo("wrapped state by dynamic boundary, state length %d", len(sent_RL_msg))
            
            # rule-based planner
            rule_trajectory_msg = self._rule_based_trajectory_model_instance.trajectory_update(dynamic_map)
            RLpoint = self.get_RL_point_from_trajectory(self._rule_based_trajectory_model_instance.last_trajectory_rule)
            sent_RL_msg.append(RLpoint.location.x)
            sent_RL_msg.append(RLpoint.location.y)
            sent_RL_msg.append(self.reward_buffer)

            print("-----------------------------",sent_RL_msg)
            rospy.loginfo("sent msg length, %d", len(sent_RL_msg))
            self.sock.sendall(msgpack.packb(sent_RL_msg))
            rospy.loginfo("sent RL msg succeeded!!!\n\n\n")


            # Prepare for calculating the reward by the planned trajectory.
            # Decoding drivable area, next drivable area and broken lanes. From dynamic boundary.
            
            last_lane_num = None
            next_last_lane_num = None
            lane_list=[]
            next_lane_list=[]
            current_lane=[]
            current_next_lane=[]
            drivable_area_list=[]
            next_drivable_area_list=[]
            for i in range(80):
                point = sent_RL_msg[(6*i):(6*i+6)]
                if point[5]/10 < 10:
                    if point[5]/10 == 0:
                        continue
                    if point[5]/10 >= 3:
                        lane_num = int((point[5]/10-3)*10)
                        if last_lane_num is None:
                            #start a new lane
                            current_lane = []
                            current_lane.append([point[0], point[1]])
                        elif lane_num != last_lane_num:
                            #save the current lane and start a new lane
                            lane_list.append(current_lane)
                            current_lane = []
                            current_lane.append([point[0], point[1]])
                        else:
                            current_lane.append([point[0], point[1]])
                    else:
                        position_point = [point[0], point[1]]
                        drivable_area_list.append(position_point)
                else:
                    if point[5]/10 >= 13:
                        next_lane_num = int((point[5]/10-13)*10)
                        if next_last_lane_num is None:
                            #start a new lane
                            current_next_lane = []
                            current_next_lane.append([point[0], point[1]])
                        elif next_lane_num != next_last_lane_num:
                            #save the current lane and start a new lane
                            next_lane_list.append(current_next_lane)
                            current_next_lane = []
                            current_next_lane.append([point[0], point[1]])
                        else:
                            current_next_lane.append([point[0], point[1]])
                    else:
                        position_point = [point[0], point[1]]
                        next_drivable_area_list.append(position_point)
            if len(current_lane) > 0:
                lane_list.append(current_lane)
            if len(current_next_lane) > 0:
                next_lane_list.append(current_next_lane)
            
            print("drivable area decoded, length ", len(drivable_area_list))
            print("next_drivable_area_decoded, length ", len(next_drivable_area_list))
            print("lanes decoded, num ", len(lane_list))
            print("next lanes decoded, num ", len(next_lane_list))

            drivable_area_array = np.array(drivable_area_list)
            next_drivable_area_array = np.array(next_drivable_area_list)

            # received RL action and plan a RL trajectory
            try:
                received_msg = msgpack.unpackb(self.sock.recv(self._buffer_size))
                rl_action = [received_msg[0], received_msg[1]]
                rl_q = received_msg[2]
                rule_q = received_msg[3]
                
                VEG_trajectory = self.generate_VEG_trajectory(rl_q, rule_q, rl_action, rule_trajectory_msg)
                trajectory_points = VEG_trajectory.trajectory.poses

                # reward: if the vehicle is running out of the drivable area, punish it.
                
                reward = 0
                for i in len(trajectory_points):
                    pointx = trajectory_points[i].pose.position.x
                    pointy = trajectory_points[i].pose.position.y

                    dist1, _, _, = dist_from_point_to_closedpolyline2d(pointx, pointy, drivable_area_array)
                    if dist1 >= 0:
                        if len(next_drivable_area_array) > 0:
                            dist2, _, _, = dist_from_point_to_closedpolyline2d(pointx, pointy, next_drivable_area_array)
                            if dist2 >= 0:
                                print("trajectory conflicts to the drivable area and next drivable area!")
                                reward = -1000
                        else:
                            print("trajectory conflicts to the drivable area!")
                            reward = -1000
                            

                self.reward_buffer = reward

                return VEG_trajectory
            
            except:
                rospy.logerr("Continous RLS Model cannot receive an action")
                return rule_trajectory_msg
        else:
            return None   
            
    def wrap_state(self):
        # ego state: ego_x(0), ego_y(1), ego_vx(2), ego_vy(3)    
        # obstacle 0 : x0(4), y0(5), vx0(6), vy0(7)
        # obstacle 1 : x0(8), y0(9), vx0(10), vy0(11)
        # obstacle 2 : x0(12), y0(13), vx0(14), vy0(15)
        state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        # ego state
        ego_ffstate = get_frenet_state(self._dynamic_map.ego_state, self.ref_path, self.ref_path_tangets)
        state[0] = ego_ffstate.s
        state[1] = -ego_ffstate.d
        state[2] = ego_ffstate.vs
        state[3] = ego_ffstate.vd

        # obs state
        closest_obs = []
        closest_obs = self.found_closest_obstacles(OBSTACLES_CONSIDERED, self._dynamic_map)
        i = 0
        for obs in closest_obs: 
            if i < OBSTACLES_CONSIDERED:               
                state[(i+1)*4+0] = obs[5]
                state[(i+1)*4+1] = obs[6]
                state[(i+1)*4+2] = obs[7]
                state[(i+1)*4+3] = obs[8]
                i = i+1
            else:
                break
        
        # if collision
        collision = int(self._collision_signal)
        self._collision_signal = False
        state.append(collision)

        # if finish
        leave_current_mmap = 0
        state.append(leave_current_mmap)
        state.append(THRESHOLD)

        return state

    
    def wrap_state_dynamic_boundary(self):
        # ego state: ego_x(0), ego_y(1), ego_vx(2), ego_vy(3)    
        # boundary points: s d vs vd omega flag
        state = [0, 0, 0, 0, 0, 0]

        # ego state
        ego_ffstate = get_frenet_state(self._dynamic_map.ego_state, self.ref_path, self.ref_path_tangets)
        state[0] = ego_ffstate.s
        state[1] = -ego_ffstate.d
        state[2] = ego_ffstate.vs
        state[3] = ego_ffstate.vd
        state[4] = 0
        state[5] = 0

        for i in range(79):
            #TODO: check the max boundary list length
            if i >= len(self._dynamic_boundary.boundary):
                for j in range(6):
                    state.append(0)
            else:
                point_ffstate = get_frenet_state_boundary_point(self._dynamic_boundary.boundary[i], self.ref_path, self.ref_path_tangets)
                s = point_ffstate.s
                d = point_ffstate.d
                vs = point_ffstate.vs
                vd = point_ffstate.vd

                omega = self._dynamic_boundary.boundary[i].omega
                flag = int(self._dynamic_boundary.boundary[i].flag*10) #socket transmit limit, must use int instead of float

                state.append(s)
                state.append(d)
                state.append(vs)
                state.append(vd)
                state.append(omega)
                state.append(flag)

        # if collision
        collision = int(self._collision_signal)
        self._collision_signal = False
        state.append(collision)

        # if finish
        leave_current_mmap = 0
        state.append(leave_current_mmap)
        state.append(THRESHOLD)

        return state

    def found_closest_obstacles(self, num, dynamic_map):
        closest_obs = []
        obs_tuples = []
        
        for obs in self._dynamic_map.jmap.obstacles: 
            # calculate distance
            p1 = np.array([self._dynamic_map.ego_state.pose.pose.position.x , self._dynamic_map.ego_state.pose.pose.position.y])
            p2 = np.array([obs.state.pose.pose.position.x , obs.state.pose.pose.position.y])
            p3 = p2 - p1
            p4 = math.hypot(p3[0],p3[1])

            # transfer to frenet
            obs_ffstate = get_frenet_state(obs.state, self.ref_path, self.ref_path_tangets)
            one_obs = (obs.state.pose.pose.position.x , obs.state.pose.pose.position.y , obs.state.twist.twist.linear.x ,
                     obs.state.twist.twist.linear.y , p4 , obs_ffstate.s , -obs_ffstate.d , obs_ffstate.vs, obs_ffstate.vd, 
                     obs.state.accel.accel.linear.x, obs.state.accel.accel.linear.y)
            obs_tuples.append(one_obs)

        # sort by distance
        sorted_obs = sorted(obs_tuples, key=lambda obs: obs[4])   
        i = 0
        for obs in sorted_obs:
            if i < num:
                closest_obs.append(obs)
                i = i + 1
            else:
                break
        
        return closest_obs

    def get_RL_point_from_trajectory(self, frenet_trajectory_rule):
        RLpoint = Transform()

        if len(frenet_trajectory_rule.t) > 15:
            RLpoint.location.x = frenet_trajectory_rule.d[15] #only works when DT param of werling is 0.15
            RLpoint.location.y = frenet_trajectory_rule.s_d[15] - ACTION_SPACE_SYMMERTY
        else:
            RLpoint.location.x = 0
            RLpoint.location.y = 0 - ACTION_SPACE_SYMMERTY
                      
        return RLpoint

    def generate_VEG_trajectory(self, rl_q, rule_q, rl_action, rule_trajectory_msg):
        
        print("rl_action", rl_action[0], rl_action[1])
        print("rl_q", rl_q)
        print("rule_q", rule_q)

        if rl_q - rule_q > THRESHOLD and rl_action[0] < 2333 and rl_action[1] < 2333:
            rl_action[1] = rl_action[1] + ACTION_SPACE_SYMMERTY
            self.kick_in_signal = self.rivz_element.draw_kick_in_circles(self._dynamic_map.ego_state.pose.pose.position.x,
                        self._dynamic_map.ego_state.pose.pose.position.y, 3.5)
            print("RL kick in!")
            return self._rule_based_trajectory_model_instance.trajectory_update_RL_kick(self._dynamic_map, rl_action)
        
        else:
            self.kick_in_signal = None
            print("RL does not kick in!")
            return rule_trajectory_msg
