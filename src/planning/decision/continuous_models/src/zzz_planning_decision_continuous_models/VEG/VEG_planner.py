
import socket
import msgpack
import rospy
import math
import numpy as np

from zzz_cognition_msgs.msg import MapState
from zzz_driver_msgs.utils import get_speed
from carla import Location, Rotation, Transform
from zzz_common.geometry import dense_polyline2d, dist_from_point_to_closedpolyline2d
from zzz_common.kinematics import get_frenet_state, get_frenet_state_boundary_point, get_frenet_state_pure_xy

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
            sent_RL_msg.append(0.0) # Rule_based_point.vs
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

            #jxy 0710: when ego vehicle is still out of the junction, do not train.
            if len(drivable_area_array) < 3:
                print("It is a false junction, return!")
                return rule_trajectory_msg

            ego_s = sent_RL_msg[0]
            ego_d = sent_RL_msg[1]
            dist_ego, _, _, = dist_from_point_to_closedpolyline2d(ego_s, ego_d, drivable_area_array)
            print("dist_ego", dist_ego)
            if dist_ego >= 0:
                print("ego vehicle has still not entered the junction, return!")
                return rule_trajectory_msg

            print("-----------------------------",sent_RL_msg)
            rospy.loginfo("sent msg length, %d", len(sent_RL_msg))
            self.sock.sendall(msgpack.packb(sent_RL_msg))
            rospy.loginfo("sent RL msg succeeded!!!\n\n\n")

            # received RL action and plan a RL trajectory
            try:
                received_msg = msgpack.unpackb(self.sock.recv(self._buffer_size))
                rl_action = [received_msg[0], received_msg[1]]
                rl_q = received_msg[2]
                rule_q = received_msg[3]
                
                VEG_trajectory = self.generate_VEG_trajectory(rl_q, rule_q, rl_action, rule_trajectory_msg)
                trajectory_points = VEG_trajectory.trajectory.poses
                print("temp safe...")

                # reward: if the vehicle is running out of the drivable area, punish it.
                
                reward = 0
                print("len(trajectory_points): ", len(trajectory_points))
                print("drivable area array: ", drivable_area_array)
                ego_s = sent_RL_msg[0]
                ego_d = sent_RL_msg[1]
                dist_ego, _, _, = dist_from_point_to_closedpolyline2d(ego_s, ego_d, drivable_area_array)
                print("dist_ego", dist_ego)
                if dist_ego >= 0:
                    print("ego vehicle has still not entered the junction, reward2 set to 0.")
                else:
                    for i in range(len(trajectory_points)):
                        if i > 20:
                            break
                        pointx = trajectory_points[i].pose.position.x
                        pointy = trajectory_points[i].pose.position.y
                        traj_point_ffstate = get_frenet_state_pure_xy(pointx, pointy, self.ref_path, self.ref_path_tangets)
                        point_s = traj_point_ffstate.s
                        point_d = traj_point_ffstate.d

                        if len(drivable_area_array) >= 3:
                            dist1, _, _, = dist_from_point_to_closedpolyline2d(point_s, point_d, drivable_area_array)
                            print("examining traj point: ", point_s, point_d)
                            print("dist1: ", dist1)
                            if dist1 >= 0:
                                if len(next_drivable_area_array) >= 3:
                                    dist2, _, _, = dist_from_point_to_closedpolyline2d(point_s, point_d, next_drivable_area_array)
                                    if dist2 >= 0:
                                        print("dist2: ", dist2)
                                        print("trajectory conflicts to the drivable area and next drivable area!")
                                        reward = -30*(20-i)
                                        break
                                else:
                                    print("trajectory conflicts to the drivable area!")
                                    reward = -15*(20-i)+1
                                    break
                        else:
                            break

                self.reward_buffer = reward #jxy0709: now the dynamic boundary is not in order, should be adjusted.
                print("temp safe...")

                return VEG_trajectory #jxy0710: dangerous attempt!
            
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
        # orientation
        x = self._dynamic_map.ego_state.pose.pose.orientation.x
        y = self._dynamic_map.ego_state.pose.pose.orientation.y
        z = self._dynamic_map.ego_state.pose.pose.orientation.z
        w = self._dynamic_map.ego_state.pose.pose.orientation.w

        rotation_mat = np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y], [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x], [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]])
        rotation_mat_inverse = np.linalg.inv(rotation_mat) #those are the correct way to deal with quaternion

        dx = [1, 0, 0]
        ego_direction_xyz = np.matmul(rotation_mat_inverse, dx)
        ego_direction = math.atan2(ego_direction_xyz[1], ego_direction_xyz[0])
        print("ego direction: ", ego_direction, "\n\n")
        ego_x = self._dynamic_map.ego_state.pose.pose.position.x
        ego_y = self._dynamic_map.ego_state.pose.pose.position.y
        print("ego xy: ", ego_x, ego_y)

        ego_ffstate = get_frenet_state(self._dynamic_map.ego_state, self.ref_path, self.ref_path_tangets)
        state[0] = ego_ffstate.s
        state[1] = -ego_ffstate.d
        state[2] = ego_ffstate.vs
        state[3] = ego_ffstate.vd
        state[4] = 0
        state[5] = 0

        #boundary points, start from those in front of ego vehicle
        start_angle = 0.78 + ego_direction
        if start_angle > math.pi:
            start_angle = start_angle - 2 * math.pi

        point_direction_diff_array = []
        for i in range(len(self._dynamic_boundary.boundary)):
            point_direction = math.atan2(self._dynamic_boundary.boundary[i].y-ego_y, self._dynamic_boundary.boundary[i].x-ego_x)
            #print("point: ", self._dynamic_boundary.boundary[i].x, self._dynamic_boundary.boundary[i].y)
            #print("point direction: ", point_direction)
            point_direction_diff_array.append(abs(point_direction - start_angle))

        start_point_index = 0
        if len(point_direction_diff_array) > 0:
            start_point_index = point_direction_diff_array.index(min(point_direction_diff_array)) #nearest to the start angle

        for i in range(79):
            if i >= len(self._dynamic_boundary.boundary):
                for j in range(6):
                    state.append(0)
            else:
                ii = start_point_index + i
                if ii >= len(self._dynamic_boundary.boundary):
                    ii = ii - len(self._dynamic_boundary.boundary)
                point_ffstate = get_frenet_state_boundary_point(self._dynamic_boundary.boundary[ii], self.ref_path, self.ref_path_tangets)
                s = point_ffstate.s
                d = point_ffstate.d
                vs = point_ffstate.vs
                vd = point_ffstate.vd

                omega = self._dynamic_boundary.boundary[ii].omega
                flag = int(self._dynamic_boundary.boundary[ii].flag*10) #socket transmit limit, must use int instead of float

                state.append(s)
                state.append(d)
                state.append(vs)
                state.append(vd)
                state.append(omega)
                state.append(flag)

                #point_direction = math.atan2(self._dynamic_boundary.boundary[ii].y-ego_y, self._dynamic_boundary.boundary[ii].x-ego_x)
                #print("point sorted: ", self._dynamic_boundary.boundary[ii].x, self._dynamic_boundary.boundary[ii].y)
                #print("point direction sorted: ", point_direction)

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

        if len(frenet_trajectory_rule.t) >= 1:
            RLpoint.location.x = frenet_trajectory_rule.d[-1] #only works when DT param of werling is 0.15
            RLpoint.location.y = frenet_trajectory_rule.s_d[1-1] - ACTION_SPACE_SYMMERTY
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
