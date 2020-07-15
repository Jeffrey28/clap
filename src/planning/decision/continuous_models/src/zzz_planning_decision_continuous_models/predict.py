import numpy as np
import rospy
import matplotlib.pyplot as plt
import copy
import math

from Werling.trajectory_structure import Frenet_path, Frenet_state
from zzz_common.kinematics import get_frenet_state
from zzz_driver_msgs.utils import get_speed, get_yaw
from common import rviz_display, convert_ndarray_to_pathmsg, convert_path_to_ndarray
from zzz_common.geometry import dense_polyline2d

#para for ego vehicle
EGO_HALF_LENGTH = 3
EGO_HALF_WIDTH = 3

def para_Line(slope,x,y):#return (a,b,c) for ax+by+c=0
    return(slope, -1, y-slope*x)

def dis_pl(A,B,C,x,y):
    if A*A+B*B==0:
        return 0
    return abs((A*x+B*y+C)/math.sqrt(A*A+B*B))

def rec_check(p_x, p_y, p_yaw, o_x, o_y, o_yaw, o_len, o_wid):#model ego and obstacle as rectangle
    o_plistx = [o_x-0.5*o_len*math.cos(o_yaw)-0.5*o_wid*math.sin(o_yaw), o_x+0.5*o_len*math.cos(o_yaw)-0.5*o_wid*math.sin(o_yaw), o_x+0.5*o_len*math.cos(o_yaw)+0.5*o_wid*math.sin(o_yaw), o_x-0.5*o_len*math.cos(o_yaw)+0.5*o_wid*math.sin(o_yaw)]
    o_plisty = [o_y-0.5*o_len*math.sin(o_yaw)+0.5*o_wid*math.cos(o_yaw), o_y+0.5*o_len*math.sin(o_yaw)+0.5*o_wid*math.cos(o_yaw), o_y+0.5*o_len*math.sin(o_yaw)-0.5*o_wid*math.cos(o_yaw), o_y-0.5*o_len*math.sin(o_yaw)-0.5*o_wid*math.cos(o_yaw)]
    ap,bp,cp = para_Line(math.tan(p_yaw),p_x,p_y)
    av,bv,cv = para_Line(-1/math.tan(p_yaw),p_x,p_y)
    for n in range(len(o_plistx)):
        if dis_pl(ap,bp,cp,o_plistx[n],o_plisty[n])<=EGO_HALF_WIDTH and dis_pl(av,bv,cv,o_plistx[n],o_plisty[n])<=EGO_HALF_LENGTH:
            return False
    return True


class predict():
    def __init__(self, dynamic_map, considered_obs_num, maxt, dt, robot_radius, radius_speed_ratio, move_gap, ego_speed):
        self.considered_obs_num = considered_obs_num
        self.maxt = maxt
        self.dt = dt
        self.check_radius = robot_radius + radius_speed_ratio * ego_speed
        self.move_gap = move_gap

        self.dynamic_map = dynamic_map
        self.initialze_fail = False

        self.rviz_collision_checking_circle = None
        self.rivz_element = rviz_display()

        try:
            self.reference_path = self.dynamic_map.jmap.reference_path.map_lane.central_path_points
            ref_path_ori = convert_path_to_ndarray(self.reference_path)
            self.ref_path = dense_polyline2d(ref_path_ori, 2)
            self.ref_path_tangets = np.zeros(len(self.ref_path))

            self.obs = self.found_closest_obstacles()
            self.obs_paths = self.prediction_obstacle(self.obs, self.maxt, self.dt)
            self.obs_pos = self.prediction_obstacle_pos(self.obs, self.maxt, self.dt)
        except:
            rospy.logdebug("continous module: fail to initialize prediction")
            self.obs_paths = []
        
        
    def check_collision(self, fp):
        if len(self.obs_pos) == 0 or len(fp.t) < 2 :
            return True

        '''    
        # two circles for a vehicle
        fp_front = copy.deepcopy(fp)
        fp_back = copy.deepcopy(fp)
        try:
            for t in range(len(fp.yaw)):
                fp_front.x[t] = fp.x[t] + math.cos(fp.yaw[t]) * self.move_gap
                fp_front.y[t] = fp.y[t] + math.sin(fp.yaw[t]) * self.move_gap
                fp_back.x[t] = fp.x[t] - math.cos(fp.yaw[t]) * self.move_gap
                fp_back.y[t] = fp.y[t] - math.sin(fp.yaw[t]) * self.move_gap

            for obsp in self.obs_paths:
                for t in range(len(fp.t)):
                    d = (obsp.x[t] - fp_front.x[t])**2 + (obsp.y[t] - fp_front.y[t])**2
                    if d <= self.check_radius**2: 
                        return False
                    d = (obsp.x[t] - fp_back.x[t])**2 + (obsp.y[t] - fp_back.y[t])**2
                    if d <= self.check_radius**2: 
                        return False
        except:
            pass
            # print("collision check fail",len(fp.yaw),len(fp_back.x),len(fp_front.x))
        '''
        fpc = copy.deepcopy(fp)
        try:
            #for t in range(len(fp.yaw)):
            for obsp in self.obs_pos:
                for t in range(len(fp.t)):
                    if rec_check(fp.x[t], fp.y[t], fp.yaw[t], obsp.x[t], obsp.y[t], obsp.yaw[t], obsp.len, obs.wid) == False:
                        return False
        except:
            pass
        # self.rviz_collision_checking_circle = self.rivz_element.draw_circles(fp_front, fp_back, self.check_radius)
        return True

    def found_closest_obstacles(self):
        closest_obs = []
        obs_tuples = []
        
        for obs in self.dynamic_map.jmap.obstacles: 
            # distance
            p1 = np.array([self.dynamic_map.ego_state.pose.pose.position.x , self.dynamic_map.ego_state.pose.pose.position.y])
            p2 = np.array([obs.state.pose.pose.position.x , obs.state.pose.pose.position.y])
            p3 = p2 - p1
            p4 = math.hypot(p3[0],p3[1])

            obs_ffstate = get_frenet_state(obs.state, self.ref_path, self.ref_path_tangets)
            obs_yaw = get_yaw(obs.state)
            one_obs = (obs.state.pose.pose.position.x , obs.state.pose.pose.position.y , obs.state.twist.twist.linear.x ,
                     obs.state.twist.twist.linear.y , p4 , obs_ffstate.s , -obs_ffstate.d , obs_ffstate.vs, obs_ffstate.vd, 
                     obs.state.accel.accel.linear.x, obs.state.accel.accel.linear.y, obs_yaw, obs.dimension.length_x, obs.dimension.length_z)
            obs_tuples.append(one_obs)
        
        #sorted by distance
        sorted_obs = sorted(obs_tuples, key=lambda obs: obs[4])   # sort by distance
        i = 0
        for obs in sorted_obs:
            if i < self.considered_obs_num:
                closest_obs.append(obs)
                i = i + 1
            else:
                break
        
        return np.array(closest_obs)

    def prediction_obstacle(self, ob, max_prediction_time, delta_t): # we should do prediciton in driving space
        
        obs_paths = []

        for one_ob in ob:
            obsp_front = Frenet_path()
            obsp_back = Frenet_path()
            obsp_front.t = [t for t in np.arange(0.0, max_prediction_time, delta_t)]
            obsp_back.t = [t for t in np.arange(0.0, max_prediction_time, delta_t)]
            ax = 0#one_ob[9]
            ay = 0#one_ob[10]

            for i in range(len(obsp_front.t)):
                vx = 0#one_ob[2] + ax * delta_t * i
                vy = 0#one_ob[3] + ax * delta_t * i
                yaw = one_ob[11]   #only for constant prediction

                obspx = one_ob[0] + i * delta_t * vx
                obspy = one_ob[1] + i * delta_t * vy

                obsp_front.x.append(obspx + math.cos(yaw) * self.move_gap)
                obsp_front.y.append(obspy + math.sin(yaw) * self.move_gap)
                obsp_back.x.append(obspx - math.cos(yaw) * self.move_gap)
                obsp_back.y.append(obspy - math.sin(yaw) * self.move_gap)
                
            obs_paths.append(obsp_front)
            obs_paths.append(obsp_back)
        self.rviz_collision_checking_circle = self.rivz_element.draw_obs_circles(obs_paths, self.check_radius)


        return obs_paths

    def prediction_obstacle_pos(self, ob, max_prediction_time, delta_t): # we should do prediciton in driving space
        
        obs_pos = []

        for one_ob in ob:
            obsp = Frenet_path()
            obsp.t = [t for t in np.arange(0.0, max_prediction_time, delta_t)]
            obsp.len = one_ob[12]
            obsp.wid = one_ob[13]
            ax = one_ob[9]
            ay = one_ob[10]

            for i in range(len(obs.t)):
                vx = one_ob[2] + ax * delta_t * i
                vy = one_ob[3] + ax * delta_t * i
                yaw = one_ob[11]   #only for constant prediction

                obspx = one_ob[0] + i * delta_t * vx
                obspy = one_ob[1] + i * delta_t * vy

                obsp.x.append(obspx)
                obsp.y.append(obspy)
                obsp.yaw.append(yaw)
                
            obs_pos.append(obsp)
        #self.rviz_collision_checking_circle = self.rivz_element.draw_obs_circles(obs_pos, self.check_radius)
        return obs_pos

