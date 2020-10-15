import numpy as np
import rospy
import matplotlib.pyplot as plt
import copy
import math
import time

from Werling.trajectory_structure import Frenet_path, Frenet_state
from zzz_common.kinematics import get_frenet_state
from zzz_driver_msgs.utils import get_speed, get_yaw
from common import rviz_display, convert_ndarray_to_pathmsg, convert_path_to_ndarray
from zzz_common.geometry import dense_polyline2d, dist_from_point_to_closedpolyline2d



class predict():
    def __init__(self, dynamic_map, dynamic_boundary, considered_obs_num, maxt, dt, robot_radius, radius_speed_ratio, move_gap, ego_speed):
        self.considered_obs_num = considered_obs_num
        self.maxt = maxt
        self.dt = dt
        self.check_radius = robot_radius + radius_speed_ratio * ego_speed
        self.move_gap = move_gap

        self.dynamic_map = dynamic_map
        self.dynamic_boundary = dynamic_boundary
        self.initialze_fail = False
        self.drivable_area_array = []

        self.rviz_collision_checking_circle = None
        self.rviz_predi_boundary = None
        self.rivz_element = rviz_display()

        #jxy: environment input here, then each path call the check_collision.

        #try:
        self.reference_path = self.dynamic_map.jmap.reference_path.map_lane.central_path_points
        ref_path_ori = convert_path_to_ndarray(self.reference_path)
        self.ref_path = dense_polyline2d(ref_path_ori, 2)
        self.ref_path_tangets = np.zeros(len(self.ref_path))

        ego_x = dynamic_map.ego_state.pose.pose.position.x
        ego_y = dynamic_map.ego_state.pose.pose.position.y

        self.drivable_area_array = self.decode(dynamic_boundary)
        self.drivable_area_array_list = self.predict_dynamic_boundary(self.drivable_area_array, maxt, dt, ego_x, ego_y)
        #except:
        #    rospy.logdebug("continous module: fail to initialize prediction")
        #    self.drivable_area_array_list = []

    def decode(self, dynamic_boundary):
        drivable_area_list=[]
        for i in range(len(dynamic_boundary.boundary)):
            if dynamic_boundary.boundary[i].flag < 10:
                point_x = dynamic_boundary.boundary[i].x
                point_y = dynamic_boundary.boundary[i].y
                point_vx = dynamic_boundary.boundary[i].vx
                point_vy = dynamic_boundary.boundary[i].vy
                point_base_x = dynamic_boundary.boundary[i].base_x
                point_base_y = dynamic_boundary.boundary[i].base_y
                point_omega = dynamic_boundary.boundary[i].omega
                point_flag = dynamic_boundary.boundary[i].flag
                
                position_point = [point_x, point_y, point_vx, point_vy, point_base_x, point_base_y, point_omega, point_flag]
                drivable_area_list.append(position_point)
        
        #print("drivable area decoded, length ", len(drivable_area_list))
        return np.array(drivable_area_list)
        
        
    def check_collision(self, fp):
        if len(fp.t) < 2 :
            return True

        t1 = time.time()

        fp_front = copy.deepcopy(fp)
        fp_back = copy.deepcopy(fp)

        self.rviz_collision_checking_circle = self.rivz_element.draw_circles(fp_front, fp_back, self.check_radius)
        try:
            for t in range(len(fp.yaw)):
                fp_front.x[t] = fp.x[t] + math.cos(fp.yaw[t]) * self.move_gap #jxy: 1m ahead
                fp_front.y[t] = fp.y[t] + math.sin(fp.yaw[t]) * self.move_gap #jxy: 1m behind
                fp_back.x[t] = fp.x[t] - math.cos(fp.yaw[t]) * self.move_gap
                fp_back.y[t] = fp.y[t] - math.sin(fp.yaw[t]) * self.move_gap
                
            drivable_area_array0 = self.drivable_area_array_list[0]
            temp_list = []
            for i in range(len(drivable_area_array0)):
                pointx = drivable_area_array0[i][0]
                pointy = drivable_area_array0[i][1]
                temp_list.append([pointx, pointy])

            drivable_area_array0_xy = np.array(temp_list)
            
            for t in range(len(fp.t)):
                #TODO: predict drivable area array
                drivable_area_array = self.drivable_area_array_list[t]
                temp_list = []
                for i in range(len(drivable_area_array)):
                    pointx = drivable_area_array[i][0]
                    pointy = drivable_area_array[i][1]
                    temp_list.append([pointx, pointy])

                drivable_area_array_xy = np.array(temp_list)
                dist_time1 = time.time()
                dist1, closest_id1, _, = dist_from_point_to_closedpolyline2d(fp_front.x[t], fp_front.y[t], drivable_area_array_xy)
                dist0, closest_id0, _, = dist_from_point_to_closedpolyline2d(fp_front.x[t], fp_front.y[t], drivable_area_array0_xy)
                dist_time2 = time.time()
                
                #jxy: hopefully this is the last bug to deal with.
                #TODO: there are sometimes the self circling problem, hopefully I can solve it in the future.
                point_flag = drivable_area_array[closest_id1][7]
                radius = self.check_radius
                if point_flag == 1: # static boundary part
                    radius = 2

                point_flag = drivable_area_array0[closest_id0][7]
                radius0 = self.check_radius
                if point_flag == 1: # static boundary part
                    radius0 = 2

                if dist1 <= radius or dist0 <= radius0:
                    t2 = time.time()
                    #print("one path check collision time: ", t2-t1)
                    return False
            
        except:
            return False #jxy: when it cannot judge whether there will be a collision or not, why should it pass the test?

        t2 = time.time()
        #print("one path check collision time: ", t2-t1)

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
                     obs.state.accel.accel.linear.x, obs.state.accel.accel.linear.y, obs_yaw)
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
                vx = one_ob[2] + ax * delta_t * i
                vy = one_ob[3] + ax * delta_t * i
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

    def predict_dynamic_boundary(self, dynamic_boundary_array, maxt, delta_t, ego_x, ego_y):
        
        dynamic_boundary_list = []
        
        steps = int(maxt / delta_t)
        rospy.logdebug("steps: %d", steps)

        dynamic_boundary = list(dynamic_boundary_array)

        for ii in range(steps):
            dt = ii * delta_t

            #step 1: reconstruct static boundary
            static_boundary = copy.deepcopy(dynamic_boundary)
            for i in range(len(dynamic_boundary)):
                j = len(dynamic_boundary) - 1 - i
                if static_boundary[j][7] == 2:
                    del static_boundary[j]

            if len(dynamic_boundary) == 0:
                # no dynamic boundary
                break

            if dynamic_boundary[0][7] == 2: #the close point is dynamic thus deleted in static boundary
                static_boundary.append(static_boundary[0])

            # step 2: predict moving obstacles
            # step 2-1: decode the moving obstacles in t0
            base_x_buffer = -1
            base_y_buffer = -1
            count_corner = 0
            count_obs = 0
            rec_obs = []
            rec_obs_list = []
            for j in range(len(dynamic_boundary)):
                if dynamic_boundary[j][7] == 2:
                    # check base point
                    pointbasex = dynamic_boundary[j][5]
                    pointbasey = dynamic_boundary[j][6]
                    if base_x_buffer == -1 and base_y_buffer == -1:
                        count_corner = 1 #initialize
                        count_obs = 1
                        base_x_buffer = pointbasex
                        base_y_buffer = pointbasey
                        rec_obs.append(list(dynamic_boundary[j]))
                    elif base_x_buffer == pointbasex and base_y_buffer == pointbasey:
                        count_corner = count_corner + 1
                        rec_obs.append(list(dynamic_boundary[j]))
                    else:
                        # change to another obs
                        count_corner = 1
                        base_x_buffer = pointbasex
                        base_y_buffer = pointbasey
                        rec_obs_list.append(rec_obs)
                        rec_obs = []
                        count_obs = count_obs + 1
                        rec_obs.append(list(dynamic_boundary[j]))

            if len(rec_obs) != 0:
                rec_obs_list.append(rec_obs)
            '''else:
                print("no obstacles in the junction")'''

            if len(rec_obs_list) != 1 and len(rec_obs_list) != 0:
                if rec_obs_list[0][0][4] == rec_obs_list[-1][0][4] and rec_obs_list[0][0][5] == rec_obs_list[-1][0][5]:
                    # the two are from one obstacle
                    last_rec_obs = rec_obs_list[-1]
                    first_rec_obs = rec_obs_list[0]
                    last_rec_obs.extend(first_rec_obs)
                    rec_obs_list[-1] = last_rec_obs
                    del rec_obs_list[0]
            
            # step 2-2: predict the moving obstacles, and modify by angle (regaining three corners)
            for i in range(len(rec_obs_list)):
                rec_obs = copy.deepcopy(rec_obs_list[i])
                corner_list_angle = []
                corner_list_dist = []
                for j in range(len(rec_obs)):
                    x = rec_obs[j][0]
                    y = rec_obs[j][1]
                    vx = rec_obs[j][2]
                    vy = rec_obs[j][3]
                    base_x = rec_obs[j][4]
                    base_y = rec_obs[j][5]
                    omega = rec_obs[j][6]
                    x_pv = x + vx * dt
                    y_pv = y + vy * dt
                    base_x = base_x + vx * dt
                    base_y = base_y + vy * dt
                    # rotate around base point
                    theta0 = math.atan2(y_pv - base_y, x_pv - base_x) + omega * dt
                    dist0 = np.linalg.norm([x_pv - base_x, y_pv - base_y])
                    x_p = base_x + dist0 * math.cos(theta0)
                    y_p = base_y + dist0 * math.sin(theta0)
                    rec_obs[j][0] = x_p
                    rec_obs[j][1] = y_p
                    corner_list_angle.append(math.atan2(y_p - ego_y, x_p - ego_x))
                    corner_list_dist.append(np.linalg.norm([x_p - ego_x, y_p - ego_y]))

                # reconstruct full obstacle and sort by angle
                if len(rec_obs) > 3: #TODO: fix it in drivable_area. Now only get 3 points if there are more than 3.
                    halfway = int(round(len(rec_obs)/2))
                    rec_obs_temp = copy.deepcopy(rec_obs[0])
                    temp1 = copy.deepcopy(rec_obs[halfway])
                    rec_obs_temp.append(temp1)
                    temp2 = copy.deepcopy(rec_obs[-1])
                    rec_obs_temp.append(temp2)

                if len(rec_obs) == 3:
                    point4_x = 2 * base_x - rec_obs[1][0]
                    point4_y = 2 * base_y - rec_obs[1][1]
                    corner_list_angle.append(math.atan2(point4_y - ego_y, point4_x - ego_x))
                    corner_list_dist.append(np.linalg.norm([point4_x - ego_x, point4_y - ego_y]))
                    temp = copy.deepcopy(rec_obs[-1])
                    rec_obs.append(temp)
                    rec_obs[3][0] = point4_x
                    rec_obs[3][1] = point4_y
                elif len(rec_obs) == 2:
                    point3_x = 2 * base_x - rec_obs[0][0]
                    point3_y = 2 * base_y - rec_obs[0][1]
                    corner_list_angle.append(math.atan2(point3_y - ego_y, point3_x - ego_x))
                    corner_list_dist.append(np.linalg.norm([point3_x - ego_x, point3_y - ego_y]))
                    #jxy: again sanctioned by python list copy.
                    temp = copy.deepcopy(rec_obs[-1])
                    rec_obs.append(temp)
                    rec_obs[2][0] = point3_x
                    rec_obs[2][1] = point3_y

                    point4_x = 2 * base_x - rec_obs[1][0]
                    point4_y = 2 * base_y - rec_obs[1][1]
                    corner_list_angle.append(math.atan2(point4_y - ego_y, point4_x - ego_x))
                    corner_list_dist.append(np.linalg.norm([point4_x - ego_x, point4_y - ego_y]))
                    temp = copy.deepcopy(rec_obs[-1])
                    rec_obs.append(temp)
                    rec_obs[3][0] = point4_x
                    rec_obs[3][1] = point4_y
                else:
                    continue # illegal obstacle (e.g. only one point)

                small_corner_id = np.argmin(corner_list_angle)
                big_corner_id = np.argmax(corner_list_angle)

                if corner_list_angle[big_corner_id] - corner_list_angle[small_corner_id] > math.pi:
                    #cross pi
                    for j in range(4):
                        if corner_list_angle[j] < 0:
                            corner_list_angle[j] += 2 * math.pi

                small_corner_id = np.argmin(corner_list_angle)
                big_corner_id = np.argmax(corner_list_angle)

                # add middle corner if we can see 3 corners
                smallest_dist_id = np.argmin(corner_list_dist)
                middle_corner_id = -1
                if not (small_corner_id == smallest_dist_id or big_corner_id == smallest_dist_id):
                    middle_corner_id = smallest_dist_id

                rec_obs_new = []
                rec_obs_new.append(rec_obs[small_corner_id])
                if middle_corner_id != -1:
                    rec_obs_new.append(rec_obs[middle_corner_id])
                rec_obs_new.append(rec_obs[big_corner_id])

                rec_obs_list[i] = rec_obs_new

            # step 3: construct the predicted boundary combining the last 2 steps
            # jxy: the angle order is different from matlab test, so reverse it
            static_boundary.reverse()
            #print("static boundary length: ", len(static_boundary))
            dynamic_boundary_p = copy.deepcopy(static_boundary)
            del dynamic_boundary_p[-1]
            angle_list_p = []
            dist_list_p = []
            for i in range(len(static_boundary) - 1):
                angle_list_p.append(math.atan2(static_boundary[i][1] - ego_y, static_boundary[i][0] - ego_x))
                dist_list_p.append(np.linalg.norm([static_boundary[i][1] - ego_y, static_boundary[i][0] - ego_x]))

            # notice that the farther ones should be considered in advance, or else the
            # shadow will not be handled correctly.
            # TODO: may still have some bugs in small probability, when a farther
            # obstacle blocked the sight of a nearer obstacle due to orientation, or
            # two dists are equal.

            center_x_list = []
            center_y_list = []
            center_dist_list = []
            for i in range(len(rec_obs_list)):
                center_x_list.append(rec_obs_list[i][0][3])
                center_y_list.append(rec_obs_list[i][0][4])
                center_dist_list.append(np.linalg.norm([center_x_list[i] - ego_x, center_y_list[i] - ego_y]))

            center_dist_list_sort = copy.deepcopy(center_dist_list)
            center_dist_list_sort.sort()
            rec_obs_list_temp = []
            for i in range(len(rec_obs_list)):
                m = center_dist_list.index(center_dist_list_sort[len(rec_obs_list) - 1 - i])
                rec_obs_list_temp.append(rec_obs_list[m])
            rec_obs_list = rec_obs_list_temp

            for i in range(len(rec_obs_list)):
                continue_flag = 0

                rec_obs = rec_obs_list[i]
                small_x = rec_obs[0][0]
                small_y = rec_obs[0][1]
                big_x = rec_obs[-1][0]
                big_y = rec_obs[-1][1]

                small_angle = math.atan2(small_y - ego_y, small_x - ego_x)
                big_angle = math.atan2(big_y - ego_y, big_x - ego_x)
                if small_angle > math.pi:
                    small_angle = small_angle + 2 * math.pi
                if big_angle > math.pi:
                    big_angle = big_angle - 2 * math.pi

                small_dist = np.linalg.norm(np.array([small_x - ego_x, small_y - ego_y]))
                big_dist = np.linalg.norm(np.array([big_x - ego_x, big_y - ego_y]))

                # create small angle and big angle shadow point
                small_cross_point_x = 0
                small_cross_point_y = 0
                big_cross_point_x = 0
                big_cross_point_y = 0
                small_wall_id = 0
                big_wall_id = 0

                for j in range(len(dynamic_boundary_p)):
                    next_id = j + 1
                    if next_id >= len(dynamic_boundary_p):
                        next_id = 0

                    if ((small_angle>angle_list_p[j] and small_angle-angle_list_p[j]<=math.pi) or (small_angle<=angle_list_p[j] and angle_list_p[j]-small_angle>=math.pi)) and \
                        ((small_angle<=angle_list_p[next_id] and angle_list_p[next_id]-small_angle<=math.pi) or (small_angle>angle_list_p[next_id] and small_angle-angle_list_p[next_id]>=math.pi)):
                        wall_id = j
                        wall2_id = next_id
                        wall_x = dynamic_boundary_p[wall_id][0]
                        wall_y = dynamic_boundary_p[wall_id][1]
                        wall2_x = dynamic_boundary_p[wall2_id][0]
                        wall2_y = dynamic_boundary_p[wall2_id][1]
                        egosmall_normal = np.array([-(ego_y - small_y), ego_x - small_x]) / np.linalg.norm(np.array([-(ego_y - small_y), ego_x - small_x]))
                        small_wall_id = wall_id

                        # cross point
                        dist_wall_egosmall = np.inner(np.array([wall_x - ego_x, wall_y - ego_y]), egosmall_normal)
                        dist_wall2_egosmall = np.inner(np.array([wall2_x - ego_x, wall2_y - ego_y]), egosmall_normal)
                        small_cross_point_x = wall_x + (wall2_x - wall_x) / (abs(dist_wall_egosmall) + abs(dist_wall2_egosmall)) * abs(dist_wall_egosmall)
                        small_cross_point_y = wall_y + (wall2_y - wall_y) / (abs(dist_wall_egosmall) + abs(dist_wall2_egosmall)) * abs(dist_wall_egosmall)
                        small_cross_point_dist = np.linalg.norm([small_cross_point_x - ego_x, small_cross_point_y - ego_y])
                        if small_cross_point_dist < small_dist:
                            continue_flag = 1
                    if ((big_angle>angle_list_p[j] and big_angle-angle_list_p[j]<=math.pi) or (big_angle<=angle_list_p[j] and angle_list_p[j]-big_angle>=math.pi)) and \
                        ((big_angle<=angle_list_p[next_id] and angle_list_p[next_id]-big_angle<=math.pi) or (big_angle>angle_list_p[next_id] and big_angle-angle_list_p[next_id]>=math.pi)):
                        wall_id = j
                        wall2_id = next_id
                        wall_x = dynamic_boundary_p[wall_id][0]
                        wall_y = dynamic_boundary_p[wall_id][1]
                        wall2_x = dynamic_boundary_p[wall2_id][0]
                        wall2_y = dynamic_boundary_p[wall2_id][1]
                        egobig_normal = np.array([-(ego_y - big_y), ego_x - big_x]) / np.linalg.norm(np.array([-(ego_y - big_y), ego_x - big_x]))
                        big_wall_id = wall_id

                        # cross point
                        dist_wall_egobig = np.inner(np.array([wall_x - ego_x, wall_y - ego_y]), egobig_normal)
                        dist_wall2_egobig = np.inner(np.array([wall2_x - ego_x, wall2_y - ego_y]), egobig_normal)
                        big_cross_point_x = wall_x + (wall2_x - wall_x) / (abs(dist_wall_egobig) + abs(dist_wall2_egobig)) * abs(dist_wall_egobig)
                        big_cross_point_y = wall_y + (wall2_y - wall_y) / (abs(dist_wall_egobig) + abs(dist_wall2_egobig)) * abs(dist_wall_egobig)
                        big_cross_point_dist = np.linalg.norm([big_cross_point_x - ego_x, big_cross_point_y - ego_y])
                        if big_cross_point_dist < big_dist:
                            continue_flag = 1

                if continue_flag == 1:
                    continue
                
                # remove the points between the cross points (if any, in increasing order
                if big_wall_id > small_wall_id:
                    # delete: small_wall_id + 1 to big_wall_id (+ 1 in python)
                    del dynamic_boundary_p[(small_wall_id + 1):(big_wall_id + 1)]
                    del angle_list_p[(small_wall_id + 1):(big_wall_id + 1)]
                    del dist_list_p[(small_wall_id + 1):(big_wall_id + 1)]
                elif big_wall_id < small_wall_id:
                    # delete: small_wall_id to the last, the first to big_wall_id (+ 1 in python)
                    if small_wall_id != len(dynamic_boundary_p) - 1:
                        del dynamic_boundary_p[(small_wall_id + 1):len(dynamic_boundary_p)]
                        del angle_list_p[(small_wall_id + 1):len(angle_list_p)]
                        del dist_list_p[(small_wall_id + 1):len(angle_list_p)]
                    del dynamic_boundary_p[0:(big_wall_id + 1)]
                    del angle_list_p[0:(big_wall_id + 1)]
                    del dist_list_p[0:(big_wall_id + 1)]

                # insert the cross points and rec obs points after the small angle cross point
                to_insert = []
                to_insert.append([small_cross_point_x, small_cross_point_y, 0, 0, 0, 0, 0, 1])
                to_insert.extend(rec_obs)
                to_insert.append([big_cross_point_x, big_cross_point_y, 0, 0, 0, 0, 0, 1])
                to_insert_angle = []
                to_insert_dist = []
                for k in range(len(to_insert)):
                    to_insert_angle.append(math.atan2(to_insert[k][1] - ego_y, to_insert[k][0] - ego_x))
                    to_insert_dist.append(np.linalg.norm([to_insert[k][0] - ego_x, to_insert[k][1] - ego_y]))

                for k in range(len(to_insert)):
                    to_insert[k] = np.array(to_insert[k])

                if small_wall_id >= (len(dynamic_boundary_p) - 1):
                    # '>' happens when big_wall_id is smaller than small_wall_id, so
                    # the first point to the big_wall_id are deleted, so simply put the
                    # to_insert at the end.
                    dynamic_boundary_p.extend(to_insert)
                    angle_list_p.extend(to_insert_angle)
                    dist_list_p.extend(to_insert_dist)
                else:
                    dynamic_boundary_p_temp = dynamic_boundary_p[0:(small_wall_id+1)]
                    dynamic_boundary_p_temp.extend(to_insert)
                    dynamic_boundary_p_temp.extend(dynamic_boundary_p[(small_wall_id+1):len(dynamic_boundary_p)])
                    angle_list_p_temp = angle_list_p[0:(small_wall_id + 1)]
                    angle_list_p_temp.extend(to_insert_angle)
                    angle_list_p_temp.extend(angle_list_p[(small_wall_id + 1):len(angle_list_p)])
                    dist_list_p_temp = dist_list_p[0:(small_wall_id + 1)]
                    dist_list_p_temp.extend(to_insert_dist)
                    dist_list_p_temp.extend(dist_list_p[(small_wall_id + 1):len(angle_list_p)])
                    
                    dynamic_boundary_p = dynamic_boundary_p_temp
                    angle_list_p = angle_list_p_temp
                    dist_list_p = dist_list_p_temp

            if len(dynamic_boundary_p) > 0:
                dynamic_boundary_p.append(dynamic_boundary_p[0])

            dynamic_boundary_p.reverse()

            dynamic_boundary_p_xy_array = np.array(dynamic_boundary_p)

            dynamic_boundary_list.append(dynamic_boundary_p_xy_array)

        self.rviz_predi_boundary = self.rivz_element.draw_predi_boundary(dynamic_boundary_list)

        return dynamic_boundary_list
