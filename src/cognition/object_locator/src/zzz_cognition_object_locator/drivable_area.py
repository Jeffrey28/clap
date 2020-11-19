import rospy
import numpy as np
from easydict import EasyDict as edict
from threading import Lock
import math
import copy
import time

from zzz_driver_msgs.msg import RigidBodyStateStamped
from zzz_navigation_msgs.msg import Map, Lane
from zzz_navigation_msgs.utils import get_lane_array, default_msg as navigation_default
from zzz_cognition_msgs.msg import MapState, LaneState, RoadObstacle
from zzz_cognition_msgs.utils import convert_tracking_box, default_msg as cognition_default
from zzz_perception_msgs.msg import TrackingBoxArray, DetectionBoxArray, ObjectSignals, DimensionWithCovariance
from zzz_common.geometry import dist_from_point_to_polyline2d, wrap_angle
from zzz_common.kinematics import get_frenet_state
from zzz_cognition_msgs.msg import DrivingSpace, DynamicBoundary, DynamicBoundaryPoint
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from zzz_driver_msgs.utils import get_speed, get_yaw

def calculate_drivable_area(tstates):
    '''
    A list of boundary points of drivable area
    '''
    tstates.drivable_area = [] #clear in every step
    ego_s = tstates.ego_s

    #step 1. generate key nodes
    ego_x = tstates.ego_vehicle_state.state.pose.pose.position.x
    ego_y = tstates.ego_vehicle_state.state.pose.pose.position.y

    #a boundary point is represented by 6 numbers, namely x, y, vx, vy, omega and flag
    angle_list = []
    dist_list = []
    vx_list = []
    vy_list = []
    id_list = []
    base_x_list = []
    base_y_list = []
    omega_list = []
    flag_list = []

    if tstates.static_map.in_junction:

        # jxy0710: try to merge the next static boundary into the junction boundary to make one closed boundary, then add dynamic objects
        #create a list of lane section, each section is defined as (start point s, end point s)
        #calculate from the right most lane to the left most lane, drawing drivable area boundary in counterclockwise
        lane_num = len(tstates.static_map.next_lanes)

        ego_s = 0 #for next unit (road section)

        lane_sections = np.zeros((lane_num, 6))
        for i in range(len(tstates.static_map.next_lanes)):
            lane_sections[i, 0] = max(ego_s - 10, 0)
            lane_sections[i, 1] = min(ego_s + 10, tstates.static_map.next_lanes[i].central_path_points[-1].s)
            lane_sections[i, 2] = 0 #vx in front
            lane_sections[i, 3] = 0 #vy in front
            lane_sections[i, 4] = 0 #vx behind
            lane_sections[i, 5] = 0 #vy behind
            #TODO: projection to the vertial direction

        next_static_area = []
        
        for i in range(len(tstates.static_map.next_lanes)):
            lane = tstates.static_map.next_lanes[i]
            if i == 0:
                next_lane_section_points_generation_united(lane_sections[i, 0], lane_sections[i, 1], lane_sections[i, 2], \
                    lane_sections[i, 3], lane_sections[i, 4], lane_sections[i, 5],lane.right_boundaries, next_static_area)
            
            if i != 0:
                next_lane_section_points_generation_united(lane_sections[i-1, 1], lane_sections[i, 1], lane_sections[i-1, 4], \
                    lane_sections[i-1, 5], lane_sections[i, 4], lane_sections[i, 5], lane.right_boundaries, next_static_area)
                if i != len(tstates.static_map.next_lanes) - 1:
                    next_lane_section_points_generation_united(lane_sections[i, 1], lane_sections[i+1, 1], lane_sections[i, 4], \
                    lane_sections[i, 5], lane_sections[i+1, 4], lane_sections[i+1, 5], lane.left_boundaries, next_static_area)
                else:
                    next_lane_section_points_generation_united(lane_sections[i, 1], lane_sections[i, 0], lane_sections[i, 4], \
                    lane_sections[i, 5], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, next_static_area)

            if len(tstates.static_map.next_lanes) == 1:
                next_lane_section_points_generation_united(lane_sections[i, 1], lane_sections[i, 0], lane_sections[i, 4], \
                    lane_sections[i, 5], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, next_static_area)

        #The lower part are removed.

        # joint point: the start point of the right most lane boundary of the next lanes. It is also a point in current drivable area.
        # It is the first point of the next static area.

        key_node_list = []

        if len(next_static_area) > 0:    
            joint_point = next_static_area[0]
            joint_point_x = joint_point[0]
            joint_point_y = joint_point[1]

            dist_array = []
            if len(tstates.static_map.drivable_area.points) >= 3:
                for i in range(len(tstates.static_map.drivable_area.points)):
                    node_point = tstates.static_map.drivable_area.points[i]
                    node_point_x = node_point.x
                    node_point_y = node_point.y
                    dist_to_joint_point = math.sqrt(pow((node_point_x - joint_point_x), 2) + pow((node_point_y - joint_point_y), 2))
                    dist_array.append(dist_to_joint_point)

            joint_point2_index = dist_array.index(min(dist_array)) # the index of the point in drivable area that equals to the joint point
        
            key_node_list = []
            for i in range(len(tstates.static_map.drivable_area.points)):
                j = len(tstates.static_map.drivable_area.points) - 1 - i
                node_point = tstates.static_map.drivable_area.points[j]
                key_node_list.append([node_point.x, node_point.y])
                if j == joint_point2_index:
                    for k in range(len(next_static_area)):
                        if k != 0: # the joint point needs not be added again
                            key_node_list.append(next_static_area[k])

            key_node_list.reverse()

        else:
            for i in range(len(tstates.static_map.drivable_area.points)):
                node_point = tstates.static_map.drivable_area.points[i]
                key_node_list.append([node_point.x, node_point.y])

    else:
        #create a list of lane section, each section is defined as (start point s, end point s)
        #calculate from the right most lane to the left most lane, drawing drivable area boundary in counterclockwise
        lane_num = len(tstates.static_map.lanes)

        lane_open_flag = [] #jxy202011: ready to include next junction boundary

        lane_sections = np.zeros((lane_num, 6))
        for i in range(len(tstates.static_map.lanes)):
            lane_sections[i, 0] = max(ego_s - 50, 0)
            lane_sections[i, 1] = min(ego_s + 50, tstates.static_map.lanes[i].central_path_points[-1].s)
            lane_sections[i, 2] = 0 #vx in front
            lane_sections[i, 3] = 0 #vy in front
            lane_sections[i, 4] = 0 #vx behind
            lane_sections[i, 5] = 0 #vy behind
            lane_open_flag.append(1) #default is open
            #TODO: projection to the vertial direction

        #next junction: paste with next junction, at joint point
        joint_point = tstates.static_map.lanes[0].right_boundaries[-1]
        joint_point_x = joint_point.boundary_point.position.x
        joint_point_y = joint_point.boundary_point.position.y

        dist_array = []
        next_key_node_list = [] #jxy: empty if still not loaded
            
        if len(tstates.static_map.next_drivable_area.points) >= 3:
            for i in range(len(tstates.static_map.next_drivable_area.points)):
                node_point = tstates.static_map.next_drivable_area.points[i]
                node_point_x = node_point.x
                node_point_y = node_point.y
                dist_to_joint_point = math.sqrt(pow((node_point_x - joint_point_x), 2) + pow((node_point_y - joint_point_y), 2))
                dist_array.append(dist_to_joint_point)

            joint_point2_index = dist_array.index(min(dist_array)) # the index of the point in drivable area that equals to the joint point

            for i in range(len(tstates.static_map.next_drivable_area.points)):
                ii = i + joint_point2_index
                if ii >= len(tstates.static_map.next_drivable_area.points) - 1:
                    ii = ii - len(tstates.static_map.next_drivable_area.points)

                node_point = tstates.static_map.next_drivable_area.points[ii]
                next_key_node_list.append([node_point.x, node_point.y])

            next_key_node_list.reverse()
            joint_point2_index = len(dist_array) - 1 - joint_point2_index
        
        print "dist array:"
        print dist_array

        key_node_list = []
        lane = tstates.static_map.lanes[0]
        lane_section_points_generation(lane_sections[0, 0], lane_sections[0, 1], lane_sections[0, 2], \
            lane_sections[0, 3], lane_sections[0, 4], lane_sections[0, 5],lane.right_boundaries, key_node_list)
        for i in range(len(next_key_node_list)):
            ii = i + joint_point2_index
            if ii >= len(next_key_node_list):
                ii = ii - len(next_key_node_list)
            key_node_list.append(next_key_node_list[ii])
        lane = tstates.static_map.lanes[-1]
        lane_section_points_generation(lane_sections[lane_num-1, 1], lane_sections[lane_num-1, 0], lane_sections[lane_num-1, 4], \
            lane_sections[lane_num-1, 5], lane_sections[lane_num-1, 2], lane_sections[lane_num-1, 3],lane.left_boundaries, key_node_list)

        print "key node list:"
        print key_node_list

    #step 2. interp in key nodes
    if len(key_node_list) >= 3:
        for i in range(len(key_node_list)):
            node_point = key_node_list[i]
            last_node_point = key_node_list[i-1]
            #point = [node_point.x, node_point.y]
            #shatter the figure
            vertex_dist = math.sqrt(pow((node_point[0] - last_node_point[0]), 2) + pow((node_point[1] - last_node_point[1]), 2))
            if vertex_dist > 0.2:
                #add interp points by step of 0.2m
                for j in range(int(vertex_dist / 0.2)):
                    x = last_node_point[0] + 0.2 * (j + 1) / vertex_dist * (node_point[0] - last_node_point[0])
                    y = last_node_point[1] + 0.2 * (j + 1) / vertex_dist * (node_point[1] - last_node_point[1])
                    angle_list.append(math.atan2(y - ego_y, x - ego_x))
                    dist_list.append(math.sqrt(pow((x - ego_x), 2) + pow((y - ego_y), 2)))
                    #the velocity of static boundary is 0
                    vx_list.append(0)
                    vy_list.append(0)
                    base_x_list.append(0)
                    base_y_list.append(0)
                    omega_list.append(0)
                    flag_list.append(1) #static boundary
                    id_list.append(-1) #static boundary, interp points (can be deleted)
            
            angle_list.append(math.atan2(node_point[1] - ego_y, node_point[0] - ego_x))
            dist_list.append(math.sqrt(pow((node_point[0] - ego_x), 2) + pow((node_point[1] - ego_y), 2)))
            vx_list.append(0)
            vy_list.append(0)
            base_x_list.append(0)
            base_y_list.append(0)
            omega_list.append(0)
            flag_list.append(1) #static boundary
            id_list.append(-2) #static boundary, nodes (cannot be deleted)
            
    else:
        return

    #step 3. consider the vehicles in the junction
    
    check_list = np.zeros(len(dist_list))

    if len(tstates.surrounding_object_list) != 0:
        for i in range(len(tstates.surrounding_object_list)):
            obs = tstates.surrounding_object_list[i]
            dist_to_ego = math.sqrt(math.pow((obs.state.pose.pose.position.x - tstates.ego_vehicle_state.state.pose.pose.position.x),2) 
                + math.pow((obs.state.pose.pose.position.y - tstates.ego_vehicle_state.state.pose.pose.position.y),2))
            
            if dist_to_ego < 35:
                #TODO: find a more robust method
                obs_x = obs.state.pose.pose.position.x
                obs_y = obs.state.pose.pose.position.y
                
                #Calculate the vertex points of the obstacle
                x = obs.state.pose.pose.orientation.x
                y = obs.state.pose.pose.orientation.y
                z = obs.state.pose.pose.orientation.z
                w = obs.state.pose.pose.orientation.w

                rotation_mat = np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y], [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x], [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]])
                rotation_mat_inverse = np.linalg.inv(rotation_mat) #those are the correct way to deal with quaternion

                vector_x = np.array([obs.dimension.length_x, 0, 0])
                vector_y = np.array([0, obs.dimension.length_y, 0])
                dx = np.matmul(rotation_mat_inverse, vector_x)
                dy = np.matmul(rotation_mat_inverse, vector_y)

                corner_list_x = np.zeros(4)
                corner_list_y = np.zeros(4)
                corner_list_x[0] = obs_x + dx[0]/2.0 + dy[0]/2.0
                corner_list_y[0] = obs_y + dx[1]/2.0 + dy[1]/2.0
                corner_list_x[1] = obs_x - dx[0]/2.0 + dy[0]/2.0
                corner_list_y[1] = obs_y - dx[1]/2.0 + dy[1]/2.0
                corner_list_x[2] = obs_x - dx[0]/2.0 - dy[0]/2.0
                corner_list_y[2] = obs_y - dx[1]/2.0 - dy[1]/2.0
                corner_list_x[3] = obs_x + dx[0]/2.0 - dy[0]/2.0
                corner_list_y[3] = obs_y + dx[1]/2.0 - dy[1]/2.0

                corner_list_angle = np.zeros(4)
                corner_list_dist = np.zeros(4)
                for j in range(4):
                    corner_list_angle[j] = math.atan2(corner_list_y[j] - ego_y, corner_list_x[j] - ego_x)
                    corner_list_dist[j] = math.sqrt(pow((corner_list_x[j] - ego_x), 2) + pow((corner_list_y[j] - ego_y), 2))

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

                for j in range(len(angle_list)):
                    if (angle_list[j] < corner_list_angle[big_corner_id] and angle_list[j] > corner_list_angle[small_corner_id]):
                        corner1 = -1
                        corner2 = -1
                        id_extra_flag = 0 # to distinguish edges in a whole object
                        if middle_corner_id == -1:
                            corner1 = big_corner_id
                            corner2 = small_corner_id
                        else:
                            if angle_list[j] < corner_list_angle[middle_corner_id]:
                                corner1 = middle_corner_id
                                corner2 = small_corner_id
                                id_extra_flag = 0.1
                            else:
                                corner1 = big_corner_id
                                corner2 = middle_corner_id
                                id_extra_flag = 0.2

                        #boundary direction
                        direction = math.atan2(corner_list_y[corner2] - corner_list_y[corner1], corner_list_x[corner2] - corner_list_x[corner1])
                        
                        cross_position_x = corner_list_x[corner2] + (corner_list_x[corner1] - corner_list_x[corner2]) * (angle_list[j] - corner_list_angle[corner2]) / (corner_list_angle[corner1] - corner_list_angle[corner2])
                        cross_position_y = corner_list_y[corner2] + (corner_list_y[corner1] - corner_list_y[corner2]) * (angle_list[j] - corner_list_angle[corner2]) / (corner_list_angle[corner1] - corner_list_angle[corner2])
                        obstacle_dist = math.sqrt(pow((cross_position_x - ego_x), 2) + pow((cross_position_y - ego_y), 2))
                        #TODO: find a more accurate method
                        if dist_list[j] > obstacle_dist:
                            
                            # Adapt to carla 0.9.8
                            vel_obs = np.array([obs.state.twist.twist.linear.x, obs.state.twist.twist.linear.y, obs.state.twist.twist.linear.z])
                            #vel_world = np.matmul(rotation_mat, vel_obs)
                            vel_world = vel_obs
                            #check if it should be reversed
                            vx = vel_world[0]
                            vy = vel_world[1]
                            omega = obs.state.twist.twist.angular.z

                            dist_list[j] = obstacle_dist
                            angle_list[j] = math.atan2(cross_position_y - ego_y, cross_position_x - ego_x) #might slightly differ
                            vx_list[j] = vx
                            vy_list[j] = vy
                            base_x_list[j] = obs_x
                            base_y_list[j] = obs_y
                            omega_list[j] = omega
                            flag_list[j] = 2 #dynamic boundary
                            id_list[j] = i + id_extra_flag #mark that this point is updated by the ith obstacle
                            check_list[j] = 1
                    elif (angle_list[j] + 2 * math.pi) > corner_list_angle[small_corner_id] and (angle_list[j] + 2 * math.pi) < corner_list_angle[big_corner_id]:
                        # cross pi
                        angle_list_plus = angle_list[j] + 2 * math.pi
                        corner1 = -1
                        corner2 = -1
                        id_extra_flag = 0
                        if middle_corner_id == -1:
                            corner1 = big_corner_id
                            corner2 = small_corner_id
                        else:
                            if angle_list_plus < corner_list_angle[middle_corner_id]:
                                corner1 = middle_corner_id
                                corner2 = small_corner_id
                                id_extra_flag = 0.1
                            else:
                                corner1 = big_corner_id
                                corner2 = middle_corner_id
                                id_extra_flag = 0.2

                        #boundary direction
                        direction = math.atan2(corner_list_y[corner2] - corner_list_y[corner1], corner_list_x[corner2] - corner_list_x[corner1])

                        cross_position_x = corner_list_x[corner2] + (corner_list_x[corner1] - corner_list_x[corner2]) * (angle_list_plus - corner_list_angle[corner2]) / (corner_list_angle[corner1] - corner_list_angle[corner2])
                        cross_position_y = corner_list_y[corner2] + (corner_list_y[corner1] - corner_list_y[corner2]) * (angle_list_plus- corner_list_angle[corner2]) / (corner_list_angle[corner1] - corner_list_angle[corner2])
                        obstacle_dist = math.sqrt(pow((cross_position_x - ego_x), 2) + pow((cross_position_y - ego_y), 2))
                        #TODO: find a more accurate method
                        if dist_list[j] > obstacle_dist:
                            # Adapt to carla 0.9.8
                            vel_obs = np.array([obs.state.twist.twist.linear.x, obs.state.twist.twist.linear.y, obs.state.twist.twist.linear.z])
                            #vel_world = np.matmul(rotation_mat, vel_obs)
                            vel_world = vel_obs
                            #check if it should be reversed
                            vx = vel_world[0]
                            vy = vel_world[1]
                            omega = obs.state.twist.twist.angular.z
                            
                            #jxy0510: it is proved to be not correct only to keep the vertical velocity.
                            dist_list[j] = obstacle_dist
                            angle_list[j] = math.atan2(cross_position_y - ego_y, cross_position_x - ego_x) #might slightly differ
                            vx_list[j] = vx
                            vy_list[j] = vy
                            base_x_list[j] = obs_x
                            base_y_list[j] = obs_y
                            omega_list[j] = omega
                            flag_list[j] = 2
                            id_list[j] = i + id_extra_flag

    # merge the points of the same object to compress the data
    length_ori = len(angle_list)
    for i in range(length_ori):
        j = length_ori - 1 - i
        next_id = j + 1
        if j == len(angle_list)-1:
            next_id = 0
        if j < 0:
            break

        if id_list[j] == -1:
            # jxy0615: downsizing more accurately
            if (id_list[j-1] == -1 or id_list[j-1] == -2) and (id_list[next_id] == -1 or id_list[next_id] == -2):
                del angle_list[j]
                del dist_list[j]
                del vx_list[j]
                del vy_list[j]
                del base_x_list[j]
                del base_y_list[j]
                del omega_list[j]
                del flag_list[j]
                continue
        elif id_list[j] == -2:
            # jxy0615: continue to downsize, static points too many
            if id_list[j-1] == -2 and id_list[next_id] == -2:
                delta_dis_prev = abs(dist_list[j] - dist_list[j-1])
                delta_dis_next = abs(dist_list[j] - dist_list[next_id])
                delta_angle_prev = abs(angle_list[j] - angle_list[j-1])
                delta_angle_next = abs(angle_list[j] - angle_list[next_id])
                if delta_dis_prev < 0.2 and delta_dis_next < 0.2 and delta_angle_prev < 2 and delta_angle_next < 2:
                    del angle_list[j]
                    del dist_list[j]
                    del vx_list[j]
                    del vy_list[j]
                    del base_x_list[j]
                    del base_y_list[j]
                    del omega_list[j]
                    del flag_list[j]
                    continue
        
        elif id_list[j] == id_list[j-1] and id_list[j] == id_list[next_id]:
            del angle_list[j]
            del dist_list[j]
            del vx_list[j]
            del vy_list[j]
            del base_x_list[j]
            del base_y_list[j]
            del omega_list[j]
            del flag_list[j]
            continue

        elif id_list[j] == id_list[j-1] and abs(id_list[j] - id_list[next_id]) < 0.5:
            angle_list[next_id] = (angle_list[next_id] + angle_list[j]) / 2
            if abs(angle_list[next_id] - angle_list[j]) > math.pi:
                angle_list[next_id] = angle_list[next_id] + math.pi
            if angle_list[next_id] > math.pi:
                angle_list[next_id] = angle_list[next_id] - 2 * math.pi
            dist_list[next_id] = (dist_list[next_id] + dist_list[j]) / 2
            del angle_list[j]
            del dist_list[j]
            del vx_list[j]
            del vy_list[j]
            del base_x_list[j]
            del base_y_list[j]
            del omega_list[j]
            del flag_list[j]
            continue #jxy0715: might not be the most reasonable. This sacrifices accuracy of observation.    
    
    for j in range(len(angle_list)):
        x = ego_x + dist_list[j] * math.cos(angle_list[j])
        y = ego_y + dist_list[j] * math.sin(angle_list[j])
        vx = vx_list[j]
        vy = vy_list[j]
        base_x = base_x_list[j]
        base_y = base_y_list[j]
        omega = omega_list[j]
        flag = flag_list[j]
        point = [x, y, vx, vy, base_x, base_y, omega, flag]
        tstates.drivable_area.append(point)
    
    
    #close the figure
    if len(tstates.drivable_area) > 0:
        tstates.drivable_area.append(tstates.drivable_area[0])

    #rospy.loginfo("drivable_area constructed with length %d", len(tstates.drivable_area))

def lane_section_points_generation(starts, ends, startvx, startvy, endvx, endvy, lane_boundaries, outpointlist):

    #set the velocity of the start point to 0, since the velocity of point i refers to the velocity of the edge between i and i+1
    startvx = 0
    startvy = 0
    if starts <= ends:
        smalls = starts
        bigs = ends
        smallvx = startvx
        smallvy = startvy
        bigvx = endvx
        bigvy = endvy
    else:
        smalls = ends
        bigs = starts
        smallvx = endvx
        smallvy = endvy
        bigvx = startvx
        bigvy = startvy
    
    pointlist = []
    for j in range(len(lane_boundaries)):
        if lane_boundaries[j].boundary_point.s <= smalls:
            if j == len(lane_boundaries) - 1:
                break
            if lane_boundaries[j+1].boundary_point.s > smalls:
                #if s < start point s, it cannot be the last point, so +1 is ok
                point1 = lane_boundaries[j].boundary_point
                point2 = lane_boundaries[j+1].boundary_point

                #projection to the longitudinal direction
                direction = math.atan2(point2.position.y - point1.position.y, point2.position.x - point1.position.x)

                v_value = smallvx * math.cos(direction) + smallvy * math.sin(direction)
                vx_s = v_value * math.cos(direction)
                vy_s = v_value * math.sin(direction)
                flag = 0
                if v_value > 0.1:
                    flag = 2
                else:
                    flag = 1

                pointx = point1.position.x + (point2.position.x - point1.position.x) * (smalls - point1.s) / (point2.s - point1.s)
                pointy = point1.position.y + (point2.position.y - point1.position.y) * (smalls - point1.s) / (point2.s - point1.s)
                point = [pointx, pointy, vx_s, vy_s, 0, 0, 0, 2]
                pointlist.append(point)
        elif lane_boundaries[j].boundary_point.s > smalls and lane_boundaries[j].boundary_point.s < bigs:
            point = [lane_boundaries[j].boundary_point.position.x, lane_boundaries[j].boundary_point.position.y, 0, 0, 0, 0, 0, 1]
            pointlist.append(point)
        elif lane_boundaries[j].boundary_point.s >= bigs:
            if j == 0:
                break
            if lane_boundaries[j-1].boundary_point.s < bigs:
                point1 = lane_boundaries[j-1].boundary_point
                point2 = lane_boundaries[j].boundary_point

                #projection to the longitudinal direction
                direction = math.atan2(point2.position.y - point1.position.y, point2.position.x - point1.position.x)

                v_value = bigvx * math.cos(direction) + bigvy * math.sin(direction)
                vx_s = v_value * math.cos(direction)
                vy_s = v_value * math.sin(direction)
                flag = 0
                if v_value > 0.1:
                    flag = 2
                else:
                    flag = 1
                #the angular velocity in lanes need not be considered, so omega = 0

                pointx = point1.position.x + (point2.position.x - point1.position.x) * (bigs - point1.s) / (point2.s - point1.s)
                pointy = point1.position.y + (point2.position.y - point1.position.y) * (bigs - point1.s) / (point2.s - point1.s)
                point = [pointx, pointy, vx_s, vy_s, 0, 0, 0, flag]
                pointlist.append(point)

    if starts <= ends:
        for i in range(len(pointlist)):
            point = pointlist[i]
            outpointlist.append(point)
    else:
        # in reverse order
        for i in range(len(pointlist)):
            j = len(pointlist) - 1 - i
            outpointlist.append(pointlist[j])

def next_lane_section_points_generation_united(starts, ends, startvx, startvy, endvx, endvy, lane_boundaries, next_static_area):

    #set the velocity of the start point to 0, since the velocity of point i refers to the velocity of the edge between i and i+1
    startvx = 0
    startvy = 0
    if starts <= ends:
        smalls = starts
        bigs = ends
        smallvx = startvx
        smallvy = startvy
        bigvx = endvx
        bigvy = endvy
    else:
        smalls = ends
        bigs = starts
        smallvx = endvx
        smallvy = endvy
        bigvx = startvx
        bigvy = startvy
    
    pointlist = []
    for j in range(len(lane_boundaries)):
        if lane_boundaries[j].boundary_point.s <= smalls:
            if j == len(lane_boundaries) - 1:
                break
            if lane_boundaries[j+1].boundary_point.s > smalls:
                #if s < start point s, it cannot be the last point, so +1 is ok
                point1 = lane_boundaries[j].boundary_point
                point2 = lane_boundaries[j+1].boundary_point

                #projection to the longitudinal direction
                direction = math.atan2(point2.position.y - point1.position.y, point2.position.x - point1.position.x)

                v_value = smallvx * math.cos(direction) + smallvy * math.sin(direction)
                vx_s = v_value * math.cos(direction)
                vy_s = v_value * math.sin(direction)
                flag = 0
                if v_value > 0.1:
                    flag = 2
                else:
                    flag = 1

                pointx = point1.position.x + (point2.position.x - point1.position.x) * (smalls - point1.s) / (point2.s - point1.s)
                pointy = point1.position.y + (point2.position.y - point1.position.y) * (smalls - point1.s) / (point2.s - point1.s)
                point = [pointx, pointy, vx_s, vy_s, 0, flag]
                pointlist.append(point)
        elif lane_boundaries[j].boundary_point.s > smalls and lane_boundaries[j].boundary_point.s < bigs:
            point = [lane_boundaries[j].boundary_point.position.x, lane_boundaries[j].boundary_point.position.y, 0, 0, 0, 1]
            pointlist.append(point)
        elif lane_boundaries[j].boundary_point.s >= bigs:
            if j == 0:
                break
            if lane_boundaries[j-1].boundary_point.s < bigs:
                point1 = lane_boundaries[j-1].boundary_point
                point2 = lane_boundaries[j].boundary_point

                #projection to the longitudinal direction
                direction = math.atan2(point2.position.y - point1.position.y, point2.position.x - point1.position.x)

                v_value = bigvx * math.cos(direction) + bigvy * math.sin(direction)
                vx_s = v_value * math.cos(direction)
                vy_s = v_value * math.sin(direction)
                flag = 0
                if v_value > 0.1:
                    flag = 2
                else:
                    flag = 1
                #the angular velocity in lanes need not be considered, so omega = 0

                pointx = point1.position.x + (point2.position.x - point1.position.x) * (bigs - point1.s) / (point2.s - point1.s)
                pointy = point1.position.y + (point2.position.y - point1.position.y) * (bigs - point1.s) / (point2.s - point1.s)
                point = [pointx, pointy] # only static
                pointlist.append(point)

    if starts <= ends:
        for i in range(len(pointlist)):
            point = pointlist[i]
            next_static_area.append(point)
    else:
        # in reverse order
        for i in range(len(pointlist)):
            j = len(pointlist) - 1 - i
            next_static_area.append(pointlist[j])