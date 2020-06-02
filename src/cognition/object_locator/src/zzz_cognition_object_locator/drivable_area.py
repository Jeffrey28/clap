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

def calculate_next_drivable_area(tstates):
    '''
    The drivable area in the next unit of road (i.e. junction or road section)
    '''
    #TODO: now only support junction, static
    tstates.next_drivable_area = [] #clear in every step
    ego_s = 0 # ego_s should be 0 since it is the startp of the next lanes
    #TODO: input s when entering the next lanes
    rospy.loginfo("Start to deal with drivable area")

    if not tstates.static_map.in_junction:
        rospy.loginfo("next unit in junction:")
        rospy.loginfo("next junction: static junction boundary point num: %d", len(tstates.static_map.next_drivable_area.points))
        current_lane_index = tstates.ego_lane_index
        current_lane = tstates.static_map.lanes[int(round(current_lane_index))]

        # Use the lane tail as reference point to update next drivable area
        ego_x = current_lane.central_path_points[-1].position.x #input the ego lane end point
        ego_y = current_lane.central_path_points[-1].position.y #input the ego lane end point

        #a boundary point is represented by 6 numbers, namely x, y, vx, vy, omega and flag
        angle_list = []
        dist_list = []
        vx_list = []
        vy_list = []
        id_list = []
        omega_list = []
        flag_list = []

        if len(tstates.static_map.next_drivable_area.points) >= 3:
            for i in range(len(tstates.static_map.next_drivable_area.points)):
                node_point = tstates.static_map.next_drivable_area.points[i]
                last_node_point = tstates.static_map.next_drivable_area.points[i-1]
                #point = [node_point.x, node_point.y]
                #shatter the figure
                vertex_dist = math.sqrt(pow((node_point.x - last_node_point.x), 2) + pow((node_point.y - last_node_point.y), 2))
                if vertex_dist > 0.2:
                    #add interp points by step of 0.2m
                    for j in range(int(vertex_dist / 0.2)):
                        x = last_node_point.x + 0.2 * (j + 1) / vertex_dist * (node_point.x - last_node_point.x)
                        y = last_node_point.y + 0.2 * (j + 1) / vertex_dist * (node_point.y - last_node_point.y)
                        angle_list.append(math.atan2(y - ego_y, x - ego_x))
                        dist_list.append(math.sqrt(pow((x - ego_x), 2) + pow((y - ego_y), 2)))
                        #the velocity of static boundary is 0
                        vx_list.append(0)
                        vy_list.append(0)
                        omega_list.append(0)
                        flag_list.append(1) #static boundary
                        id_list.append(-1) #static boundary, interp points (can be deleted)
                
                angle_list.append(math.atan2(node_point.y - ego_y, node_point.x - ego_x))
                dist_list.append(math.sqrt(pow((node_point.x - ego_x), 2) + pow((node_point.y - ego_y), 2)))
                vx_list.append(0)
                vy_list.append(0)
                omega_list.append(0)
                flag_list.append(1) #static boundary
                id_list.append(-2) #static boundary, nodes (cannot be deleted)
                
        else:
            return

        #consider the vehicles in the junction
        check_list = np.zeros(len(dist_list))

        if len(tstates.surrounding_object_list) != 0:
            for i in range(len(tstates.surrounding_object_list)):
                obs = tstates.surrounding_object_list[i]
                dist_to_ego = math.sqrt(math.pow((obs.state.pose.pose.position.x - ego_x),2) 
                    + math.pow((obs.state.pose.pose.position.y - ego_y),2))
                
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
                                vel_world = np.matmul(rotation_mat, vel_obs)
                                #check if it should be reversed
                                vx = vel_world[0]
                                vy = vel_world[1]
                                omega = obs.state.twist.twist.angular.z

                                dist_list[j] = obstacle_dist
                                angle_list[j] = math.atan2(cross_position_y - ego_y, cross_position_x - ego_x) #might slightly differ
                                vx_list[j] = vx
                                vy_list[j] = vy
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
                                vel_world = np.matmul(rotation_mat, vel_obs)
                                #check if it should be reversed
                                vx = vel_world[0]
                                vy = vel_world[1]
                                omega = obs.state.twist.twist.angular.z
                                
                                #jxy0510: it is proved to be not correct only to keep the vertical velocity.
                                dist_list[j] = obstacle_dist
                                angle_list[j] = math.atan2(cross_position_y - ego_y, cross_position_x - ego_x) #might slightly differ
                                vx_list[j] = vx
                                vy_list[j] = vy
                                omega_list[j] = omega
                                flag_list[j] = 2
                                id_list[j] = i + id_extra_flag

        # merge the points of the same object to compress the data
        length_ori = len(angle_list)
        for i in range(length_ori):
            j = length_ori - 1 - i
            next_id = j + 1
            if id_list[j] == -2:
                # key points, should not delete
                continue
            if j == len(angle_list)-1:
                next_id = 0
            if j < 0:
                break
            if id_list[j] == id_list[j-1] and id_list[j] == id_list[next_id]:
                del angle_list[j]
                del dist_list[j]
                del vx_list[j]
                del vy_list[j]
                del omega_list[j]
                del flag_list[j]
            if id_list[j] >= 0 and id_list[next_id] != id_list[j]:
                # velocity of point i means the velocity of the edge between point i and point i+1
                vx_list[j] = 0
                vy_list[j] = 0

        for j in range(len(angle_list)):
            x = ego_x + dist_list[j] * math.cos(angle_list[j])
            y = ego_y + dist_list[j] * math.sin(angle_list[j])
            vx = vx_list[j]
            vy = vy_list[j]
            omega = omega_list[j]
            flag = flag_list[j]
            point = [x, y, vx, vy, omega, flag]
            tstates.next_drivable_area.append(point)
        
        #close the figure
        tstates.next_drivable_area.append(tstates.next_drivable_area[0])

        rospy.loginfo("next_drivable_area constructed with length %d", len(tstates.next_drivable_area))

    else:
        rospy.loginfo("next unit in lanes:")
        #create a list of lane section, each section is defined as (start point s, end point s)
        #calculate from the right most lane to the left most lane, drawing drivable area boundary in counterclockwise
        lane_num = len(tstates.static_map.next_lanes)
        #jxy: velocity of the vehicle in front and the vehicle behind are included in the lane sections
        #velocity is initialized to 0
        lane_sections = np.zeros((lane_num, 6))
        for i in range(len(tstates.static_map.next_lanes)):
            lane_sections[i, 0] = max(ego_s - 50, 0)
            lane_sections[i, 1] = min(ego_s + 50, tstates.static_map.next_lanes[i].central_path_points[-1].s)
            lane_sections[i, 2] = 0 #vx in front
            lane_sections[i, 3] = 0 #vy in front
            lane_sections[i, 4] = 0 #vx behind
            lane_sections[i, 5] = 0 #vy behind
            #TODO: projection to the vertial direction

        for obstacle in tstates.obstacles:
            if obstacle.lane_index == -1:
                continue
            else:
                #the obstacle in on the same road as the ego vehicle
                lane_index_rounded = int(round(obstacle.lane_index))
                #TODO: consider those on the lane boundary
                if obstacle.lane_dist_s > ego_s and obstacle.lane_dist_s < lane_sections[lane_index_rounded, 1]:
                    lane_sections[lane_index_rounded, 1] = obstacle.lane_dist_s - obstacle.dimension.length_x / 2.0
                    lane_sections[lane_index_rounded, 4] = obstacle.state.twist.twist.linear.x
                    lane_sections[lane_index_rounded, 5] = obstacle.state.twist.twist.linear.y
                elif obstacle.lane_dist_s <= ego_s and obstacle.lane_dist_s > lane_sections[lane_index_rounded, 0]:
                    lane_sections[lane_index_rounded, 0] = obstacle.lane_dist_s + obstacle.dimension.length_x / 2.0
                    lane_sections[lane_index_rounded, 2] = obstacle.state.twist.twist.linear.x
                    lane_sections[lane_index_rounded, 3] = obstacle.state.twist.twist.linear.y
        
        for i in range(len(tstates.static_map.next_lanes)):
            lane = tstates.static_map.next_lanes[i]
            if i == 0:
                next_lane_section_points_generation(lane_sections[i, 0], lane_sections[i, 1], lane_sections[i, 2], \
                    lane_sections[i, 3], lane_sections[i, 4], lane_sections[i, 5],lane.right_boundaries, tstates)
            
            if i != 0:
                next_lane_section_points_generation(lane_sections[i-1, 1], lane_sections[i, 1], lane_sections[i-1, 4], \
                    lane_sections[i-1, 5], lane_sections[i, 4], lane_sections[i, 5], lane.right_boundaries, tstates)
                if i != len(tstates.static_map.next_lanes) - 1:
                    next_lane_section_points_generation(lane_sections[i, 1], lane_sections[i+1, 1], lane_sections[i, 4], \
                    lane_sections[i, 5], lane_sections[i+1, 4], lane_sections[i+1, 5], lane.left_boundaries, tstates)
                else:
                    next_lane_section_points_generation(lane_sections[i, 1], lane_sections[i, 0], lane_sections[i, 4], \
                    lane_sections[i, 5], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, tstates)

            if len(tstates.static_map.next_lanes) == 1:
                next_lane_section_points_generation(lane_sections[i, 1], lane_sections[i, 0], lane_sections[i, 4], \
                    lane_sections[i, 5], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, tstates)

        for j in range(len(tstates.static_map.next_lanes)):
            i = len(tstates.static_map.next_lanes) - 1 - j
            lane = tstates.static_map.next_lanes[i]                
            if i != len(tstates.static_map.next_lanes) - 1:
                next_lane_section_points_generation(lane_sections[i+1, 0], lane_sections[i, 0], lane_sections[i+1, 2], \
                    lane_sections[i+1, 3], lane_sections[i, 4], lane_sections[i, 5], lane.left_boundaries, tstates)
                if i != 0:
                    next_lane_section_points_generation(lane_sections[i, 0], lane_sections[i-1, 0], lane_sections[i, 2], \
                    lane_sections[i, 3], lane_sections[i-1, 4], lane_sections[i-1, 5], lane.right_boundaries, tstates)

        #close the figure
        if len(tstates.next_drivable_area) > 0:
            tstates.next_drivable_area.append(tstates.next_drivable_area[0])

        rospy.loginfo("next_drivable_area constructed with length %d", len(tstates.next_drivable_area))


def next_lane_section_points_generation(starts, ends, startvx, startvy, endvx, endvy, lane_boundaries, tstates):

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

                pointx = point1.position.x + (point2.position.x - point1.position.x) * (smalls - point1.s) / (point2.s - point1.s)
                pointy = point1.position.y + (point2.position.y - point1.position.y) * (smalls - point1.s) / (point2.s - point1.s)
                point = [pointx, pointy, vx_s, vy_s, 0, 2]
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
                #the angular velocity in lanes need not be considered, so omega = 0

                pointx = point1.position.x + (point2.position.x - point1.position.x) * (bigs - point1.s) / (point2.s - point1.s)
                pointy = point1.position.y + (point2.position.y - point1.position.y) * (bigs - point1.s) / (point2.s - point1.s)
                point = [pointx, pointy, vx_s, vy_s, 0, 2]
                pointlist.append(point)

    if starts <= ends:
        for i in range(len(pointlist)):
            point = pointlist[i]
            tstates.next_drivable_area.append(point)
    else:
        # in reverse order
        for i in range(len(pointlist)):
            j = len(pointlist) - 1 - i
            tstates.next_drivable_area.append(pointlist[j])

def calculate_drivable_area(tstates):
    '''
    A list of boundary points of drivable area
    '''
    tstates.drivable_area = [] #clear in every step
    ego_s = tstates.ego_s
    rospy.loginfo("Start to deal with drivable area")

    if tstates.static_map.in_junction:
        rospy.loginfo("in junction:")
        rospy.loginfo("static junction boundary point num: %d", len(tstates.static_map.drivable_area.points))
        ego_x = tstates.ego_vehicle_state.state.pose.pose.position.x
        ego_y = tstates.ego_vehicle_state.state.pose.pose.position.y

        #a boundary point is represented by 6 numbers, namely x, y, vx, vy, omega and flag
        angle_list = []
        dist_list = []
        vx_list = []
        vy_list = []
        id_list = []
        omega_list = []
        flag_list = []

        if len(tstates.static_map.drivable_area.points) >= 3:
            for i in range(len(tstates.static_map.drivable_area.points)):
                node_point = tstates.static_map.drivable_area.points[i]
                last_node_point = tstates.static_map.drivable_area.points[i-1]
                #point = [node_point.x, node_point.y]
                #shatter the figure
                vertex_dist = math.sqrt(pow((node_point.x - last_node_point.x), 2) + pow((node_point.y - last_node_point.y), 2))
                if vertex_dist > 0.2:
                    #add interp points by step of 0.2m
                    for j in range(int(vertex_dist / 0.2)):
                        x = last_node_point.x + 0.2 * (j + 1) / vertex_dist * (node_point.x - last_node_point.x)
                        y = last_node_point.y + 0.2 * (j + 1) / vertex_dist * (node_point.y - last_node_point.y)
                        angle_list.append(math.atan2(y - ego_y, x - ego_x))
                        dist_list.append(math.sqrt(pow((x - ego_x), 2) + pow((y - ego_y), 2)))
                        #the velocity of static boundary is 0
                        vx_list.append(0)
                        vy_list.append(0)
                        omega_list.append(0)
                        flag_list.append(1) #static boundary
                        id_list.append(-1) #static boundary, interp points (can be deleted)
                
                angle_list.append(math.atan2(node_point.y - ego_y, node_point.x - ego_x))
                dist_list.append(math.sqrt(pow((node_point.x - ego_x), 2) + pow((node_point.y - ego_y), 2)))
                vx_list.append(0)
                vy_list.append(0)
                omega_list.append(0)
                flag_list.append(1) #static boundary
                id_list.append(-2) #static boundary, nodes (cannot be deleted)
                
        else:
            return

        #consider the vehicles in the junction
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
                                vel_world = np.matmul(rotation_mat, vel_obs)
                                #check if it should be reversed
                                vx = vel_world[0]
                                vy = vel_world[1]
                                omega = obs.state.twist.twist.angular.z

                                dist_list[j] = obstacle_dist
                                angle_list[j] = math.atan2(cross_position_y - ego_y, cross_position_x - ego_x) #might slightly differ
                                vx_list[j] = vx
                                vy_list[j] = vy
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
                                vel_world = np.matmul(rotation_mat, vel_obs)
                                #check if it should be reversed
                                vx = vel_world[0]
                                vy = vel_world[1]
                                omega = obs.state.twist.twist.angular.z
                                
                                #jxy0510: it is proved to be not correct only to keep the vertical velocity.
                                dist_list[j] = obstacle_dist
                                angle_list[j] = math.atan2(cross_position_y - ego_y, cross_position_x - ego_x) #might slightly differ
                                vx_list[j] = vx
                                vy_list[j] = vy
                                omega_list[j] = omega
                                flag_list[j] = 2
                                id_list[j] = i + id_extra_flag

        # merge the points of the same object to compress the data
        length_ori = len(angle_list)
        for i in range(length_ori):
            j = length_ori - 1 - i
            next_id = j + 1
            if id_list[j] == -2:
                # key points, should not delete
                continue
            if j == len(angle_list)-1:
                next_id = 0
            if j < 0:
                break
            if id_list[j] == id_list[j-1] and id_list[j] == id_list[next_id]:
                del angle_list[j]
                del dist_list[j]
                del vx_list[j]
                del vy_list[j]
                del omega_list[j]
                del flag_list[j]
            if id_list[j] >= 0 and id_list[next_id] != id_list[j]:
                # velocity of point i means the velocity of the edge between point i and point i+1
                vx_list[j] = 0
                vy_list[j] = 0

        for j in range(len(angle_list)):
            x = ego_x + dist_list[j] * math.cos(angle_list[j])
            y = ego_y + dist_list[j] * math.sin(angle_list[j])
            vx = vx_list[j]
            vy = vy_list[j]
            omega = omega_list[j]
            flag = flag_list[j]
            point = [x, y, vx, vy, omega, flag]
            tstates.drivable_area.append(point)
        
        #close the figure
        tstates.drivable_area.append(tstates.drivable_area[0])

        rospy.loginfo("drivable_area constructed with length %d", len(tstates.drivable_area))

    else:
        rospy.loginfo("in lanes:")
        #create a list of lane section, each section is defined as (start point s, end point s)
        #calculate from the right most lane to the left most lane, drawing drivable area boundary in counterclockwise
        lane_num = len(tstates.static_map.lanes)
        #jxy: velocity of the vehicle in front and the vehicle behind are included in the lane sections
        #velocity is initialized to 0
        lane_sections = np.zeros((lane_num, 6))
        for i in range(len(tstates.static_map.lanes)):
            lane_sections[i, 0] = max(ego_s - 50, 0)
            lane_sections[i, 1] = min(ego_s + 50, tstates.static_map.lanes[i].central_path_points[-1].s)
            lane_sections[i, 2] = 0 #vx in front
            lane_sections[i, 3] = 0 #vy in front
            lane_sections[i, 4] = 0 #vx behind
            lane_sections[i, 5] = 0 #vy behind
            #TODO: projection to the vertial direction

        for obstacle in tstates.obstacles:
            if obstacle.lane_index == -1:
                continue
            else:
                #the obstacle in on the same road as the ego vehicle
                lane_index_rounded = int(round(obstacle.lane_index))
                #TODO: consider those on the lane boundary
                if obstacle.lane_dist_s > ego_s and obstacle.lane_dist_s < lane_sections[lane_index_rounded, 1]:
                    lane_sections[lane_index_rounded, 1] = obstacle.lane_dist_s - obstacle.dimension.length_x / 2.0
                    lane_sections[lane_index_rounded, 4] = obstacle.state.twist.twist.linear.x
                    lane_sections[lane_index_rounded, 5] = obstacle.state.twist.twist.linear.y
                elif obstacle.lane_dist_s <= ego_s and obstacle.lane_dist_s > lane_sections[lane_index_rounded, 0]:
                    lane_sections[lane_index_rounded, 0] = obstacle.lane_dist_s + obstacle.dimension.length_x / 2.0
                    lane_sections[lane_index_rounded, 2] = obstacle.state.twist.twist.linear.x
                    lane_sections[lane_index_rounded, 3] = obstacle.state.twist.twist.linear.y
        
        for i in range(len(tstates.static_map.lanes)):
            lane = tstates.static_map.lanes[i]
            if i == 0:
                lane_section_points_generation(lane_sections[i, 0], lane_sections[i, 1], lane_sections[i, 2], \
                    lane_sections[i, 3], lane_sections[i, 4], lane_sections[i, 5],lane.right_boundaries, tstates)
            
            if i != 0:
                lane_section_points_generation(lane_sections[i-1, 1], lane_sections[i, 1], lane_sections[i-1, 4], \
                    lane_sections[i-1, 5], lane_sections[i, 4], lane_sections[i, 5], lane.right_boundaries, tstates)
                if i != len(tstates.static_map.lanes) - 1:
                    lane_section_points_generation(lane_sections[i, 1], lane_sections[i+1, 1], lane_sections[i, 4], \
                    lane_sections[i, 5], lane_sections[i+1, 4], lane_sections[i+1, 5], lane.left_boundaries, tstates)
                else:
                    lane_section_points_generation(lane_sections[i, 1], lane_sections[i, 0], lane_sections[i, 4], \
                    lane_sections[i, 5], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, tstates)

            if len(tstates.static_map.lanes) == 1:
                lane_section_points_generation(lane_sections[i, 1], lane_sections[i, 0], lane_sections[i, 4], \
                    lane_sections[i, 5], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, tstates)

        for j in range(len(tstates.static_map.lanes)):
            i = len(tstates.static_map.lanes) - 1 - j
            lane = tstates.static_map.lanes[i]                
            if i != len(tstates.static_map.lanes) - 1:
                lane_section_points_generation(lane_sections[i+1, 0], lane_sections[i, 0], lane_sections[i+1, 2], \
                    lane_sections[i+1, 3], lane_sections[i, 4], lane_sections[i, 5], lane.left_boundaries, tstates)
                if i != 0:
                    lane_section_points_generation(lane_sections[i, 0], lane_sections[i-1, 0], lane_sections[i, 2], \
                    lane_sections[i, 3], lane_sections[i-1, 4], lane_sections[i-1, 5], lane.right_boundaries, tstates)

        #close the figure
        if len(tstates.drivable_area) > 0:
            tstates.drivable_area.append(tstates.drivable_area[0])

        rospy.loginfo("drivable_area constructed with length %d", len(tstates.drivable_area))

def lane_section_points_generation(starts, ends, startvx, startvy, endvx, endvy, lane_boundaries, tstates):

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

                pointx = point1.position.x + (point2.position.x - point1.position.x) * (smalls - point1.s) / (point2.s - point1.s)
                pointy = point1.position.y + (point2.position.y - point1.position.y) * (smalls - point1.s) / (point2.s - point1.s)
                point = [pointx, pointy, vx_s, vy_s, 0, 2]
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
                #the angular velocity in lanes need not be considered, so omega = 0

                pointx = point1.position.x + (point2.position.x - point1.position.x) * (bigs - point1.s) / (point2.s - point1.s)
                pointy = point1.position.y + (point2.position.y - point1.position.y) * (bigs - point1.s) / (point2.s - point1.s)
                point = [pointx, pointy, vx_s, vy_s, 0, 2]
                pointlist.append(point)

    if starts <= ends:
        for i in range(len(pointlist)):
            point = pointlist[i]
            tstates.drivable_area.append(point)
    else:
        # in reverse order
        for i in range(len(pointlist)):
            j = len(pointlist) - 1 - i
            tstates.drivable_area.append(pointlist[j])