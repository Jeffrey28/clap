#!/usr/bin/env python

import rospy
import time
import numpy as np
from zzz_common.geometry import dense_polyline2d, dist_from_point_to_polyline2d
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from zzz_cognition_msgs.msg import MapState, DynamicBoundary, DynamicBoundaryPoint
from zzz_driver_msgs.utils import get_speed, get_yaw
from carla_msgs.msg import CarlaWorldInfo


from predict import predict

class MainDecision(object):
    def __init__(self, trajectory_planner=None):
        self._dynamic_map_buffer = None
        self._dynamic_boundary_buffer = None
        self._world_info_flag = time.time()
        self.current_time = time.time()
        self.last_time = time.time()
        self._trajectory_planner = trajectory_planner
        self._bridge_lock = 0

    def receive_dynamic_map(self, dynamic_map):
        self._dynamic_map_buffer = dynamic_map
    
    def receive_dynamic_boundary(self, dynamic_boundary):
        rospy.loginfo("Received dynamic boundary")
        rospy.loginfo("received point num: %d", len(dynamic_boundary.boundary))
        rospy.loginfo("header: %s", dynamic_boundary.header.frame_id)
        self._dynamic_boundary_buffer = dynamic_boundary

    def receive_world_info(self, world_info):
        rospy.loginfo("Received world info")
        self._world_info_flag = time.time()
        self._bridge_lock = 0 # Open the lock when world info comes in

    def update_trajectory(self):
        
        # This function generate trajectory

        # Update_dynamic_local_map
        if self._dynamic_map_buffer is None:
            rospy.loginfo("No dynamic map!")
            return None
        dynamic_map = self._dynamic_map_buffer

        # Update_dynamic_boundary
        if self._dynamic_boundary_buffer is None:
            rospy.loginfo("No dynamic boundary!")
            return None
        dynamic_boundary = self._dynamic_boundary_buffer

        # Check whether world info is dead
        self.current_time = time.time()
        if self.current_time - self.last_time > 10 or self._bridge_lock == 1:
            print("Bridge restart!")
            if self.current_time - self._world_info_flag > 10 or self._bridge_lock == 1:
                #the world info is dead
                print("World info is dead!")
                self._bridge_lock = 1
                return None

        self.last_time = self.current_time
        #if current_time - self._world_info_flag > 5:
        #    rospy.loginfo("World info is dead!")
        #    return None

        #jxy0715: use rule in false junctions, not training or clearing buff.
        ego_x = dynamic_map.ego_state.pose.pose.position.x
        ego_y = dynamic_map.ego_state.pose.pose.position.y
        if (-55 < ego_x < -45 and 17 < ego_y < 27) or (25 < ego_x < 35 and 25 < ego_y < 35):
            print("Rule decision in false junction!")
            return self._trajectory_planner.trajectory_update(dynamic_map, dynamic_boundary)

        else:
            if dynamic_map.model == dynamic_map.MODEL_MULTILANE_MAP:
                self._trajectory_planner.clear_buff(dynamic_map)
                return None
            else:
                return self._trajectory_planner.trajectory_update(dynamic_map, dynamic_boundary)

        


    

    