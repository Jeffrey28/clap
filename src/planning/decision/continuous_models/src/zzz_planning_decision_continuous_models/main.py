#!/usr/bin/env python

import rospy
import numpy as np
from zzz_common.geometry import dense_polyline2d, dist_from_point_to_polyline2d
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from zzz_cognition_msgs.msg import MapState, DynamicBoundary, DynamicBoundaryPoint
from zzz_driver_msgs.utils import get_speed, get_yaw

from predict import predict

class MainDecision(object):
    def __init__(self, trajectory_planner=None):
        self._dynamic_map_buffer = None
        self._dynamic_boundary_buffer = None
        self._trajectory_planner = trajectory_planner

    def receive_dynamic_map(self, dynamic_map):
        self._dynamic_map_buffer = dynamic_map

    def receive_dynamic_boundary(self, dynamic_boundary):
        rospy.loginfo("Received dynamic boundary")
        rospy.loginfo("received point num: %d", len(dynamic_boundary.boundary))
        rospy.loginfo("header: %s", dynamic_boundary.header.frame_id)
        self._dynamic_boundary_buffer = dynamic_boundary

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

        if dynamic_map.model == dynamic_map.MODEL_MULTILANE_MAP or len(dynamic_boundary.boundary) == 0:
            rospy.loginfo("Start to clear buff!!!\n\n\n")
            self._trajectory_planner.clear_buff(dynamic_map)
            return None
        else:
            rospy.loginfo("Start to carry out RL in the junction!!!\n\n\n")
            return self._trajectory_planner.trajectory_update(dynamic_map, dynamic_boundary)
