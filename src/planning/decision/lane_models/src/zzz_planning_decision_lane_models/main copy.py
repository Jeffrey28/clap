#!/usr/bin/env python

import rospy
from zzz_common.geometry import dense_polyline2d, dist_from_point_to_polyline2d
from zzz_planning_msgs.msg import DecisionTrajectory
from threading import Lock
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from zzz_cognition_msgs.msg import MapState
from zzz_navigation_msgs.msg import Map
from zzz_driver_msgs.utils import get_speed, get_yaw
from zzz_planning_decision_lane_models.local_trajectory import PolylineTrajectory, Werling_planner # TODO(Temps): Should seperate into continous models

# Make lat lon model as parameter


class MainDecision(object):
    def __init__(self, lon_decision=None, lat_decision=None, local_trajectory=None):
        self._dynamic_map_buffer = None
        self._static_map_buffer = None
        self._dynamic_boundary_buffer = None

        self._longitudinal_model_instance = lon_decision
        self._lateral_model_instance = lat_decision
        self._local_trajectory_instance = Werling_planner() # MPCTrajectory()

        self._dynamic_map_lock = Lock()

        self._load_next_junction_flag = 0
        self._load_next_road_flag = 0
        self._last_lane_index = -1
        self._initialize_flag = 0

    def receive_dynamic_map(self, dynamic_map):
        assert type(dynamic_map) == MapState
        self._dynamic_map_buffer = dynamic_map

    def receive_static_map(self, static_map):
        assert type(static_map) == Map
        self._static_map_buffer = static_map

    # def receive_dynamic_boundary(self, dynamic_boundary):
    #     assert type(dynamic_boundary) == DynamicBoundaryList
    #     self._dynamic_boundary_buffer = dynamic_boundary

    # update running in main node thread loop
    def update(self, close_to_lane=5):
        '''
        This function generate trajectory
        '''
        # update_dynamic_local_map
        if self._dynamic_map_buffer is None:
            return None
        else:
            dynamic_map = self._dynamic_map_buffer
            static_map = self._static_map_buffer
            dynamic_boundary = dynamic_map.jmap.boundary_list

        #TODO: jxy202011: we are achieving united decision, but will always go ahead. Fix when the vehicle is getting to the destination.

        if dynamic_map.model == dynamic_map.MODEL_JUNCTION_MAP: #jxy: enter junction, ready to enter another road

            self._initialize_flag = 0 #only initialize in the lanes

            #changing_lane_index, desired_speed = self._lateral_model_instance.lateral_decision(dynamic_map)
            changing_lane_index = self._last_lane_index
            desired_speed = 15.0 / 3.6
            #TODO: construct several virtual lanes in junction, and put in front and rear vehicles
            
            ego_speed = get_speed(dynamic_map.ego_state)

            rospy.logdebug("Planning (junction): target_speed = %f km/h, current_speed: %f km/h", desired_speed*3.6, ego_speed*3.6)

            if static_map is None:
                return None

            if len(static_map.drivable_area.points) < 3:
                return None

            #TODO: what if start point is in the junction?
            if self._load_next_road_flag == 0:
                self._local_trajectory_instance.remove_useless_lane(changing_lane_index) #only one remains
                self._local_trajectory_instance.prolong_frenet_lane(dynamic_map, static_map)
                self._load_next_road_flag = 1
            
            trajectory, local_desired_speed = self._local_trajectory_instance.get_trajectory(dynamic_map, dynamic_boundary, 0, desired_speed, 0)
            #TODO: consider next lanes: which to enter?

            msg = DecisionTrajectory()
            msg.trajectory = self.convert_ndarray_to_pathmsg(trajectory) # TODO: move to library
            msg.desired_speed = local_desired_speed
            msg.RLS_action = self._lateral_model_instance.decision_action

            return msg

            '''
            if dynamic_map.jmap.distance_to_lanes < close_to_lane and self._cleared_buff_flag == 1:
                print "built frenet lane in the junction"
                self._local_trajectory_instance.build_frenet_lane(dynamic_map, static_map)
                return None
            else:
                self._local_trajectory_instance.clean_frenet_lane()
                self._cleared_buff_flag = 1
                self._load_next_junction_flag = 0
                return None
            '''

        else:

            if self._initialize_flag == 0: #jxy: initialize
                self._local_trajectory_instance.clean_frenet_lane()
                self._local_trajectory_instance.build_frenet_lane(dynamic_map, static_map)
                self._load_next_junction_flag = 0
                print "built frenet lane"
                self._initialize_flag = 1
                return None
            
            if static_map is not None and self._load_next_junction_flag == 0:
                if len(static_map.next_drivable_area.points) > 3: #jxy: last is sticking point, equal to point 0, thus requiring 4 points
                    print "next junction loaded"
                    self._local_trajectory_instance.clean_frenet_lane() #TODO: modify the Werling planner by prolonging the path rather than clean it
                    self._local_trajectory_instance.build_frenet_lane(dynamic_map, static_map)
                    self._load_next_junction_flag = 1

            self._load_next_road_flag = 0

            changing_lane_index, desired_speed = self._lateral_model_instance.lateral_decision(dynamic_map)
            self._last_lane_index = changing_lane_index
            
            ego_speed = get_speed(dynamic_map.ego_state)

            rospy.logdebug("Planning (lanes): target_lane = %d, target_speed = %f km/h, current_speed: %f km/h", changing_lane_index, desired_speed*3.6, ego_speed*3.6)
            
            trajectory, local_desired_speed = self._local_trajectory_instance.get_trajectory(dynamic_map, dynamic_boundary, changing_lane_index, desired_speed, 1)


            msg = DecisionTrajectory()
            msg.trajectory = self.convert_ndarray_to_pathmsg(trajectory) # TODO: move to library
            msg.desired_speed = local_desired_speed
            msg.RLS_action = self._lateral_model_instance.decision_action

            return msg

    def convert_ndarray_to_pathmsg(self, path): 

        msg = Path()
        for wp in path:
            pose = PoseStamped()
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            msg.poses.append(pose)

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        return msg

    
