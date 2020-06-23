
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

#jxy 20191125: first output the driving space, then use the driving space for cognition. 
#For this demo version, it will be a unified module, in future versions, this will be split into 2 modules.

class DrivingSpaceConstructor:
    def __init__(self, lane_dist_thres=5):
        self._static_map_lock = Lock()
        self._static_map_buffer = None

        self._ego_vehicle_state_lock = Lock()
        self._ego_vehicle_state_buffer = None

        self._surrounding_object_list_lock = Lock()
        self._surrounding_object_list_buffer = None

        self._traffic_light_detection_lock = Lock()
        self._traffic_light_detection_buffer = None

        self._driving_space = None
        self._obstacles_markerarray = None
        self._lanes_boundary_markerarray = None
        
        self._lane_dist_thres = lane_dist_thres

        self._ego_vehicle_distance_to_lane_head = [] # distance from vehicle to lane start
        self._ego_vehicle_distance_to_lane_tail = [] # distance from vehicle to lane end

    @property
    def driving_space(self):
        return self._driving_space

    # ====== Data Receiver =======

    def receive_static_map(self, static_map):
        assert type(static_map) == Map
        with self._static_map_lock:
            self._static_map_buffer = static_map
            rospy.loginfo("Updated Local Static Map: lanes_num = %d, in_junction = %d, target_lane_index = %d",
                len(static_map.lanes), int(static_map.in_junction), static_map.target_lane_index)

    def receive_object_list(self, object_list):
        assert type(object_list) == TrackingBoxArray
        with self._surrounding_object_list_lock:
            if self._ego_vehicle_state_buffer != None:
                self._surrounding_object_list_buffer = convert_tracking_box(object_list, self._ego_vehicle_state_buffer)
                #jxy: the converted objects are in the RoadObstacle() format

    def receive_ego_state(self, state):
        assert type(state) == RigidBodyStateStamped
        with self._ego_vehicle_state_lock:
            self._ego_vehicle_state_buffer = state
            #TODO: wrap ego vehicle just like wrapping obstacle

    def receive_traffic_light_detection(self, detection):
        assert type(detection) == DetectionBoxArray
        with self._traffic_light_detection_lock:
            self._traffic_light_detection_buffer = detection

    # ====== Data Updator =======

    def update_driving_space(self):

        tstates = edict()

        # Skip if not ready
        if not self._ego_vehicle_state_buffer:
            return False

        with self._ego_vehicle_state_lock:
            tstates.ego_vehicle_state = copy.deepcopy(self._ego_vehicle_state_buffer) 

        # Update buffer information
        tstates.surrounding_object_list = copy.deepcopy(self._surrounding_object_list_buffer or [])

        tstates.static_map = copy.deepcopy(self._static_map_buffer or navigation_default(Map)) 
        static_map = tstates.static_map # for easier access
        tstates.static_map_lane_path_array = get_lane_array(tstates.static_map.lanes)
        tstates.static_map_lane_tangets = [[point.tangent for point in lane.central_path_points] for lane in tstates.static_map.lanes]
        tstates.obstacles = [] #about to add in the following steps
        tstates.ego_lane_index = -1 #about to modify in the following steps
        tstates.ego_s = -1
        tstates.drivable_area = []
        tstates.next_drivable_area = []
        self._driving_space = DrivingSpace()

        # Update driving_space with tstate
        if static_map.in_junction or len(static_map.lanes) == 0:
            rospy.logdebug("In junction due to static map report junction location")
            self.calculate_drivable_area(tstates)
        else:
            for lane in tstates.static_map.lanes:
                self._driving_space.lanes.append(lane)
            #jxy: why is target lane in static map?
            self.locate_ego_vehicle_in_lanes(tstates)
            self.locate_obstacle_in_lanes(tstates)
            self.locate_stop_sign_in_lanes(tstates)
            self.locate_speed_limit_in_lanes(tstates)
            self.calculate_drivable_area(tstates)
            self.calculate_next_drivable_area(tstates)
        
        self._driving_space.header.frame_id = "map"
        self._driving_space.header.stamp = rospy.Time.now()
        self._driving_space.ego_state = tstates.ego_vehicle_state.state
        #TODO: drivable area. It should be updated by obstacles. Only static drivable area is not OK.
        self._driving_space.obstacles = tstates.obstacles
        rospy.logdebug("len(self._static_map.lanes): %d", len(tstates.static_map.lanes))

        self.dynamic_boundary = DynamicBoundary()
        self.dynamic_boundary.header.frame_id = "map"
        self.dynamic_boundary.header.stamp = rospy.Time.now()
        for i in range(len(tstates.drivable_area)):
            drivable_area_point = tstates.drivable_area[i]
            boundary_point = DynamicBoundaryPoint()
            boundary_point.x = drivable_area_point[0]
            boundary_point.y = drivable_area_point[1]
            boundary_point.vx = drivable_area_point[2]
            boundary_point.vy = drivable_area_point[3]
            boundary_point.omega = drivable_area_point[4]
            boundary_point.flag = drivable_area_point[5]
            self.dynamic_boundary.boundary.append(boundary_point)

        #jxy0510: extend the dynamic boundary by lanes
        '''
        if not (tstates.static_map.in_junction):
            for lane in tstates.static_map.lanes:
                if len(lane.right_boundaries) > 0 and len(lane.left_boundaries) > 0:
                    #the left most lane boundary line cannot be broken, or else it won't be the left most lane
                    if lane.right_boundaries[0].boundary_type == 1:
                        for lb in lane.right_boundaries:
                            lane_point = DynamicBoundaryPoint()
                            lane_point.x = lb.boundary_point.position.x
                            lane_point.y = lb.boundary_point.position.y
                            lane_point.vx = 0
                            lane_point.vy = 0
                            lane_point.omega = 0
                            lane_point.flag = 3
                            self.dynamic_boundary.boundary.append(lane_point
                            '''
        
        rospy.loginfo("---------------------dynamic boundary updated!")

        #visualization
        #1. lanes
        self._lanes_markerarray = MarkerArray()

        count = 0
        if not (tstates.static_map.in_junction):
            biggest_id = 0 #TODO: better way to find the smallest id
            
            for lane in tstates.static_map.lanes:
                if lane.index > biggest_id:
                    biggest_id = lane.index
                tempmarker = Marker() #jxy: must be put inside since it is python
                tempmarker.header.frame_id = "map"
                tempmarker.header.stamp = rospy.Time.now()
                tempmarker.ns = "zzz/cognition"
                tempmarker.id = count
                tempmarker.type = Marker.LINE_STRIP
                tempmarker.action = Marker.ADD
                tempmarker.scale.x = 0.12
                tempmarker.color.r = 1.0
                tempmarker.color.g = 0.0
                tempmarker.color.b = 0.0
                tempmarker.color.a = 0.5
                tempmarker.lifetime = rospy.Duration(0.5)

                for lanepoint in lane.central_path_points:
                    p = Point()
                    p.x = lanepoint.position.x
                    p.y = lanepoint.position.y
                    p.z = lanepoint.position.z
                    tempmarker.points.append(p)
                self._lanes_markerarray.markers.append(tempmarker)
                count = count + 1

        #2. lane boundary line
        self._lanes_boundary_markerarray = MarkerArray()

        count = 0
        if not (tstates.static_map.in_junction):
            #does not draw lane when ego vehicle is in the junction
            
            for lane in tstates.static_map.lanes:
                if len(lane.right_boundaries) > 0 and len(lane.left_boundaries) > 0:
                    tempmarker = Marker() #jxy: must be put inside since it is python
                    tempmarker.header.frame_id = "map"
                    tempmarker.header.stamp = rospy.Time.now()
                    tempmarker.ns = "zzz/cognition"
                    tempmarker.id = count

                    #each lane has the right boundary, only the lane with the smallest id has the left boundary
                    tempmarker.type = Marker.LINE_STRIP
                    tempmarker.action = Marker.ADD
                    tempmarker.scale.x = 0.15
                    if lane.right_boundaries[0].boundary_type == 1: #broken lane is set gray
                        tempmarker.color.r = 0.6
                        tempmarker.color.g = 0.6
                        tempmarker.color.b = 0.5
                        tempmarker.color.a = 0.5
                    else:
                        tempmarker.color.r = 1.0
                        tempmarker.color.g = 1.0
                        tempmarker.color.b = 1.0
                        tempmarker.color.a = 0.5
                    tempmarker.lifetime = rospy.Duration(0.5)

                    for lb in lane.right_boundaries:
                        p = Point()
                        p.x = lb.boundary_point.position.x
                        p.y = lb.boundary_point.position.y
                        p.z = lb.boundary_point.position.z
                        tempmarker.points.append(p)
                    self._lanes_boundary_markerarray.markers.append(tempmarker)
                    count = count + 1

                    #biggest id: draw left lane
                    if lane.index == biggest_id:
                        tempmarker = Marker() #jxy: must be put inside since it is python
                        tempmarker.header.frame_id = "map"
                        tempmarker.header.stamp = rospy.Time.now()
                        tempmarker.ns = "zzz/cognition"
                        tempmarker.id = count

                        #each lane has the right boundary, only the lane with the biggest id has the left boundary
                        tempmarker.type = Marker.LINE_STRIP
                        tempmarker.action = Marker.ADD
                        tempmarker.scale.x = 0.3
                        if lane.left_boundaries[0].boundary_type == 1: #broken lane is set gray
                            tempmarker.color.r = 0.6
                            tempmarker.color.g = 0.6
                            tempmarker.color.b = 0.6
                            tempmarker.color.a = 0.5
                        else:
                            tempmarker.color.r = 1.0
                            tempmarker.color.g = 1.0
                            tempmarker.color.b = 1.0
                            tempmarker.color.a = 0.5
                        tempmarker.lifetime = rospy.Duration(0.5)

                        for lb in lane.left_boundaries:
                            p = Point()
                            p.x = lb.boundary_point.position.x
                            p.y = lb.boundary_point.position.y
                            p.z = lb.boundary_point.position.z
                            tempmarker.points.append(p)
                        self._lanes_boundary_markerarray.markers.append(tempmarker)
                        count = count + 1

        #3. obstacle
        self._obstacles_markerarray = MarkerArray()
        
        count = 0
        if tstates.surrounding_object_list is not None:
            for obs in tstates.surrounding_object_list:
                dist_to_ego = math.sqrt(math.pow((obs.state.pose.pose.position.x - tstates.ego_vehicle_state.state.pose.pose.position.x),2) 
                    + math.pow((obs.state.pose.pose.position.y - tstates.ego_vehicle_state.state.pose.pose.position.y),2))
                
                if dist_to_ego < 50:
                    tempmarker = Marker() #jxy: must be put inside since it is python
                    tempmarker.header.frame_id = "map"
                    tempmarker.header.stamp = rospy.Time.now()
                    tempmarker.ns = "zzz/cognition"
                    tempmarker.id = count
                    tempmarker.type = Marker.CUBE
                    tempmarker.action = Marker.ADD
                    tempmarker.pose = obs.state.pose.pose
                    tempmarker.scale.x = obs.dimension.length_x
                    tempmarker.scale.y = obs.dimension.length_y
                    tempmarker.scale.z = obs.dimension.length_z
                    if obs.lane_index == -1:
                        tempmarker.color.r = 0.5
                        tempmarker.color.g = 0.5
                        tempmarker.color.b = 0.5
                    elif obs.lane_dist_left_t == 0 or obs.lane_dist_right_t == 0:
                        # those who is on the lane boundary, warn by yellow
                        tempmarker.color.r = 1.0
                        tempmarker.color.g = 1.0
                        tempmarker.color.b = 0.0
                    else:
                        tempmarker.color.r = 1.0
                        tempmarker.color.g = 0.0
                        tempmarker.color.b = 1.0
                    if tstates.static_map.in_junction:
                        tempmarker.color.r = 1.0
                        tempmarker.color.g = 0.0
                        tempmarker.color.b = 1.0
                    tempmarker.color.a = 0.5
                    tempmarker.lifetime = rospy.Duration(0.5)

                    self._obstacles_markerarray.markers.append(tempmarker)
                    count = count + 1
            
            for obs in tstates.surrounding_object_list:
                dist_to_ego = math.sqrt(math.pow((obs.state.pose.pose.position.x - tstates.ego_vehicle_state.state.pose.pose.position.x),2) 
                    + math.pow((obs.state.pose.pose.position.y - tstates.ego_vehicle_state.state.pose.pose.position.y),2))
                
                if dist_to_ego < 50:
                    tempmarker = Marker() #jxy: must be put inside since it is python
                    tempmarker.header.frame_id = "map"
                    tempmarker.header.stamp = rospy.Time.now()
                    tempmarker.ns = "zzz/cognition"
                    tempmarker.id = count
                    tempmarker.type = Marker.ARROW
                    tempmarker.action = Marker.ADD
                    tempmarker.scale.x = 0.4
                    tempmarker.scale.y = 0.7
                    tempmarker.scale.z = 0.75
                    tempmarker.color.r = 1.0
                    tempmarker.color.g = 1.0
                    tempmarker.color.b = 0.0
                    tempmarker.color.a = 0.5
                    tempmarker.lifetime = rospy.Duration(0.5)

                    #quaternion transform for obs velocity in carla 0.9.8

                    x = obs.state.pose.pose.orientation.x
                    y = obs.state.pose.pose.orientation.y
                    z = obs.state.pose.pose.orientation.z
                    w = obs.state.pose.pose.orientation.w

                    rotation_mat = np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y], [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x], [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]])
                    rotation_mat_inverse = np.linalg.inv(rotation_mat) #those are the correct way to deal with quaternion

                    vel_obs = np.array([obs.state.twist.twist.linear.x, obs.state.twist.twist.linear.y, obs.state.twist.twist.linear.z])
                    #vel_world = np.matmul(rotation_mat, vel_obs)
                    vel_world = vel_obs
                    #check if it should be reversed
                    obs_vx_world = vel_world[0]
                    obs_vy_world = vel_world[1]
                    obs_vz_world = vel_world[2]

                    startpoint = Point()
                    endpoint = Point()
                    startpoint.x = obs.state.pose.pose.position.x
                    startpoint.y = obs.state.pose.pose.position.y
                    startpoint.z = obs.state.pose.pose.position.z
                    endpoint.x = obs.state.pose.pose.position.x + obs_vx_world
                    endpoint.y = obs.state.pose.pose.position.y + obs_vy_world
                    endpoint.z = obs.state.pose.pose.position.z + obs_vz_world
                    tempmarker.points.append(startpoint)
                    tempmarker.points.append(endpoint)

                    self._obstacles_markerarray.markers.append(tempmarker)
                    count = count + 1

        #4. the labels of objects
        self._obstacles_label_markerarray = MarkerArray()

        count = 0
        if tstates.surrounding_object_list is not None:                    
            for obs in tstates.surrounding_object_list:
                dist_to_ego = math.sqrt(math.pow((obs.state.pose.pose.position.x - tstates.ego_vehicle_state.state.pose.pose.position.x),2) 
                    + math.pow((obs.state.pose.pose.position.y - tstates.ego_vehicle_state.state.pose.pose.position.y),2))
                
                if dist_to_ego < 50:
                    tempmarker = Marker() #jxy: must be put inside since it is python
                    tempmarker.header.frame_id = "map"
                    tempmarker.header.stamp = rospy.Time.now()
                    tempmarker.ns = "zzz/cognition"
                    tempmarker.id = count
                    tempmarker.type = Marker.TEXT_VIEW_FACING
                    tempmarker.action = Marker.ADD
                    hahaha = obs.state.pose.pose.position.z + 1.0
                    tempmarker.pose.position.x = obs.state.pose.pose.position.x
                    tempmarker.pose.position.y = obs.state.pose.pose.position.y
                    tempmarker.pose.position.z = hahaha
                    tempmarker.scale.z = 0.6
                    tempmarker.color.r = 1.0
                    tempmarker.color.g = 0.0
                    tempmarker.color.b = 1.0
                    tempmarker.color.a = 0.5
                    tempmarker.text = " lane_index: " + str(obs.lane_index) + "\n lane_dist_right_t: " + str(obs.lane_dist_right_t) + "\n lane_dist_left_t: " + str(obs.lane_dist_left_t) + "\n lane_anglediff: " + str(obs.lane_anglediff)
                    tempmarker.lifetime = rospy.Duration(0.5)

                    self._obstacles_label_markerarray.markers.append(tempmarker)
                    count = count + 1


        #5. ego vehicle visualization
        self._ego_markerarray = MarkerArray()

        #rospy.loginfo("We are at position: %f %f\n\n\n\n", tstates.ego_vehicle_state.state.pose.pose.position.x, tstates.ego_vehicle_state.state.pose.pose.position.y)

        tempmarker = Marker()
        tempmarker.header.frame_id = "map"
        tempmarker.header.stamp = rospy.Time.now()
        tempmarker.ns = "zzz/cognition"
        tempmarker.id = 1
        tempmarker.type = Marker.CUBE
        tempmarker.action = Marker.ADD
        tempmarker.pose = tstates.ego_vehicle_state.state.pose.pose
        tempmarker.scale.x = 4.0 #jxy: I don't know...
        tempmarker.scale.y = 2.0
        tempmarker.scale.z = 1.8
        tempmarker.color.r = 1.0
        tempmarker.color.g = 0.0
        tempmarker.color.b = 0.0
        tempmarker.color.a = 0.5
        tempmarker.lifetime = rospy.Duration(0.5)

        self._ego_markerarray.markers.append(tempmarker)

        #quaternion transform for ego velocity

        x = tstates.ego_vehicle_state.state.pose.pose.orientation.x
        y = tstates.ego_vehicle_state.state.pose.pose.orientation.y
        z = tstates.ego_vehicle_state.state.pose.pose.orientation.z
        w = tstates.ego_vehicle_state.state.pose.pose.orientation.w

        # rotation_mat = np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y], [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x], [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]])
        # rotation_mat_inverse = np.linalg.inv(rotation_mat) #those are the correct way to deal with quaternion

        vel_self = np.array([[tstates.ego_vehicle_state.state.twist.twist.linear.x], [tstates.ego_vehicle_state.state.twist.twist.linear.y], [tstates.ego_vehicle_state.state.twist.twist.linear.z]])
        # vel_world = np.matmul(rotation_mat_inverse, vel_self)
        # #check if it should be reversed
        ego_vx_world = vel_self[0]
        ego_vy_world = vel_self[1]
        ego_vz_world = vel_self[2]

        # ego_vx_world = self._ego_vehicle_state.state.twist.twist.linear.x
        # ego_vy_world = self._ego_vehicle_state.state.twist.twist.linear.y
        # ego_vz_world = self._ego_vehicle_state.state.twist.twist.linear.z

        tempmarker = Marker()
        tempmarker.header.frame_id = "map"
        tempmarker.header.stamp = rospy.Time.now()
        tempmarker.ns = "zzz/cognition"
        tempmarker.id = 2
        tempmarker.type = Marker.ARROW
        tempmarker.action = Marker.ADD
        tempmarker.scale.x = 0.4
        tempmarker.scale.y = 0.7
        tempmarker.scale.z = 0.75
        tempmarker.color.r = 1.0
        tempmarker.color.g = 1.0
        tempmarker.color.b = 0.0
        tempmarker.color.a = 0.5
        tempmarker.lifetime = rospy.Duration(0.5)

        startpoint = Point()
        endpoint = Point()
        startpoint.x = tstates.ego_vehicle_state.state.pose.pose.position.x
        startpoint.y = tstates.ego_vehicle_state.state.pose.pose.position.y
        startpoint.z = tstates.ego_vehicle_state.state.pose.pose.position.z
        endpoint.x = tstates.ego_vehicle_state.state.pose.pose.position.x + ego_vx_world
        endpoint.y = tstates.ego_vehicle_state.state.pose.pose.position.y + ego_vy_world
        endpoint.z = tstates.ego_vehicle_state.state.pose.pose.position.z + ego_vz_world
        tempmarker.points.append(startpoint)
        tempmarker.points.append(endpoint)

        self._ego_markerarray.markers.append(tempmarker)

        #6. drivable area
        rospy.loginfo("Start to draw drivable area:")
        rospy.loginfo("drivable area point num: %d", len(tstates.drivable_area))
        self._drivable_area_markerarray = MarkerArray()

        count = 0
        if len(tstates.drivable_area) != 0:
            
            tempmarker = Marker() #jxy: must be put inside since it is python
            tempmarker.header.frame_id = "map"
            tempmarker.header.stamp = rospy.Time.now()
            tempmarker.ns = "zzz/cognition"
            tempmarker.id = count
            tempmarker.type = Marker.LINE_STRIP
            tempmarker.action = Marker.ADD
            tempmarker.scale.x = 0.20
            tempmarker.color.r = 1.0
            tempmarker.color.g = 1.0
            tempmarker.color.b = 0.0
            tempmarker.color.a = 0.5
            tempmarker.lifetime = rospy.Duration(0.5)

            for i in range(len(tstates.drivable_area)):
                point = tstates.drivable_area[i]
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = 0 #TODO: the map does not provide z value
                tempmarker.points.append(p)
            self._drivable_area_markerarray.markers.append(tempmarker)
            count = count + 1
            
            #stress the dynamic parts
            for i in range(len(tstates.drivable_area)):
                point = tstates.drivable_area[i]
                if abs(point[2]) < 0.1 and abs(point[3]) < 0.1:
                    continue

                if i != len(tstates.drivable_area) - 1:
                    next_point = tstates.drivable_area[i+1]
                else:
                    # this is actually tstates.drivable_area[0] for closing the figure
                    continue

                tempmarker = Marker() #jxy: must be put inside since it is python
                tempmarker.header.frame_id = "map"
                tempmarker.header.stamp = rospy.Time.now()
                tempmarker.ns = "zzz/cognition"
                tempmarker.id = count
                tempmarker.type = Marker.ARROW
                tempmarker.action = Marker.ADD
                tempmarker.scale.x = 0.40
                tempmarker.scale.y = 0.75
                tempmarker.scale.z = 0.75
                tempmarker.color.r = 0.0
                tempmarker.color.g = 0.0
                tempmarker.color.b = 1.0
                tempmarker.color.a = 1.0
                tempmarker.lifetime = rospy.Duration(0.1)

                #the velocity of i is the section velocity between point i and point i+1
                startpoint = Point()
                endpoint = Point()
                startpoint.x = (point[0] + next_point[0])/2.0
                startpoint.y = (point[1] + next_point[1])/2.0
                startpoint.z = 0.0
                endpoint.x = (point[0] + next_point[0])/2.0 + point[2]
                endpoint.y = (point[1] + next_point[1])/2.0 + point[3]
                endpoint.z = 0.0
                tempmarker.points.append(startpoint)
                tempmarker.points.append(endpoint)

                self._drivable_area_markerarray.markers.append(tempmarker)
                count = count + 1

        #7. next drivable area
        self._next_drivable_area_markerarray = MarkerArray()

        count = 0
        if len(tstates.next_drivable_area) != 0:
            
            tempmarker = Marker() #jxy: must be put inside since it is python
            tempmarker.header.frame_id = "map"
            tempmarker.header.stamp = rospy.Time.now()
            tempmarker.ns = "zzz/cognition"
            tempmarker.id = count
            tempmarker.type = Marker.LINE_STRIP
            tempmarker.action = Marker.ADD
            tempmarker.scale.x = 0.20
            tempmarker.color.r = 0.5
            tempmarker.color.g = 0.5
            tempmarker.color.b = 0.0
            tempmarker.color.a = 0.5
            tempmarker.lifetime = rospy.Duration(0.1)

            for point in tstates.next_drivable_area:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = 0 #TODO: the map does not provide z value
                tempmarker.points.append(p)
            self._next_drivable_area_markerarray.markers.append(tempmarker)
            count = count + 1

        #8. next lanes
        self._next_lanes_markerarray = MarkerArray()

        count = 0
        if len(tstates.static_map.next_lanes) != 0:
            biggest_id = 0 #TODO: better way to find the smallest id
            
            for lane in tstates.static_map.next_lanes:
                if lane.index > biggest_id:
                    biggest_id = lane.index
                tempmarker = Marker() #jxy: must be put inside since it is python
                tempmarker.header.frame_id = "map"
                tempmarker.header.stamp = rospy.Time.now()
                tempmarker.ns = "zzz/cognition"
                tempmarker.id = count
                tempmarker.type = Marker.LINE_STRIP
                tempmarker.action = Marker.ADD
                tempmarker.scale.x = 0.12
                tempmarker.color.r = 0.7
                tempmarker.color.g = 0.0
                tempmarker.color.b = 0.0
                tempmarker.color.a = 0.5
                tempmarker.lifetime = rospy.Duration(0.5)

                for lanepoint in lane.central_path_points:
                    p = Point()
                    p.x = lanepoint.position.x
                    p.y = lanepoint.position.y
                    p.z = lanepoint.position.z
                    tempmarker.points.append(p)
                self._next_lanes_markerarray.markers.append(tempmarker)
                count = count + 1

        #9. next lane boundary line
        self._next_lanes_boundary_markerarray = MarkerArray()

        count = 0
        if len(tstates.static_map.next_lanes) != 0:
            
            for lane in tstates.static_map.next_lanes:
                if len(lane.right_boundaries) > 0 and len(lane.left_boundaries) > 0:
                    tempmarker = Marker() #jxy: must be put inside since it is python
                    tempmarker.header.frame_id = "map"
                    tempmarker.header.stamp = rospy.Time.now()
                    tempmarker.ns = "zzz/cognition"
                    tempmarker.id = count

                    #each lane has the right boundary, only the lane with the smallest id has the left boundary
                    tempmarker.type = Marker.LINE_STRIP
                    tempmarker.action = Marker.ADD
                    tempmarker.scale.x = 0.15
                    
                    if lane.right_boundaries[0].boundary_type == 1: #broken lane is set gray
                        tempmarker.color.r = 0.4
                        tempmarker.color.g = 0.4
                        tempmarker.color.b = 0.4
                        tempmarker.color.a = 0.5
                    else:
                        tempmarker.color.r = 0.7
                        tempmarker.color.g = 0.7
                        tempmarker.color.b = 0.7
                        tempmarker.color.a = 0.5
                    tempmarker.lifetime = rospy.Duration(0.5)

                    for lb in lane.right_boundaries:
                        p = Point()
                        p.x = lb.boundary_point.position.x
                        p.y = lb.boundary_point.position.y
                        p.z = lb.boundary_point.position.z
                        tempmarker.points.append(p)
                    self._next_lanes_boundary_markerarray.markers.append(tempmarker)
                    count = count + 1

                    #biggest id: draw left lane
                    if lane.index == biggest_id:
                        tempmarker = Marker() #jxy: must be put inside since it is python
                        tempmarker.header.frame_id = "map"
                        tempmarker.header.stamp = rospy.Time.now()
                        tempmarker.ns = "zzz/cognition"
                        tempmarker.id = count

                        #each lane has the right boundary, only the lane with the biggest id has the left boundary
                        tempmarker.type = Marker.LINE_STRIP
                        tempmarker.action = Marker.ADD
                        tempmarker.scale.x = 0.3
                        if lane.left_boundaries[0].boundary_type == 1: #broken lane is set gray
                            tempmarker.color.r = 0.4
                            tempmarker.color.g = 0.4
                            tempmarker.color.b = 0.4
                            tempmarker.color.a = 0.5
                        else:
                            tempmarker.color.r = 0.7
                            tempmarker.color.g = 0.7
                            tempmarker.color.b = 0.7
                            tempmarker.color.a = 0.5
                        tempmarker.lifetime = rospy.Duration(0.5)

                        for lb in lane.left_boundaries:
                            p = Point()
                            p.x = lb.boundary_point.position.x
                            p.y = lb.boundary_point.position.y
                            p.z = lb.boundary_point.position.z
                            tempmarker.points.append(p)
                        self._next_lanes_boundary_markerarray.markers.append(tempmarker)
                        count = count + 1

        #10. traffic lights
        self._traffic_lights_markerarray = MarkerArray()

        #TODO: now no lights are in. I'll check it when I run the codes.
        
        #lights = self._traffic_light_detection.detections
        #rospy.loginfo("lights num: %d\n\n", len(lights))
        
        rospy.logdebug("Updated driving space")

        return True

    # ========= For in lane =========

    def locate_object_in_lane(self, object, tstates, dimension, dist_list=None):
        '''
        Calculate (continuous) lane index for a object.
        Parameters: dist_list is the distance buffer. If not provided, it will be calculated
        '''

        if not dist_list:
            dist_list = np.array([dist_from_point_to_polyline2d(
                object.pose.pose.position.x,
                object.pose.pose.position.y,
                lane) for lane in tstates.static_map_lane_path_array]) # here lane is a python list of (x, y)
        
        # Check if there's only two lanes
        if len(tstates.static_map.lanes) < 2:
            closest_lane = second_closest_lane = 0
        else:
            closest_lane, second_closest_lane = np.abs(dist_list[:, 0]).argsort()[:2]

        # Signed distance from target to two closest lane
        closest_lane_dist, second_closest_lane_dist = dist_list[closest_lane, 0], dist_list[second_closest_lane, 0]

        if abs(closest_lane_dist) > self._lane_dist_thres:
            return -1, -99, -99, -99, -99 # TODO: return reasonable value

        lane = tstates.static_map.lanes[closest_lane]
        left_boundary_array = np.array([(lbp.boundary_point.position.x, lbp.boundary_point.position.y) for lbp in lane.left_boundaries])
        right_boundary_array = np.array([(lbp.boundary_point.position.x, lbp.boundary_point.position.y) for lbp in lane.right_boundaries])

        if len(left_boundary_array) == 0:
            ffstate = get_frenet_state(object,
                            tstates.static_map_lane_path_array[closest_lane],
                            tstates.static_map_lane_tangets[closest_lane]
                        )
            lane_anglediff = ffstate.psi
            lane_dist_s = ffstate.s
            return closest_lane, -1, -1, lane_anglediff, lane_dist_s
        else:
            # Distance to lane considering the size of the object
            x = object.pose.pose.orientation.x
            y = object.pose.pose.orientation.y
            z = object.pose.pose.orientation.z
            w = object.pose.pose.orientation.w

            rotation_mat = np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y], [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x], [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]])
            rotation_mat_inverse = np.linalg.inv(rotation_mat) #those are the correct way to deal with quaternion

            vector_x = np.array([dimension.length_x, 0, 0])
            vector_y = np.array([0, dimension.length_y, 0])
            dx = np.matmul(rotation_mat_inverse, vector_x)
            dy = np.matmul(rotation_mat_inverse, vector_y)

            #the four corners of the object, in bird view: left front is 0, counterclockwise.
            #TODO: may consider 8 corners in the future
            corner_list_x = np.zeros(4)
            corner_list_y = np.zeros(4)
            corner_list_x[0] = object.pose.pose.position.x + dx[0]/2.0 + dy[0]/2.0
            corner_list_y[0] = object.pose.pose.position.y + dx[1]/2.0 + dy[1]/2.0
            corner_list_x[1] = object.pose.pose.position.x - dx[0]/2.0 + dy[0]/2.0
            corner_list_y[1] = object.pose.pose.position.y - dx[1]/2.0 + dy[1]/2.0
            corner_list_x[2] = object.pose.pose.position.x - dx[0]/2.0 - dy[0]/2.0
            corner_list_y[2] = object.pose.pose.position.y - dx[1]/2.0 - dy[1]/2.0
            corner_list_x[3] = object.pose.pose.position.x + dx[0]/2.0 - dy[0]/2.0
            corner_list_y[3] = object.pose.pose.position.y + dx[1]/2.0 - dy[1]/2.0

            dist_left_list_all = np.array([dist_from_point_to_polyline2d(
                    corner_list_x[i],
                    corner_list_y[i],
                    left_boundary_array) for i in range(4)])
            dist_right_list_all = np.array([dist_from_point_to_polyline2d(
                    corner_list_x[i],
                    corner_list_y[i],
                    right_boundary_array) for i in range(4)])

            dist_left_list = dist_left_list_all[:, 0]
            dist_right_list = dist_right_list_all[:, 0]

            lane_dist_left_t = -99
            lane_dist_right_t = -99

            if np.min(dist_left_list) * np.max(dist_left_list) <= 0:
                # the object is on the left boundary of lane
                lane_dist_left_t = 0
            else:
                lane_dist_left_t = np.sign(np.min(dist_left_list)) * np.min(np.abs(dist_left_list))

            if np.min(dist_right_list) * np.max(dist_right_list) <= 0:
                # the object is on the right boundary of lane
                lane_dist_right_t = 0
            else:
                lane_dist_right_t = np.sign(np.min(dist_right_list)) * np.min(np.abs(dist_right_list))

            if np.min(dist_left_list) * np.max(dist_left_list) > 0 and np.min(dist_right_list) * np.max(dist_right_list) > 0:
                if np.min(dist_left_list) * np.max(dist_right_list) >= 0:
                    # the object is out of the road
                    closest_lane = -1
                
            ffstate = get_frenet_state(object,
                            tstates.static_map_lane_path_array[closest_lane],
                            tstates.static_map_lane_tangets[closest_lane]
                        )
            lane_anglediff = ffstate.psi
            lane_dist_s = ffstate.s # this is also helpful in getting ego s coordinate in the road

            # Judge whether the point is outside of lanes
            if closest_lane == -1:
                # The object is at left or right most
                return closest_lane, lane_dist_left_t, lane_dist_right_t, lane_anglediff, lane_dist_s
            else:
                # The object is between center line of lanes
                a, b = closest_lane, second_closest_lane
                la, lb = abs(closest_lane_dist), abs(second_closest_lane_dist)
                if lb + la == 0:
                    lane_index_return = -1
                else:
                    lane_index_return = (b*la + a*lb)/(lb + la)
                return lane_index_return, lane_dist_left_t, lane_dist_right_t, lane_anglediff, lane_dist_s
        

    def locate_obstacle_in_lanes(self, tstates):
        tstates.obstacles = [] #clear in every step
        if tstates.surrounding_object_list == None:
            return
        for obj in tstates.surrounding_object_list:
            if len(tstates.static_map.lanes) != 0:
                obj.lane_index, obj.lane_dist_left_t, obj.lane_dist_right_t, obj.lane_anglediff, obj.lane_dist_s = self.locate_object_in_lane(obj.state, tstates, obj.dimension)
            else:
                obj.lane_index = -1
            tstates.obstacles.append(obj)

    def locate_ego_vehicle_in_lanes(self, tstates, lane_end_dist_thres=2, lane_dist_thres=5):
        dist_list = np.array([dist_from_point_to_polyline2d(
            tstates.ego_vehicle_state.state.pose.pose.position.x, tstates.ego_vehicle_state.state.pose.pose.position.y,
            lane, return_end_distance=True)
            for lane in tstates.static_map_lane_path_array])
        ego_dimension = DimensionWithCovariance()
        ego_dimension.length_x = 4.0
        ego_dimension.length_y = 2.0 #jxy: I don't know
        ego_dimension.length_z = 1.8
        ego_lane_index, _, _, _, ego_s = self.locate_object_in_lane(tstates.ego_vehicle_state.state, tstates, ego_dimension)
        #TODO: should be added to converted ego msg
        ego_lane_index_rounded = int(round(ego_lane_index))

        self._ego_vehicle_distance_to_lane_head = dist_list[:, 3]
        self._ego_vehicle_distance_to_lane_tail = dist_list[:, 4]
        if ego_lane_index < 0 or self._ego_vehicle_distance_to_lane_tail[ego_lane_index_rounded] <= lane_end_dist_thres:
            # Drive into junction, wait until next map
            tstates.ego_lane_index = -1
            tstates.ego_s = ego_s
            rospy.logdebug("In junction due to close to intersection, ego_lane_index = %f, dist_to_lane_tail = %f", ego_lane_index, self._ego_vehicle_distance_to_lane_tail[int(ego_lane_index)])
            return
        else:
            tstates.ego_lane_index = ego_lane_index
            tstates.ego_s = ego_s
            #TODO: this is not modified!
        rospy.logdebug("Distance to end: (lane %f) %f", ego_lane_index, self._ego_vehicle_distance_to_lane_tail[ego_lane_index_rounded])

    def calculate_next_drivable_area(self, tstates):
        '''
        The drivable area in the next unit of road (i.e. junction or road section)
        '''
        #TODO: now only support junction, static
        tstates.next_drivable_area = []
        if len(tstates.static_map.next_drivable_area.points) != 0:
            for i in range(len(tstates.static_map.next_drivable_area.points)):
                node_point = tstates.static_map.next_drivable_area.points[i]
                tstates.next_drivable_area.append([node_point.x, node_point.y])
        #close the figure
        if len(tstates.static_map.next_drivable_area.points) != 0:
            tstates.next_drivable_area.append(tstates.next_drivable_area[0])

    def calculate_drivable_area(self, tstates):
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
                    self.lane_section_points_generation(lane_sections[i, 0], lane_sections[i, 1], lane_sections[i, 2], \
                        lane_sections[i, 3], lane_sections[i, 4], lane_sections[i, 5],lane.right_boundaries, tstates)
                
                if i != 0:
                    self.lane_section_points_generation(lane_sections[i-1, 1], lane_sections[i, 1], lane_sections[i-1, 4], \
                        lane_sections[i-1, 5], lane_sections[i, 4], lane_sections[i, 5], lane.right_boundaries, tstates)
                    if i != len(tstates.static_map.lanes) - 1:
                        self.lane_section_points_generation(lane_sections[i, 1], lane_sections[i+1, 1], lane_sections[i, 4], \
                        lane_sections[i, 5], lane_sections[i+1, 4], lane_sections[i+1, 5], lane.left_boundaries, tstates)
                    else:
                        self.lane_section_points_generation(lane_sections[i, 1], lane_sections[i, 0], lane_sections[i, 4], \
                        lane_sections[i, 5], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, tstates)

                if len(tstates.static_map.lanes) == 1:
                    self.lane_section_points_generation(lane_sections[i, 1], lane_sections[i, 0], lane_sections[i, 4], \
                        lane_sections[i, 5], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, tstates)

            for j in range(len(tstates.static_map.lanes)):
                i = len(tstates.static_map.lanes) - 1 - j
                lane = tstates.static_map.lanes[i]                
                if i != len(tstates.static_map.lanes) - 1:
                    self.lane_section_points_generation(lane_sections[i+1, 0], lane_sections[i, 0], lane_sections[i+1, 2], \
                        lane_sections[i+1, 3], lane_sections[i, 4], lane_sections[i, 5], lane.left_boundaries, tstates)
                    if i != 0:
                        self.lane_section_points_generation(lane_sections[i, 0], lane_sections[i-1, 0], lane_sections[i, 2], \
                        lane_sections[i, 3], lane_sections[i-1, 4], lane_sections[i-1, 5], lane.right_boundaries, tstates)

            #close the figure
            if len(tstates.drivable_area) > 0:
                tstates.drivable_area.append(tstates.drivable_area[0])

            rospy.loginfo("drivable_area constructed with length %d", len(tstates.drivable_area))

    def lane_section_points_generation(self, starts, ends, startvx, startvy, endvx, endvy, lane_boundaries, tstates):

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

    def locate_traffic_light_in_lanes(self, tstates):
        # TODO: Currently it's a very simple rule to locate the traffic lights
        if tstates.traffic_light_detection is None:
            return
        lights = tstates.traffic_light_detection.detections
        #jxy: demanding that the lights are in the same order as the lanes.

        total_lane_num = len(tstates.static_map.lanes)
        if len(lights) == 1:
            for i in range(total_lane_num):
                if lights[0].signal == ObjectSignals.TRAFFIC_LIGHT_RED:
                    tstates.static_map.lanes[i].map_lane.stop_state = Lane.STOP_STATE_STOP
                elif lights[0].signal == ObjectSignals.TRAFFIC_LIGHT_YELLOW:
                    tstates.static_map.lanes[i].map_lane.stop_state = Lane.STOP_STATE_YIELD
                elif lights[0].signal == ObjectSignals.TRAFFIC_LIGHT_GREEN:
                    tstates.static_map.lanes[i].map_lane.stop_state = Lane.STOP_STATE_THRU
        elif len(lights) > 1 and len(lights) == total_lane_num:
            for i in range(total_lane_num):
                if lights[i].signal == ObjectSignals.TRAFFIC_LIGHT_RED:
                    tstates.static_map.lanes[i].map_lane.stop_state = Lane.STOP_STATE_STOP
                elif lights[i].signal == ObjectSignals.TRAFFIC_LIGHT_YELLOW:
                    tstates.static_map.lanes[i].map_lane.stop_state = Lane.STOP_STATE_YIELD
                elif lights[i].signal == ObjectSignals.TRAFFIC_LIGHT_GREEN:
                    tstates.static_map.mmap.lanes[i].map_lane.stop_state = Lane.STOP_STATE_THRU
        elif len(lights) > 1 and len(lights) != total_lane_num:
            red = True
            for i in range(len(lights)):
                if lights[i].signal == ObjectSignals.TRAFFIC_LIGHT_GREEN:
                    red = False
            for i in range(total_lane_num):
                if red:
                    tstates.static_map.lanes[i].map_lane.stop_state = Lane.STOP_STATE_STOP
                else:
                    tstates.static_map.lanes[i].map_lane.stop_state = Lane.STOP_STATE_THRU
        
    def locate_stop_sign_in_lanes(self, tstates):
        '''
        Put stop sign detections into lanes
        '''
        # TODO: Implement this
        pass

    def locate_speed_limit_in_lanes(self, tstates):
        '''
        Put stop sign detections into lanes
        '''
        # TODO(zhcao): Change the speed limit according to the map or the traffic sign(perception)
        # Now we set the multilane speed limit as 40 km/h.
        total_lane_num = len(tstates.static_map.lanes)
        for i in range(total_lane_num):
            tstates.static_map.lanes[i].speed_limit = 40
