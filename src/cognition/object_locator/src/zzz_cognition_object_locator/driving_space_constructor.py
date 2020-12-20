
import rospy
import numpy as np
from easydict import EasyDict as edict
from threading import Lock
import math
import copy
import time

from zzz_driver_msgs.msg import RigidBodyStateStamped
from zzz_navigation_msgs.msg import Map, Lane, LanePoint
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
from drivable_area import calculate_drivable_areas, predict_obstacles

#jxy 20191125: first output the driving space, then use the driving space for cognition. 
#For this demo version, it will be a unified module, in future versions, this will be split into 2 modules.

DT = 0.3  # time tick [s]
STEPS = 10 # predict time steps

class DrivingSpaceConstructor:
    def __init__(self, lane_dist_thres=5):
        self._static_map_lock = Lock()
        self._static_map_buffer = None

        self._static_map_updated_flag = 0

        self._ego_vehicle_state_lock = Lock()
        self._ego_vehicle_state_buffer = None

        self._surrounding_object_list_lock = Lock()
        self._surrounding_object_list_buffer = None

        self._traffic_light_detection_lock = Lock()
        self._traffic_light_detection_buffer = None

        self._driving_space = None
        self._obstacles_markerarray = None
        self._lanes_boundary_markerarray = None
        self._dynamic_map = None

        self._lanes_memory = []
        
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
            self._static_map_updated_flag = 1
            rospy.loginfo("Updated Local Static Map: lanes_num = %d, in_junction = %d, exit_lane_index = %d",
                len(static_map.lanes), int(static_map.in_junction), static_map.exit_lane_index[0])

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

        t1 = time.time()

        # Skip if not ready
        if not self._ego_vehicle_state_buffer:
            return False

        with self._ego_vehicle_state_lock:
            tstates.ego_vehicle_state = self._ego_vehicle_state_buffer

        # Update buffer information
        tstates.surrounding_object_list = self._surrounding_object_list_buffer or [] #jxy20201202: remove deepcopy to accelerate

        tstates.static_map = self._static_map_buffer or navigation_default(Map)
        static_map = tstates.static_map # for easier access
        tstates.static_map_lane_path_array = get_lane_array(tstates.static_map.lanes)
        tstates.static_map_lane_tangets = [[point.tangent for point in lane.central_path_points] for lane in tstates.static_map.lanes]
        tstates.next_drivable_area = []
        tstates.surrounding_object_list_timelist = []
        tstates.drivable_area_timelist = []
        tstates.ego_s = 0
        tstates.dynamic_map = cognition_default(MapState)
        self._driving_space = DrivingSpace()

        t2 = time.time()

        #jxy20201202: move prediction here
        predict_obstacles(tstates, STEPS)
        print "predicted obstacles, steps: "
        print len(tstates.surrounding_object_list_timelist)

        t3 = time.time()

        #jxy20201218: add virtual lanes in the junction, move the cohering execution here (from lane decision)
        status_flag = 0
        if self._static_map_updated_flag == 1:
            self._static_map_updated_flag = 0 #only prolong once when received new static map
            if len(static_map.virtual_lanes) != 0 and len(static_map.next_lanes) != 0:
                # in lanes, prolong with the virtual lanes in the junction
                # TODO: consider when the numbers do not equal
                for i in range(min(len(static_map.virtual_lanes), len(static_map.next_lanes))):
                    if len(static_map.next_drivable_area.points) > 3:
                        point_x = static_map.next_lanes[i].central_path_points[0].position.x
                        point_y = static_map.next_lanes[i].central_path_points[0].position.y
                        next_point_x = static_map.next_lanes[i].central_path_points[1].position.x
                        next_point_y = static_map.next_lanes[i].central_path_points[1].position.y
                        exit_junction_point = [point_x, point_y]
                        exit_junction_direction = [next_point_x - point_x, next_point_y - point_y]
                        original_length = len(static_map.virtual_lanes[i].central_path_points)
                        self.extend_junction_path(tstates, i, static_map.lanes[i].central_path_points, \
                            exit_junction_point, exit_junction_direction)
                        status_flag = 0.5 #loaded next junction
                        rospy.loginfo("extension successful, original length: %d, extended length: %d", \
                            original_length, len(static_map.virtual_lanes[i].central_path_points))

                for i in range(len(static_map.virtual_lanes)): #should be as many as next lanes
                    static_map.virtual_lanes[i].central_path_points.extend(static_map.next_lanes[i].central_path_points)

                self._lanes_memory = static_map.virtual_lanes

            if len(static_map.lanes) == 0: #in junction, keep the virtual lanes
                static_map.virtual_lanes = self._lanes_memory #jxy: check whether it will change as python

        #jxy1216: merge obstacle locator inside
        dynamic_map = tstates.dynamic_map # for easier access

        # Create dynamic maps and add static map elements
        dynamic_map.header.frame_id = "map"
        dynamic_map.header.stamp = rospy.Time.now()
        dynamic_map.ego_state = tstates.ego_vehicle_state.state
        
        if static_map.in_junction:
            rospy.logdebug("Cognition: In junction due to static map report junction location")
            dynamic_map.model = MapState.MODEL_JUNCTION_MAP
            status_flag = 1
        else:
            dynamic_map.model = MapState.MODEL_MULTILANE_MAP

        if len(static_map.virtual_lanes) != 0: #jxy20201218: in junction model, the virtual lanes are still considered.
            for lane in static_map.virtual_lanes:
                dlane = cognition_default(LaneState)
                dlane.map_lane = lane
                dynamic_map.mmap.lanes.append(dlane)
            dynamic_map.mmap.exit_lane_index = copy.deepcopy(static_map.exit_lane_index)

        dynamic_map.status_flag = status_flag

        #tstates.dynamic_map.jmap.obstacles = tstates.surrounding_object_list #no longer required

        # Update driving_space with tstate
        if static_map.in_junction or len(static_map.lanes) == 0:
            rospy.logdebug("In junction due to static map report junction location")
        else:
            for lane in tstates.static_map.lanes:
                self._driving_space.lanes.append(lane)
            #jxy: why is target lane in static map?
            self.locate_ego_vehicle_in_lanes(tstates) #TODO: consider ego_s and front/rear vehicle in virtual lanes!!
            self.locate_surrounding_objects_in_lanes(tstates) # TODO: here lies too much repeated calculation, change it and lateral decision
            for i in range(STEPS):
                self.locate_obstacle_in_lanes(tstates, i)
            self.locate_stop_sign_in_lanes(tstates)
            self.locate_speed_limit_in_lanes(tstates)

        t4 = time.time()
        
        for i in range(STEPS):
            calculate_drivable_areas(tstates, i)

        t5 = time.time()
        
        #jxy1202: will change output at final step
        #TODO: remove driving space and merge dynamic boundary into dynamic map, and optimize dynamic map (lane-obs-lane structure)
        #TODO: prolonging the lanes: move here (less urgent)

        self._dynamic_map = dynamic_map

        #jxy1202: will change output at final step
        for tt in range(STEPS):
            dynamic_boundary = DynamicBoundary()
            dynamic_boundary.header.frame_id = "map"
            dynamic_boundary.header.stamp = rospy.Time.now()
            for i in range(len(tstates.drivable_area_timelist[tt])):
                drivable_area_point = tstates.drivable_area_timelist[tt][i]
                boundary_point = DynamicBoundaryPoint()
                boundary_point.x = drivable_area_point[0]
                boundary_point.y = drivable_area_point[1]
                boundary_point.vx = drivable_area_point[2]
                boundary_point.vy = drivable_area_point[3]
                boundary_point.base_x = drivable_area_point[4]
                boundary_point.base_y = drivable_area_point[5]
                boundary_point.omega = drivable_area_point[6]
                boundary_point.flag = drivable_area_point[7]
                dynamic_boundary.boundary.append(boundary_point)
            dynamic_map.jmap.boundary_list.append(dynamic_boundary)

        self.visualization(tstates)

        t6 = time.time()
        rospy.loginfo("initialize time consumption: %f ms", (t2 - t1) * 1000)
        rospy.loginfo("predict obstacles time consumption: %f ms", (t3 - t2) * 1000)
        rospy.loginfo("locate obstacle time consumption: %f ms", (t4 - t3) * 1000)
        rospy.loginfo("dynamic boundary construction time consumption: %f ms", (t5 - t4) * 1000)
        rospy.loginfo("output and visualization time consumption: %f ms", (t6 - t5) * 1000)
        rospy.loginfo("total time: %f ms", (t6 - t1) * 1000)
        
        rospy.logdebug("Updated driving space")

        return True

    # ========= For virtual lane =========

    def extend_junction_path(self, tstates, extension_index, path, exit_junction_point, exit_junction_direction):

        #TODO: consider strange junctions, or U turn.

        if len(exit_junction_point) == 0:
            return path

        x1 = path[-1].position.x
        y1 = path[-1].position.y
        x2 = exit_junction_point[0]
        y2 = exit_junction_point[1]
        dx1 = path[-1].position.x - path[-2].position.x
        dy1 = path[-1].position.y - path[-2].position.y
        dx2 = exit_junction_direction[0]
        dy2 = exit_junction_direction[1]
        
        print "extension params:"
        print x1
        print y1
        print x2
        print y2
        print dx1
        print dy1
        print dx2
        print dy2

        #change coordinate, plan a 3rd polyline
        l = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
        if l == 0:
            return
        m = (dx1 * (x2 - x1) + dy1 * (y2 - y1)) / l
        n = (dx2 * (x2 - x1) + dy2 * (y2 - y1)) / l
        tt1 = math.sqrt(((dx1 * dx1 + dy1 * dy1) / (m * m)) - 1)
        tt2 = math.sqrt(((dx2 * dx2 + dy2 * dy2) / (n * n)) - 1)

        extended_path = tstates.static_map.virtual_lanes[extension_index].central_path_points

        for i in range(11):
            u = l - l / 10 * i
            v = -(u - l) * u * ((tt1 - tt2) / (l * l) * u + tt2 / l)
            x = ((x1 - x2) * u + (y2 - y1) * v) / l + x2
            y = ((y1 - y2) * u + (x1 - x2) * v) / l + y2
            point = LanePoint()
            point.position.x = x
            point.position.y = y
            extended_path.append(point)

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
        

    def locate_obstacle_in_lanes(self, tstates, time_step):
        obstacles_step = tstates.surrounding_object_list_timelist[time_step]
        if obstacles_step == None:
            return
        for obj in obstacles_step:
            if len(tstates.static_map.lanes) != 0:
                obj.lane_index, obj.lane_dist_left_t, obj.lane_dist_right_t, obj.lane_anglediff, obj.lane_dist_s = self.locate_object_in_lane(obj.state, tstates, obj.dimension)
            else:
                obj.lane_index = -1

    def locate_surrounding_objects_in_lanes(self, tstates, lane_dist_thres=3):
        lane_front_vehicle_list = [[] for _ in tstates.static_map.virtual_lanes]
        lane_rear_vehicle_list = [[] for _ in tstates.static_map.virtual_lanes]

        # TODO: separate vehicle and other objects?
        if tstates.surrounding_object_list is not None:
            for vehicle_idx, vehicle in enumerate(tstates.surrounding_object_list):
                dist_list = np.array([dist_from_point_to_polyline2d(
                    vehicle.state.pose.pose.position.x,
                    vehicle.state.pose.pose.position.y,
                    lane, return_end_distance=True)
                    for lane in tstates.static_map_lane_path_array])
                closest_lane = np.argmin(np.abs(dist_list[:, 0]))

                # Determine if the vehicle is close to lane enough
                if abs(dist_list[closest_lane, 0]) > lane_dist_thres:
                    continue 
                if dist_list[closest_lane, 3] < self._ego_vehicle_distance_to_lane_head[closest_lane]:
                    # The vehicle is behind if its distance to lane start is smaller
                    lane_rear_vehicle_list[closest_lane].append((vehicle_idx, dist_list[closest_lane, 3]))
                if dist_list[closest_lane, 4] < self._ego_vehicle_distance_to_lane_tail[closest_lane]:
                    # The vehicle is ahead if its distance to lane end is smaller
                    lane_front_vehicle_list[closest_lane].append((vehicle_idx, dist_list[closest_lane, 4]))
        
        # Put the vehicles onto lanes
        for lane_id in range(len(tstates.static_map.virtual_lanes)):
            front_vehicles = np.array(lane_front_vehicle_list[lane_id])
            rear_vehicles = np.array(lane_rear_vehicle_list[lane_id])

            if len(front_vehicles) > 0:
                # Descending sort front objects by distance to lane end
                for vehicle_row in reversed(front_vehicles[:,1].argsort()):
                    front_vehicle_idx = int(front_vehicles[vehicle_row, 0])
                    front_vehicle = tstates.surrounding_object_list[front_vehicle_idx]
                    front_vehicle.ffstate = get_frenet_state(front_vehicle.state,
                        tstates.static_map_lane_path_array[lane_id],
                        tstates.static_map_lane_tangets[lane_id]
                    )
                    # Here we use relative frenet coordinate
                    front_vehicle.ffstate.s = self._ego_vehicle_distance_to_lane_tail[lane_id] - front_vehicles[vehicle_row, 1]
                    front_vehicle.behavior = self.predict_vehicle_behavior(front_vehicle, tstates)
                    tstates.dynamic_map.mmap.lanes[lane_id].front_vehicles.append(front_vehicle)
                    break #jxy1217: only keep one, since only one is used in IDM
                
                front_vehicle = tstates.dynamic_map.mmap.lanes[lane_id].front_vehicles[0]
                rospy.logdebug("Lane index: %d, Front vehicle id: %d, behavior: %d, x:%.1f, y:%.1f, d:%.1f", 
                                lane_id, front_vehicle.uid, front_vehicle.behavior,
                                front_vehicle.state.pose.pose.position.x,front_vehicle.state.pose.pose.position.y,
                                front_vehicle.ffstate.s)

            if len(rear_vehicles) > 0:
                # Descending sort rear objects by distance to lane end
                for vehicle_row in reversed(rear_vehicles[:,1].argsort()):
                    rear_vehicle_idx = int(rear_vehicles[vehicle_row, 0])
                    rear_vehicle = tstates.surrounding_object_list[rear_vehicle_idx]
                    rear_vehicle.ffstate = get_frenet_state(rear_vehicle.state,
                        tstates.static_map_lane_path_array[lane_id],
                        tstates.static_map_lane_tangets[lane_id]
                    )
                    # Here we use relative frenet coordinate
                    rear_vehicle.ffstate.s = rear_vehicles[vehicle_row, 1] - self._ego_vehicle_distance_to_lane_head[lane_id] # negative value
                    rear_vehicle.behavior = self.predict_vehicle_behavior(rear_vehicle, tstates)
                    tstates.dynamic_map.mmap.lanes[lane_id].rear_vehicles.append(rear_vehicle)
                    break
                
                rear_vehicle = tstates.dynamic_map.mmap.lanes[lane_id].rear_vehicles[0]
                rospy.logdebug("Lane index: %d, Rear vehicle id: %d, behavior: %d, x:%.1f, y:%.1f, d:%.1f", 
                                lane_id, rear_vehicle.uid, rear_vehicle.behavior, 
                                rear_vehicle.state.pose.pose.position.x,rear_vehicle.state.pose.pose.position.y,
                                rear_vehicle.ffstate.s)

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

        #from obstacle locator
        tstates.ego_s = ego_s
        if ego_lane_index < 0 or self._ego_vehicle_distance_to_lane_tail[ego_lane_index_rounded] <= lane_end_dist_thres:
            # Drive into junction, wait until next map
            rospy.logdebug("Cognition: Ego vehicle close to intersection, ego_lane_index = %f, dist_to_lane_tail = %f", ego_lane_index, self._ego_vehicle_distance_to_lane_tail[int(ego_lane_index)])
            tstates.dynamic_map.model = MapState.MODEL_JUNCTION_MAP
            # TODO: Calculate frenet coordinate here or in put_buffer?
            return
        else:
            tstates.dynamic_map.model = MapState.MODEL_MULTILANE_MAP
            tstates.dynamic_map.ego_ffstate = get_frenet_state(tstates.ego_vehicle_state, 
                tstates.static_map_lane_path_array[ego_lane_index_rounded],
                tstates.static_map_lane_tangets[ego_lane_index_rounded])
            tstates.dynamic_map.mmap.ego_lane_index = ego_lane_index
            tstates.dynamic_map.mmap.distance_to_junction = self._ego_vehicle_distance_to_lane_tail[ego_lane_index_rounded]
        rospy.logdebug("Distance to end: (lane %f) %f", ego_lane_index, self._ego_vehicle_distance_to_lane_tail[ego_lane_index_rounded])

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

    def locate_speed_limit_in_lanes(self, tstates, ref_stop_thres = 10):
        '''
        Put stop sign detections into lanes
        '''
        # TODO(zhcao): Change the speed limit according to the map or the traffic sign(perception)
        # Now we set the multilane speed limit as 40 km/h.
        total_lane_num = len(tstates.static_map.lanes)
        for i in range(total_lane_num):
            tstates.dynamic_map.mmap.lanes[i].map_lane.speed_limit = 15

    #TODO: optimize it, and find where is it used
    def predict_vehicle_behavior(self, vehicle, tstates, lane_change_thres = 0.2):
        '''
        Detect the behaviors of surrounding vehicles
        '''

        dist_list = np.array([dist_from_point_to_polyline2d(vehicle.state.pose.pose.position.x, vehicle.state.pose.pose.position.y, lane)
            for lane in tstates.static_map_lane_path_array])
        dist_list = np.abs(dist_list)
        closest_lane = dist_list[:, 0].argsort()[0]
        closest_idx = int(dist_list[closest_lane, 1])
        closest_point = tstates.dynamic_map.mmap.lanes[closest_lane].map_lane.central_path_points[closest_idx]

        vehicle_driving_direction = get_yaw(vehicle.state)
        lane_direction = closest_point.tangent
        d_theta = vehicle_driving_direction - lane_direction
        d_theta = wrap_angle(d_theta)

        # rospy.logdebug("id:%d, vehicle_direction:%.2f, lane_direction:%.2f",vehicle.uid,vehicle_driving_direction,lane_direction)
        if abs(d_theta) > lane_change_thres:
            if d_theta > 0:
                behavior = RoadObstacle.BEHAVIOR_MOVING_LEFT
            else:
                behavior = RoadObstacle.BEHAVIOR_MOVING_RIGHT
        else:
            behavior = RoadObstacle.BEHAVIOR_FOLLOW
        
        return behavior

    def visualization(self, tstates):

        #visualization
        #1. lanes #jxy20201219: virtual lanes
        self._lanes_markerarray = MarkerArray()

        count = 0
        if len(tstates.static_map.virtual_lanes) != 0:
            biggest_id = 0 #TODO: better way to find the smallest id
            
            for lane in tstates.static_map.virtual_lanes:
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
        if tstates.surrounding_object_list_timelist[0] is not None:
            for obs in tstates.surrounding_object_list_timelist[0]:
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
            
            for obs in tstates.surrounding_object_list_timelist[0]:
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

                    #rotation_mat = np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y], [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x], [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]])
                    #rotation_mat_inverse = np.linalg.inv(rotation_mat) #those are the correct way to deal with quaternion

                    vel_obs = np.array([obs.state.twist.twist.linear.x, obs.state.twist.twist.linear.y, obs.state.twist.twist.linear.z])
                    #vel_world = np.matmul(rotation_mat, vel_obs)
                    #vel_world = vel_obs
                    #check if it should be reversed
                    obs_vx_world = vel_obs[0]
                    obs_vy_world = vel_obs[1]
                    obs_vz_world = vel_obs[2]

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
        if tstates.surrounding_object_list_timelist[0] is not None:                    
            for obs in tstates.surrounding_object_list_timelist[0]:
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
        self._drivable_area_markerarray = MarkerArray()

        count = 0
        if len(tstates.drivable_area_timelist[0]) != 0:

            for i in range(len(tstates.drivable_area_timelist[0])):
                
                #part 1: boundary section
                tempmarker = Marker() #jxy: must be put inside since it is python
                tempmarker.header.frame_id = "map"
                tempmarker.header.stamp = rospy.Time.now()
                tempmarker.ns = "zzz/cognition"
                tempmarker.id = count
                tempmarker.type = Marker.LINE_STRIP
                tempmarker.action = Marker.ADD
                tempmarker.scale.x = 0.20
                tempmarker.color.a = 0.5
                tempmarker.lifetime = rospy.Duration(0.5)

                point = tstates.drivable_area_timelist[0][i]
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = 0 #TODO: the map does not provide z value
                tempmarker.points.append(p)

                next_id = i + 1
                if next_id >= len(tstates.drivable_area_timelist[0]):
                    continue #closed line, the first point equal to the last point

                next_point = tstates.drivable_area_timelist[0][next_id]
                p = Point()
                p.x = next_point[0]
                p.y = next_point[1]
                p.z = 0 #TODO: the map does not provide z value
                tempmarker.points.append(p)

                tempmarker.color.r = 1.0
                tempmarker.color.g = 1.0
                tempmarker.color.b = 0.0

                if next_point[7] == 2:
                    tempmarker.color.r = 1.0
                    tempmarker.color.g = 0.0
                    tempmarker.color.b = 0.0
                elif next_point[7] == 3:
                    tempmarker.color.r = 0.0
                    tempmarker.color.g = 0.0
                    tempmarker.color.b = 1.0
                
                self._drivable_area_markerarray.markers.append(tempmarker)
                count = count + 1

                #part 2: boundary section motion status
                if next_point[7] == 2 or next_point[7] == 3: # and (abs(next_point[2]) + abs(next_point[3])) > 0.3:
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
                    tempmarker.color.r = 0.5
                    tempmarker.color.g = 0.5
                    tempmarker.color.b = 1.0
                    tempmarker.color.a = 0.8
                    tempmarker.lifetime = rospy.Duration(0.5)

                    startpoint = Point()
                    endpoint = Point()
                    startpoint.x = (point[0] + next_point[0]) / 2
                    startpoint.y = (point[1] + next_point[1]) / 2
                    startpoint.z = 0
                    endpoint.x = startpoint.x + next_point[2]
                    endpoint.y = startpoint.y + next_point[3]
                    endpoint.z = 0
                    tempmarker.points.append(startpoint)
                    tempmarker.points.append(endpoint)

                    self._drivable_area_markerarray.markers.append(tempmarker)
                    count = count + 1

        if len(self._drivable_area_markerarray.markers) < 100:
            for i in range(100 - len(self._drivable_area_markerarray.markers)):
                tempmarker = Marker()
                tempmarker.header.frame_id = "map"
                tempmarker.header.stamp = rospy.Time.now()
                tempmarker.ns = "zzz/cognition"
                tempmarker.id = count
                tempmarker.type = Marker.SPHERE
                tempmarker.action = Marker.ADD
                tempmarker.scale.x = 0.4
                tempmarker.scale.y = 0.7
                tempmarker.scale.z = 0.75
                tempmarker.color.r = 0.5
                tempmarker.color.g = 0.5
                tempmarker.color.b = 1.0
                tempmarker.color.a = 0.8
                tempmarker.lifetime = rospy.Duration(0.5)

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
            tempmarker.color.r = 0.0
            tempmarker.color.g = 1.0
            tempmarker.color.b = 0.0
            tempmarker.color.a = 0.5
            tempmarker.lifetime = rospy.Duration(0.5)

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