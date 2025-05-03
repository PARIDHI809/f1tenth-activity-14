#!/usr/bin/env python3

import rospy
import numpy as np
import math
import csv
import os
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan

class PurePursuitController:
    def __init__(self):
        rospy.init_node('pure_pursuit_controller', anonymous=True) # init node
        
        self.odom_topic = rospy.get_param("~odom_topic", "/odom") # load params
        self.command_topic = rospy.get_param("~command_topic", "/drive")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.base_lookahead = rospy.get_param("~lookahead_distance", 1.5)  # ctrl params
        self.max_velocity = rospy.get_param("~max_velocity", 4.0)  # m/s
        self.min_velocity = rospy.get_param("~min_velocity", 1.0)  # m/s
        self.wheelbase = rospy.get_param("~wheelbase", 0.33)  # mets, F1TENTH car wb
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.4)
        
        self.k_smooth = 0.8  # path smoothing and velocity planning parameters, smoothing factor
        self.velocity_scale = 2.5  # scales vel based on curv
        self.min_lookahead = 0.8  # min lA dist
        self.max_lookahead = 3.0  # max lA dist
        self.velocity_lookahead = 10  # num of wayp to lA for vel plan

        self.use_obstacle_avoidance = True # obstacle avoidance params
        self.obstacle_threshold = 1.0  # this is the dist to consider an obstacle
        self.obstacle_weight = 0.4  
        self.lap_start_time = None # racing params
        self.current_lap_time = 0
        self.best_lap_time = float('inf')
        self.checkpoint_errors = []
        self.collision_count = 0
        self.lap_count = 0
        
        csv_path = os.path.join(os.path.dirname(__file__), "..", "csv", "gp_centerline.csv")
        
        # load wayp
        self.waypoints = self.load_waypoints(csv_path)
        self.total_waypoints = len(self.waypoints)

        self.waypoints_x = [wp[0] for wp in self.waypoints] # extract X Y for faster access
        self.waypoints_y = [wp[1] for wp in self.waypoints]
        
        self.checkpoints = [i for i in range(0, self.total_waypoints, 100)] # gen cp, curv data
        self.next_checkpoint_idx = 0
        self.passed_checkpoints = set()
        self.curvatures = self.calculate_path_curvature()
        
        self.velocity_profile = self.generate_velocity_profile() # smooth vel profile based on curv
        self.race_started = False  # state vars
        self.race_finished = False
        self.current_waypoint_idx = 0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.velocity = 0.0
        self.steering = 0.0
        self.lidar_data = None
        
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)  # create vis markers
        self.waypoint_markers = self.create_waypoint_markers()
        self.lookahead_marker = self.create_lookahead_marker()
        self.drive_pub = rospy.Publisher(self.command_topic, AckermannDriveStamped, queue_size=10) # pubs and subs
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)
        self.race_start_sub = rospy.Subscriber('/race_start', Bool, self.rs_cb)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb)
        
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)  # timer for ctrl loop
        
        rospy.loginfo("Pure Pursuit Controller initialized!")
        rospy.loginfo(f"Loaded {self.total_waypoints} waypoints with {len(self.checkpoints)} checkpoints")
    
    def load_waypoints(self, filename):
        waypoints = []
        try:
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                if csv.Sniffer().has_header(file.read(1024)): # skip header
                    file.seek(0)  # s.o.f
                    next(csv_reader)  # skip header
                else:
                    file.seek(0)  # s.o.f
                
                for row in csv_reader:
                    if len(row) >= 2:  # must ensure X and Y
                        x, y = float(row[0]), float(row[1])
                        waypoints.append((x, y))
            
            rospy.loginfo(f"Successfully loaded {len(waypoints)} waypoints from {filename}")
            return waypoints
        except Exception as e:
            rospy.logerr(f"Error loading waypoints: {e}")
            return []
    
    def calculate_path_curvature(self):
        """ Calc curv at eAch wayp for vel planning"""
        curvatures = np.zeros(self.total_waypoints)
        
        for i in range(self.total_waypoints):
            prev_idx = (i - 1) % self.total_waypoints
            next_idx = (i + 1) % self.total_waypoints
            
            x1, y1 = self.waypoints[prev_idx] # get 3 cons pts
            x2, y2 = self.waypoints[i]
            x3, y3 = self.waypoints[next_idx]
            
            dx1, dy1 = x2 - x1, y2 - y1 # calc vect bw pts
            dx2, dy2 = x3 - x2, y3 - y2
            
            l1 = np.sqrt(dx1**2 + dy1**2) # calc dist bw pts
            l2 = np.sqrt(dx2**2 + dy2**2)
            
            if l1 < 0.01 or l2 < 0.01: # skip calc pts are too close
                curvatures[i] = 0.0
                continue
            
            cos_angle = (dx1*dx2 + dy1*dy2) / (l1 * l2) # calc angle bw vect
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # ensure in valid rng
            angle = np.arccos(cos_angle)
            if angle <= 0.001:  # calc curv inv of rad
                curvatures[i] = 0.0
            else:
                curvatures[i] = angle / min(l1, l2)
        
        # smooth the curv
        smoothed_curvatures = curvatures.copy()
        window_size = 5
        for i in range(self.total_waypoints):
            sum_curvature = 0.0
            count = 0
            for j in range(-window_size, window_size + 1):
                idx = (i + j) % self.total_waypoints
                sum_curvature += curvatures[idx]
                count += 1
            smoothed_curvatures[i] = sum_curvature / count
        
        return smoothed_curvatures
    
    def generate_velocity_profile(self):
        """gen vel prof """
        velocities = np.zeros(self.total_waypoints)
        max_curvature = np.max(self.curvatures)
        
        if max_curvature < 0.001:
            return np.full(self.total_waypoints, self.max_velocity)
        
        for i in range(self.total_waypoints): # vel based on curv
            normalized_curvature = self.curvatures[i] / max_curvature # higher curv = lwr vel
            velocities[i] = self.max_velocity - normalized_curvature * self.velocity_scale
            velocities[i] = np.clip(velocities[i], self.min_velocity, self.max_velocity)
        
        smoothed_velocities = velocities.copy() # smooth vel prof
        window_size = 10
        for i in range(self.total_waypoints):
            sum_velocity = 0.0
            count = 0
            for j in range(-window_size, window_size + 1):
                idx = (i + j) % self.total_waypoints
                sum_velocity += velocities[idx]
                count += 1
            smoothed_velocities[i] = sum_velocity / count
        forward_planning = np.zeros(self.total_waypoints) # lA to slow down
        for i in range(self.total_waypoints):
            min_velocity = smoothed_velocities[i]
            for j in range(1, self.velocity_lookahead + 1):
                idx = (i + j) % self.total_waypoints
                min_velocity = min(min_velocity, smoothed_velocities[idx])
            forward_planning[i] = min_velocity
        
        return forward_planning
    
    def create_waypoint_markers(self):
        """vis markers"""
        marker_array = MarkerArray()
        waypoints_marker = Marker() # wayp marker
        waypoints_marker.header.frame_id = "map"
        waypoints_marker.id = 0
        waypoints_marker.type = Marker.LINE_STRIP
        waypoints_marker.action = Marker.ADD
        waypoints_marker.scale.x = 0.05  # line width
        waypoints_marker.color.r = 0.0
        waypoints_marker.color.g = 1.0
        waypoints_marker.color.b = 0.0
        waypoints_marker.color.a = 0.5
        waypoints_marker.pose.orientation.w = 1.0
        
        # add all wayp
        for wp in self.waypoints:
            p = PoseStamped()
            p.pose.position.x = wp[0]
            p.pose.position.y = wp[1]
            p.pose.position.z = 0.0
            waypoints_marker.points.append(p.pose.position)
        
        marker_array.markers.append(waypoints_marker)
        
        for i, idx in enumerate(self.checkpoints): # check pt marker
            checkpoint_marker = Marker()
            checkpoint_marker.header.frame_id = "map"
            checkpoint_marker.id = i + 1
            checkpoint_marker.type = Marker.SPHERE
            checkpoint_marker.action = Marker.ADD
            checkpoint_marker.scale.x = 0.5
            checkpoint_marker.scale.y = 0.5
            checkpoint_marker.scale.z = 0.5
            checkpoint_marker.color.r = 1.0
            checkpoint_marker.color.g = 0.0
            checkpoint_marker.color.b = 1.0
            checkpoint_marker.color.a = 0.7
            checkpoint_marker.pose.orientation.w = 1.0
            checkpoint_marker.pose.position.x = self.waypoints[idx][0]
            checkpoint_marker.pose.position.y = self.waypoints[idx][1]
            checkpoint_marker.pose.position.z = 0.0
            
            marker_array.markers.append(checkpoint_marker)
        
        return marker_array
    
    def create_lookahead_marker(self):
        """marker for lA pt"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = 100
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.pose.orientation.w = 1.0
        
        return marker
    
    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        quat = msg.pose.pose.orientation # yaw from quat
        ign, ign, self.yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        
        self.velocity = np.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y) # curr vel
    
    def scan_cb(self, msg):
        """ """
        self.lidar_data = msg
    
    def rs_cb(self, msg):
        """Callback race start signal"""
        if msg.data and not self.race_started:
            self.race_started = True
            self.lap_start_time = rospy.Time.now()
            rospy.loginfo("Race started!")
    
    def find_close_wayp(self):
        """ closest wayp to the curr pos"""
        min_dist = float('inf')
        closest_idx = 0
        search_window = 100 # search window for eff
        start_idx = max(0, self.current_waypoint_idx - search_window)
        end_idx = min(self.total_waypoints, self.current_waypoint_idx + search_window)
        
        for i in range(start_idx, end_idx):
            wp_x, wp_y = self.waypoints[i]
            dist = np.hypot(self.x - wp_x, self.y - wp_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        if closest_idx < self.current_waypoint_idx and abs(closest_idx - self.current_waypoint_idx) > self.total_waypoints / 2:
            # we back to start yet..?
            if not self.race_finished and len(self.passed_checkpoints) == len(self.checkpoints):
                self.finish_lap()
        
        self.current_waypoint_idx = closest_idx
        return closest_idx
    
    def check_for_checkpoints(self, next_idx):
        """if passed a cp"""
    
        for chk_idx, checkpoint in enumerate(self.checkpoints): # all cp to see if passed any
            if checkpoint == next_idx and checkpoint not in self.passed_checkpoints:
                # calc distance to cp
                chk_x, chk_y = self.waypoints[checkpoint]
                distance = np.hypot(self.x - chk_x, self.y - chk_y)
                
                if distance < 1.0:  # close to cp
                    self.passed_checkpoints.add(checkpoint)
                    self.checkpoint_errors.append(distance)
                    rospy.loginfo(f"Passed checkpoint {chk_idx + 1}/{len(self.checkpoints)}, distance: {distance:.2f}m")
        
                    self.next_checkpoint_idx = (chk_idx + 1) % len(self.checkpoints) # update next cp
                    if len(self.passed_checkpoints) == len(self.checkpoints):# check if completed all cp
                        rospy.loginfo("All checkpoints passed!")
    
    def finish_lap(self):
        """Handle lap completion"""
        if self.lap_start_time is not None:
            current_time = rospy.Time.now()
            lap_time = (current_time - self.lap_start_time).to_sec()
            
            self.lap_count += 1
            self.current_lap_time = lap_time
            
            if lap_time < self.best_lap_time:
                self.best_lap_time = lap_time
            
            rospy.loginfo(f"Lap {self.lap_count} completed in {lap_time:.2f} seconds")
            
            # calc avg cp err
            if self.checkpoint_errors:
                avg_error = sum(self.checkpoint_errors) / len(self.checkpoint_errors)
                rospy.loginfo(f"Average checkpoint error: {avg_error:.2f}m")
            # reset
            self.lap_start_time = current_time
            self.passed_checkpoints = set()
            self.checkpoint_errors = []
    
    def calculate_dynamic_lookahead(self):
        """calc lA dist based on curr vel"""
        
        lookahead = self.base_lookahead + 0.3 * self.velocity # faster = look further ahead
        return np.clip(lookahead, self.min_lookahead, self.max_lookahead) # lA is within bounds
    
    def find_lookahead_point(self):
        """lA pt at spec dist ahead on  path"""
        
        lookahead_distance = self.calculate_dynamic_lookahead() # get dynamic lA dist
        
        idx = self.find_close_wayp() # find closest wayp
        
        self.check_for_checkpoints(idx) # check for cp

        for i in range(self.total_waypoints): # search for wayp at lA dist
            next_idx = (idx + i) % self.total_waypoints
            wp_x, wp_y = self.waypoints[next_idx]
            
            dist = np.hypot(self.x - wp_x, self.y - wp_y) # calc dist to this wayp
            
            if dist >= lookahead_distance: # check if wayp is apx lA dist
                self.update_lookahead_marker(wp_x, wp_y)
                return wp_x, wp_y, next_idx
        next_idx = (idx + 10) % self.total_waypoints # if no pt found at lA dist
        wp_x, wp_y = self.waypoints[next_idx]
        self.update_lookahead_marker(wp_x, wp_y)
        return wp_x, wp_y, next_idx
    
    def update_lookahead_marker(self, x, y):
        """ update vis marker for lA pt """
        self.lookahead_marker.pose.position.x = x
        self.lookahead_marker.pose.position.y = y
        self.lookahead_marker.pose.position.z = 0.1
        self.lookahead_marker.header.stamp = rospy.Time.now()
        
        marker_array = MarkerArray()
        marker_array.markers = [self.lookahead_marker]
        self.marker_pub.publish(marker_array)
    
    def calculate_steering_angle(self, lookahead_x, lookahead_y):
        """ Calc steer angle """
        
        dx = lookahead_x - self.x # transform lA pt to vehicle coords
        dy = lookahead_y - self.y
        target_x = dx * np.cos(-self.yaw) - dy * np.sin(-self.yaw) # rotate to car's frame
        target_y = dx * np.sin(-self.yaw) + dy * np.cos(-self.yaw)
        lookahead_distance = np.hypot(dx, dy) # calc curv (1/R)
        curvature = 2.0 * target_y / (lookahead_distance ** 2) # calc steer angle
        steering_angle = np.arctan(curvature * self.wheelbase)
        return np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle) # lim steer angle
    
    def check_obstacles(self):
        """check Lidar data, adjust """
        if self.lidar_data is None or not self.use_obstacle_avoidance:
            return 0.0
        min_distance = float('inf') # init vars
        obstacle_angle = 0.0
        ranges = self.lidar_data.ranges # processin LIDAR data
        angle_min = self.lidar_data.angle_min
        angle_increment = self.lidar_data.angle_increment
        
        # range of -45 to +45 deg
        front_angle_range = np.pi/4  # 45 degrees
        
        for i, r in enumerate(ranges):
            angle = angle_min + i * angle_increment # calc angle of this ray
            
            if abs(angle) > front_angle_range: # Only obst within our front view
                continue
            if r < self.lidar_data.range_min or r > self.lidar_data.range_max: # skip inv meas
                continue
            if r < min_distance and r < self.obstacle_threshold: # check if close obst
                min_distance = r
                obstacle_angle = angle
        if min_distance == float('inf'): # if no obst det
            return 0.0
        obstacle_steering = -np.sign(obstacle_angle) * (self.obstacle_threshold - min_distance) / self.obstacle_threshold
        
        return obstacle_steering * self.obstacle_weight
    
    def check_for_collision(self):
        """check idf vehicle collided with wall """
        if self.lidar_data is None:
            return False 
    # check for v close reads in LIDAR data
        collision_threshold = 0.2  # 20cm 
        for r in self.lidar_data.ranges: # all ranges
            if r < self.lidar_data.range_min or r > self.lidar_data.range_max:
                continue # skip inv meas
            if r < collision_threshold:
                rospy.logwarn("Collision detected!")
                self.collision_count += 1
                return True
        return False
    
    

if __name__ == '__main__':
    try:
        controller = PurePursuitController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass