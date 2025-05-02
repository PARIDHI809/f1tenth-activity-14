#!/usr/bin/env python3

import rospy
import numpy as np
import math
import csv
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion

class PurePursuitController:
    def __init__(self):
        # Initialize node
        rospy.init_node('pure_pursuit_controller', anonymous=True)
        
        # Load parameters from params.yaml
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.command_topic = rospy.get_param("~command_topic", "/drive")
        
        # Controller parameters (tune these!)
        self.lookahead_distance = 1.5  # meters
        self.max_velocity = 3.0  # m/s
        self.wheelbase = 0.33  # meters, F1TENTH car wheelbase
        
        # Load waypoints from CSV
        self.waypoints = self.load_waypoints("gp_centerline.csv")
        self.total_waypoints = len(self.waypoints)
        self.checkpoints = [i for i in range(0, self.total_waypoints, 100)]
        self.next_checkpoint_idx = 0
        
        # State variables
        self.race_started = False
        self.current_waypoint_idx = 0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        
        # ROS Publishers and Subscribers
        self.drive_pub = rospy.Publisher(self.command_topic, AckermannDriveStamped, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        self.race_start_sub = rospy.Subscriber('/race_start', Bool, self.race_start_callback)
        
        # Timer for control loop
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("Pure Pursuit Controller initialized!")
    
    def load_waypoints(self, filename):
        waypoints = []
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header if present
            for row in csv_reader:
                x, y = float(row[0]), float(row[1])
                waypoints.append((x, y))
        return waypoints
    
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        orientation_q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
    
    def race_start_callback(self, msg):
        self.race_started = msg.data
        if self.race_started:
            rospy.loginfo("Race started!")
    
    def find_closest_waypoint(self):
        min_dist = float('inf')
        closest_idx = 0
        
        # Search in a window around the current waypoint for efficiency
        search_window = 50
        start_idx = max(0, self.current_waypoint_idx - search_window)
        end_idx = min(self.total_waypoints, self.current_waypoint_idx + search_window)
        
        for i in range(start_idx, end_idx):
            wp_x, wp_y = self.waypoints[i]
            dist = np.hypot(self.x - wp_x, self.y - wp_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Check if we need to wrap around the track
        if closest_idx == self.total_waypoints - 1:
            closest_idx = 0
        
        self.current_waypoint_idx = closest_idx
        return closest_idx
    
    def find_lookahead_point(self):
        # Start searching from the closest waypoint
        idx = self.find_closest_waypoint()
        
        # Loop through waypoints ahead to find one at lookahead distance
        for i in range(self.total_waypoints):
            next_idx = (idx + i) % self.total_waypoints
            wp_x, wp_y = self.waypoints[next_idx]
            
            # Calculate distance to this waypoint
            dist = np.hypot(self.x - wp_x, self.y - wp_y)
            
            # Check if this waypoint is approximately at our lookahead distance
            if dist >= self.lookahead_distance:
                # Check if we passed a checkpoint
                if next_idx in self.checkpoints and next_idx == self.checkpoints[self.next_checkpoint_idx]:
                    rospy.loginfo(f"Passed checkpoint {self.next_checkpoint_idx}")
                    self.next_checkpoint_idx = (self.next_checkpoint_idx + 1) % len(self.checkpoints)
                return wp_x, wp_y
        
        # If no suitable point found, use the furthest point
        return self.waypoints[(idx + 10) % self.total_waypoints]
    
    def calculate_steering_angle(self, lookahead_x, lookahead_y):
        # Transform the lookahead point from global to vehicle coordinates
        dx = lookahead_x - self.x
        dy = lookahead_y - self.y
        
        # Rotate the point to align with vehicle's heading
        target_x = dx * np.cos(-self.yaw) - dy * np.sin(-self.yaw)
        target_y = dx * np.sin(-self.yaw) + dy * np.cos(-self.yaw)
        
        # Calculate curvature/steering angle using pure pursuit formula
        curvature = 2.0 * target_y / (self.lookahead_distance ** 2)
        steering_angle = np.arctan(curvature * self.wheelbase)
        
        return steering_angle
    
    def calculate_velocity(self, steering_angle):
        # Simple velocity control - slow down in curves
        abs_steering = abs(steering_angle)
        velocity = self.max_velocity * (1.0 - abs_steering / (np.pi/4))
        return max(1.0, velocity)  # Minimum velocity to keep moving
    
    def control_loop(self, event):
        if not self.race_started:
            return
        
        # Find lookahead point
        lookahead_x, lookahead_y = self.find_lookahead_point()
        
        # Calculate steering angle
        steering_angle = self.calculate_steering_angle(lookahead_x, lookahead_y)
        
        # Calculate velocity based on steering angle
        velocity = self.calculate_velocity(steering_angle)
        
        # Create and publish drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = velocity
        
        self.drive_pub.publish(drive_msg)

if __name__ == '__main__':
    try:
        controller = PurePursuitController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass