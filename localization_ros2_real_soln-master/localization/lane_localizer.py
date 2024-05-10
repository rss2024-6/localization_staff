import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, TransformStamped, Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.stats import circmean
from sensor_msgs.msg import LaserScan
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from visualization_msgs.msg import Marker
import json
import math


assert rclpy


class LaneLocalizer(Node):
    def __init__(self):
        super().__init__("lane_localizer")
        

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/sample_pose", 10)
        self.odom_sub = self.create_subscription(Odometry, "/pf/pose/odom", self.update_location, 1)
        self.goal_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/goal_pose",
            self.test_location_update,
            10
        )

        self.simulation = True
    
    def update_location(self, msg):
        """
        Publishes message to particle filter to add in a sample with given location estimate
        Params: Odometry
        Publishes PoseWithCovarianceStamped of new location estimate
        """
        pose = msg.pose.pose
        if self.simulation:
            measured_line_location = self.get_simulated_lane_location(pose)
        else:
            measured_line_location = self.get_lane_location(pose)
        
        expected_line_location = self.get_expected_lane_location(pose)


        measured_expected_vector = np.array([expected_line_location.x - measured_line_location.x, expected_line_location.y - measured_line_location.y])
        curr_pose = np.array([pose.position.x, pose.position.y])

        new_pose = curr_pose + measured_expected_vector

        new_pose_msg = PoseWithCovarianceStamped()
        new_pose_msg.header.frame_id = "map"
        new_pose_msg.header.stamp = self.get_clock().now().to_msg()
        new_pose_msg.pose.pose.position.x = new_pose[0]
        new_pose_msg.pose.pose.position.y = new_pose[1]
        new_pose_msg.pose.pose.position.z = pose.position.z
        new_pose_msg.pose.pose.orientation = pose.orientation
        new_pose_msg.pose.covariance = msg.pose.covariance

        self.pose_pub.publish(new_pose_msg)

    def get_lane_location(self, pose):
        """
        Returns measured lane location
        """
        pass


    def get_expected_lane_location(self, pose):
        """
        Returns a projection of the robots location on to the nearest lane line
        """
        position = pose.position
        lane_line = self.find_closest_line(position)
        projection = self.project_point(position, lane_line[0], lane_line[1])
        
        return projection
        

    def get_simulated_lane_location(self, pose):
        """
        Returns a simulated measurement of a projection of the robots location on to the nearest lane line
        """
        projection = self.get_expected_lane_location(pose)
        # TODO: Add noise to the projection
        return projection
    
    def project_point(P, P1, P2, type='same'):
        """
        type: 'same', 'point', 'np'
        """
        # Convert points to numpy arrays
        if isinstance(P, Point):
            originally_point = True
            P = np.array([P.x, P.y])
            P1 = np.array([P1.x, P1.y])
            P2 = np.array([P2.x, P2.y])
        else:
            P = np.array(P)
            P1 = np.array(P1)
            P2 = np.array(P2)

        # Direction vector from P1 to P2
        line_vector = P2 - P1

        # Vector from P1 to P
        point_vector = P - P1

        # Project point_vector onto line_vector using dot product
        scalar_projection = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)

        # Calculate the projection point
        projection = P1 + scalar_projection * line_vector

        if (originally_point and type == 'same') or type == 'point':
            projection = Point(x=projection[0], y=projection[1])
        return projection

    def find_closest_line(self, point):
        """
        Returns closest lane line to a point in the map space
        """
        min_distance = float('inf')
        closest_segment = None

        # Iterate over pairs of points to define line segments
        for i in range(len(self.lane_points) - 1):
            p1 = self.lane_points[i]
            p2 = self.lane_points[i + 1]
            distance = self.point_line_segment_distance(point.x, point.y, p1.x, p1.y, p2.x, p2.y)
            if distance < min_distance:
                min_distance = distance
                closest_segment = (p1, p2)
        return closest_segment
    
    def point_line_segment_distance(self, px, py, x1, y1, x2, y2):
        # Line coefficients
        A = y2 - y1
        B = x1 - x2
        C = A * x1 + B * y1

        # Perpendicular distance
        dist = abs(A * px + B * py - C) / math.sqrt(A**2 + B**2)
        # Projection point on the line
        dx = x2 - x1
        dy = y2 - y1
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        if t < 0.0 or t > 1.0:  # Outside the segment, use the closest endpoint
            if t < 0.0:
                dist = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            else:
                dist = math.sqrt((px - x2) ** 2 + (py - y2) ** 2)
        return dist

        
    def visualize_lanes(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lines"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        with open('src/path_planning/lanes/full-lane.traj', 'r') as file:
            # Load its content and convert it into a dictionary
            data = json.load(file)

        # Define the points
        data_points = data['points']
        points = [Point(x=dp['x'], y=dp['y']) for dp in data_points]

        marker.points = points

        # Define the line color and scale
        marker.scale.x = 0.1  # Line width
        marker.color.a = 1.0  # Alpha
        marker.color.r = 1.0  # Red
        marker.color.g = 0.0  # Green
        marker.color.b = 0.0  # Blue

        self.lanes_pub.publish(marker)
        # self.show_lane_directions()
    
    def test_location_update(self, pose):
        self.pose_pub.publish(pose)

    def test_lane_location_update(self, point):
        pass


def main(args=None):
    rclpy.init(args=args)
    ll = LaneLocalizer()
    rclpy.spin(ll)
    rclpy.shutdown()