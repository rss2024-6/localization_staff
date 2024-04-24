import numpy as np
import rclpy
import tf2_ros
import threading
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.stats import circmean
from sensor_msgs.msg import LaserScan
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from localization.motion_model import MotionModel
from localization.sensor_model import SensorModel

assert rclpy


class ParticleFilter(Node):

    def __init__(self, log_name=""):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.declare_parameter('num_particles', "default")
        self.declare_parameter('max_viz_particles', "default")
        self.declare_parameter('publish_odom', True)
        self.declare_parameter('angle_step', 1)
        self.declare_parameter('do_viz', "default")

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value
        self.MAX_PARTICLES = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.MAX_VIZ_PARTICLES = self.get_parameter('max_viz_particles').get_parameter_value().integer_value
        self.PUBLISH_ODOM = self.get_parameter('publish_odom').get_parameter_value().bool_value
        self.ANGLE_STEP = self.get_parameter('angle_step').get_parameter_value().integer_value
        self.DO_VIZ = self.get_parameter('do_viz').get_parameter_value().bool_value

        self.initiated = False
        self.TEST_MODE = True
        self.SAMPLE_RATE = .1

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer, self)
        self.best_p = -1

        # Get parameters
        self.num_particles = 1000.0
        self.weights = np.ones(
            int(self.num_particles)) / self.num_particles  # start with uniform weight for each particle
        self.particles = np.zeros((int(self.num_particles), 3))
        self.lock = threading.Lock()

        # Initialize publishers/subscribers

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.prev_time = self.T0 = self.prev_log_time = self.get_clock().now()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.estimate_scan_pub = self.create_publisher(LaserScan, "/estimate_scan", 1)

        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        self.particles_pub = self.create_publisher(PoseArray, "/debug", 1)

        self.get_logger().info("%s" % odom_topic)
        self.get_logger().info("%s" % scan_topic)

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        self.get_logger().info("=============meow +READY+ meow=============")

    def getOdometryMsg(self, debug=False):
        """
        Uses self.particles to get pose prediction. Puts prediction in Odometry message.

        If debug, then also publish the particles as PoseArray msg. 
        """
        # self.get_logger().info("getting odom msg")
        now = self.get_clock().now()

        x_avg, y_avg = np.mean(self.particles[:, :2], axis=0)
        theta_avg = circmean(self.particles[:, 2])
        msg = Odometry()
        msg.pose.pose.position.x, msg.pose.pose.position.y = x_avg, y_avg
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w, = quaternion_from_euler(
            0, 0, theta_avg)
        msg.header.frame_id = '/map'
        msg.header.stamp = now.to_msg()

        # get the spread of the particles (N x 3)
        if self.initiated and self.not_converged:
            std_pre = np.std(self.particles, axis=0)

            std = np.sqrt(np.sum(std_pre[:2] ** 2))  # only care about x,y

            if .01 < std < 0.2 and self.best_p > 1e-78:
                delta_t = ((now - self.start_time).nanoseconds) / 1e9

                self.not_converged = False

        ############# We want to compare the base_link to the estimate pose to get the error of our implementation, but we're running into issues with base_link not being a valid name
        # transform = self.tfBuffer.lookup_transform('base_link', 'map', rclpy.time.Time())

        # ground truth pose w.r.t. map coordinate frame

        # write data to file
        # publish transform from map to base_link_pf
        transform = TransformStamped()
        transform.header.stamp = now.to_msg()
        transform.header.frame_id = "/map"
        transform.child_frame_id = self.particle_filter_frame

        transform.transform.translation.x = x_avg
        transform.transform.translation.y = y_avg
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = quaternion_from_euler(
            0, 0, theta_avg)
        self.tf_broadcaster.sendTransform(transform)

        if debug:
            array = PoseArray()
            array.header.frame_id = '/map'
            array.poses = []
            for particle in self.particles:
                pose = Pose()
                pose.position.x, pose.position.y = particle[:2]
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_euler(
                    0, 0, particle[2])
                array.poses.append(pose)

            self.particles_pub.publish(array)

        return msg

    def laser_callback(self, scan):
        """
        update + resample
        """
        # self.get_logger().info("In laser callback")
        if not self.sensor_model.map_set:
            return

        now = self.get_clock().now()

        self.lock.acquire()
        # Update Step (distribution over the self.num_particles points)

        self.unnormed_weights = self.sensor_model.evaluate(self.particles, scan.ranges)  # P(z|x) 
        self.best_p = max(self.unnormed_weights)
        self.weights = self.unnormed_weights / np.sum(self.unnormed_weights)

        # Resampling
        idxs = np.random.choice(int(self.num_particles), int(self.num_particles), p=self.weights)
        self.particles = self.particles[idxs]
        # will sample from 0->num_particles with weights defined as self.weights

        # publish average pose
        msg = self.getOdometryMsg()
        self.odom_pub.publish(msg)

        estimate_particle = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y,
                                      euler_from_quaternion(
                                          [0, 0, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]])
        scan_new = self.sensor_model.scan_sim.scan(np.array([estimate_particle]))
        scan_msg = LaserScan()
        scan_msg.header.frame_id = self.particle_filter_frame.lstrip("/")
        scan_msg.header.stamp = now.to_msg()
        scan_msg.angle_min = -4.71 / 2  # scan.angle_min
        scan_msg.angle_max = 4.71 / 2  # scan.angle_max 
        scan_msg.angle_increment = 4.71 / 99  # scan.angle_increment
        scan_msg.time_increment = scan.time_increment
        scan_msg.range_min = scan.range_min
        scan_msg.range_max = scan.range_max
        scan_msg.scan_time = scan.scan_time

        scan_msg.ranges = scan_new[0].tolist()

        self.estimate_scan_pub.publish(scan_msg)

        self.prev_time = now

        self.lock.release()

    def odom_callback(self, odom):
        """
        Prediction Step
        """

        self.get_logger().info("In odom callback")
        if not self.sensor_model.map_set:
            return

        self.lock.acquire()

        now = self.get_clock().now()

        vx, vy = odom.twist.twist.linear.x, odom.twist.twist.linear.y,
        wz = odom.twist.twist.angular.z

        dt = (now - self.prev_time).nanoseconds / 1e9

        self.get_logger().info(f'dt {dt} vx {vx} vy {vy}')

        dx, dy = vx * dt, vy * dt
        dtheta = wz * dt

        delta_x = [dx, dy, dtheta]

        self.prev_time = now

        self.particles = self.motion_model.evaluate(self.particles, delta_x)

        msg = self.getOdometryMsg()
        self.odom_pub.publish(msg)

        self.lock.release()

    def pose_callback(self, pose):
        """
        This is done whenever the green arrow is placed down in RVIZ. 

        pose: guess from YOU (?)

        Sample around the pose (x+eps_x, y+eps_y, theta+eps_theta) eps_? ~  N(0,sigma)
        """
        if not self.sensor_model.map_set:
            return
        self.lock.acquire()

        std_trans = 1.
        std_theta = np.pi / 4
        x, y = pose.pose.pose.position.x, pose.pose.pose.position.y
        theta = euler_from_quaternion(
            [pose.pose.pose.orientation.x, pose.pose.pose.orientation.y, pose.pose.pose.orientation.z,
             pose.pose.pose.orientation.w])[-1]

        x_samples = (std_trans * np.random.randn(int(self.num_particles)) + x)[:, None]
        y_samples = (std_trans * np.random.randn(int(self.num_particles)) + y)[:, None]

        theta_samples = (std_theta * np.random.randn(int(self.num_particles)) + theta)[:, None]

        self.particles = np.hstack((x_samples, y_samples, theta_samples))
        msg = self.getOdometryMsg()
        self.odom_pub.publish(msg)

        self.lock.release()


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
