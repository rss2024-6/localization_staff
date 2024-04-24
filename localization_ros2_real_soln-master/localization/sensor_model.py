import numpy as np
import sys
from nav_msgs.msg import OccupancyGrid
from tf_transformations import euler_from_quaternion

from scan_simulator_2d import PyScanSimulator2D

# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        self.node = node

        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', "default")
        node.declare_parameter('scan_theta_discretization', "default")
        node.declare_parameter('scan_field_of_view', "default")
        node.declare_parameter('lidar_scale_to_map_scale', 1)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        self.alpha_hit = .74
        self.alpha_short = .07
        self.alpha_max = .07
        self.alpha_rand = .12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        self.z_max = 200.0
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)
        
        
        node.get_logger().info("finished setting up sensor model")

    def pmax(self, z_k):
        return 1 if z_k == self.z_max else 0

    def phit(self, z_k, d):
        return (1 / np.sqrt(2 * np.pi * self.sigma_hit * self.sigma_hit)) * np.exp(
            -(np.square(z_k - d)) / (2 * self.sigma_hit ** 2)) if 0 <= z_k and z_k <= self.z_max else 0

    def pshort(self, z_k, d):
        return 2 * (1 - z_k / d) / d if 0 <= z_k and z_k <= d and d != 0 else 0

    def prand(self, z_k):
        return 1 / self.z_max if 0 <= z_k and z_k <= self.z_max else 0

    def meters_to_pixels(self, meters):
        return (meters / float(self.resolution * self.lidar_scale_to_map_scale))

    def pdf_without_phit(self, z_k, d):
        return (self.alpha_max * self.pmax(z_k) + self.alpha_rand * self.prand(z_k) + self.alpha_short * self.pshort(
            z_k, d))

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """

        # x = [] + 1
        # 
        for z_k in range(0, 201):
            for d in range(0, 201):
                self.sensor_model_table[z_k, d] = self.phit(float(z_k), float(d))

        nu = np.sum(self.sensor_model_table, axis=0)
        self.sensor_model_table /= nu
        # self.node.get_logger().info(f"nu_value {nu}")

        for z_k in range(0, 201):
            for d in range(0, 201):
                # self.sensor_model_table[z_k, d] = self.alpha_hit * self.sensor_model_table[
                #     z_k, d] + self.pdf_without_phit(float(z_k), float(d))
                self.sensor_model_table[z_k, d] = self.alpha_hit * self.sensor_model_table[
                    z_k, d] + self.alpha_short * self.pshort(z_k, d) + self.alpha_max * self.pmax(
                    z_k) + self.alpha_rand * self.prand(
                    z_k)

        thesum = np.sum(self.sensor_model_table, axis=0)
        self.node.get_logger().info(f"value {self.sensor_model_table[:, 0].sum()}")

        # for z_k in range(0, self.table_width):
        #     for d in range(0, self.table_width):
        #         self.sensor_model_table[z_k, d] = self.alpha_hit * self.phit(float(z_k),
        #                                                                      float(d)) + self.pdf_without_phit(
        #             float(z_k), float(d))

        self.sensor_model_table /= np.sum(self.sensor_model_table, axis=0)

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """
        
        # self.node.get_logger().info("evaluating sensor model")
        
        if not self.map_set:
            return

        # scans: d (depending on i) for each particle. [N x num_beams]
        # Calculate likelihood P(zk | xk, "i") (how likely is zk if we scan from angle i and position xk).

        # zk: the actual scan from the lidar 
        # xk: the potential positions (particles)
        # d_i: the ground truth for this angle i. 
    
        stride = 11 # 10

        scans = self.scan_sim.scan(np.array(particles))
        N, m = scans.shape
        # self.node.get_logger().info(str(np.array(particles).shape))
        # self.node.get_logger().info(str(np.array(scans).shape))
        # self.node.get_logger().info(str(np.array(observation).shape))
        observation = self.meters_to_pixels(np.array(observation[::11]))# [::-1]  # 1 x num_beams
        # self.node.get_logger().info(str(observation.shape))
        observation = np.repeat(observation[np.newaxis, :], N, axis=0)  # N x num_beams
        observation = np.rint(np.clip(observation, 0, 200)).astype(int)

        scans = self.meters_to_pixels(scans)
        scans = np.rint(np.clip(scans, 0, 200)).astype(int)
        
        probs = self.sensor_model_table[observation, scans]  # length N

        # probs = np.prod(probs, axis = 1) #intersect of all d^i's in each particle simulated scan
        probs = np.exp(np.sum(np.log(probs), axis=1))
        return np.array(np.power(probs, 1 / 2.2))
        # return probs

    def map_callback(self, map_msg):
        self.node.get_logger().info('yo i got the map')
        
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        self.node.get_logger().info('about to set')

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        self.node.get_logger().info('done setting map')

        # Make the map set
        self.map_set = True

        self.node.get_logger().info("Map initialized")
