import numpy as np


class MotionModel:

    def __init__(self, node):
        ####################################
        node.declare_parameter("deterministic", "default")
        self.node = node
        self.deterministic = node.get_parameter("deterministic").get_parameter_value().bool_value
        ####################################

    def rotation_matrix(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################

        # self.node.get_logger().info("in motion eval")

        dx, dy, dtheta = odometry

        x_std = .05 * 2
        y_std = .01 * 2
        thetastd = np.pi / 30 * 2

        particles = np.array(particles)
        ret = np.empty(particles.shape)

        count = 0

        for prtcl in particles:
            xnoise = x_std * np.random.normal() if not self.deterministic else 0
            ynoise = y_std * np.random.normal() if not self.deterministic else 0
            thetanoise = thetastd * np.random.normal() if not self.deterministic else 0

            x, y, theta = prtcl[0], prtcl[1], prtcl[2]
            prev_trans_vec = np.array([[x], [y]])
            rot = self.rotation_matrix(theta)

            new_trans_vec = np.dot(rot, np.array([[dx + xnoise], [dy + ynoise]])) + prev_trans_vec
            new_row = np.append(new_trans_vec, theta + (-dtheta + thetanoise))
            ret[count] = new_row

            count += 1

        return ret

        ####################################
