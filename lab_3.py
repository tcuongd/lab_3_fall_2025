import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

np.set_printoptions(precision=3, suppress=True)

Kp = 3
Kd = 0.1


class InverseKinematics(Node):
    def __init__(self):
        super().__init__("inverse_kinematics")
        self.joint_subscription = self.create_subscription(
            JointState, "joint_states", self.listener_callback, 10
        )
        self.joint_subscription  # prevent unused variable warning

        self.command_publisher = self.create_publisher(
            Float64MultiArray, "/forward_command_controller/commands", 10
        )

        self.pd_timer_period = 1.0 / 200  # 200 Hz
        self.ik_timer_period = 1.0 / 20  # 20 Hz
        self.pd_timer = self.create_timer(self.pd_timer_period, self.pd_timer_callback)
        self.ik_timer = self.create_timer(self.ik_timer_period, self.ik_timer_callback)

        self.joint_positions = None
        self.joint_velocities = None
        self.target_joint_positions = None

        self.ee_triangle_positions = np.array(
            [
                [0.05, 0.0, -0.12],  # Touchdown
                [-0.05, 0.0, -0.12],  # Liftoff
                [0.0, 0.0, -0.06],  # Mid-swing
            ]
        )

        center_to_rf_hip = np.array([0.07500, -0.08350, 0])
        self.ee_triangle_positions = self.ee_triangle_positions + center_to_rf_hip
        self.current_target = 0
        self.t = 0

    def listener_callback(self, msg):
        joints_of_interest = ["leg_front_r_1", "leg_front_r_2", "leg_front_r_3"]
        self.joint_positions = np.array(
            [msg.position[msg.name.index(joint)] for joint in joints_of_interest]
        )
        self.joint_velocities = np.array(
            [msg.velocity[msg.name.index(joint)] for joint in joints_of_interest]
        )

    def forward_kinematics(self, theta1: float, theta2: float, theta3: float) -> np.ndarray:
        def rotation_x(angle):
            # rotation about the x-axis implemented for you
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, np.cos(angle), -np.sin(angle), 0],
                    [0, np.sin(angle), np.cos(angle), 0],
                    [0, 0, 0, 1],
                ]
            )

        def rotation_y(angle):
            return np.array(
                [
                    [np.cos(angle), 0, np.sin(angle), 0],
                    [0, 1, 0, 0],
                    [-np.sin(angle), 0, np.cos(angle), 0],
                    [0, 0, 0, 1],
                ]
            )

        def rotation_z(angle):
            return np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0, 0],
                    [np.sin(angle), np.cos(angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

        def translation(x, y, z):
            return np.array(
                [
                    [1, 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1],
                ]
            )

        # theta is positive when the Z-axis points out of the BOTTOM (i.e. uncovered metal parts) of the BLDC motor.
        # Since we keep the frame orientation the same on both left and right sides, motors 1 and 3 will use negative angles on the left and positive angles on the right.

        T_0_1 = translation(+0.07500, -0.0335, 0) @ rotation_x(1.57080) @ rotation_z(+theta1)
        T_1_2 = translation(0, 0, -0.039) @ rotation_y(-1.57080) @ rotation_z(+theta2)
        T_2_3 = translation(0, -0.0494, 0.0685) @ rotation_y(+1.57080) @ rotation_z(+theta3)
        T_3_ee = translation(0.06231, -0.06216, -0.018)
        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee

        return T_0_ee[:3, 3].copy()

    def inverse_kinematics(self, target_ee, initial_guess=[0, 0, 0]):
        def cost_function(theta) -> tuple[float, np.ndarray]:
            """
            Use the forward_kinematics method to get the current end-effector position.
            Calculate the L1 distance between the current and target end-effector positions.
            Return the sum of squared L1 distances as the cost (AKA the squared L2 norm of the error vector).
            """
            current_ee = self.forward_kinematics(*theta)
            l1_errors = np.abs(current_ee - np.array(target_ee))
            return np.sqrt(np.sum(l1_errors**2)), l1_errors

        def gradient(theta: np.ndarray, epsilon=1e-3) -> np.ndarray:
            grads: list[float] = []

            for i, angle in enumerate(theta):
                theta_back = theta.copy()
                theta_back[i] = angle - epsilon

                theta_forward = theta.copy()
                theta_forward[i] = angle + epsilon

                c_back, _ = cost_function(theta_back)
                c_forward, _ = cost_function(theta_forward)

                grads.append((c_forward - c_back) / 2 * epsilon)

            return np.array(grads)

        theta = np.array(initial_guess)
        learning_rate = 5.0
        max_iterations = 10
        tolerance = 0.01

        cost_l = []
        for _ in range(max_iterations):
            grad = gradient(theta)
            theta -= learning_rate * grad

            # Use mean L1 to check convergence
            _, l1_errors = cost_function(theta)
            cost_l.append(np.mean(l1_errors))

            if abs(cost_l[-1] - cost_l[-2]) < tolerance:
                print(f"Converged: {cost_l[-1]:.4f} vs {cost_l[-2]:.4f}")
                break

            # TODO (BONUS): Implement the (quasi-)Newton's method instead of finite differences for faster convergence

        print(f"Cost: {cost_l}")

        return theta

    def interpolate_triangle(self, t: float) -> np.ndarray:
        # Intepolate between the three triangle positions in the self.ee_triangle_positions
        # based on the current time t
        ################################################################################################
        # TODO: Implement the interpolation function
        ################################################################################################
        if t >= 0 and t <= 1:
            shift = t * (self.ee_triangle_positions[1] - self.ee_triangle_positions[0])
            return self.ee_triangle_positions[0] + shift
        elif t > 1 and t <= 2:
            shift = (2 - t) * (self.ee_triangle_positions[2] - self.ee_triangle_positions[1])
            return self.ee_triangle_positions[1] + shift
        elif t > 2 and t <= 3:
            shift = (3 - t) * (self.ee_triangle_positions[0] - self.ee_triangle_positions[2])
            return self.ee_triangle_positions[2] + shift
        else:
            raise ValueError(f"t out of bounds: {t} must be in [0, 3].")

    def ik_timer_callback(self):
        if self.joint_positions is not None:
            target_ee = self.interpolate_triangle(self.t)
            self.target_joint_positions = self.inverse_kinematics(target_ee, self.joint_positions)
            current_ee = self.forward_kinematics(*self.joint_positions)

            next_t = self.t + self.ik_timer.timer_period_ns / 1e9
            if next_t > 3:
                self.t = 0.0
            else:
                self.t = next_t

            self.get_logger().info(
                f"Target EE: {target_ee}, Current EE: {current_ee}, Target Angles: {self.target_joint_positions}, Target Angles to EE: {self.forward_kinematics(*self.target_joint_positions)}, Current Angles: {self.joint_positions}"
            )

    def pd_timer_callback(self):
        if self.target_joint_positions is not None:
            command_msg = Float64MultiArray()
            command_msg.data = self.target_joint_positions.tolist()
            self.command_publisher.publish(command_msg)


def main():
    rclpy.init()
    inverse_kinematics = InverseKinematics()

    try:
        rclpy.spin(inverse_kinematics)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        # Send zero torques
        zero_torques = Float64MultiArray()
        zero_torques.data = [0.0, 0.0, 0.0]
        inverse_kinematics.command_publisher.publish(zero_torques)

        inverse_kinematics.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
