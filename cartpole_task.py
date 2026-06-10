import mujoco
from mujoco import rollout
import numpy as np
import matplotlib.pyplot as plt


class CartpoleSwingUp:
    """Cartpole swing-up optimal control cost."""

    def __init__(self, N_ENVS):

        # Load model
        model_xml = "model/cartpole/scene.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(model_xml)

        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_datas = [mujoco.MjData(self.mj_model) for _ in range(N_ENVS)]

        self.dt = self.mj_model.opt.timestep

        # Problem size
        self.N_ENVS = N_ENVS
        self.N_STEPS = 20
        self.N_HOLD = 4

        # Initial state
        state_size = mujoco.mj_stateSize(
            self.mj_model,
            mujoco.mjtState.mjSTATE_FULLPHYSICS
        )

        self.q_init = np.array([0, np.pi])

        self.state_init = np.zeros((N_ENVS, state_size))

        # Initialize near hanging-down state
        self.state_init[:, 1:1+self.mj_model.nq] = self.q_init

        # Desired goal state
        # qpos = [cart_position, pole_angle]
        self.q_desired = np.array([0.0, 0.0])

        # Cost weights
        self.w_cart = 10.0
        self.w_pole = 10.0
        self.w_vel = 10.0
        self.w_ctrl = 1e-4

        self.x_max = 0.95
        self.u_max = 2.

        # Held controls
        self.control_hold = np.zeros(
            (N_ENVS,
             self.N_STEPS * self.N_HOLD,
             self.mj_model.nu)
        )

    def reset_datas(self):

        for data in self.mj_datas:
            data.qpos[:] = self.q_init
            data.qvel[:] = 0.0
            mujoco.mj_forward(self.mj_model, data)

    def state_penalty(self, q):
        
        excess = np.maximum(np.abs(q[:, 0]) - self.x_max, 0.0)
        return 100.0 * excess**2


    def state_cost(self, q, qvel):

        cart_cost = self.w_cart * q[:, 0]**2
        pole_cost = self.w_pole * q[:, 1]**2
        vel_cost = self.w_vel * np.sum(qvel**2, axis=1)

        return cart_cost + pole_cost + vel_cost

    def control_cost(self, u):
        excess = np.maximum(np.abs(u) - self.u_max, 0.0)
        penalty = 100.0 * np.sum(excess**2, axis=1)
        return self.w_ctrl * np.sum(u**2, axis=1) + penalty

    def zero_order_hold(self, control):
        # TODO: fill self.control_hold with zero-order hold expanded controls
        #   self.control_hold has shape (N_ENVS, N_STEPS * N_HOLD, nu)
        #   control has shape (N_ENVS, N_STEPS, nu)
        #   For each control step t and each hold step j, assign:
        #     self.control_hold[:, t * self.N_HOLD + j] = control[:, t]
        #   i.e. each control is repeated N_HOLD times before the next one is applied
        raise NotImplementedError

    def rollout(self, control):
        self.zero_order_hold(control)

        states, _ = rollout.rollout(
            self.mj_model,
            self.mj_datas,
            self.state_init,
            self.control_hold,
            nstep=self.N_STEPS * self.N_HOLD
        )

        total_cost = np.zeros(self.N_ENVS)

        for t in range(self.N_STEPS):

            sim_t = (t + 1) * self.N_HOLD - 1
            qpos = states[:, sim_t, 1:1+self.mj_model.nq]
            # qvel = states[:, sim_t, 1 + self.mj_model.nq:]

            state_c = self.state_penalty(qpos)
            ctrl_c = self.control_cost(control[:, t])

            total_cost += state_c + ctrl_c

        # terminal bonus
        q_final = states[:, -1, 1:1+self.mj_model.nq]
        qvel_final = states[:, -1,
                            1+self.mj_model.nq:
                            1+self.mj_model.nq+self.mj_model.nv]

        total_cost += self.state_cost(q_final, qvel_final)

        return total_cost

    def get_warm_start(self):

        return np.zeros(
            (self.N_STEPS, self.mj_model.nu)
        )

    def cost_function(self, control):

        self.reset_datas()

        cost = self.rollout(control)

        return cost
    


    def plot_solution(self, q_traj, v_traj, control_traj):
        """
        Plot cartpole state and control trajectories.

        q_traj shape:
            (T, 2)
            [x, theta]

        v_traj shape:
            (T, 2)
            [x_dot, theta_dot]

        control_traj shape:
            (T-1,) or (T-1, 1)
        """

        control_traj = np.asarray(control_traj).squeeze()

        T = q_traj.shape[0]

        time_state = np.arange(T) * self.dt * self.N_HOLD
        time_ctrl = np.arange(len(control_traj)) * self.dt * self.N_HOLD

        x = q_traj[:, 0]
        theta = q_traj[:, 1]

        x_dot = v_traj[:, 0]
        theta_dot = v_traj[:, 1]

        fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

        # Cart position
        axs[0].plot(time_state, x)
        axs[0].axhline(self.x_max, linestyle="--")
        axs[0].axhline(-self.x_max, linestyle="--")
        axs[0].plot(time_state[-1], 0, "o")
        axs[0].set_ylabel("x [m]")
        axs[0].grid(True)

        # Pole angle
        axs[1].plot(time_state, theta)
        axs[1].plot(time_state[-1], 0, "o")
        axs[1].set_ylabel("theta [rad]")
        axs[1].grid(True)

        # Cart velocity
        axs[2].plot(time_state, x_dot)
        axs[2].set_ylabel("x_dot [m/s]")
        axs[2].grid(True)

        # Pole angular velocity
        axs[3].plot(time_state, theta_dot)
        axs[3].set_ylabel("theta_dot [rad/s]")
        axs[3].grid(True)

        # Control
        axs[4].plot(time_ctrl, control_traj)
        axs[4].axhline(self.u_max, linestyle="--")
        axs[4].axhline(-self.u_max, linestyle="--")
        axs[4].set_ylabel("u")
        axs[4].set_xlabel("time [s]")
        axs[4].grid(True)

        plt.tight_layout()
        plt.show()