import mujoco
from mujoco import rollout
import numpy as np


class Humanoid:
    """The Unitree G1 humanoid must move to a given config."""

    def __init__(
        self,
        N_ENVS,
    ) -> None:

        # Load model
        model_xml = "model/g1/scene_23dof.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(model_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_datas = [mujoco.MjData(self.mj_model) for _ in range(N_ENVS)]
        self.dt = self.mj_model.opt.timestep

        # init problem size
        self.N_ENVS = N_ENVS
        self.N_STEPS = 10
        self.N_HOLD = 5

        # Get sensor IDs
        self.left_foot_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"
        )
        self.right_foot_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"
        )

        # Cost
        cost_weights = 1 * np.ones(self.mj_model.nv)
        cost_weights[:6] = 100  # Base pose is more important
        self.cost_weights = cost_weights

        # Initial state

        q_goal = np.load("model/g1/reference_position.npy")

        self.q_init = np.load(
            "model/g1/reference_position.npy"
        )  # self.mj_model.qpos0.copy()
        state_size = mujoco.mj_stateSize(
            self.mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        self.state_init = np.zeros((self.N_ENVS, state_size))
        self.state_init[:, 1 : 1 + self.mj_model.nq] = self.q_init

        # q_goal = self.mj_model.qpos0.copy()

        self.set_goal_pos(q_goal)

        # control hold
        self.control_hold = np.zeros(
            (N_ENVS, self.N_STEPS * self.N_HOLD, self.mj_model.nu)
        )

    def config_error(self, q, q_goal):
        pos_err = q[:, :3] - q_goal[:3]
        rot_err = np.zeros((self.N_ENVS, 3))
        for i in range(self.N_ENVS):
            mujoco.mju_subQuat(rot_err[i], q[i, 3:7], q_goal[3:7])
        rest_err = q[:, 7:] - q_goal[7:]

        error = np.concatenate([pos_err, rot_err, rest_err], axis=1)
        error_sq = error**2

        return error_sq @ self.cost_weights

    def reset_datas(self):
        for data in self.mj_datas:
            data.qpos[:] = self.q_init
            data.qvel[:] = 0
            mujoco.mj_forward(self.mj_model, data)

    def get_warm_start(self):
        control_init = np.zeros((self.N_STEPS, self.mj_model.nu))
        control_init += self.q_init[7:]
        return control_init

    def set_goal_pos(self, q_goal):
        self.q_desired = np.array(q_goal, dtype=float)
        self.mj_data.qpos[:] = q_goal.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.ref_left_foot_pos = np.array(
            self.mj_data.site_xpos[self.left_foot_site_id]
        )
        self.ref_right_foot_pos = np.array(
            self.mj_data.site_xpos[self.right_foot_site_id]
        )

    def _get_foot_position_errors(self, datas):

        left_err = np.array(
            [
                np.linalg.norm(
                    d.site_xpos[self.left_foot_site_id] - self.ref_left_foot_pos
                )
                ** 2
                for d in datas
            ]
        )
        right_err = np.array(
            [
                np.linalg.norm(
                    d.site_xpos[self.right_foot_site_id] - self.ref_right_foot_pos
                )
                ** 2
                for d in datas
            ]
        )

        return left_err + right_err

    def _get_acc_cost(self, qacc):
        # Acceleration penalty
        return np.linalg.norm(qacc, axis=1) ** 2

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
            nstep=self.N_STEPS * self.N_HOLD,
        )

        cost = np.zeros(self.N_ENVS)

        for sim_t in range(self.N_STEPS * self.N_HOLD - 1):
            qacc = (
                states[:, sim_t + 1, self.mj_model.nq + 1 :]
                - states[:, sim_t, self.mj_model.nq + 1 :]
            ) / self.dt
            cost += 0.01 * self._get_acc_cost(qacc)

        qpos = states[:, -1, 1 : self.mj_model.nq + 1]
        cost += self.config_error(qpos, self.q_desired)
        cost += 100 * self._get_foot_position_errors(self.mj_datas)
        return cost

    def cost_function(self, control):
        self.reset_datas()
        cost = self.rollout(control)
        return cost
