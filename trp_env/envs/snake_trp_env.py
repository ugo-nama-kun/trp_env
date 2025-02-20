import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from trp_env.envs.two_resource_env import TwoResourceEnv, AgentType

DEFAULT_CAMERA_CONFIG = {
    "distance": 20.0,
}


class MySnakeEnv(MujocoEnv, utils.EzPickle):
    FILE = "snake.xml"
    ORI_IND = 2
    TYPE = AgentType.SWIMMER
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps"  : 7,
    }

    def __init__(self,
                 xml_path,
                 ctrl_cost_coeff=1e-2,
                 ego_obs=True,
                 sparse_rew=False,
                 vision=False,
                 width=64,
                 height=64,
                 *args, **kwargs):

        utils.EzPickle.__init__(
            self,
            xml_path,
            ctrl_cost_coeff,  # gym has 1 here!
            ego_obs,
            sparse_rew,
            vision,
            width,
            height,
            **kwargs
        )

        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.ego_obs = ego_obs
        self.sparse_rew = sparse_rew

        frame_skip = 150
        obs_shape = 12 + 10 + 10 + 2
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip,
            observation_space,
            width=width if vision else 480,
            height=height if vision else 480,
            default_camera_config=DEFAULT_CAMERA_CONFIG
        )
        
    def get_current_obs(self):
        if self.ego_obs:
            return np.concatenate([
                self.data.qpos.flat.copy()[2:],
                self.data.qvel.flat.copy(),
            ]).reshape(-1)
        else:
            # Comment from Yossy:
            # 17 dim observation is descried in HiPPO paper. However, ego-centric observation (less than 17 dim) is used in the code (^^;)
            # https://github.com/alexlioralexli/rllab-finetuning/blob/2dae9141d0fdc284d04f18931907131d66b43023/sandbox/finetuning/runs/pg_test.py#L220
            return np.concatenate([
                self.data.qpos.flat.copy(),
                self.data.qvel.flat.copy(),
                self.get_body_com("torso").flat.copy(),
            ]).reshape(-1)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        next_obs = self.get_current_obs()
        lb, ub = self.action_space.low, self.action_space.high
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(a / scaling))
        forward_reward = np.linalg.norm(self.get_body_comvel("torso"))  # swimmer has no problem of jumping reward
        reward = forward_reward - ctrl_cost
        done = False
        # Written in original Snake.
        # https://github.com/alexlioralexli/rllab-finetuning/blob/2dae9141d0fdc284d04f18931907131d66b43023/sandbox/finetuning/envs/mujoco/snake_env.py#L12
        if self.sparse_rew:
            if abs(self.get_body_com("torso")[0]) > 100.0:
                reward = 1.0
                done = True
            else:
                reward = 0.
        com = np.concatenate([self.get_body_com("torso").flat]).reshape(-1)
        ori = self.get_ori()
        info = dict(reward_fwd=forward_reward, reward_ctrl=ctrl_cost, com=com, ori=ori)
        return next_obs, reward, done, False, info

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self.get_current_obs()

    def get_ori(self):
        return self.data.qpos[self.__class__.ORI_IND]

    def get_body_comvel(self, body_name):
        # Imported from https://github.com/alexlioralexli/rllab-finetuning/blob/2dae9141d0fdc284d04f18931907131d66b43023/rllab/envs/mujoco/mujoco_env.py#L236
        # idx = self.sim.body_names.index(body_name)
        # return self.sim.body_comvels[idx]
        # from mujoco document: https://mujoco.readthedocs.io/en/latest/APIreference.html?highlight=cvel#mjdata
        return self.data.body(body_name).cvel[3:]


class SnakeTwoResourceEnv(TwoResourceEnv):
    """
    Two Resource Problem with approximated scale of Konidaris & Barto paper
    """
    MODEL_CLASS = MySnakeEnv
    ORI_IND = MySnakeEnv.ORI_IND
    
    def __init__(self,
                 vision=False,
                 width=64,
                 height=64,
                 *args, **kwargs):
        super().__init__(
            vision=vision,
            width=width,
            height=height,
            *args, **kwargs
        )


class SnakeSmallTwoResourceEnv(TwoResourceEnv):
    """
    Small-sized Two Resource Problem with the scale of original GatherEnv
    """
    MODEL_CLASS = MySnakeEnv
    ORI_IND = MySnakeEnv.ORI_IND

    def __init__(self,
                 n_blue=6,
                 n_red=4,
                 activity_range=6.,
                 n_bins=20,
                 sensor_range=16,
                 vision=False,
                 width=64,
                 height=64,
                 *args, **kwargs):
        super().__init__(
            n_blue=n_blue,
            n_red=n_red,
            n_bins=n_bins,
            activity_range=activity_range,
            sensor_range=sensor_range,
            vision=vision,
            width=width,
            height=height,
            *args, **kwargs
        )
