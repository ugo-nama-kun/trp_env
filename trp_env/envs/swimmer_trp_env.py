import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from trp_env.envs.two_resource_env import TwoResourceEnv, AgentType

DEFAULT_CAMERA_CONFIG = {
    "distance": 20.0,
}


class MySwimmerEnv(MujocoEnv, utils.EzPickle):
    FILE = "swimmer.xml"
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
                 ctrl_cost_coeff=1e-4,
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
        obs_shape = 8 + 10 + 10 + 2
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

    def step(self, a):
        xposbefore = self.data.qpos.flat.copy()[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.data.qpos.flat.copy()[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - self.ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self.get_current_obs()
        return ob, reward, False, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def get_current_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self.get_current_obs()


class SwimmerTwoResourceEnv(TwoResourceEnv):
    """
    Two Resource Problem with approximated scale of Konidaris & Barto paper
    """
    MODEL_CLASS = MySwimmerEnv
    ORI_IND = 2
    
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


class SwimmerSmallTwoResourceEnv(TwoResourceEnv):
    """
    Small-sized Two Resource Problem with the scale of original GatherEnv
    """
    MODEL_CLASS = MySwimmerEnv
    ORI_IND = 2

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
