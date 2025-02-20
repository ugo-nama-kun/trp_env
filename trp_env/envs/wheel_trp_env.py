import math
from typing import Optional

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from trp_env.envs.two_resource_env import TwoResourceEnv, AgentType

# Code is mostly based on the original paper and provided code
# https://github.com/alexlioralexli/rllab-finetuning/blob/2dae9141d0fdc284d04f18931907131d66b43023/sandbox/finetuning/envs/mujoco/ant_env.py


DEFAULT_CAMERA_CONFIG = {
    "distance": 20.0,
}


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


class WheelEnv(MujocoEnv, utils.EzPickle):
    FILE = "wheel.xml"
    ORI_IND = 3
    TYPE = AgentType.WALKER
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps"  : 20,
    }
    
    def __init__(self,
                 xml_path,
                 ctrl_cost_coeff=1e-2,  # gym has 1 here!
                 rew_speed=False,  # if True the dot product is taken with the speed instead of the position
                 rew_dir=None,  # (x,y,z) -> Rew=dot product of the CoM SPEED with this dir. Otherwise, DIST to 0
                 ego_obs=True,
                 sparse=False,
                 vision=False,
                 width=64,
                 height=64,
                 *args, **kwargs):
        
        utils.EzPickle.__init__(
            self,
            xml_path,
            ctrl_cost_coeff,  # gym has 1 here!
            rew_speed,  # if True the dot product is taken with the speed instead of the position
            rew_dir,  # (x,y,z) -> Rew=dot product of the CoM SPEED with this dir. Otherwise, DIST to 0
            ego_obs,
            sparse,
            vision,
            width,
            height,
            **kwargs
        )
        
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.reward_dir = rew_dir
        self.rew_speed = rew_speed
        self.ego_obs = ego_obs
        self.sparse = sparse
        
        frame_skip = 5
        obs_shape = 15
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
    
    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
    
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.data.qpos.flat.copy()[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori
    
    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        if self.rew_speed:
            direction_com = self.get_body_com('torso')
        else:
            direction_com = self.get_body_com('torso')
        if self.reward_dir:
            direction = np.array(self.reward_dir, dtype=float) / np.linalg.norm(self.reward_dir)
            forward_reward = np.dot(direction, direction_com)
        else:
            forward_reward = np.linalg.norm(
                direction_com[0:-1])  # instead of comvel[0] (does this give jumping reward??)
        lb, ub = self.action_space.low, self.action_space.high
        scaling = (ub - lb) * 0.5
        ctrl_cost = .5 * self.ctrl_cost_coeff * np.square(a / scaling).sum()
        
        if self.sparse:  # strip the forward reward, but keep the other costs/rewards!
            if np.linalg.norm(self.get_body_com("torso")[0:2]) > np.inf:  # potentially could specify some distance
                forward_reward = 1.0
            else:
                forward_reward = 0.
        reward = forward_reward - ctrl_cost
        
        state = self.state_vector()
        # notdone = np.isfinite(state).all() \
        #     and state[2] >= 0.3 and state[2] <= 1.0  # Different from Gym Ant 0.2 --> 0.3
        notdone = np.isfinite(state).all()  # Agent dies only if the agent broken
        terminated = not notdone
        truncated = False
        ob = self.get_current_obs()
        com = np.concatenate([self.get_body_com("torso").flat]).reshape(-1)
        ori = self.get_ori()
        return ob, reward, terminated, truncated, dict(
            com=com,
            ori=ori,
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost)
    
    def get_current_obs(self):
        return np.concatenate([
            self.data.qpos.flat.copy()[2:],  # qpos[:1] is (x,y) position of the agent
            self.data.qvel.flat.copy(),
        ]).reshape(-1)
    
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.standard_normal(size=self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self.get_current_obs()


class WheelTwoResourceEnv(TwoResourceEnv):
    """
    Two Resource Problem with approximated scale of Konidaris & Barto paper
    """
    MODEL_CLASS = WheelEnv
    ORI_IND = 3
    
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


class WheelSmallTwoResourceEnv(TwoResourceEnv):
    """
    Small-sized Two Resource Problem with the scale of original GatherEnv
    """
    MODEL_CLASS = WheelEnv
    ORI_IND = 3
    
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
