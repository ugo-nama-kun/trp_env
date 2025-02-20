import pytest
import numpy as np

from pytest import approx

from trp_env.envs.two_resource_env import FoodClass
from trp_env.envs.wheel_trp_env import WheelTwoResourceEnv, WheelSmallTwoResourceEnv


class TestEnv:

    def test_instance(self):
        env = WheelSmallTwoResourceEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = WheelSmallTwoResourceEnv()
        env.reset()

    def test_run_env(self):
        env = WheelSmallTwoResourceEnv(show_sensor_range=True)
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())
            env.render()
        env.close()

    def test_dim(self):
        env = WheelTwoResourceEnv()
        obs, _ = env.reset()
        
        print(env.wrapped_env.get_current_obs(), env.wrapped_env.get_current_obs().shape)

        assert len(env.observation_space.high) == 15 + 10 + 10 + 2
        assert len(env.action_space.high) == 2
        assert len(obs) == 15 + 10 + 10 + 2
        assert len(env.action_space.sample()) == 2

    def test_dim_small(self):
        env = WheelSmallTwoResourceEnv()
        obs, _ = env.reset()

        assert len(env.observation_space.high) == 15 + 20 + 20 + 2
        assert len(env.action_space.high) == 2
        assert len(obs) == 15 + 20 + 20 + 2
        assert len(env.action_space.sample()) == 2
