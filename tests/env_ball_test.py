import pytest
import numpy as np

from pytest import approx

from trp_env.envs.ball_trp_env import BallTwoResourceEnv, BallSmallTwoResourceEnv


class TestEnv:

    def test_instance(self):
        env = BallSmallTwoResourceEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = BallSmallTwoResourceEnv()
        env.reset()

    def test_run_env(self):
        env = BallSmallTwoResourceEnv(show_sensor_range=True, random_position=True)
        for _ in range(5):
            env.reset()
            for i in range(10):
                action = env.action_space.sample()
                obs, _, _, _, _ = env.step(action)
                # print(obs)
                env.render()
        env.close()

    def test_dim(self):
        env = BallTwoResourceEnv()
        obs, _ = env.reset()
        
        print(env.wrapped_env.get_current_obs(), env.wrapped_env.get_current_obs().shape)

        assert len(env.observation_space.high) == 6 + 10 + 10 + 2
        assert len(env.action_space.high) == 2
        assert len(obs) == 6 + 10 + 10 + 2
        assert len(env.action_space.sample()) == 2

    def test_dim_small(self):
        env = BallSmallTwoResourceEnv()
        obs, _ = env.reset()

        assert len(env.observation_space.high) == 6 + 20 + 20 + 2
        assert len(env.action_space.high) == 2
        assert len(obs) == 6 + 20 + 20 + 2
        assert len(env.action_space.sample()) == 2
