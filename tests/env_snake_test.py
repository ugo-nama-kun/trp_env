from trp_env.envs.snake_trp_env import SnakeTwoResourceEnv, SnakeSmallTwoResourceEnv


class TestEnv:

    def test_instance(self):
        env = SnakeTwoResourceEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = SnakeTwoResourceEnv()
        env.reset()

    def test_run_env(self):
        env = SnakeSmallTwoResourceEnv(n_blue=12, n_red=12)
        env.reset()
        for i in range(5):
            env.step(env.action_space.sample())
            env.render()

    def test_dim(self):
        env = SnakeTwoResourceEnv()
        obs, _ = env.reset()

        assert len(env.observation_space.high) == 12 + 10 + 10 + 2  # 17 + 10 + 10 if non-ego-centric observation
        assert len(env.action_space.high) == 4
        assert len(obs) == 12 + 10 + 10 + 2 # 17 + 10 + 10 if non-ego-centric observation
        assert len(env.action_space.sample()) == 4
