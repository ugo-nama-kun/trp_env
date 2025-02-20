from trp_env.envs.humanoid_trp_env import HumanoidSmallTwoResourceEnv, HumanoidTwoResourceEnv


class TestEnv:

    def test_instance(self):
        env = HumanoidTwoResourceEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = HumanoidTwoResourceEnv()
        env.reset()

    def test_run_env(self):
        env = HumanoidTwoResourceEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_dim(self):
        env = HumanoidTwoResourceEnv()
        obs, _ = env.reset()

        assert len(env.wrapped_env.data.sensordata) == 66

        assert len(env.observation_space.high) == 66 + 10 + 10 + 2
        assert len(env.action_space.high) == 21
        assert len(obs) == 66 + 10 + 10 + 2
        assert len(env.action_space.sample()) == 21

    def test_dim_small(self):
        env = HumanoidSmallTwoResourceEnv()
        obs, _ = env.reset()

        assert len(env.observation_space.high) == 66 + 20 + 20 + 2
        assert len(env.action_space.high) == 21
        assert len(obs) == 66 + 20 + 20 + 2
        assert len(env.action_space.sample()) == 21

    def test_render_env(self):
        env = HumanoidSmallTwoResourceEnv(show_sensor_range=True,
                                          n_bins=20,
                                          sensor_range=16.)
        for n in range(5):
            env.reset()
            for i in range(10):
                env.step(env.action_space.sample())
                env.render()
        env.close()

    def test_rgbd_render_and_capture(self):
        env = HumanoidSmallTwoResourceEnv(vision=True)
        env.reset()

        im = None
        for i in range(10):
            env.step(env.action_space.sample())
            env.render()
            im = env.render(mode='rgbd_array', camera_id=0)

        import matplotlib.pyplot as plt
        plt.subplot(2, 2, 1)
        plt.imshow(im[:, :, :3] / 255.)
        plt.subplot(2, 2, 2)
        plt.hist(im[:, :, :3].flatten())
        plt.subplot(2, 2, 3)
        plt.imshow(im[:, :, 3], "gray")
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.hist(im[:, :, 3].flatten())
        plt.savefig("test_rgbd_render_humanoid.png")

        assert im.shape == (64, 64, 4)
        assert im[:, :, :3].shape == (64, 64, 3)

        env.close()
