import gymnasium
import trp_env

env = gymnasium.make("SmallLowGearAntTRP-v1", on_texture=True)

env.reset()

done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    
    env.render()
