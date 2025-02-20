from gymnasium.envs.registration import register

register(
    id='AntTRP-v1',
    entry_point='trp_env.envs:AntTwoResourceEnv',
)

register(
    id='SmallAntTRP-v1',
    entry_point='trp_env.envs:AntSmallTwoResourceEnv',
)

register(
    id='SensorAntTRP-v1',
    entry_point='trp_env.envs:SensorAntTwoResourceEnv',
)

register(
    id='SmallSensorAntTRP-v1',
    entry_point='trp_env.envs:SensorAntSmallTwoResourceEnv',
)

register(
    id='LowGearAntTRP-v1',
    entry_point='trp_env.envs:LowGearAntTwoResourceEnv',
)

register(
    id='SmallLowGearAntTRP-v1',
    entry_point='trp_env.envs:LowGearAntSmallTwoResourceEnv',
)

register(
    id='SnakeTRP-v1',
    entry_point='trp_env.envs:SnakeTwoResourceEnv',
)

register(
    id='SmallSnakeTRP-v1',
    entry_point='trp_env.envs:SnakeSmallTwoResourceEnv',
)

register(
    id='SwimmerTRP-v1',
    entry_point='trp_env.envs:SwimmerTwoResourceEnv',
)

register(
    id='SmallSwimmerTRP-v1',
    entry_point='trp_env.envs:SwimmerSmallTwoResourceEnv',
)

register(
    id='HumanoidTRP-v1',
    entry_point='trp_env.envs:HumanoidTwoResourceEnv',
)

register(
    id='SmallHumanoidTRP-v1',
    entry_point='trp_env.envs:HumanoidSmallTwoResourceEnv',
)

register(
    id='RealAntTRP-v1',
    entry_point='trp_env.envs:RealAntTwoResourceEnv',
)

register(
    id='SmallRealAntTRP-v1',
    entry_point='trp_env.envs:RealAntSmallTwoResourceEnv',
)

register(
    id='WheelTRP-v1',
    entry_point='trp_env.envs:WheelTwoResourceEnv',
)

register(
    id='SmallWheelTRP-v1',
    entry_point='trp_env.envs:WheelSmallTwoResourceEnv',
)

register(
    id='BallTRP-v1',
    entry_point='trp_env.envs:BallTwoResourceEnv',
)

register(
    id='SmallBallTRP-v1',
    entry_point='trp_env.envs:BallSmallTwoResourceEnv',
)
