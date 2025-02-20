from setuptools import setup

setup(name='trp_env',
      version='8.0.1',
      install_requires=['gymnasium==0.27.1', 'mujoco>=2.3.1.post1', "torch", "scipy"],
      package_data = {'trp_env': ['envs/models/*.xml', 'envs/models/texture/*.png']},
)
