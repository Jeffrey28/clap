from setuptools import setup

setup(name='carla_trainning_dqn',
      version='0.0.3',
      install_requires=['gym'],  # And any other dependencies foo needs
      packages=['gym_routing', 'gym_routing.envs']
)
