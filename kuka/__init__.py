from gym.envs.registration import register

register(
	id='kuka-v0',
	entry_point='kuka.envs:KukaEnv')