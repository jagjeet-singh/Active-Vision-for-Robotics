import os
import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import pybullet_data
from PIL import Image

CAMERA_IMG_SCALE = 1000

class KukaEnv(gym.Env):
	metadata = {
	'render.modes': ['human', 'rgb_array'],
	'video.frames_per_second' : 100
	}
	def __init__(self):
		# Setting up env variables
		self.num_envs = 4
		# self.numJoints = p.getNumJoints(self.botId, self.physicsClient)
		self.numJoints= 7

		self.action_map = None

		# Action and observation spaces
		self.action_space = spaces.Discrete(6)
		self.observation_space = spaces.Box(low=0., high=1., shape=(4,))

		# State quantities that are updated after evey simulation step
		self.end_effector_pos = None
		self.rot_matrix = None
		self.curr_camera_img = None

		# self.projection_matrix will always be a fixed quantity
		fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
		self.projection_matrix = p.computeProjectionMatrixFOV(
			fov, aspect, nearplane, farplane)

		self._seed()
	
	def init_bullet(self, render=False, delta=0.3):
		# Setting up pybullet
		self.render = render
		p.connect(p.GUI if self.render else p.DIRECT)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

		# Create action map
		list_actions = []
		for i in range(3):
			for j in [-delta, delta]:
				action = np.zeros(3)
				action[i] = j
				list_actions.append(action)
		self.action_map = dict(zip(range(6), list_actions))

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _init_simulation(self):
		p.setGravity(0, 0, -10)
		p.setTimeStep(1)

		# Add plane
		self.planeId = p.loadURDF("plane.urdf")

		# Add kuka bot
		start_pos = [0, 0, 0.001]
		start_orientation = p.getQuaternionFromEuler([0, 0, 0])
		self.botId = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation)

		# Add table
		# start_pos = [1, 0, 0.001]
		# start_orientation = p.getQuaternionFromEuler([0, 0, 0])
		# self.tableId = p.loadURDF("table/table.urdf", start_pos, start_orientation)

		# Add object
		start_pos = [3.0, 0.0, 0.5]
		start_orientation = p.getQuaternionFromEuler([0, 0, 0])
		self.teddyId = p.loadURDF("sphere2.urdf", start_pos, start_orientation)
		# self.teddyId = p.loadURDF("teddy_vhacd.urdf", start_pos, start_orientation)

		self._update_state_quantities()

	def _update_state_quantities(self):
		com_p, com_o, _, _, _, _ = p.getLinkState(
			self.botId, 6, computeForwardKinematics=True)

		# Update end effector position
		self.end_effector_pos = com_p

		# Update rotation matrix of end effector's frame
		# from beginning of episode to now.
		rot_matrix = p.getMatrixFromQuaternion(com_o)
		rot_matrix = np.array(rot_matrix).reshape(3, 3)
		self.rot_matrix = rot_matrix

		# Update current camera image.
		self.curr_camera_img = self._get_curr_camera_img()

	def _get_curr_camera_img(self):
		# Initial vectors
		init_camera_vector = (0, 0, 1) # z-axis
		init_up_vector = (0, 1, 0) # y-axis
		# Rotated vectors
		camera_vector = self.rot_matrix.dot(init_camera_vector)
		up_vector = self.rot_matrix.dot(init_up_vector)
		view_matrix = p.computeViewMatrix(
			self.end_effector_pos, self.end_effector_pos + 0.1 * camera_vector,
			up_vector)
		img = p.getCameraImage(
			CAMERA_IMG_SCALE, CAMERA_IMG_SCALE, view_matrix, self.projection_matrix)
		return img

	def _step_simulation(self):
		"""Step simulation in pybullet and update state quantities"""
		p.stepSimulation() # step simulation in pybullet
		self._update_state_quantities() # update state quantities

	def _step(self, action):
		ee_frame_disp = self.action_map[action]
		world_frame_disp = self.rot_matrix.dot(ee_frame_disp)
		ee_target_pos = self.end_effector_pos + world_frame_disp
		self._assign_throttle(ee_target_pos)
		self._step_simulation()
		self._gt_bbox = self._compute_observation()
		reward = self._compute_reward(print_reward=self.render)
		done = self._compute_done()
		self._envStepCounter += 1
		return self._gt_bbox, reward, done, {}

	def _reset(self):
		self._envStepCounter = 0

		p.resetSimulation()
		self._init_simulation()

		self._gt_bbox = self._compute_observation()
		return self._gt_bbox

	def _assign_throttle(self, ee_target_pos):
		# Calculate joint positions using inverse kinematics
		joint_pos = p.calculateInverseKinematics(self.botId, 6, ee_target_pos)
		p.setJointMotorControlArray(bodyIndex=self.botId,
			jointIndices=range(self.numJoints),
			controlMode=p.POSITION_CONTROL,
			targetPositions=joint_pos,
			forces=[500] * self.numJoints,
			positionGains=[0.03] * self.numJoints,
			velocityGains=[1] * self.numJoints)

	def _compute_observation(self):
		# Reshape curr_camera_img to (CAMERA_IMG_SCALE ,CAMERA_IMG_SCALE, 3)
		seg = (self.curr_camera_img[4] == self.teddyId)
		x_indices = np.where(np.max(seg, axis=0))[0]
		y_indices = np.where(np.max(seg, axis=1))[0]
		# If object occurs in the current image
		if x_indices.size > 0:
			x_min = np.min(x_indices) / CAMERA_IMG_SCALE
			x_max = np.max(x_indices) / CAMERA_IMG_SCALE
			y_min = np.min(y_indices) / CAMERA_IMG_SCALE
			y_max = np.max(y_indices) / CAMERA_IMG_SCALE
		else:
		 	x_min, x_max, y_min, y_max = 0., 0., 0., 0.
		return [x_min, y_min, x_max, y_max]

	def _compute_reward(self, print_reward=True):
		# Get the similarity between the current image is the target image
		bbox = self._gt_bbox
		reward = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
		if print_reward: print("reward: {}".format(reward))
		return reward

	def _compute_done(self):
		# return 1 if the intersection is above a particular threshold
		return self._envStepCounter >= 200

	def _render(self, mode='human', close=False):
		pass
