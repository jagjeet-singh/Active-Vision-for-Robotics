import os
import math
import numpy as np
import pdb
import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import pybullet_data
from PIL import Image
import pickle

CAMERA_IMG_SCALE = 1000

class KukaEnv(gym.Env):
	metadata = {
	'render.modes': ['human', 'rgb_array'],
	'video.frames_per_second' : 50
	}
	def __init__(self):
		# Setting up env variables
		self.num_envs = 4
		# self.numJoints = p.getNumJoints(self.botId, self.physicsClient)
		self.numJoints= 7
		# 1 value for angular velocity of each joint
		self.action_space = spaces.Box(low=-1., high=1., shape=(self.numJoints,))
		# get sample image from random initialization of joint angles
		self.observation_space = spaces.Box(low=0., high=1., shape=(4,))
		
		# Setting up pybullet 
		p.connect(p.GUI)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

		fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
		self.projection_matrix = p.computeProjectionMatrixFOV(
			fov, aspect, nearplane, farplane)

		self._seed()
	
	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _init_simulation(self):
		p.setGravity(0, 0, -10)
		p.setTimeStep(0.01)

		# Add plane
		self.planeId = p.loadURDF("plane.urdf")

		# Add kuka bot
		start_pos = [0, 0, 0.001]
		start_orientation = p.getQuaternionFromEuler([0, 0, 0])
		self.botId = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation)

		# Add table 
		start_pos = [1, 0, 0.001]
		start_orientation = p.getQuaternionFromEuler([0, 0, 0])
		self.tableId = p.loadURDF("table/table.urdf", start_pos, start_orientation)

		# Add object 
		start_pos = [0.5, 0, 1.001]
		start_orientation = p.getQuaternionFromEuler([0, 0, 0])
		self.teddyId = p.loadURDF("teddy_vhacd.urdf", start_pos, start_orientation)

	def _step(self, action):
		self._assign_throttle(action)
		p.stepSimulation(); self.get_camera()
		self._observation = self._compute_observation()
		reward = self._compute_reward()
		done = self._compute_done()
		self._envStepCounter += 1
		return self._observation, reward, done, {}

	def _reset(self):
		self.vt = np.zeros(self.numJoints)
		self._envStepCounter = 0

		p.resetSimulation()
		self._init_simulation()
		
		self._observation = self._compute_observation()
		return self._observation

	def _assign_throttle(self, action):
		# Action is the change in angular velocity of each joint
		for i in range(3, self.numJoints):
			self.vt[i] = self.vt[i] + action[i]
			p.setJointMotorControl2(bodyUniqueId=self.botId,
				jointIndex=i,
				controlMode=p.VELOCITY_CONTROL,
				targetVelocity=self.vt[i])

	def get_camera(self):
		# Center of mass position and orientation (of link-7)
		com_p, com_o, _, _, _, _ = p.getLinkState(
			self.botId, 6, computeForwardKinematics=True)
		rot_matrix = p.getMatrixFromQuaternion(com_o)
		rot_matrix = np.array(rot_matrix).reshape(3, 3)
		# Initial vectors
		init_camera_vector = (0, 0, 1) # z-axis
		init_up_vector = (0, 1, 0) # y-axis
		# Rotated vectors
		camera_vector = rot_matrix.dot(init_camera_vector)
		up_vector = rot_matrix.dot(init_up_vector)
		view_matrix = p.computeViewMatrix(
			com_p, com_p + 0.1 * camera_vector, up_vector)
		img = p.getCameraImage(
			CAMERA_IMG_SCALE, CAMERA_IMG_SCALE, view_matrix, self.projection_matrix)
		return img

	def _compute_observation(self):
		# get Camera image as per current link state and reshape to
		# (CAMERA_IMG_SCALE ,CAMERA_IMG_SCALE, 3)
		curr_img = self.get_camera()
		seg = (curr_img[4] == 3)
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

	def _compute_reward(self):
		# Get the similarity between the current image is the target image 
		bbox = self._compute_observation()
		reward = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
		print(reward)
		return reward 

	def _compute_done(self):
		# return 1 if the intersection is above a particular threshold
		return self._envStepCounter >= 1500 

	def _render(self, mode='human', close=False):
		pass