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
	'video.frames_per_second' : 100
	}
	def __init__(self):

		# Action map to make action space discrete
		delta = 0.3
		self.action_map={
		0:[0.0,0.0,0.0],
		1:[0.0,0.0,delta],
		2:[0.0,0.0,-delta],
		3:[0.0,delta,0.0],
		4:[0.0,delta,delta],
		5:[0.0,delta,-delta],
		6:[0.0,-delta,0.0],
		7:[0.0,-delta,delta],
		8:[0.0,-delta,-delta],
		9:[delta,0.0,0.0],
		10:[delta,0.0,delta],
		11:[delta,0.0,-delta],
		12:[delta,delta,0.0],
		13:[delta,delta,delta],
		14:[delta,delta,-delta],
		15:[delta,-delta,0.0],
		16:[delta,-delta,delta],
		17:[delta,-delta,-delta],
		18:[-delta,0.0,0.0],
		19:[-delta,0.0,delta],
		20:[-delta,0.0,-delta],
		21:[-delta,delta,0.0],
		22:[-delta,delta,delta],
		23:[-delta,delta,-delta],
		24:[-delta,-delta,0.0],
		25:[-delta,-delta,delta],
		26:[-delta,-delta,-delta]
		}

		# Setting up env variables
		self.num_envs = 4
		# self.numJoints = p.getNumJoints(self.botId, self.physicsClient)
		self.numJoints= 7
		self.endEffector = [0.0,0.0,0.0]
		# 1 value for angular velocity of each joint
		self.action_space = spaces.Discrete(27)
		# get sample image from random initialization of joint angles
		self.observation_space = spaces.Box(low=0., high=1., shape=(4,))
		
		# Setting up pybullet 
		p.connect(p.DIRECT)
		# p.connect(p.GUI)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

		fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
		self.projection_matrix = p.computeProjectionMatrixFOV(
			fov, aspect, nearplane, farplane)
		self.endEffector = [0.0,0.0,0.0]
		self._seed()
	
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

	def _step(self, action):
		displacement = self.action_map[action]
		self.endEffector = [sum(x) for x in zip(self.get_current_endEffector(), displacement)]
		# self.endEffector = [sum(x) for x in zip(self.endEffector, displacement)]
		self._assign_throttle(self.endEffector)
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

	def _assign_throttle(self, endEffector):
		# Action is the change in angular velocity of each joint
		print(endEffector)
		jointPoses = p.calculateInverseKinematics(self.botId,6,endEffector)
		for i in range(self.numJoints):
			p.setJointMotorControl2(bodyIndex=self.botId,
				jointIndex=i,
				controlMode=p.POSITION_CONTROL,
				targetPosition=jointPoses[i],
				targetVelocity=0,
				force=500,
				positionGain=0.03,
				velocityGain=1)
			
	def get_current_endEffector(self):
		com_p, com_o, _, _, _, _ = p.getLinkState(self.botId, 6, computeForwardKinematics=True)
		rot_matrix = p.getMatrixFromQuaternion(com_o)
		rot_matrix = np.array(rot_matrix).reshape(3, 3)
		# Initial vectors
		init_camera_vector = (0, 0, 1) # z-axis
		# Rotated vectors
		camera_vector = rot_matrix.dot(init_camera_vector)
		return com_p+0.1*camera_vector

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
		seg = (curr_img[4] == self.teddyId)
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
		# print("Reward:{}".format(reward))
		return reward*10.0 if reward>0.0 else -1

	def _compute_done(self):
		# return 1 if the intersection is above a particular threshold
		return self._envStepCounter >= 200 

	def _render(self, mode='human', close=False):
		pass