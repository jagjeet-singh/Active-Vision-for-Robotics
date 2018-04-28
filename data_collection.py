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
import cv2
from time import sleep
import random
from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse


CAMERA_IMG_SCALE = 256

parser = argparse.ArgumentParser()
parser.add_argument('--data-type',type=str, default='target',help="Segmentation data (seg) or target (target) data")
args = parser.parse_args()

class KukaEnv():
	metadata = {
	'render.modes': ['human', 'rgb_array'],
	'video.frames_per_second' : 100
	}
	def __init__(self):

		# Setting up env variables
		self.numJoints= 7
		
		# p.connect(p.DIRECT)
		p.connect(p.GUI)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

		fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
		self.projection_matrix = p.computeProjectionMatrixFOV(
			fov, aspect, nearplane, farplane)
		self._seed()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	
	def _init_simulation(self, objects):
		p.setGravity(0, 0, -10)
		p.setTimeStep(1)

		# Add plane
		self.planeId = p.loadURDF("plane.urdf")

		# Add kuka bot
		start_pos = [0, 0, 0.001]
		start_orientation = p.getQuaternionFromEuler([0, 0, 0])
		self.botId = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orientation)



		if args.data_type == 'seg':

			self.objectIDs = []
			self.object_names = []
			self.start_pos_all = []
			# Add objects:
			for ob in objects:
				to_add = np.random.randint(0,2)
				if to_add:
					x = [np.random.uniform(-4.0, -1.0), np.random.uniform(1.0, 4.0)]
					y = [np.random.uniform(-4.0, -1.0), np.random.uniform(1.0, 4.0)]
					z = 0.0 if 'tray' in ob else 0.5
					start_pos = [random.choice(x), random.choice(y), z]
					self.start_pos_all.append(start_pos)
					start_orientation = p.getQuaternionFromEuler([0, 0, 0])
					ob_id = p.loadURDF(ob, start_pos, start_orientation)
					self.objectIDs.append(ob_id)
					self.object_names.append(ob)

			if len(self.objectIDs)==0:
				return 0
			self.target_object_id = random.choice(self.objectIDs)
			self.target_object_name = self.object_names[self.objectIDs.index(self.target_object_id)]
			self.target_object_pos = self.start_pos_all[self.objectIDs.index(self.target_object_id)]
			print('objects present:')
			print(self.object_names)
			print('target object:'+ self.target_object_name)
			print('Target object position:')
			print(self.target_object_pos)
			return 1

		else:

			y = 0.0
			x = 2.0
			z = 0.0
			# x = 1.7 if 'tray' in objects else 1.95
			# z = 0.0 if 'tray' in objects else 0.5
			self.target_object_pos = [x,y,z]
			start_orientation = p.getQuaternionFromEuler([0, 0, 0])
			self.target_object_id = p.loadURDF(objects, self.target_object_pos, start_orientation)
			print('Collecting target images for {}'.format(objects))
			return 1


	def _assign_throttle(self, endEffector):
		# Action is the change in angular velocity of each joint
		# print(endEffector)
		joint_pos = p.calculateInverseKinematics(self.botId,6,endEffector)
		p.setJointMotorControlArray(bodyIndex=self.botId,
			jointIndices=range(self.numJoints),
			controlMode=p.POSITION_CONTROL,
			targetPositions=joint_pos,
			forces=[500] * self.numJoints,
			positionGains=[0.03] * self.numJoints,
			velocityGains=[1] * self.numJoints)

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
			com_p + 0.005 * camera_vector, com_p + 0.1 * camera_vector, up_vector)
		img = p.getCameraImage(
			CAMERA_IMG_SCALE, CAMERA_IMG_SCALE, view_matrix, self.projection_matrix)
		return img


def main():

	objects = ["sphere2.urdf", "tray/traybox.urdf", "sphere2red.urdf"]
	# objects = ["rascecar/racecar.urdf"]
	total=0

	env = KukaEnv()


	if args.data_type == 'target':
		steps_per_run = 100
		obj_dir = 'objects/'
		if not os.path.isdir(obj_dir):
			os.makedirs(obj_dir)
		for ob in objects:
			p.resetSimulation()
			env._init_simulation(ob)
			[x,y,z] = env.target_object_pos
			x = x+(-0.2 if 'tray' in ob else +0.1)
			ob_idx = objects.index(ob)
			best_seg = 0
			for i in range(steps_per_run):
				delta = 0.1
				deltax = np.random.uniform(-delta, delta)
				deltay = np.random.uniform(-delta, delta)
				deltaz = np.random.uniform(-delta, delta)
				endEffector = [x+deltax, y+deltay, z+deltaz]
				print(endEffector)
				env._assign_throttle(endEffector)
				# env._assign_throttle([2.0,0,1])
				p.stepSimulation();
				camera_res = env.get_camera()
				img = camera_res[2][:,:,:-1].astype(np.uint8)
				# img = camera_res[2][:,:,-2::-1]
				seg = (camera_res[4] == env.target_object_id).astype(np.uint8)
				total_seg = np.sum(seg)
				if total_seg > best_seg:
					best_view_idx = i
					best_image = img
					best_seg = total_seg
			ob_name = ob.split('/')[-1].split('.')[0]
			plt.imsave(obj_dir+'/'+ob_name+'.png',best_image)		


	elif args.data_type == 'seg':
		num_of_runs = 50
		steps_per_run = 100
		image_dir = 'images/'
		seg_dir = 'seg/'
		obj_dir = 'objects/'
		target_dir = 'targets/'
		for j in range(num_of_runs):
		
			atLeast_one = 0
			while not atLeast_one:
				p.resetSimulation()
				atLeast_one =  env._init_simulation(objects)

			num_images = 100
			[x,y,z] = env.target_object_pos
			ob_idx = objects.index(env.target_object_name)
			ob_name = env.target_object_name.split('/')[-1].split('.')[0]
			ob_img = Image.open(obj_dir+'/'+ob_name+'.png').convert('RGB')

			for i in range(steps_per_run):
				deltax = np.random.uniform(-1.0, 1.0)
				deltay = np.random.uniform(-1.0, 1.0)
				deltaz = np.random.uniform(-1.0, 1.0)
				endEffector = [x+deltax-0.1, y+deltay, z+deltaz+0.1]
				env._assign_throttle(endEffector)
				# env._assign_throttle([2.0,0,1])
				p.stepSimulation();
				camera_res = env.get_camera()
				img = camera_res[2][:,:,:-1].astype(np.uint8)
				# img = camera_res[2][:,:,-2::-1]
				seg = (camera_res[4] == env.target_object_id).astype(np.uint8)
				# folder = env.target_object_name.split('/')[-1].split('.')[0]
				total_seg = np.sum(seg)
				# image_dir = 'images/'+folder
				# seg_dir = 'seg/'+folder


				if not os.path.isdir(image_dir):
					os.makedirs(image_dir)
				if not os.path.isdir(seg_dir):
					os.makedirs(seg_dir)
				if np.count_nonzero(seg)>0:
					# objects_images[ob_idx]+=1
					if np.count_nonzero(seg)>500:
						pdb.set_trace()
					total+=1
					plt.imsave(image_dir+str(total)+'.png',img)
					plt.imsave(seg_dir+str(total)+'.png',seg, cmap=cm.gray)
					plt.imsave(target_dir+str(total)+'.png', ob_img)
					# plt.imsave(image_dir+'/'+str(objects_images[ob_idx])+'.png',img)
					# plt.imsave(seg_dir+'/'+str(objects_images[ob_idx])+'.png',seg, cmap=cm.gray)
				# imsave(image_dir+'/'+str(j*steps_per_run+i)+'.png',img)
				# imsave(seg_dir+'/'+str(j*steps_per_run+i)+'.png',seg)
				print(endEffector)


if __name__ == '__main__':
    main()