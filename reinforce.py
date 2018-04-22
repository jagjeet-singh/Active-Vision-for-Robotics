import sys
import argparse
import numpy as np
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from termcolor import cprint
import pdb
import math
from logger import Logger
import shutil
import os
import kuka

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

class Reinforce(nn.Module):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, nS, nA):
        super(Reinforce, self).__init__()
        self.linear1 = nn.Linear(nS, nS*2)
        self.linear2 = nn.Linear(nS*2,nS*2)
        self.linear3 = nn.Linear(nS*2,nS*2)
        self.linear4 = nn.Linear(nS*2,nA)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))        
        x = F.softmax(self.linear4(x), dim=1)
        return x

def generate_episode(env, model, render=False, sample=True):
    # Generates an episode by executing the current policy in the given env.
    # Returns:
    # - a list of states, indexed by time step
    # - a list of actions, indexed by time step
    # - a list of rewards, indexed by time step
    # TODO: Implement this method.
    done = False
    nS = env.observation_space.shape[0]
    nA = env.action_space.n
    states = np.zeros((0,nS))
    # actions = torch.zeros((0, nA))
    actions = np.zeros((0))
    rewards = np.zeros((0))
    state = env.reset()
    # state = Variable(torch.from_numpy(env.reset()).type(torch.FloatTensor)).view(-1,nS)
    while not done:
        if render:
            env.render()
        state = np.reshape(state, (1,nS))
        action_softmax = model(Variable(torch.from_numpy(state).type(FloatTensor)))
        if sample:
	        action = np.random.choice(nA, 1, p=action_softmax.data.numpy().reshape(nA,))[0]
        else:
        	action = np.argmax(action_softmax.data.numpy())
        next_state, reward, done, _ = env.step(action)
        states = np.append(states, state, axis=0)
        # action = torch.eye(nA)[action].view(1,nA)            
        actions = np.append(actions, action)
        rewards = np.append(rewards,reward)
        state = np.copy(next_state)
    return states, actions, rewards

def save_checkpoint(state, is_best, env):
    filename='saved_models/'+env+'_reinforce_checkpoint.pth.tar'
    bestFilename = 'saved_models/'+env+'_reinforce_model_best.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)

def load_checkpoint(checkpoint_file, model, optimizer):
    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        start_episode = checkpoint['epoch']
        best_reward = checkpoint['best_reward']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (episode {})"
              .format(checkpoint_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_file))

    return start_episode, best_reward

def gradient_clipping(model, clip):
	#Clip the gradient
	if clip is None:
		return
	totalnorm = 0
	for p in model.parameters():
		if p.grad is None:
			continue
		p.grad.data.clamp_(-clip,clip)
		if np.isnan(np.min(p.grad.data.numpy())) or np.isnan(np.max(p.grad.data.numpy())):
			print("#### NaN Encountered ####")
			pdb.set_trace()

def evaluate(env, model, num_episodes = 100, render=False, sample=False):

	total_rewards = []
	for i in range(num_episodes):
		_,_,rewards = generate_episode(env, model, render, sample)
		total_rewards.append(np.sum(rewards))
	return total_rewards


def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser()
	parser.add_argument('--model-config-path', dest='model_config_path',
	                    type=str, default='LunarLander-v2-config.json',
	                    help="Path to the model config file.")
	parser.add_argument('--end-episode', dest='end_episode', type=int,
	                    default=50000, help="Number of episodes to train on.")
	parser.add_argument('--start-episode', dest='start_episode', type=int,
	                default=0, help="Starting episode")
	parser.add_argument('--lr', dest='lr', type=float,
	                    default=5e-4, help="The learning rate.")
	parser.add_argument('--gamma', dest='gamma', 
	                    type=float,default=0.99, help='Discount factor')
	# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	parser_group = parser.add_mutually_exclusive_group(required=False)
	parser_group.add_argument('--render', dest='render',
	                          action='store_true',
	                          help="Whether to render the environment.")
	parser_group.add_argument('--no-render', dest='render',
	                          action='store_false',
	                          help="Whether to render the environment.")
	parser.set_defaults(render=False)
	parser.add_argument('--tb-logdir', default='reinforce', 
		help='Name of Tensorboard log directory')
	parser.add_argument('--eval-freq', default=1000, help='Frequency of evaluation')
	parser.add_argument('--eval-episodes', default=100, help='Number of episodes to evaluate on')
	parser.add_argument('--train-plot-freq', default=10, type=int)
	parser.add_argument('--log-freq', default=10, type=int)
	parser.add_argument('--env', default='LunarLander-v2', help='Name of the gym environment')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')

	return parser.parse_args()


def main(args):
    # Parse command-line arguments.
	args = parse_arguments()
	render = args.render

	# Create the environment.
	env = gym.make(args.env)
	nS = env.observation_space.shape[0]
	nA = env.action_space.n
	# Declare the model
	model = Reinforce(nS, nA)
	# criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	logger = Logger('hw3_logs', name=args.env+'_'+args.tb_logdir)

	if args.resume:
		start_episode, best_reward = load_checkpoint(args.resume, model, optimizer)
	else:
		start_episode = args.start_episode
		best_reward = -np.inf
	
	# Start with policy model 
	for i in range(start_episode, args.end_episode):
		is_best = 0
		# Generate the episode
		states, actions, rewards = generate_episode(env, model, render=False)
		print(actions.shape[0])
		states = Variable(torch.from_numpy(states).type(FloatTensor), requires_grad=False).view(-1,nS)
		actions = Variable(torch.from_numpy(actions).type(LongTensor), requires_grad=False).view(-1,1)
		episode_length = states.shape[0]

		# Array to store returns from each time step
		G = torch.zeros(episode_length,1)

		# First calculating return and loss for the last step of episode
		G[episode_length-1] = rewards[episode_length-1]/100.0

		for step in range(episode_length-2,-1,-1):
		    G[step]=rewards[step]/100.0 + args.gamma*G[step+1]
		    
		# Calculating log of policy for the entire episode
		output_var = model(states).log() # log of Probabilities
		logPolicy = output_var.gather(1,actions)
		G = Variable(G, requires_grad=False)
		loss = -torch.sum(logPolicy*G)/episode_length

		# Backprop
		optimizer.zero_grad()
		loss.backward()
		# gradient_clipping(model, 0.5)
		clip_grad.clip_grad_norm(model.parameters(), 0.5)
		optimizer.step()


		#Plotting train results on Tensorboard
		if i%args.train_plot_freq == 0:
		    logger.scalar_summary(tag='Train/Rewards', value=np.sum(rewards), step=i)
		    logger.scalar_summary(tag='Train/Loss', value=loss, step=i)
		    logger.model_param_histo_summary(model=model, step=i)

		# Evaluating and plotting eval results
		# if i%args.eval_freq==0:
		# 	eval_rewards = evaluate(env, model, num_episodes = args.eval_episodes)
		# 	mean = np.mean(eval_rewards)
		# 	std= np.std(eval_rewards)
		    
		# 	if mean > best_reward:
		# 		is_best = 1

		# 	# Save checkpoint
		# 	save_checkpoint({
		# 		'epoch': i + 1,
		# 		'state_dict': model.state_dict(),
		# 		'best_reward': best_reward,
		# 		'optimizer' : optimizer.state_dict(),
		# 	}, is_best, args.env)

		# 	# Plot on tensorboard
		# 	logger.scalar_summary(tag='Test/Mean Reward', value=mean, step=i)
		# 	logger.scalar_summary(tag='Test/Std', value=std, step=i)

		# 	# Print results
		# 	if mean > 150:
		# 		cprint('Evaluate - Episode:{}, Mean:{}, Std:{}'.format(i, mean, std), color='green')
		# 	else:
		# 		print('Evaluate - Episode:{}, Mean:{}, Std:{}'.format(i, mean, std))

		# Printing train results
		if i%args.log_freq==0:
			if np.sum(rewards) > 150:
				cprint('Train - Episode:{}, Reward:{}, Loss:{}'.format(i, np.sum(rewards), loss.data[0]), color='green')
			else:
				print('Train - Episode:{}, Reward:{}, Loss:{}'.format(i, np.sum(rewards), loss.data[0]))


if __name__ == '__main__':
    main(sys.argv)
