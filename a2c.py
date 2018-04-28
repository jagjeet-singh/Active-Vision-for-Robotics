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
import time

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',type=str, default='LunarLander-v2-config.json',help="Path to the model config file.")
    parser.add_argument('--end-episode', dest='end_episode', type=int, default=50000, help="Number of episodes to train on.")
    parser.add_argument('--start-episode', dest='start_episode', type=int,default=0, help="Starting episode")
    parser.add_argument('--actor-lr', dest='actor_lr', type=float,default=5e-4, help="Actor learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,default=1e-4, help="Critic learning rate.")
    parser.add_argument('--gamma', dest='gamma',type=float,default=0.99, help='Discount factor')
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',action='store_true',help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',action='store_false',help="Whether to render the environment.")
    parser.set_defaults(render=False)
    parser.add_argument('--tb-logdir', default='reinforce', help='Name of Tensorboard log directory')
    parser.add_argument('--eval-freq', default=500, help='Frequency of evaluation')
    parser.add_argument('--eval-episodes', default=100, help='Number of episodes to evaluate on')
    parser.add_argument('--train-plot-freq', type=int, default=10)
    parser.add_argument('--log-freq', type=int, default=10)
    parser.add_argument('--env', default='LunarLander-v2', help='Name of the gym environment')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    parser.add_argument('--N', default=100, type=int, help='N for N-step return')
    parser.add_argument('--critic-overtrained', dest='critic_overtrained', action='store_true', help='Overtrain Critic')
    parser.add_argument('--overtrain-episodes',default=1000, type=int, help='Number episodes for overtrainig Critic')
    parser.add_argument('--overtrain-freq',default=10, type=int, help='Frequency of overtraining')
    parser.add_argument('--a2c-variant',default='', type=str, help='Variant of A2C used')
    parser.add_argument('--as-baseline', action='store_true', help='Following gym baseline implementation')
    parser.add_argument('--value-loss-coef', type=float,default=0.5, help='Coefficient of value loss in the total loss')
    parser.add_argument('--entropy-coef', type=float,default=0.1, help='Coefficient of entropy in the total loss')
    parser.add_argument('--use-entropy', action='store_true', help='Use entropy in the loss function?')

    return parser.parse_args()


class Actor(nn.Module):

    def __init__(self, nS, nA):
        super(Actor, self).__init__()
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

class Critic(nn.Module):

    def __init__(self, nS, nA):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(nS, nS*2)
        self.linear2 = nn.Linear(nS*2,nS*4)
        self.linear3 = nn.Linear(nS*4,nS*4)
        self.linear4 = nn.Linear(nS*4,nS*2)
        self.linear5 = nn.Linear(nS*2,1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))        
        x = self.linear5(x)
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
    print("## Generating Episode ##")
    start_time = time.time()
    step = 0
    # state = Variable(torch.from_numpy(env.reset()).type(torch.FloatTensor)).view(-1,nS)
    while not done:
        if render:
            env.render()
        state = np.reshape(state, (1,nS))
        action_softmax = model(Variable(torch.from_numpy(state).type(FloatTensor)))
        if sample:
            if use_cuda:
                action = np.random.choice(nA, 1, p=action_softmax.data.cpu().numpy().reshape(nA,))[0]
            else:
                action = np.random.choice(nA, 1, p=action_softmax.data.numpy().reshape(nA,))[0]
        else:
            if use_cuda:
                action = np.argmax(action_softmax.data.cpu().numpy())
            else:
                action = np.argmax(action_softmax.data.numpy())

        next_state, reward, done, _ = env.step(action)
        print("Action:{}, Reward:{}".format(action, reward))
        # print("Step:{}, state{}, action:{}, next state:{}, reward:{}".format(step, state, action, next_state, reward))
        states = np.append(states, state, axis=0)
        # action = torch.eye(nA)[action].view(1,nA)            
        actions = np.append(actions, action)
        rewards = np.append(rewards,reward)
        state = np.copy(next_state)
        step+=1
    print("Episode completed in {} seconds".format(time.time()-start_time))
    return states, actions, rewards

def save_checkpoint(state, is_best, env, a2c_variant):
    filename='saved_models/'+env+'-a2c-'+a2c_variant+'-checkpoint.pth.tar'
    bestFilename = 'saved_models/'+env+'-a2c-'+a2c_variant+'-model-best.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)

def load_checkpoint(checkpoint_file, actor_model, critic_model, optimizer=None, optimizer_actor=None, optimizer_critic=None):
    args = parse_arguments()
    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        start_episode = checkpoint['epoch']
        best_reward = checkpoint['best_reward']
        actor_model.load_state_dict(checkpoint['state_dict'][0])
        critic_model.load_state_dict(checkpoint['state_dict'][1])
        if args.as_baseline:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            optimizer_actor.load_state_dict(checkpoint['optimizer'][0])
            optimizer_critic.load_state_dict(checkpoint['optimizer'][1])
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


def train(i, env, actor_model, critic_model,logger, best_reward, optimizer=None, optimizer_actor=None, optimizer_critic=None, sample=True):

        args = parse_arguments()
        nS = env.observation_space.shape[0]
        nA = env.action_space.n
        # Start with policy model 
    
        is_best = 0
        # Generate the episode
        states, actions, rewards = generate_episode(env, actor_model, render=False)
        print(actions.shape[0])
        states = Variable(torch.from_numpy(states).type(FloatTensor), requires_grad=False).view(-1,nS)
        # rewards = Variable(torch.from_numpy(rewards).type(FloatTensor), requires_grad=False).view(-1,1)
        actions = Variable(torch.from_numpy(actions).type(LongTensor), requires_grad=False).view(-1,1)
        episode_length = states.shape[0]

        # Array to store N-step returns from each time step
        # R = Variable(torch.zeros(episode_length,1), requires_grad=True)

        state_values = critic_model(states)

        V_end = []
        bootstrap = np.zeros((episode_length,1))
        # Calculating N-step return for each time step
        for step in range(episode_length-1,-1,-1):
            if step+args.N >= episode_length:
                V_end.append(Variable(FloatTensor([0]), requires_grad=False))
            else:
                V_end.append(state_values[step+args.N])
        
            for k in range(args.N):
                if step+k >=episode_length:
                    continue
                else:
                    bootstrap[step] += (args.gamma**k)*rewards[step+k]/100.0

        V_end = torch.stack(V_end)
        bootstrap = Variable(torch.from_numpy(bootstrap).type(FloatTensor))
        R = (args.gamma**args.N)*V_end + bootstrap
        # Calculating log of policy for the entire episode
        actor_output = actor_model(states)
        actor_output_log = actor_output.log() # log of Probabilities
        logPolicy = actor_output_log.gather(1,actions)
        dist_entropy = -(actor_output_log * actor_output).sum(-1).mean()
        actor_loss = -torch.mean((R - state_values)*logPolicy) - (dist_entropy * args.entropy_coef if args.use_entropy else 0)
        critic_loss = torch.mean((R - state_values)**2)
        

        # Inheriting some implementation tricks from gym baseline
        if args.as_baseline:
            
            loss = critic_loss * args.value_loss_coef + actor_loss - dist_entropy * args.entropy_coef
            optimizer.zero_grad()
            loss.backward()
            clip_grad.clip_grad_norm(actor_model.parameters(), 0.5)
            clip_grad.clip_grad_norm(critic_model.parameters(), 0.5)
            optimizer.step()

        else:
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()

            if not args.critic_overtrained or i>args.overtrain_episodes or i%args.overtrain_freq==0:
                actor_loss.backward(retain_graph=True)
                clip_grad.clip_grad_norm(actor_model.parameters(), 0.5)
                optimizer_actor.step()

            critic_loss.backward()
            clip_grad.clip_grad_norm(critic_model.parameters(), 0.5)
            optimizer_critic.step()


        #Plotting train results on Tensorboard
        if i%args.train_plot_freq == 0:
            logger.scalar_summary(tag='Train/Rewards', value=np.sum(rewards), step=i)
            logger.scalar_summary(tag='Train/Actor/Loss', value=actor_loss, step=i)
            logger.scalar_summary(tag='Train/Critic/Loss', value=critic_loss, step=i)
            logger.model_param_histo_summary(model=actor_model, step=i)
            logger.model_param_histo_summary(model=critic_model, step=i)


        # Evaluating and plotting eval results
        # if i%args.eval_freq==0:
        #     eval_rewards = evaluate(env, actor_model, num_episodes = args.eval_episodes, sample=sample)
        #     mean = np.mean(eval_rewards)
        #     std= np.std(eval_rewards)
        #     if mean>200.0:
        #         sample=False
        #     if mean > best_reward:
        #         is_best = 1

        #     # Save checkpoint
        #     save_checkpoint({
        #         'epoch': i + 1,
        #         'state_dict': [actor_model.state_dict(), critic_model.state_dict()],
        #         'best_reward': best_reward,
        #         'optimizer' :  optimizer.state_dict() if args.as_baseline else [optimizer_actor.state_dict(),optimizer_critic.state_dict()]
        #     }, is_best, args.env, args.a2c_variant)

        #     # Plot on tensorboard
        #     logger.scalar_summary(tag='Test/Mean Reward', value=mean, step=i)
        #     logger.scalar_summary(tag='Test/Std', value=std, step=i)

        #     # Print results
        #     if mean > 190:
        #         cprint('Evaluate - Episode:{}, Mean:{}, Std:{}'.format(i, mean, std), color='green')
        #     else:
        #         print('Evaluate - Episode:{}, Mean:{}, Std:{}'.format(i, mean, std))

        # Printing train results
        if i%args.log_freq==0:
            if np.sum(rewards) > 190:
                cprint('Train - Episode:{}, Reward:{}'.format(i, np.sum(rewards)), color='green')

            else:
                print('Train - Episode:{}, Reward:{}'.format(i, np.sum(rewards)))


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    render = args.render

    # Create the environment.
    env = gym.make(args.env)

    # Declare the model
    nS = env.observation_space.shape[0]
    nA = env.action_space.n
    actor_model = Actor(nS, nA)
    critic_model = Critic(nS, nA)
    if use_cuda:
        actor_model.cuda()
        critic_model.cuda()

    if args.as_baseline:
        optimizer = torch.optim.RMSprop(list(actor_model.parameters()) + list(critic_model.parameters()), args.actor_lr)
        optimizer_actor = None
        optimizer_critic = None
    else:
        optimizer = None
        optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=args.actor_lr)
        optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=args.critic_lr)
    logger = Logger('tb_logs', name=args.env+'_'+args.tb_logdir)

    if args.resume:
        if args.as_baseline:
            start_episode, best_reward = load_checkpoint(args.resume, actor_model, critic_model, optimizer=optimizer)
        else:
            start_episode, best_reward = load_checkpoint(args.resume, actor_model, critic_model, optimizer_actor=optimizer_actor, optimizer_critic=optimizer_critic)
    else:
        start_episode = args.start_episode
        best_reward = -np.inf

    for i in range(start_episode, args.end_episode):
        train(i, env, actor_model, critic_model, optimizer=optimizer, 
            optimizer_actor=optimizer_actor, optimizer_critic=optimizer_critic,
            logger=logger, best_reward=best_reward, sample=True)



if __name__ == '__main__':
    main(sys.argv)
