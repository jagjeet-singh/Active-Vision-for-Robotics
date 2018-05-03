import argparse
import numpy as np
import os
import gym
import kuka

from baselines import deepq

def main():
  parser = argparse.ArgumentParser(description='DQN for Active Vision')
  parser.add_argument('--commit', type=str, help='commit SHA of this run')
  parser.add_argument('--exp-name', type=str, default=None,
    help='name of experiment. model will be loaded from this folder')
  parser.add_argument('--render', type=bool, nargs='?', const=True,
    help='whether to render GUI or not')
  parser.add_argument('--max-episodes', type=int, help='maximum episodes')
  parser.add_argument('--epoch', type=int, default=None,
    help='training epoch of model to load. default will load kuka_model.pkl')
  args = parser.parse_args()

  print('This run is based upon #commit {}'.format(args.commit))
  exp_dir = os.path.join('exp/deepq', args.exp_name)
  models_dir = os.path.join(exp_dir, 'models')

  env = gym.make("kuka-v0")
  env.init_bullet(render=args.render, delta=1.0)

  if args.epoch is None:
    load_path = os.path.join(exp_dir, 'kuka_model.pkl')
  else:
    load_path = os.path.join(models_dir, 'ckpt_{}.pkl'.format(args.epoch))
    print(load_path)
  act = deepq.load(load_path)
  for _ in range(args.max_episodes):
    obs, done = env.reset(), False
    while not done:
      obs, _, _, _ = env.step(act(np.array(obs).reshape(1, -1))[0])

if __name__ == '__main__':
  main()