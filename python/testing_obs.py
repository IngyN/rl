## Copyright (C) 2016-17 Google Inc.
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License along
## with this program; if not, write to the Free Software Foundation, Inc.,
## 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
################################################################################
"""A working example of deepmind_lab using python."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint
import sys
import numpy as np
import six
import cv2
import random
import time

import deepmind_lab

def _action(*entries):
    return np.array(entries, dtype=np.intc)

# class QTable:
#   def __init__(state_size, action_size):
#     self.state_size = state_size
#     self.action_size = action_size
#     self.qt = np.zeros((state_size, action_size))

#   def 

class QLearningAgent:
  ACTIONS = {
      'look_left': _action(-35, 0, 0, 0, 0, 0, 0),
      'look_right': _action(35, 0, 0, 0, 0, 0, 0),
      'forward': _action(0, 0, 0, 1, 0, 0, 0)
  }

  ACTION_LIST = list(six.viewvalues(ACTIONS))

  def __init__(self, coord_map):
    coord_map_shape = get_coord_map_shape(coord_map)
    state_size = coord_map_shape[0] * coord_map_shape[1]
    action_size = len(QLearningAgent.ACTION_LIST)
    self.qtable = np.zeros((state_size, action_size))
    self.action_count = len(QLearningAgent.ACTIONS)

  def get_action(self, action_index):
    return QLearningAgent.ACTION_LIST[action_index]

  def random_action_index(self):
    return random.randint(0, self.action_count - 1)

def get_q_table_state(i, j, coord_map):
    cols = get_coord_map_shape(coord_map)[1]
    return i * cols + j

def get_coord_map_shape(coord_map):
  return (len(coord_map), len(coord_map[0]))

def get_coord_map(map_string):
  """Given a map string, returns a 2D array of map items (G, A,..). Bottom-left of map is taken as (0,0)."""

  map = map_string.splitlines()
  
  rows = len(map)
  cols = len(map[0])

  coord_map = [[0]*cols for i in range(rows)]
  
  for i in range(rows):
    for j in range(cols):
      coord_map[j][rows-1-i] = map[i][j]

  return coord_map

def get_coord_x_y(coord_map, real_world_x, real_world_y, world_width, world_height):
  """Given a coordinate map, real_world coordinates (x,y) and real world width and height, returns coordinates in coord_map"""

  coord_map_shape = get_coord_map_shape(coord_map)
  
  rows = coord_map_shape[0]
  cols = coord_map_shape[1]

  sf_col=world_width/cols
  sf_row=world_height/rows

  #floor the floating pt
  coord_x = int(real_world_x/sf_col)
  coord_y = int(real_world_y/sf_row)

  #boundary condition
  if coord_x >= rows:
    coord_x -= 1
  if coord_y >= cols:
    coord_y -= 1

  return (coord_x, coord_y)

def print_step(obs, step, action):

  # print('---------------------------- Step : ', step, '--------------------')
  # print(obs.keys())
  # print maze layout:
  # print(f'Action taken = {action}')

  # for key in obs.keys():
  #   if key != 'RGB_INTERLEAVED' and key != 'DEBUG.CAMERA_INTERLEAVED.TOP_DOWN':
  #     print('Key :', key, obs[key])

  img = obs['DEBUG.CAMERA_INTERLEAVED.TOP_DOWN']
  cv2.circle(img, (120, 50), 3, (0,255,0), -1)
  cv2.imshow('map', img)
  cv2.waitKey(2)

  
def run(level_script, config, num_episodes):
  start_time = time.time()	
  print('Start time = {}'.format(start_time))
  """Construct and start the environment."""

  world_width = int(config["width"])
  world_height = int(config["height"])
    
  #initialize world
  env = deepmind_lab.Lab(level_script, ['RGB_INTERLEAVED','DEBUG.CAMERA_INTERLEAVED.TOP_DOWN' ,'DEBUG.MAZE.LAYOUT', 'DEBUG.POS.TRANS'], config)

  #score = 0

  # for _ in six.moves.range(num_episodes):
  #   while env.is_running():
  #     #get the next action from the agent
  #     action = agent.step()

  #     # Advance the environment 4 frames while executing the action.
  #     reward = env.step(action, num_steps=1)
  #     print_step(env.observations(),env.num_steps(), action)
  #     if reward != 0:
  #       score += reward
  #       print('Score =', score)
  #       sys.stdout.flush()
  
  #hyperparameters
  learning_rate = 0.8           # Learning rate

  max_steps = 1000                # Max steps per episode

  gamma = 0.95                  # Discounting rate

  # Exploration parameters
  epsilon = 1.0                 # Exploration rate
  max_epsilon = 1.0             # Exploration probability at start
  min_epsilon = 0.01            # Minimum exploration probability 

  decay_rate = 0.025             # Exponential decay rate for exploration prob

  rewards = []

  env.reset()
  obs = env.observations()
  map_string = obs['DEBUG.MAZE.LAYOUT']
  current_pos = obs['DEBUG.POS.TRANS']
  coord_map = get_coord_map(map_string)

  #initialize agent
  agent = QLearningAgent(coord_map)
  qtable = agent.qtable

  for episode in six.moves.range(num_episodes):
    coord_x_y = get_coord_x_y(coord_map, current_pos[0], current_pos[1], world_width, world_height)
    state = get_q_table_state(coord_x_y[0], coord_x_y[1], coord_map)

    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):

      print('------------------Episode: {}, Step: {}------------------'.format(episode, step))

      # 3. Choose an action a in the current world state (s)
      ## First we randomize a number
      exp_exp_tradeoff = random.uniform(0, 1)
      
      ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
      if exp_exp_tradeoff > epsilon:
          print("exploitation")
          action_index = np.argmax(qtable[state,:])

      # Else doing a random choice --> exploration
      else:
          print("exploration")
          action_index = agent.random_action_index()

      action = agent.get_action(action_index)

      # Take the action (a) and observe the outcome state(s') and reward (r)
      # new_state, reward, done, info = env.step(action)
      reward = env.step(action, num_steps=4)

      print_step(env.observations(),env.num_steps(), action)

      obs = env.observations()  # dict of Numpy arrays
      current_pos = obs['DEBUG.POS.TRANS']
      new_coord_x_y = get_coord_x_y(coord_map, current_pos[0], current_pos[1], world_width, world_height)
      new_state = get_q_table_state(coord_x_y[0], coord_x_y[1], coord_map)

      #calculate penalties
      penalty = 1 #penalize for each step

      #TODO nabeel: rewrite wall bumping logic
      if coord_map[new_coord_x_y[0]][new_coord_x_y[1]] == "*":

        print('bumped into wall!')

        penalty += 10
        break

      reward -= penalty

      print('action = {}'.format(action))
      print('reward => \n{}'.format(reward))
      print('coord_x_y = {}, new_coord_x_y = {}'.format(coord_x_y,new_coord_x_y))
      print('state = {}, new_state = {}'.format(state, new_state))

      # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
      # qtable[new_state,:] : all the actions we can take from new state
      qtable[state, action_index] = qtable[state, action_index] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action_index])
      
      print('qtable => \n{}'.format(qtable))

      total_rewards += reward
      
      # Our new state is state
      state = new_state
      coord_x_y = new_coord_x_y
      
      # if coord_map[coord_x_y[0]][coord_x_y[1]] == "G":
      #   current_time = time.time()
      #   time_elapsed = current_time - start_time
      #   print(f'elapsed time = [time_elapsed], current_ts = {current_time} ---- reached goal!')
      #   done = True 
      #   break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)

    env.reset()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-l', '--level_script', type=str,
                      default='demos/random_maze',
                      help='The level that is to be played. Levels'
                      'are Lua scripts, and a script called \"name\" means that'
                      'a file \"assets/game_scripts/name.lua is loaded.')
  parser.add_argument('-s', '--level_settings', type=str, default=None,
                      action='append',
                      help='Applies an opaque key-value setting. The setting is'
                      'available to the level script. This flag may be provided'
                      'multiple times. Universal settings are `width` and '
                      '`height` which give the screen size in pixels, '
                      '`fps` which gives the frames per second, and '
                      '`random_seed` which can be specified to ensure the '
                      'same content is generated on every run.')
  parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
  parser.add_argument('--num_episodes', type=int, default=1,
                      help='The number of episodes to play.')
  args = parser.parse_args()

  # Convert list of level setting strings (of the form "key=value") into a
  # `config` key/value dictionary.
  config = {k:v for k, v in [s.split('=') for s in args.level_settings]}

  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.level_script, config, args.num_episodes)
