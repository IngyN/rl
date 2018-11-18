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

import deepmind_lab

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


def get_map_item(coord_map, real_world_x, real_world_y, world_width, world_height):
  """Given a coordinate map, real_world coordinates (x,y) and real world width and height, returns what map item is present at (x,y)"""

  rows = len(coord_map)
  cols = len(coord_map[0])

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

  return coord_map[coord_x][coord_y]

def print_step(obs, step):

  print('---------------------------- Step : ', step, '--------------------')
  # print(obs.keys())
  # print maze layout:
  for key in obs.keys():
    if key != 'RGB_INTERLEAVED' and key != 'DEBUG.CAMERA_INTERLEAVED.TOP_DOWN':
      print('Key :', key, obs[key])

  img = obs['DEBUG.CAMERA_INTERLEAVED.TOP_DOWN']
  cv2.circle(img, (120, 50), 3, (0,255,0), -1)
  cv2.imshow('map', img)
  cv2.waitKey(2)

  
def run(level_script, config, num_episodes):
  """Construct and start the environment."""
  env = deepmind_lab.Lab(level_script, ['RGB_INTERLEAVED','DEBUG.CAMERA_INTERLEAVED.TOP_DOWN' ,'DEBUG.MAZE.LAYOUT', 'DEBUG.POS.TRANS'], config)
  env.reset()

  observation_spec = env.observation_spec()
  print('Observation spec:')
  pprint.pprint(observation_spec)

  action_spec = env.action_spec()
  print('Action spec:')
  pprint.pprint(action_spec)

  obs = env.observations()  # dict of Numpy arrays
  rgb_i = obs['RGB_INTERLEAVED']
  print('Observation shape:', rgb_i.shape)
  sys.stdout.flush()

  # Create an action to move forwards.
  action = np.zeros([7], dtype=np.intc)
  action[3] = 1

  score = 0

  for _ in six.moves.range(num_episodes):
    while env.is_running():
      # Advance the environment 4 frames while executing the action.
      reward = env.step(action, num_steps=4)
      print_step(env.observations(),env.num_steps())
      if reward != 0:
        score += reward
        print('Score =', score)
        sys.stdout.flush()


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
