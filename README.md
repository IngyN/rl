# Reinforcement Learning for the visually impaired of Virginia

## Introduction
Navigation using geolocation technology has become an essential function of our personal smartphones. It comes in handy when walking, cycling, driving or using public transport in areas we are unfamiliar with. However, because of GPS limitations it falls short when it comes to indoor navigation. In our project, we focus on indoor navigation specifically for visually impaired individuals who face this challenge every day even when moving within their homes. The idea is to use the camera of a handheld phone for vision, and a reinforcement learning based algorithm for navigation. After the modeling of an indoor environment and training, the resident will be able to navigate from any point in the indoor environment to another, avoiding walls and obstacles. The purpose of this document is to report our findings and approach to solving the problem. This project uses a simulation of the desired indoor environment for training. In this document, we propose several experiments using Q-Learning and our results.

## Files 
- python/testing_obs.py: implements the Q-Learning agent and learning
- python/static_goal_maze.lua: adapted Lua file for maze generation
- python/plot.py: plot results from agent run and training with matplotlib
- python/logs/ : logs from the agent. 
- report.pdf : report detailing our work.

Most other files are common with the Deepmind lab repo and are not our work. Here is a short explanation of our work in the form of a [video] (https://www.youtube.com/watch?v=8KGUOTYc1Qc).

## Requirements
- Python 3 (recommended 3.5 or more)
- Opencv 2 and opencv for python
- [DeepmindLab](https://github.com/deepmind/lab) 

## Coming Soon
Tutorial on how to install Deepmind lab and start an agent and your own map.
