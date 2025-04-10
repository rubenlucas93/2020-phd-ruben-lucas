---
title: "Migration to RLStudio of mountain car problem proposed by openAI-gym"
excerpt: "The mountain car exercise was migrated to RL-Studio"

sidebar:
  nav: "docs"

toc: true
toc_label: "TOC installation"
toc_icon: "cog"


categories:
- your category
tags:
- tag 1
- tag 2
- tag 3
- tag 4

author: Rubén Lucas
pinned: false
---
<strong>MIGRATION</strong>

The previous implementation (no RL-Studio related) is documented [in this post](https://roboticslaburjc.github.io/2020-phd-ruben-lucas/projects/2021-02-07-customized-mountainCar/) which is an evolution of [this one](https://roboticslaburjc.github.io/2020-phd-ruben-lucas/projects/2021-01-22-mountainCar_qlearning/) in which you can see more details of this problem solution.
This migration consisted of:
  -  Creating models and world for mountain car problem.
  -  Ensuring actions doesnt provoke and unconsistent state (robot must always be within the "mountain" platform and move just to right and left).
  -  Ensure actions efforts make the problem reachable but considerably difficult so we can take benefit of the algorithm to solve it.
  -  Migrating the learning algorithm to make the robot behave well in mountain car problem.
  -  Adapt rewards as a non-stationary problem.
  -  perform several tests to conclude that the problem was succesfully migrated and solved using qlearning.

you can find all the iterations tested in the [results uploaded](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/RL_Unibotics/RL-Studio/results/mountain_car) in the repository.

In there you will notice that there is not need to give plenty of information to the agent through the reward function.
   - If you give a reward of 1 when the goal is achieved and 0 otherwise, the robot finally learn the optimal path.
   - If you stop the episode when the robot perform 0 steps you encourage the robot to reach the goal before 20 steps are accomplished.

The optimal reward configuration and hyperparameters can be found in the [uploaded agent code](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/RL_Unibotics/RL-Studio/)
In the same way, there you will find the [worlds and models](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/RL_Unibotics/RL-Studio/CustomRobots) used.

<strong>RL-Studio related<strong>

Additionally, the following steps were accomplished to adapt the problem to RL-Studio:
- Create a .yml configuration file
- Including the algorithm in "algorithms" folder
- Including the agent in "agents" folder
- Push the model and world referenced in the configuration file to the [CustomRobots repository](https://github.com/JdeRobot/CustomRobots)
- Adding the environment in /rl-studio/__init__ file
- Adding a folder in /rl-studio/envs (no major modifications with respect to the other implementations in rl-studio)

<strong>DEMO</strong>

<iframe width="560" height="315" src="https://www.youtube.com/embed/KZjDe6N-d0k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

