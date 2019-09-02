# DRL applied to "Reacher" Environment 
This repo trains a Deep Reinforcement Learning (DRL) agent to solve the Unity ML-Agents "Reacher" environment. 
The motivation for this program was the 2nd project in the Udacity Deep Reinforcement Learning 
[Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). 


## Reacher Environment
The [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) 
environment provided by [Unity](https://unity3d.com/machine-learning/) contains a double-jointed arm can move to target 
locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
Thus, the goal of the DRL agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of 
the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the 
action vector should be a number between -1 and 1.   

The gif below illustrate the environment with 10 identical arms.  This repo solves the environment with a single arm.  

![Trained Agent](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

### Solving the Environment (Single Arm)
The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 
consecutive episodes.

### Getting Started 
This repo is setup for a Windows 10 Home 64-bit environment.  If you prefer 32-bit Windows, OS X or Linux, please see 
the source Udacity [repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).
Note to determine if your Windows is 64-bit or 32-bit follow this 
[link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64).
If you install one of the other environments, the following operations should be the same.  Only the Reacher environment
downloaded into the `p2_continuous-control` folder will be different. 

### Instructions
Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!  
