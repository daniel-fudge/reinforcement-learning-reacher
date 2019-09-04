# DRL applied to "Reacher" Environment 
This repo trains a Deep Reinforcement Learning (DRL) agent to solve the Unity ML-Agents "Reacher" environment. 
The motivation for this program was the 2nd project in the Udacity Deep Reinforcement Learning 
[Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). 

## Project Details
### Reacher Environment
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

## Getting Started 
This repo is setup for a Windows 10 Home 64-bit environment.  If you prefer 32-bit Windows, OS X or Linux, please see 
the source Udacity [repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).
Note to determine if your Windows is 64-bit or 32-bit follow this 
[link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64).
If you install one of the other environments, the following operations should be the same.  Only the Reacher environment
downloaded into the `p2_continuous-control` folder will be different. 

### Python 'drlnd' Virtual Environment (venv)

To set up your python environment to run the code in this repository, follow the instructions below.  Note if you have 
already created the 'drlnd' venv for other Udacity Deep Reinforcement Learning NanoDegree (DRLND), you can simply 
activate it. 

1. If not already installed, install the Anaconda Python distribution from [here](https://www.anaconda.com/distribution/). 

2. Create (and activate) a new environment with Python 3.6.

	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
3. Ensure you have the "Build Tools for Visual Studio 2019" installed from this 
[site](https://visualstudio.microsoft.com/downloads/).  
This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) may also 
be very helpful.  This was confirmed to work in Windows 10 Home.  

4. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
5. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.  
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

## Instructions
### Training the Agent
The agent is trained and tested with the `run_test.py` script, which can be executed in your favorite IDE of the 
following command:

    python run_test.py 

### Pretty plot
The raw scores are save as `scores.npz` and can be plotted and saved as `scores.png` with the command:

    python plot.py  

### Saved model Weights
The trained model is saved as `checkpoint.pth` in the root directory and can be loaded with the command:  

    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    
### Report
A [report](Report.md) providing a description of the implementation, a plot of the rewards and ideas for future work can
be found in `Report.md`.

## License
This code is copyright under the [MIT License](LICENSE).

## Contributions
Please feel free to raise issues against this repo if you have any questions or suggestions for improvement.
