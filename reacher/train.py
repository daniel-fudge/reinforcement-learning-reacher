"""
This script trains and saves the model and plots its performance.

Note:  You need to verify the env path is correct for you PC and OS.
"""

from collections import deque
from reacher.ddpg_agent import Agent
import numpy as np
import torch


def make_plot(show=False):
    """Makes a pretty training plot call score.png.

    Args:
        show (bool):  If True, show the image.  If False, save the image.
    """

    import matplotlib.pyplot as plt

    # Load the previous scores and calculated running mean of 100 runs
    # ---------------------------------------------------------------------------------------
    with np.load('scores.npz') as data:
        scores = data['arr_0']
    cum_sum = np.cumsum(np.insert(scores, 0, 0))
    rolling_mean = (cum_sum[100:] - cum_sum[:-100]) / 100

    # Make a pretty plot
    # ---------------------------------------------------------------------------------------
    plt.figure()
    x_max = len(scores)
    y_min = scores.min() - 1
    x = np.arange(x_max)
    plt.scatter(x, scores, s=2, c='k', label='Raw Scores', zorder=4)
    plt.plot(x[99:], rolling_mean, lw=2, label='Rolling Mean', zorder=3)
    plt.scatter(x_max, rolling_mean[-1], c='g', s=40, marker='*', label='Episode {}'.format(x_max), zorder=5)
    plt.plot([0, x_max], [30, 30], lw=1, c='grey', ls='--', label='Target Score = 30', zorder=1)
    plt.plot([x_max, x_max], [y_min, rolling_mean[-1]], lw=1, c='grey', ls='--', label=None, zorder=2)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.xlim([0, x_max + 5])
    plt.ylim(bottom=y_min)
    if show:
        plt.show()
    else:
        plt.savefig('scores.png', dpi=200)
    plt.close()


def train(agent, env, n_episodes=2000, max_t=1000):
    """This function trains the given agent in the given environment.

    Args:
        agent (Agent):  The agent to train.
        env (unityagents.UnityEnvironment):  The training environment
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of time steps per episode
    """

    scores = list()
    scores_window = deque(maxlen=100)
    brain_name = env.brain_names[0]
    for i_episode in range(1, n_episodes + 1):
        brain_info = env.reset(train_mode=True)[brain_name]
        state = brain_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state=state)
            brain_info = env.step(action)[brain_name]
            next_state = brain_info.vector_observations[0]
            reward = brain_info.rewards[0]
            done = brain_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            break

    # Save models weights and scores
    torch.save(agent.actor_target.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_target.state_dict(), 'checkpoint_critic.pth')
    np.savez('scores.npz', scores)


def setup(env):
    """Setups up the environment to train.

    Args:
        env (unityagents.UnityEnvironment):  The training environment
    """
    # Setup the environment and print of some information for reference
    # -----------------------------------------------------------------------------------
    print('Setting up the environment.')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]

    # Setup the agent and return it
    # -----------------------------------------------------------------------------------
    print('Setting up the agent.')
    return Agent(state_size=state_size, action_size=action_size, random_seed=42)
