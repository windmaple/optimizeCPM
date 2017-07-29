from tensorforce import Configuration
from tensorforce.agents import *
from tensorforce.agents import PPOAgent
from optimizeCPM_env import optimizeCPM
from tensorforce.execution import Runner
from tensorforce.core.networks import layered_network_builder

import numpy as np
import pylab

ts = []

env = optimizeCPM()

agents = []
agent = VPGAgent(config=Configuration(
    loglevel='debug',
    batch_size=1,
    states=env.states,
    actions=env.actions,
    network=layered_network_builder([
        dict(type='dense', size=32),
        dict(type='dense', size=32),
    ]),
))
agents.append(agent)

# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.timestep,
                                                                                 reward=r.episode_rewards[-1]))
    ts.append(r.timestep)
    return True

for agent in agents:
    # Create the runner
    runner = Runner(agent=agent, environment=env)
    # Start learning
    runner.run(episodes=500000, max_timesteps=5, episode_finished=episode_finished)

    # Print statistics
    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(ep=runner.episode,
                                                                                                       ar=np.mean(
                                                                                                           runner.episode_rewards[
                                                                                                       -100:])))
    WINDOW_SIZE = 20
    tmp = [np.mean(runner.episode_rewards[i:i + WINDOW_SIZE]) for i in range(len(runner.episode_rewards) - WINDOW_SIZE)]
    pylab.plot(tmp, label=agent.name.replace('Agent', ''))
    # tmp = [np.mean(ts[i:i + WINDOW_SIZE]) for i in range(len(ts) - WINDOW_SIZE)]
    # pylab.plot(ts)

pylab.legend(loc='lower right')
pylab.show()