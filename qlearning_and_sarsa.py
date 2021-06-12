import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from os import system
from time import sleep

alpha = 0.7
gamma = 0.95
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01
train_episodes = 2500
max_steps = 100
visualize_policy = False


def get_e_greedy_action(env, Q, state, episode):
    global epsilon, max_epsilon, min_epsilon, decay

    exp_tradeoff = random.uniform(0,1)
    if exp_tradeoff > epsilon:
        action = np.argmax(Q[state,:])
    else:
        action = env.action_space.sample()
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay*episode)

    return action


def qlearning(env):
    global alpha, gamma, epsilon, max_epsilon, min_epsilon, decay, train_episodes, max_steps

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    training_rewards = []

    for episode in range(train_episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            # choose next action according to e-greedy policy
            action = get_e_greedy_action(env, Q, state, episode)

            # env step
            new_state, reward, done, _ = env.step(action)

            # update Q and crt state
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            state = new_state

            episode_reward += reward

            if done:
                break

        # just for generating graphs
        training_rewards.append(episode_reward)

    return Q, training_rewards



def sarsa(env):
    global alpha, gamma, epsilon, max_epsilon, min_epsilon, decay, train_episodes, max_steps

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    training_rewards = []

    for episode in range(train_episodes):
        state = env.reset()
        episode_reward = 0

        # choose init action according to e-greedy policy
        action = get_e_greedy_action(env, Q, state, episode)

        for _ in range(max_steps):
            # env step
            new_state, reward, done, _ = env.step(action)

            # choose action in new_state (a' for s')
            new_action = get_e_greedy_action(env, Q, new_state, episode)

            #update Q
            Q[state, action] = Q[state, action] + alpha * (reward + gamma*Q[new_state, new_action] - Q[state, action])
            state = new_state
            action = new_action

            episode_reward += reward

            if done:
                break

        # just for generating graphs
        training_rewards.append(episode_reward)

    return Q, training_rewards



def main():
    global visualize_policy
    env = gym.make("Taxi-v3").env

    Q, training_rewards_ql = qlearning(env)
    Q, training_rewards_sarsa = sarsa(env)

    #Visualizing results and total reward over all episodes
    x = range(train_episodes)
    plt.plot(x, training_rewards_sarsa, label="SARSA")
    plt.plot(x, training_rewards_ql, label="Q Learning")
    plt.xlabel('Episode')
    plt.ylabel('Training total reward')
    plt.title('Total rewards over all episodes in training')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.legend()
    plt.savefig("q_learning_and_sarsa_taxi-v3_rewards.png", dpi=100)
    plt.close()

    if visualize_policy:
        #Visualizing the gameplay
        state = env.reset()
        system('clear')
        env.render()
        done = False
        while not done:
            action = np.argmax(Q[state,:])
            new_state, reward, done, _ = env.step(action)
            state = new_state

            sleep(2)
            system('clear')
            env.render()

if __name__ == "__main__":
    main()