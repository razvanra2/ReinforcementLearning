import matplotlib
import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from mpl_toolkits.mplot3d import Axes3D
import sklearn.pipeline
import sklearn.preprocessing

env = gym.make('MountainCar-v0')
num_episodes = 1000
gamma = 0.95
alpha = 0.1
num_actions = env.action_space.n

w = np.zeros((num_actions,100))

observation_examples = np.array([env.observation_space.sample() for x in range(1000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=25)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=25)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=25)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=25))
        ])

featurizer.fit(scaler.transform(observation_examples))

def featurize_state(state):
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized

def policy(state, weight, epsilon=0.1):
    A = np.ones(num_actions,dtype=float) * epsilon/num_actions
    bestAction =  np.argmax([state.dot(w[a]) for a in range(num_actions)])
    A[bestAction] += (1.0-epsilon)
    sample = np.random.choice(num_actions,p=A)
    return sample

def plotCostToGo(numTiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=numTiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=numTiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max([featurize_state(_).dot(w[a]) for a in range(num_actions)]), 2, np.dstack([X, Y]))

    ax = plt.axes(projection='3d')
    x = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')

    plt.show()


for e in range(num_episodes):

    state = env.reset()
    state = featurize_state(state)

    while True:
        action = policy(state,w)
        observation, reward, done, info = env.step(action)
        observation = featurize_state(observation)

        next_action = policy(observation,w)

        target = reward + gamma * observation.dot(w[next_action])
        td_error = state.dot(w[action]) - target

        dw = (td_error).dot(state)

        w[action] -= alpha * dw

        state = observation

        if done:
            break

plotCostToGo()

observation = env.reset()
observation = featurize_state(observation)
done = False
while not done:
    action = policy(observation, w)
    observation, reward, done, _ = env.step(action)
    observation = featurize_state(observation)

    env.render()