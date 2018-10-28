import math
import ddqn as dqn
#import dqn
import py2048
import numpy as np
from collections import deque
from matplotlib import pyplot as plt

step = 0
height = 4
width = 4
board = py2048.GameBoard(height=height, width=width)

agent = dqn.DQNAgent(state_size = height*width, action_size = 4)
agent.discount_factor = 0.95 # discount factor

n_training_moves = 100000000


def calculateReward(state, next_state, action, done):
    reward = 0
    reward += ((next_state == 0).sum() - (state == 0).sum())/7
    reward -= int(done)
    return reward

def calculateScore(board,exp = False):
    if exp:
        board = board.exponentiate()
    else:
        board = board.board
    return board.flatten().sum()
    # return max(board.flatten())

smoothness = 1500

scores = deque(maxlen = 4000)
expScores = deque(maxlen = 4000)
try:
    agent.load_model("./save_model/py2048_a3c")
    print("LOADED NETWORK")
except:
    print("FAILED TO LOAD NETWORK")

for i in range(n_training_moves):

    state = (board.board.flatten()/10)
    next_state = (board.board.flatten() / 10)

    action = agent.get_action(state)
    board.performAction(action)


    done = int(board.gameOver)
    reward = calculateReward(state, next_state, action, done)

    agent.memory(state, action, reward)

    values = agent.actor.predict(np.array(agent.states))
    #values = np.reshape(values, len(values))

    agent.avg_p_max += np.amax(values)
    state = next_state
    step += 1
    agent.t += 1


    if i % 1000 == 0:
        agent.save_model("./save_model/py2048_a3c")

    if agent.t >= agent.t_max or done:
        agent.train_episode(done)
        agent.update_localmodel()
        agent.t = 0

    if done:
        MaxQ = (agent.avg_p_max / float(step))
        scores.append(calculateScore(board))
        expScores.append(calculateScore(board,exp=True))
        MaxNumber = 2 ** board.Max_number()
        print("score: {}".format(np.mean(list(scores)[-1*smoothness:])))
        print("expScore: {}".format(np.mean(list(expScores)[-1*smoothness:])))
        print("board: \n{}".format(board.exponentiate()))
        print("MaxNumber :", MaxNumber)
        print("#"*100)
        board.reset()
        agent.save_learning_result(format(np.mean(list(scores)[-1 * smoothness:])), MaxNumber, MaxQ)
        step = 0
        agent.avg_p_max = 0


print("done training")