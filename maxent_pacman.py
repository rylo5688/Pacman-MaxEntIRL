import numpy as np
import matplotlib.pyplot as plt

from maxentIRL import maxent_irl

smallClassicStates = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,0],
                      [0,1,0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,1,0],
                      [0,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,0],
                      [0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0],
                      [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

actions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)] # Row-column

def state_index_to_coords(index, n_rows=18):
    return (index//n_rows, index%n_rows)


def transition_probability(n_states):
    transition_probs = np.empty((n_states, n_states, len(actions)))
    for curState in range(n_states):
        for nextState in range(n_states):
            for (aIndex, (xAct, yAct)) in enumerate(actions):
                xCur, yCur = state_index_to_coords(curState)
                xNext, yNext = state_index_to_coords(nextState)

                if (xCur + xAct, yCur + yAct) == (xNext, yNext):
                    transition_probs[curState][nextState][aIndex] = 1.0
                else:
                    transition_probs[curState][nextState][aIndex] = 0.0
    return transition_probs

def normalize(vals):
    """
    normalize array
    """

    return vals/sum(vals)

if __name__ == '__main__':
    width, height = 20, 7   # This is the width x height for the smallClassic layout without the walls
    discount = 0.01         # MDP discount factor
    n_trajectories = 10     # Number of samples
    gradient_iterations = 1   # Gradient descent iterations
    learning_rate = 0.01    # Gradient descent learning rate

    feature_matrix = np.eye(width*height) # Identity matrix

    transition_probs = transition_probability(width*height)

    path1 = [(114, (0, 0), 114), (114, (0, 1), 114), (114, (0, -1), 114), (114, (0, 0), 114), (114, (0, 1), 114), (114, (0, -1), 114), (114, (0, 0), 114), (114, (0, 1), 114), (114, (0, -1), 114), (114, (0, 0), 114), (114, (0, -1), 114), (114, (0, 1), 114), (114, (0, 0), 114), (114, (0, -1), 114), (114, (-1, 0), 114), (114, (0, -1), 113), (113, (-1, 0), 113), (113, (-1, 0), 113), (113, (0, -1), 112), (112, (0, 1), 112), (112, (0, -1), 112), (112, (0, -1), 111), (111, (-1, 0), 111), (111, (0, -1), 111), (111, (1, 0), 90), (90, (0, 1), 90), (90, (0, -1), 90), (90, (1, 0), None)]
    path2 = [(114, (0, 0), 114), (114, (0, 1), 114), (114, (0, -1), 114), (114, (0, 0), 114), (114, (0, 1), 114), (114, (0, -1), 114), (114, (0, 0), 114), (114, (0, 1), 114), (114, (-1, 0), 114), (114, (0, 0), 114), (114, (0, -1), 114), (114, (0, 1), 114), (114, (0, 0), 114), (114, (0, -1), 114), (114, (1, 0), 114), (114, (0, 0), 114), (114, (0, -1), 114), (114, (0, 1), 114), (114, (0, 0), 114), (114, (0, 1), 114), (114, (0, -1), 114), (114, (0, 0), 114), (114, (0, 1), 114), (114, (-1, 0), 114), (114, (0, 1), 115), (115, (-1, 0), 115), (115, (0, -1), 115), (115, (0, 1), 116)]
    path3 = [(114, (0, 0), 114), (114, (0, 1), 114), (114, (0, -1), 114), (114, (0, 0), 114), (114, (0, 1), 114), (114, (0, -1), 114), (114, (0, 0), 114), (114, (0, 1), 114), (114, (-1, 0), 114), (114, (0, 0), 114), (114, (0, -1), 114), (114, (-1, 0), 114), (114, (0, 0), 114), (114, (-1, 0), 114), (114, (0, 1), 114), (114, (0, 0), 114), (114, (0, -1), 114), (114, (1, 0), 114), (114, (0, 0), 114), (114, (1, 0), 114), (114, (0, -1), 114), (114, (0, 0), 114), (114, (0, -1), 114), (114, (1, 0), 114), (114, (0, -1), 113), (113, (0, 1), 113), (113, (0, 1), 113), (113, (0, -1), 112)]
    sample_paths = np.array([path1, path2, path3])

    rewards_maxent = maxent_irl(sample_paths, feature_matrix, transition_probs, discount=discount, iterations=gradient_iterations, learning_rate=learning_rate )
    # maxent_irl()    # for row in smallClassicStates:
    print(normalize(rewards_maxent))

