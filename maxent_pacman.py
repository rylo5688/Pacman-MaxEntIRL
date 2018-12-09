import numpy as np

smallClassicStates = [[1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1],
                      [1,0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,1],
                      [1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1],
                      [1,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,1],
                      [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]

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


if __name__ == '__main__':
    width, height = 18, 5   # This is the width x height for the smallClassic layout without the walls
    discount = 0.01         # MDP discount factor
    n_trajectories = 10     # Number of samples
    gradient_iterations = 100   # Gradient descent iterations
    learning_rate = 0.01    # Gradient descent learning rate

    feature_matrix = np.eye(width*height) # Identity matrix

    transition_probs = transition_probability(width*height)
    print(transition_probs)
    # for row in smallClassicStates:

