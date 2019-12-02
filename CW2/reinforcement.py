import numpy as np
import random
import time
import matplotlib.pyplot as plt

def init_policy(value, W, shape):
    policy = np.empty(shape, dtype=int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            policy[i,j] = random.randint(0,3)
            policy = update_policy((i,j), policy, value, W)
    return policy


def init_reward(shape):
    reward = np.zeros(shape, dtype=int)
    reward[5,0] = 0 # Start-point
    reward[2,2] = 1 # End-point
    return reward

def init_value(shape):
    value = np.zeros(shape, dtype=float)
    return value

def add_all_indicies(shape):
    record_poss = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            record_poss.append((i,j))
    return record_poss

def valid_pos(pos, W, shape): # Validated
    return False if (pos in W or pos[0] < 0 or pos[1] < 0 or pos[0] >= shape[0] or pos[1] >= shape[1]) else True

def move_pos(pos, action, W, shape):
    if pos == (2,2):
        pos_next = (5,0)
    elif action == 0:
        pos_next = (pos[0]-1,pos[1])
    elif action == 1:
        pos_next = (pos[0]+1,pos[1])
    elif action == 2:
        pos_next = (pos[0],pos[1]-1)
    elif action == 3:
        pos_next = (pos[0],pos[1]+1)
    return pos_next if valid_pos(pos_next,W,shape) else pos

def update_value(state, policy, reward, value, W, alpha=0.1, discount=0.9):
    a = alpha
    d = discount
    value_new = np.copy(value)
    pos = state
    if valid_pos(pos,W,value.shape):
        pos_next = move_pos(pos, policy[pos], W, policy.shape)
        value_new[pos] = value[pos] + a*(reward[pos_next] + d*value[pos_next] - value[pos])
    return value_new

def update_policy(state, policy, value, W, epoch=100, epochs=1000, epsilon_explore=0.7, epsilon_exploit=0.1, explore_prop=1/3):
    if state == (2,2):
        policy[state] = -10000000
    elif valid_pos(state,W,policy.shape):
        next_poss = [move_pos(state,k, W, policy.shape) for k in range(4)]
        next_vals = [value[pos] for pos in next_poss]

        # Switch from explore to exploit
        if epoch > epochs*explore_prop:
            epsilon = epsilon_exploit
        else:
            epsilon = epsilon_explore

        # Prevent moving to self
        move_to_self = True
        while move_to_self:
            if random.uniform(0,1) <= epsilon:
                next_move = random.randint(0,3)
            else:
                next_move = np.argmax(next_vals)
            if next_poss[next_move] == state:
                next_vals[next_move] = -float("inf")
            else:
                move_to_self = False

        policy[state] = next_move
    else:
        policy[state] = -1
    return policy


def convert_policy_format(policy):
    policy_new = np.empty(policy.shape, dtype='U25')
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            if policy[i,j] == 0:
                policy_new[i,j] = u"\u2191"
            elif policy[i,j] == 1:
                policy_new[i,j] = u"\u2193"
            elif policy[i,j] == 2:
                policy_new[i,j] = u"\u2190"
            elif policy[i,j] == 3:
                policy_new[i,j] = u"\u2192"
            elif policy[i,j] == -1:
                policy_new[i,j] = u"\u2022"
            else:
                policy_new[i,j] = "E"
    return policy_new

def record_results(value, results, poss):
    for i in poss:
        results.append([])
    for i in range(len(poss)):
        results[i].append(value[poss[i]])
    return results

def plot_results(results, poss, shape, epochs, explore_prop, top_amount=3):
    top_results = []
    top_poss = []
    title=f"Value at highest {top_amount} states"

    # # Remove end state (2,2)
    # del results[2*shape[1] + 2]
    # del poss[2*shape[1] + 2]

    for i in range(top_amount):
        index = np.argmax(results)
        top_results.append(results[index])
        top_poss.append(poss[index])
        del results[index]
        del poss[index]

    for i in range(len(top_poss)):
        plt.plot(top_results[i], label=(f"{top_poss[i]}" + (" #Start" if top_poss[i]==(5,0) else "  #End" if top_poss[i] == (2,2) else "")))
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    xmin, xmax, ymin, ymax = plt.axis()
    plt.vlines(epochs*explore_prop, ymin, ymax, label="explore to exploit", linestyle="dotted")
    plt.legend(loc="best", framealpha=1)
    plt.title(title)
    plt.show()


def tests():
    shape = [6,8]
    W = [(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,2),(3,5),(3,6),(4,2),(5,1)] # Walls
    reward = init_reward(shape)
    value = init_value(shape)
    policy = init_policy(value, W, shape)

    # Testing validation function
    print ("Testing validation function")
    validation = np.empty([10,12],dtype=bool)

    for i in range (validation.shape[0]):
        for j in range (validation.shape[1]):
            validation[i,j] = valid_pos((i-2,j-2), W, reward.shape)

    print (validation)

    # Testing move function
    print ("Testing move function")
    validation = np.empty([6,8,4,2], dtype=int)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(4):
                validation[i,j,k] = move_pos((i,j),k,W,shape)
    print (validation[0])



def main():
    shape = [6,8]
    W = [(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,2),(3,5),(3,6),(4,2),(5,1)] # Walls
    reward = init_reward(shape)
    value = init_value(shape)
    policy = init_policy(value, W, shape)
    record_poss = add_all_indicies(shape)
    results = []

    # Hyperparams for learning
    alpha = 0.4
    discount = 0.9
    epsilon_explore = 0.7
    epsilon_exploit = 0
    explore_prop = 2/3

    # Main epochs for learning
    start = time.time()
    epochs = 100
    for i in range(0,epochs):
        if i%(int(epochs/10)) == 0:
            print (f"Epoch: {i}")
        state = (5,0)
        results = record_results(value, results, record_poss)
        while state != (2,2):
            # if i==91:
            #     print (f"{state},")
            value = update_value(state, policy, reward, value, W, alpha=alpha, discount=discount)
            policy = update_policy(state, policy, value, W, epoch=i, epochs=epochs, epsilon_explore=epsilon_explore, epsilon_exploit=epsilon_exploit, explore_prop=explore_prop)
            state = move_pos(state, policy[state], W, shape)

    # Final outcome
    print(f"Time taken: {time.time() - start}")
    print ("Final policy: ")
    print(convert_policy_format(policy))
    print ("Final values: ")
    print (value)
    input("Press Enter to continue...")

    # Plot results of top valued states
    plot_results(results, record_poss, shape, epochs, explore_prop, top_amount=4)
    plt.imshow(value)
    plt.show()
    
main()
