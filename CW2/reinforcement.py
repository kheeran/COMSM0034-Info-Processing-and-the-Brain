import numpy as np
import random
import time
import matplotlib.pyplot as plt

def init_policy(shape):
    policy = np.empty(shape, dtype=int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            policy[i,j] = random.randint(0,3)

    return policy

# Ensure that initial policy converges
def init_converging_policy(shape, W, reward):
    policy = init_policy(shape)
    check = True
    while check:
        if reward[move_pos((2,2), policy[2,2], W, shape)] == 0:
            policy = init_policy(shape)
        else:
            check = False
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
    if action == 0:
        pos_next = (pos[0]-1,pos[1])
    if action == 1:
        pos_next = (pos[0]+1,pos[1])
    if action == 2:
        pos_next = (pos[0],pos[1]-1)
    if action == 3:
        pos_next = (pos[0],pos[1]+1)
    return pos_next if valid_pos(pos_next,W,shape) else pos

def update_values(policy, reward, value, W):
    a = 0.2
    d = 0.2
    value_new = np.copy(value)
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            pos = (i,j)
            if pos == (2,2):
                value_new[pos] = 1
            elif valid_pos(pos,W,value.shape):
                pos_next = move_pos(pos, policy[pos], W, policy.shape)
                value_new[pos] = value[pos] + a*(reward[pos_next] + d*value[pos_next] - value[pos])
    return value_new

def update_policy(policy, value, W):
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            if valid_pos((i,j),W,policy.shape):
                next_poss = [move_pos((i,j),k, W, policy.shape) for k in range(4)]
                next_vals = [value[pos] for pos in next_poss]
                policy[i,j] = np.argmax(next_vals)
            else:
                policy[i,j] = -1
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
            else:
                policy_new[i,j] = u"\u2022"
    return policy_new

def record_results(value, results, poss):
    for i in poss:
        results.append([])
    for i in range(len(poss)):
        results[i].append(value[poss[i]])
    return results

def plot_results(results, poss, shape):
    top_results = []
    top_poss = []
    top_amount = 3
    title=f"Value of highest {top_amount} positions"

    del results[2*shape[1] + 2]
    del poss[2*shape[1] + 2]

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
    plt.legend()
    plt.title(title)
    plt.show()


def tests():
    shape = [6,8]
    W = [(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,2),(3,5),(3,6),(4,2),(5,1)] # Walls
    policy = init_policy(shape)
    reward = init_reward(shape)
    value = init_value(shape)

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
    policy = init_converging_policy(shape, W, reward)
    value = init_value(shape)
    record_poss = add_all_indicies(shape)
    results = []

    # Main epochs for learning
    start = time.time()
    epochs = 10
    for i in range(0,epochs):
        value = update_values(policy, reward, value, W)
        policy = update_policy(policy, value, W)
        results = record_results(value, results, record_poss)



    print(f"Time taken: {time.time() - start}")
    print ("Final policy: ")
    print(convert_policy_format(policy))
    print ("Final values: ")
    print (value)
    input("Press Enter to continue...")
    plot_results(results, record_poss, shape)


main()
