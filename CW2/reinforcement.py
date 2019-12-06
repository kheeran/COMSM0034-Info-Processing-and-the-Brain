import numpy as np
import random
import time
import matplotlib.pyplot as plt

def init_policy(reward, value, W, shape):
    policy = np.empty(shape, dtype=int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            policy[i,j] = random.randint(0,3)
            policy = update_policy((i,j), reward, policy, value, W)
    return policy


def init_reward(shape):
    reward = np.zeros(shape, dtype=int)
    reward[5,0] = 0 # Start-point
    reward[2,2] = 1 # End-point
    return reward

def init_value(shape):
    value = np.zeros(shape, dtype=float)
    return value

def init_results(poss):
    results = []
    for i in poss:
        results.append([])
    return results

def init_shortest_path():
    a = np.array([[18, 17, 16, 15, 14, 13, 12, 11],
        [19, -1000, -1000, -1000, -1000, -1000, 11, 10],
        [20, -1000, 0, 1, 2, -1000, 10, 9],
        [21, 22, -1000, 2, 3, -1000, -1000, 8],
        [22, 23, -1000, 3, 4, 5, 6, 7],
        [23, -1000, 5, 4, 5, 6, 7, 8]])
    return a


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

def update_policy(state, reward, policy, value, W, discount=0.9, epoch=100, epochs=1000, epsilon_explore=0.7, epsilon_exploit=0.1, explore_prop=1/3):
    if state == (2,2):
        policy[state] = -10000000
    elif valid_pos(state,W,policy.shape):
        next_poss = [move_pos(state,k, W, policy.shape) for k in range(4)]
        next_vals = [reward[pos] + discount*value[pos] for pos in next_poss]

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
    for i in range(len(poss)):
        results[i].append(value[poss[i]])
    return results

def plot_results(results, W, poss, shape, epochs, explore_prop,performance_check, sample_amount=3):

    # Check if the samples taken from the results is not more than the results
    sample_amount_check = True
    while sample_amount_check:
        if sample_amount*2 > len(results):
            sample_amount -= 1
            print("sample_amount reduced by 1")
        else:
            sample_amount_check = False

    top_results = []
    mean_var_top_results = []
    top_poss = []
    bottom_results = []
    mean_var_bottom_results = []
    bottom_poss = []
    title=f"Value at highest {sample_amount} states"

    # Remove end state (2,2)
    del results[2*shape[1] + 2]
    del poss[2*shape[1] + 2]

    for i in range(sample_amount):
        index_top = np.argmax(results, axis=0)[epochs-1]
        top_results.append(results[index_top])
        mean_var_top_results.append((np.mean(results[index_top][:int(explore_prop*epochs)]),np.var(results[index_top][:int(explore_prop*epochs)])))
        top_poss.append(poss[index_top])

        del results[index_top]
        del poss[index_top]

        is_wall = True
        while is_wall:
            index_bottom = np.argmin(results, axis=0)[epochs-1]
            if poss[index_bottom] in W:
                del results[index_bottom]
                del poss[index_bottom]
            else:
                is_wall = False

        bottom_results.append(results[index_bottom])
        mean_var_bottom_results.append((np.mean(results[index_bottom][:int(explore_prop*epochs)]),np.var(results[index_bottom][:int(explore_prop*epochs)])))
        bottom_poss.append(poss[index_bottom])

        del results[index_bottom]
        del poss[index_bottom]


    print_var = False
    for i in range(len(top_poss)):
        plt.plot(top_results[i], label=(f"{top_poss[i]}" + f" Top No. {i+1}" + (" #Start" if top_poss[i]==(5,0) else "  #End" if top_poss[i] == (2,2) else "") + (f" {mean_var_top_results[i]}" if print_var else "") ))
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    xmin, xmax, ymin, ymax = plt.axis()
    # plt.vlines(epochs*explore_prop, ymin, ymax, label="explore to exploit", linestyle="dotted")
    plt.legend(loc="best", framealpha=1)
    plt.title(f"Value at the highest {sample_amount} states")
    plt.draw()
    plt.waitforbuttonpress()
    plt.clf()

    plt.plot(performance_check)
    plt.title("Learning curve")
    plt.xlabel("Epochs")
    plt.ylabel("Error in shortest paths")
    plt.draw()
    plt.waitforbuttonpress()
    plt.clf()

    # for i in range(len(bottom_poss)):
    #     plt.plot(bottom_results[i], label=(f"{bottom_poss[i]}" + f" Bottom No. {i+1}" + (" #Start" if bottom_poss[i]==(5,0) else "  #End" if bottom_poss[i] == (2,2) else "") + (f" {mean_var_bottom_results[i]}" if print_var else "")))
    # plt.xlabel("Epochs")
    # plt.ylabel("Value")
    # xmin, xmax, ymin, ymax = plt.axis()
    # plt.vlines(epochs*explore_prop, ymin, ymax, label="explore to exploit", linestyle="dotted")
    # plt.legend(loc="best", framealpha=1)
    # plt.title(f"Value at the lowest {sample_amount} states")
    # plt.draw()
    # plt.waitforbuttonpress()
    # plt.clf()

    plt.close()

def tests():
    shape = [6,8]
    W = [(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,2),(3,5),(3,6),(4,2),(5,1)] # Walls
    reward = init_reward(shape)
    value = init_value(shape)
    policy = init_policy(reward, value, W, shape)

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

def performance(state, policy, W, shape):
    no_steps = 0
    while state != (2,2):
        state = move_pos(state, policy[state], W, shape)
        no_steps += 1

    return no_steps


def main():
    shape = [6,8]
    W = [(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,2),(3,5),(3,6),(4,2),(5,1)] # Walls
    reward = init_reward(shape)
    value = init_value(shape)
    policy = init_policy(reward, value, W, shape)
    record_poss = add_all_indicies(shape)
    results = init_results(record_poss)
    performance_check = []
    shortest_path = init_shortest_path()

    # Hyperparams for learning
    alpha = 0.1
    discount = 0.9
    epsilon_explore = 0.7
    epsilon_exploit = 0.3
    explore_prop = 1

    # Main epochs for learning
    start = time.time()
    epochs = 500
    for i in range(0,epochs):
        epsilon_explore = 1 - i/epochs
        if i%(int(epochs/10)) == 0:
            print (f"Epoch: {i}")
        state = (5,0)
        results = record_results(value, results, record_poss)
        while state != (2,2):
            value = update_value(state, policy, reward, value, W, alpha=alpha, discount=discount)
            policy = update_policy(state, reward, policy, value, W, discount=discount, epoch=i, epochs=epochs, epsilon_explore=epsilon_explore, epsilon_exploit=epsilon_exploit, explore_prop=explore_prop)
            state = move_pos(state, policy[state], W, shape)
        # performance check
        error = 0
        for state in record_poss:
            if valid_pos(state,W,value.shape):
                error += (performance(state, policy, W, shape) - shortest_path[state])
        performance_check.append(error)

        if i%(int(epochs/5)) == int(epochs/5)-1:
            plt.imshow(value, cmap='gray')
            plt.title(f"Estimated shortest paths to reward. Error = {error} moves. Epoch = {i+1}")
            plt.show()



    print(performance_check)
    # Final outcome
    print(f"Time taken: {time.time() - start}")
    print ("Final policy: ")
    print(convert_policy_format(policy))
    print ("Final values: ")
    print (value)
    input("Press Enter to continue...")

    # Plot results of top valued states
    plot_results(results, W, record_poss, shape, epochs, explore_prop, performance_check, sample_amount=4)
    # plt.imshow(value, cmap='gray')
    # plt.title(f"Estimated shortest paths to reward. Error = {error} moves")
    # plt.show()

main()
