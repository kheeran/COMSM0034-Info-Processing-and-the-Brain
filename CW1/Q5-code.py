import matplotlib.pyplot as plt
import numpy as np
import math
import random

def calc_entropy(probs):
    if np.sum(probs) != 1:
        print ("WARNING: Probability distribution doesn't sum to 1, it sums to {}".format(np.sum(probs)))
    entropy = 0
    for px in probs:
        if px != 0:
            entropy -= px*math.log(px,2)
    return entropy

def trial_calculations(actual_entropy, repetitions, trials, laplace = False, alpha = 0.25):
    actual = []
    estimate = []

    # # case n = 0
    # actual.append(actual_entropy)
    # estimate.append(0.0)

    # case n > 0
    for n in range(1,trials):

        estimate_X = np.zeros((repetitions,8))

        for rep in range(0,repetitions):
            for i in range(0,n):
                estimate_X[rep][random.randint(0,7)] += 1

        # print (estimate_X)
        if laplace == True:
            estimate_X_avg = (np.average(estimate_X, axis=0)+alpha)/(n+8*alpha)
        else:
            estimate_X_avg = np.average(estimate_X, axis=0)/n

        # print (estimate_X_avg)
        # print (np.sum(estimate_X_avg))

        actual.append(actual_entropy)
        estimate.append(calc_entropy(estimate_X_avg))

    # print (actual)
    # print (estimate)

    return actual, estimate, laplace, alpha

def plot_fig(actual, estimate, laplace, alpha, location = 111):
    plt.figure(1)
    plt.subplot(location)
    plt.title("Estimate VS Actual Entropy (with {} repetition(s) per sample)".format(repetitions))
    plt.plot(actual)
    plt.plot(estimate)
    plt.legend(["Actual", "Estimate with{} Smoothing {}".format(("" if laplace else "out"), "(alpha = {})".format(alpha) if laplace else "")])
    # plt.xlim(-1,trials+1)
    plt.ylim(1.5-0.1,3.1)
    plt.xlabel("No. of Trials")
    plt.ylabel("H(X)")
    plt.show()


# MAIN
dist_X = np.full((8),1/8)
actual_entropy = calc_entropy(dist_X)
# print (actual_entropy)

repetitions = 3
trials = 100

'''
print ("Repetition(s) per Sample: {}\nNo. of Trials: {} ".format(repetitions, trials))


actual, estimate, laplace, alpha = trial_calculations(actual_entropy, repetitions, trials)
plot_fig(actual, estimate, laplace, alpha)

actual, estimate, laplace, alpha = trial_calculations(actual_entropy, repetitions, trials, laplace=True, alpha = 0.25)
plot_fig(actual, estimate, laplace, alpha)

actual, estimate, laplace, alpha = trial_calculations(actual_entropy, repetitions, trials, laplace=True, alpha = 0.5)
plot_fig(actual, estimate, laplace, alpha)

actual, estimate, laplace, alpha = trial_calculations(actual_entropy, repetitions, trials, laplace=True, alpha = 1.0)
plot_fig(actual, estimate, laplace, alpha)

actual, estimate, laplace, alpha = trial_calculations(actual_entropy, repetitions, trials, laplace=True, alpha = 2.0)
plot_fig(actual, estimate, laplace, alpha)
'''
actual, estimate, laplace, alpha = trial_calculations(actual_entropy, repetitions, trials, laplace=True, alpha = 2.0)
plot_fig(actual, estimate, laplace, alpha)
#
# X = [1/8, 7/8]
# HX = calc_entropy(X)
# Y = [9/16, 1/4, 3/16]
# HY = calc_entropy(Y)
# XY = [1/16, 1/16, 1/2, 1/4,1/8]
# HXY = calc_entropy(XY)
#
# Xg1 = [1/9, 8/9]
# HXg1 = calc_entropy(Xg1)
#
# Xg2 = [0,1]
# HXg2 = calc_entropy(Xg2)
#
# Xg3 = [1/3,2/3]
# HXg3 = calc_entropy(Xg3)
#
# HXgY = 9/16 * HXg1 + 1/4 * HXg2 + 3/16 * HXg3
#
# X = [1/8, 1/4, 5/16, 5/16]
# HX = calc_entropy(X)
#
# print (HX)
