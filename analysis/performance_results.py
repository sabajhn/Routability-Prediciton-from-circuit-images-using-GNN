import matplotlib.pyplot as plt
import numpy as np
def plot_results(train, test):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    y1 = [[] for i in range(4)]
    y2 = [[] for i in range(4)]

    for i in range(len(train)):
        acc = list(train[i])
        for j in range(4):
            y1[j].append(acc[j+1])

    for i in range(len(test)):
        acc = list(test[i])
        for j in range(4):
            y2[j].append(acc[j+1])


    n = len(y1[0])
    x = [i for i in range(0,n)]
    labels = ["acc", "Prec", "Rec", "Spec"]
    for j in range(4):
        ax[0].plot(x, y1[j],label=labels[j])
        ax[1].plot(x, y2[j],label=labels[j])
        
    plt.legend()
    plt.show()


