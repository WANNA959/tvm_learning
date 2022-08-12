from matplotlib import pyplot as plt
import numpy as np


def plot_bar_time(data):
    y1_data = []
    y2_data = []
    for item in data.keys():
        y1_data.append(data[item]['build'])
        y2_data.append(data[item]['create'])

    labels = data.keys()

    plt.subplot(1, 2, 1)
    plt.title("vgg model build/create time")
    width = 0.25
    x = np.arange(len(y1_data))
    plt.bar(x-width/2, y1_data, color="yellow", width=width)
    plt.bar(x+width/2, y2_data, color="blue", width=width)
    plt.xticks(range(len(labels)), labels)
    for i in range(len(y1_data)):
        plt.text(x=i-width, y=y1_data[i] + 2, s='%d' % y1_data[i])
    for i in range(len(y2_data)):
        plt.text(x=i, y=y2_data[i] + 2, s='%d' % y2_data[i])
    plt.xlabel("vgg-n")
    plt.ylabel("times/ms")


def plot_model_time(data):

    print(data)

    plt.figure(figsize=(20, 8), dpi=80)
    plot_bar_time(data)

    x = range(0, len(data['vgg-11']['run'])-1)
    plt.suptitle("vgg model time")
    plt.subplot(1, 2, 2)
    for label in data.keys():
        y = data[label]['run']
        plt.plot(x, y[1:], label=label)

        plt.xlabel("index")
        plt.ylabel("time/ms")
        title = "vgg model run time"
        plt.title(title)
        plt.grid(alpha=0.4)

    plt.savefig("./vgg_images/vgg_run.png")
    plt.show()
