import numpy as np
import matplotlib.pyplot as plt

def draw_line_chart(epoch,loss_list):
    x = [i for i in range(epoch)]
    y = loss_list
    l=plt.plot(x,y)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("loss"+".png")