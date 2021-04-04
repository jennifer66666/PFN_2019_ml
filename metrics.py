import numpy as np
import matplotlib.pyplot as plt

def draw_line_chart(epoch,result_list,title):
    x = [i for i in range(epoch)]
    y = result_list
    plt.figure()
    l=plt.plot(x,y)
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.savefig(title+".png")