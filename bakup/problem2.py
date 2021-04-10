from .utils import *
from .graph import *
from .metrics import *
import math

def forward(graph,w,A,b):
    p,s = y_hat(graph,A,b,w)
    return p,s

def compute_loss(label,s,s_threshold=500):
    if abs(s) > s_threshold:
        return label * (-s) + (1-label) * s
    else: 
        return label * math.log(1+math.exp(-s)) + (1-label)*math.log(1+math.exp(s))

#会自动调用forward和compute_loss
def compute_gradient(graph,w,A,b,eta=0.001,D=8):
    label = graph.label
    gradient = []
    p,s = forward(graph,w,A,b)
    l_no_eta = compute_loss(label,s)
    # think w,A,b as a flattened list
    for i in range(D*D+D+1):
        # for item in w
        if i < D*D:
            w_eta = w
            w_eta[i//D][i%D] += eta
            p,s = forward(graph,w_eta,A,b)
        # for item in A
        elif D*D <= i < D*D+D:
            A_eta = A
            A_eta[i-D*D] += eta
            p,s = forward(graph,w,A_eta,b)
        elif D*D+D <= i:
            b_eta = b + eta
            p,s = forward(graph,w,A,b_eta)
        l_eta = compute_loss(label,s)
        d_loss = l_eta - l_no_eta 
        gradient.append(d_loss/eta)
    return gradient,p,s,l_no_eta

def update_theta(w,A,b,gradient,alpha=0.0001,D=8):   
    for i in range(D*D+D+1):
        # for item in w
        if i < D*D:
            w[i//D][i%D] -= alpha * gradient[i]
        # for item in A
        elif D*D <= i < D*D+D:
            A[i-D*D] -= alpha * gradient[i]
        elif D*D+D <= i:
            b -= alpha * gradient[i]
    return w,A,b

def problem2_test():
    epoch = 10
    file_index = 4
    graph = Graph(file_index)
    loss_list = []
    for i in range(epoch):
        if i == 0:
            w = create_W()
            A = create_A()
            b = create_b()
        gradient,_,_,_ = compute_gradient(graph,w,A,b)
        w,A,b = update_theta(w,A,b,gradient)
        p,s = forward(graph,w,A,b)
        loss_list.append(compute_loss(graph.label,s))
    draw_line_chart(epoch,loss_list,"p2")

if __name__ == '__main__':
    problem2_test()
