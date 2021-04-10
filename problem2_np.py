from .utils_np import *
from .metrics import *

def forward(graph,w,A,b,T=2,updating_x=False):
    xv_t_matrix = graph.xv_t_matrix
    for t in range(T):
        xv_t_matrix_new = []
        for v_index in range(graph.n):
            av_t = aggregate_1(graph,xv_t_matrix,v_index)
            xv_t_new = aggregate_2(av_t,w)
            xv_t_matrix_new.append(xv_t_new)
        xv_t_matrix_new = np.array(xv_t_matrix_new).reshape([-1,8])
        xv_t_matrix = xv_t_matrix_new    
    if updating_x:
        # aggregate xv_t should NOT give it back to the graph
        # NEVER UPDATING X
        graph.xv_t_matrix = xv_t_matrix_new
    hG = read_out(xv_t_matrix_new)
    s = compute_s(A,hG,b)
    p = compute_p(s)
    y_hat = compute_y_hat(p)
    return {"s":s,"p":p,"y_hat":y_hat}

def compute_loss(label,s,s_threshold=500):
    if abs(s) > s_threshold:
        return label * (-s) + (1-label) * s
    else: 
        return label * math.log(1+math.exp(-s)) + (1-label)*math.log(1+math.exp(s))

# pert refers to perturbation
def compute_gradient(graph,w,A,b,epsilon=0.001,D=8):
    label = graph.label
    forward_result = forward(graph,w,A,b)
    p = forward_result["p"]
    s = forward_result["s"]
    y_hat = forward_result["y_hat"]
    loss_ori = compute_loss(label,s).reshape([1,1])
    gradient = []
    for pert_index in range(D*D+D+1):
        if pert_index < D*D:
            # w
            w_perted = w + epsilon * pert_mask(pert_index)
            s_perted = forward(graph,w_perted,A,b)["s"]
        elif D*D <= pert_index < D*D+D:
            # A 
            A_perted = A + epsilon * pert_mask(pert_index)
            s_perted = forward(graph,w,A_perted,b)["s"]
        elif D*D+D <= pert_index:
            # b
            b_perted = b + epsilon * pert_mask(pert_index)
            s_perted = forward(graph,w,A,b_perted)["s"]
        loss_perted = compute_loss(label,s_perted)
        gradient.append((loss_perted - loss_ori)/epsilon)
    gradient_w = np.array(gradient[:D*D]).reshape([D,D])
    gradient_A = np.array(gradient[D*D:D*D+D]).reshape(D)
    gradient_b = np.array(gradient[D*D+D:]).reshape([1,1])
    return {"theta":(gradient_w,gradient_A,gradient_b),"loss":loss_ori,"p":p,"y_hat":y_hat}

def pert_mask(pert_index,D=8):
    if pert_index < D*D:
        mask = np.zeros([D,D])
        mask[pert_index//D][pert_index%D] = 1
    elif D*D <= pert_index < D*D+D:
        mask = np.zeros(D)
        mask[pert_index - D*D] = 1
    else:
        mask = np.array([1])
    return mask

def update_theta(w,A,b,gradient,alpha=0.0001,momentum=False,cache=None,eta=0.9):   
    gradient_w,gradient_A,gradient_b = gradient
    if momentum:
        cache_w,cache_A,cache_b = cache
        w = w - alpha*gradient_w + eta*cache_w
        A = A - alpha*gradient_A + eta*cache_A
        b = b - alpha*gradient_b + eta*cache_b
    else:
        w = w - alpha*gradient_w
        A = A - alpha*gradient_A
        b = b - alpha*gradient_b
    return w,A,b

def problem2_test():
    epoch = 100
    file_index = 1
    graph = Graph_np(file_index)
    loss_list = []
    for i in range(epoch):
        if i == 0:
            w = create_w()
            A = create_A()
            b = create_b()
        gradient = compute_gradient(graph,w,A,b)
        w,A,b = update_theta(w,A,b,gradient)
        # not sure whether need to update x
        step = forward(graph,w,A,b,updating_x=False)
        p = step["p"]
        s = step["s"]
        loss_list.append(compute_loss(graph.label,s))
    draw_line_chart(epoch,loss_list,"p2")

if __name__ == '__main__':
    problem2_test()
    
