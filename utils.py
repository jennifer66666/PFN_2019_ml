from .graph import *

def aggregate_1(graph,xv_t,v_index,D=8):
    n = graph.n
    av_t = [0]*D
    for i in range(n):
        if graph.is_adjacent(v_index,i):
                av_t = sum_two_list(av_t,xv_t[i])
    return av_t

def aggregate_2(av_t,w,D=8):
    xv_t_new = []
    for i in range(D):
        element = sum_dot_two_list(w[i],av_t)
        element = max(0,element)
        xv_t_new.append(element)
    return xv_t_new


def read_out(xv_T_matrix):
    # assume xv_T_matrix = [x1_T,x2_T,x3_T...]
    return np.sum(xv_T_matrix,axis = -2)

def compute_s(A,hG,b):
    return np.dot(A,hG)+b

def compute_p(s,s_threshold = 500):
    s = np.clip(s, -s_threshold, s_threshold)
    return 1.0/(1+np.exp(-s))

def compute_y_hat(p):
    

def y_hat(graph,A,b,w):
    hG = read_out(graph,w)
    s = sum_dot_two_list(A,hG) + b
    p = 1/(1+math.exp(-clip(s,-500,500)))
    if p > 0.5:
        return 1,s
    else:
        return 0,s

def sum_two_list(l1,l2):
    return [p+q for p,q in zip(l1,l2)]

def sum_dot_two_list(l1,l2):
    return sum([p*q for p,q in zip(l1,l2)])

def clip(x,left,right):
    if x>right:
        return right
    elif x<left:
        return left
    else:
        return x

def create_W(D=8,mode="gaussian"):
    if mode == "ones":
        W = []
        for _ in range(D):
            W.append([1]*D)
    elif mode == "gaussian":
        mean, var = 0, 0.4
        W = np.random.normal(mean, var, [D,D]).tolist()
    return W

def create_A(D=8):
    mean = 0
    var = 0.4
    return np.random.normal(mean, var, D).tolist()

def create_b():
        return 0

def problem1_test():
    file_index = 1
    graph = Graph(file_index)
    w = create_W()
    A = create_A()
    b = create_b()
    assert len(read_out(graph,w)) == 8
    y_h,_ = y_hat(graph,A,b,w)
    assert y_h == 0 or 1

if __name__ == '__main__':
    problem1_test()