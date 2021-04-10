from .graph_np import *

def aggregate_1(graph,xv_t_matrix,v_index):
    # xv_t 13x8
    # av_t 1x8
    v_line_adj = graph.adjmat[v_index]
    return np.dot(v_line_adj , xv_t_matrix)

def aggregate_2(av_t,w):
    # w 8x8
    # result 8x1
    result = np.dot(w,av_t)
    result = np.reshape(result,[1,-1])
    return np.maximum(result, 0)

def read_out(xv_T_matrix):
    # assume xv_T_matrix = [x1_T,x2_T,x3_T...]
    return np.sum(xv_T_matrix,axis = -2)

def compute_s(A,hG,b):
    return np.dot(A,hG)+b

def compute_p(s,s_threshold = 500):
    s = np.clip(s, -s_threshold, s_threshold)
    return 1.0/(1+np.exp(-s))

def compute_y_hat(p):
    result = p > 0.5
    return result.astype(int)

def create_w(mode="gaussian",D=8):
    if mode == "gaussian":
        mean, var = 0, 0.4
        w = np.random.normal(mean, var, [D,D])
    elif mode == "zero":
        w = np.zeros([D,D])
    return w

def create_A(mode="gaussian",D=8):
    if mode == "gaussian":
        mean = 0
        var = 0.4
        A = np.random.normal(mean, var, D)
    elif mode == "zero":
        A = np.zeros(D)
    return A

def create_b():
    return np.array([0])

def problem1_test(file_index,T=2):
    graph = Graph_np(file_index) 
    xv_t_matrix = graph.xv_t
    w = create_w()
    xv_T_matrix = []
    for v_index in range(graph.n):
        av_t = aggregate_1(graph,xv_t_matrix,v_index)
        xv_t = aggregate_2(av_t,w)
        xv_T_matrix.append(xv_t)
    xv_T_matrix = np.array(xv_T_matrix).reshape([-1,8])
    assert xv_T_matrix.shape == (13,8)
    hG = read_out(xv_T_matrix)
    assert hG.shape == (8,)
    A = create_A()
    b = create_b()
    s = compute_s(A,hG,b)
    assert s.shape == (1,)
    p = compute_p(s)
    y_hat = compute_y_hat(p)

if __name__ == '__main__':
    problem1_test(1)