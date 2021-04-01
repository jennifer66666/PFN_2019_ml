import numpy as np

class Graph:
    def __init__(self,D,adjmat_filepath,w_mode):
        self.D = D
        self.n, self.adjmat = self.readin_adjmat_file(adjmat_filepath)
        self.xv_0 = self.create_x()
        self.w = self.create_W(mode = w_mode)
    
    def readin_adjmat_file(self,adjmat_filepath):
        with open(adjmat_filepath) as f:
            n = int(f.readline().strip())
            adjmat = []
            for _ in range(n):
                line = f.readline().strip().split(" ")
                line = [int(i) for i in line]
                adjmat.append(line)
        return n,adjmat
    
    def create_x(self):
        vector = [0] * self.D
        vector[0] = 1
        result = []
        for _ in range(self.n):
            result.append(vector)
        return result

    def create_W(self,mode="ones"):
        if mode == "ones":
            W = []
            for _ in range(self.D):
                W.append([1]*self.D)
        elif mode == "gaussian":
            mean, var = 0, 0.4
            W = np.random.normal(mean, var, [self.D,self.D]).tolist()
        return W

    def is_adjacent(self,v1_index,v2_index):
        if self.adjmat[v1_index][v2_index] == 1:
            return True
        else:
            return False

    def aggregate_1(self,xv_t,v_index):
        av_t = [0] * self.D
        for i in range(self.n):
            if self.is_adjacent(v_index,i):
                    av_t = sum_two_list(av_t,xv_t[i])
        return av_t

    def aggregate_2(self,av_t):
        xv_t_new = []
        for i in range(self.D):
            element = sum_dot_tow_list(self.w[i],av_t)
            element = max(0,element)
            xv_t_new.append(element)
        return xv_t_new
    
    def read_out(self,T):
        xv_t = self.xv_0
        for t in range(T):
            # every time update every v
            xv_t_new_list = []
            for v in range(self.n):
                av_t = self.aggregate_1(xv_t,v)
                xv_t_new_list.append(self.aggregate_2(av_t))
            xv_t = xv_t_new_list
        hG = [0]*self.D
        for v in range(self.n):
            hG = sum_two_list(hG,xv_t[v])
        return hG

def sum_two_list(l1,l2):
    return [p+q for p,q in zip(l1,l2)]

def sum_dot_tow_list(l1,l2):
    return sum([p*q for p,q in zip(l1,l2)])

def graph_test():
    D = 4
    adjmat_filepath = "datasets/train/0_graph.txt"
    mode = "ones"
    T = 5
    graph = Graph(D,adjmat_filepath,mode)
    assert len(graph.read_out(T)) == 4

if __name__ == '__main__':
    graph_test()