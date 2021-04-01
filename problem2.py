from .utils import *
from .metrics import *
import math

class Learning:
    def __init__(self,fileindex):
        self.eta = 0.001
        self.alpha = 0.0001
        self.epoch = 50
        self.s_threshold = 500
        self.D = 8
        # read in a file for test
        graph_file_name = "datasets/train/"+str(fileindex)+"_graph.txt"
        self.graph = Graph(self.D,graph_file_name)
        label_file_name = "datasets/train/"+str(fileindex)+"_label.txt"
        self.label = self.readin_label_file(label_file_name)

    def readin_label_file(self,labels_filepath):
        with open(labels_filepath) as f:
            label = int(f.readline().strip())
        return label

    def forward(self,w,A,b,T=2):
        _,s = self.graph.y_hat(T,A,b,w)
        return s

    def loss(self,s):
        # when s_threshold = 100, sometimes overflow
        # so we set it 80
        if abs(s) > self.s_threshold:
            return self.label * (-s) + (1-self.label) * s
        else: 
            return self.label * math.log(1+math.exp(-s)) + (1-self.label)*math.log(1+math.exp(s))

    def backward(self,w,A,b):
        gradient = []
        l_no_eta = self.loss(self.forward(w,A,b))
        # think w,A,b as a flatten list
        for i in range(self.D*self.D+self.D+1):
            # for item in w
            if i < self.D*self.D:
                w_eta = w
                w_eta[i//self.D][i%self.D] += self.eta
                l_eta = self.loss(self.forward(w_eta,A,b))
            # for item in A
            elif self.D*self.D <= i < self.D*self.D+self.D:
                A_eta = A
                A_eta[i-self.D*self.D] += self.eta
                l_eta = self.loss(self.forward(w,A_eta,b))
            elif self.D*self.D+self.D <= i:
                b_eta = b + self.eta
                l_eta = self.loss(self.forward(w,A,b_eta))
            d_loss = l_eta - l_no_eta 
            gradient.append(d_loss/self.eta)
        
        for i in range(self.D*self.D+self.D+1):
            # for item in w
            if i < self.D*self.D:
                w[i//self.D][i%self.D] -= self.alpha * gradient[i]
            # for item in A
            elif self.D*self.D <= i < self.D*self.D+self.D:
                A[i-self.D*self.D] -= self.alpha * gradient[i]
            elif self.D*self.D+self.D <= i:
                b -= self.alpha * gradient[i]
        return w,A,b

    def __call__(self):
        loss_list = []
        for i in range(self.epoch):
            if i == 0:
                w = self.graph.create_W()
                A = self.graph.create_A()
                b = self.graph.create_b()
            w,A,b = self.backward(w,A,b)
            loss_list.append(self.loss(self.forward(w,A,b)))
        draw_line_chart(self.epoch,loss_list)


if __name__ == '__main__':
    learning_test = Learning(7)
    learning_test()
