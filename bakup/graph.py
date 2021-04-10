import numpy as np
import math

class Graph:
    def __init__(self,file_index):
        graph_file_name = "datasets/train/"+str(file_index)+"_graph.txt"
        label_file_name = "datasets/train/"+str(file_index)+"_label.txt"
        self.n, self.adjmat = self.readin_adjmat_file(graph_file_name )
        self.label = self.readin_label_file(label_file_name)
        self.xv_0 = self.create_x()
    
    def readin_adjmat_file(self,adjmat_filepath):
        with open(adjmat_filepath) as f:
            n = int(f.readline().strip())
            adjmat = []
            for _ in range(n):
                line = f.readline().strip().split(" ")
                line = [int(i) for i in line]
                adjmat.append(line)
        return n,adjmat

    def readin_label_file(self,labels_filepath):
        with open(labels_filepath) as f:
            label = int(f.readline().strip())
        return label
    
    def create_x(self,D=8):
        #vector = [0] * D
        #vector[0] = 1
        # [vector,vector...] is shallow copy
        result = []
        for _ in range(self.n):
            result.append([1,0,0,0,0,0,0,0])
        return result

    def is_adjacent(self,v1_index,v2_index):
        if self.adjmat[v1_index][v2_index] == 1:
            return True
        else:
            return False

if __name__ == '__main__':
    graph = Graph(1)
    print(graph.n)
    print(graph.adjmat)
    print(graph.is_adjacent(0,3))
    print(graph.is_adjacent(3,1))