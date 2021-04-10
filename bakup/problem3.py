import random
from .utils import *
from .metrics import *
from .problem2 import *
from .graph import *

def shuffle_and_batches(train,batch_size):
    shuffle = random.sample(train,len(train))
    batches = []
    for i in range(int(len(train)/batch_size)):
        batches.append(shuffle[i*batch_size:(i+1)*batch_size])
    return batches

def SGD(split,epochs,batch_size,D=8,all=2000):
    #init theta
    w = create_W()
    A = create_A()
    b = create_b()
    #split train and test
    train = [i for i in range(int(all*split))]
    validation = [i for i in range(int(all*split),all)]
    #loop epochs 
    loss_epochs_val = []
    loss_epochs_train = []
    acc_epochs_val = []
    acc_epochs_train = []
    for epoch in range(epochs):
        one_epoch = loop_epoch(train,validation,batch_size,w,A,b)
        loss_epochs_train.append(one_epoch["loss_train"])
        acc_epochs_train.append(one_epoch["acc_train"])
        loss_epochs_val.append(one_epoch["loss_val"])
        acc_epochs_val.append(one_epoch["acc_val"])
        w,A,b = one_epoch["theta"]
    draw_line_chart(epochs,loss_epochs_val,"loss_val")
    draw_line_chart(epochs,acc_epochs_val,"acc_val")
    draw_line_chart(epochs,loss_epochs_train,"loss_train")
    draw_line_chart(epochs,acc_epochs_train,"acc_train")
    return w,A,b

def loop_epoch(train,validation,batch_size,w,A,b):
    right_train = 0
    loss_train = 0
    #shuffle and batches
    batches = shuffle_and_batches(train,batch_size)
    #loop batches
    for batch in batches:
        one_batch = loop_batch(w,A,b,batch)
        right_train += one_batch["right_train"]
        loss_train += one_batch["loss_train"]
        w,A,b = one_batch["theta"]
    # evaluate when every epoch finish
    loss_val,acc_val = evaluate(w,A,b,validation)
    loss_train,acc_train = loss_train/len(train),right_train/len(train)
    return {"loss_train":loss_train,"acc_train":acc_train,"loss_val":loss_val,"acc_val":acc_val,"theta":(w,A,b)}

def loop_batch(w,A,b,batch,D=8):
    gradient_batch = [0]*(D*D+D+1)
    right_train = 0
    loss_train = 0
    #for every sample in batch
    for sample_index in batch:
        #calculate gradient list
        graph = Graph(sample_index)
        label = graph.label
        gradient_sample,p,s,l_no_eta = compute_gradient(graph,w,A,b)
        gradient_batch = sum_two_list(gradient_batch,gradient_sample)
        if p == label:
            right_train += 1
        print(p)
        loss_train += l_no_eta
    #average the batch's gradient
    gradient_batch = [i/batch_size for i in gradient_batch]
    #update params
    w,A,b = update_theta(w,A,b,gradient_batch)
    return {"right_train":right_train,"loss_train":loss_train,"theta":(w,A,b)}

def evaluate(w,A,b,validation):
    loss = 0
    right = 0
    for index in validation:
        graph = Graph(index)
        label = graph.label
        p,s = forward(graph,w,A,b)
        loss += compute_loss(label,s)
        if p == label:
            right += 1
    return loss/len(validation),right/len(validation)

if __name__ == '__main__':
    batch_size = 4
    epochs = 10
    split = 0.8
    SGD(split,epochs,batch_size,all=100)



#momentum SGD
#initialize w = 0 
#loop epoch 
    #shuffule and batches
    #loop batches
        #for every sample in batch
            #calculate gradient list
        #average the batch's gradient 
        #update params theta = theta - alpha*gradient_average + eta*w
        #cache: w =  - alpha*gradient_average + eta*w