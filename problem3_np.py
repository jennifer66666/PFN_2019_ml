from .problem2_np import *
from .adam_np import *
import random

def shuffle_and_batches(train,batch_size):
    shuffle = random.sample(train,len(train))
    batches = []
    for i in range(int(len(train)/batch_size)):
        batches.append(shuffle[i*batch_size:(i+1)*batch_size])
    return batches

def SGD(split,epochs,batch_size,D=8,all=2000,momentum=False,adam=False):
    #init theta
    w = create_w()
    A = create_A()
    b = create_b()
    adam_t = None
    if momentum:
        # momentum is initialized as zero
        cache_w = create_w(mode="zero")
        cache_A = create_A(mode="zero")
        cache_b = create_b()
        cache = (cache_w,cache_A,cache_b)
    if adam:
        (m_pre_w,m_pre_A,m_pre_b) = (create_w(mode="zero"),create_A(mode="zero"),create_b())
        m_pre = (m_pre_w,m_pre_A,m_pre_b) 
        (v_pre_w,v_pre_A,v_pre_b) = (create_w(mode="zero"),create_A(mode="zero"),create_b())
        v_pre = (v_pre_w,v_pre_A,v_pre_b)
        # IMPORTANT: not 0
        adam_t = 1
        adam_cache = (m_pre,v_pre,adam_t)
    #split train and test
    train = [i for i in range(int(all*split))]
    validation = [i for i in range(int(all*split),all)]
    #loop epochs 
    loss_epochs_val = []
    loss_epochs_train = []
    acc_epochs_val = []
    acc_epochs_train = []
    for epoch in range(epochs):
        print("Be in epoch "+ str(epoch)+"     ")
        if momentum:
            one_epoch = loop_epoch(train,validation,batch_size,w,A,b,momentum=momentum,cache=cache)
        elif adam:
            one_epoch = loop_epoch(train,validation,batch_size,w,A,b,adam=adam,adam_cache=adam_cache)
        else:
            one_epoch = loop_epoch(train,validation,batch_size,w,A,b)
        loss_epochs_train.append(one_epoch["loss_train"].reshape(1))
        acc_epochs_train.append(one_epoch["acc_train"])
        loss_epochs_val.append(one_epoch["loss_val"].reshape(1))
        acc_epochs_val.append(one_epoch["acc_val"])
        w,A,b = one_epoch["theta"]
        cache = one_epoch["cache"]
        adam_cache = one_epoch["adam_cache"]
    draw_line_chart(epochs,loss_epochs_val,"loss_val")
    draw_line_chart(epochs,acc_epochs_val,"acc_val")
    draw_line_chart(epochs,loss_epochs_train,"loss_train")
    draw_line_chart(epochs,acc_epochs_train,"acc_train")
    return w,A,b

def loop_epoch(train,validation,batch_size,w,A,b,momentum=False,adam=False,cache=None,adam_cache=None):
    right_train = 0
    loss_train = 0
    #shuffle and batches
    batches = shuffle_and_batches(train,batch_size)
    #loop batches
    for i,batch in enumerate(batches):
        one_batch = loop_batch(w,A,b,batch,momentum=momentum,adam=adam,cache=cache,adam_cache=adam_cache)
        record_process_batch(i)
        right_train += one_batch["right_train"]
        loss_train += one_batch["loss_train"]
        w,A,b = one_batch["theta"]
        cache = one_batch["cache"]
        adam_cache = one_batch["adam_cache"]
    # evaluate when every epoch finish
    loss_val,acc_val = evaluate(w,A,b,validation)
    loss_train,acc_train = loss_train/len(train),right_train/len(train)
    return {"loss_train":loss_train,"acc_train":acc_train,"loss_val":loss_val,"acc_val":acc_val,"theta":(w,A,b),"cache":cache,"adam_cache":adam_cache}

def loop_batch(w,A,b,batch,D=8,momentum=False,adam=False,cache=None,adam_cache=None):
    right_train = 0
    loss_train = 0
    w_batch = np.zeros([D,D])
    A_batch = np.zeros(D)
    b_batch = np.array([0.]).reshape([1,1])
    #for every sample in batch
    for sample_index in batch:
        #calculate gradient list
        graph = Graph_np(sample_index)
        label = graph.label
        output = compute_gradient(graph,w,A,b)
        gradient_w,gradient_A,gradient_b = output["theta"]
        w_batch += gradient_w
        A_batch += gradient_A
        b_batch += gradient_b
        y_hat = output["y_hat"]
        if y_hat == label:
            right_train += 1
        loss_train += output["loss"]
    #average the batch's gradient
    batch_size = len(batch)
    w_batch = w_batch / batch_size
    A_batch = A_batch / batch_size
    b_batch = b_batch / batch_size
    gradient_batch = (w_batch,A_batch,b_batch)
    #update params
    if adam:
        m_pre,v_pre,adam_t = adam_cache
        adam_out = adam_optimizer((w,A,b),gradient_batch,m_pre,v_pre,adam_t)
        (w,A,b) = adam_out["theta"]
        m_pre = adam_out["m"]
        v_pre = adam_out["v"]
        adam_t += 1
        adam_cache = (m_pre,v_pre,adam_t)
    else:
        w,A,b = update_theta(w,A,b,gradient_batch,momentum=momentum,cache=cache)

    if momentum:
        cache = cache_momentum(gradient_batch,cache)
    return {"right_train":right_train,"loss_train":loss_train,"theta":(w,A,b),"cache":cache,"adam_cache":adam_cache}

def cache_momentum(gradient_batch,cache,alpha=0.0001,eta=0.9):
    gradient_w,gradient_A,gradient_b = gradient_batch
    cache_w,cache_A,cache_b = cache
    cache_w_new = -alpha*gradient_w + eta*cache_w
    cache_A_new = -alpha*gradient_A + eta*cache_A
    cache_b_new = -alpha*gradient_b + eta*cache_b
    return (cache_w_new,cache_A_new,cache_b_new)

def evaluate(w,A,b,validation):
    loss = 0
    right = 0
    for index in validation:
        graph = Graph_np(index)
        label = graph.label
        forward_result = forward(graph,w,A,b)
        y_hat = forward_result["y_hat"]
        s = forward_result["s"]
        loss += compute_loss(label,s)
        if y_hat == label:
            right += 1
    return loss/len(validation),right/len(validation)

if __name__ == '__main__':
    batch_size = 10
    epochs = 30
    split = 0.8
    SGD(split,epochs,batch_size,all=2000,adam=True)

