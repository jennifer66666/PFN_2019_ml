from .problem3_np import *

# pre detotes previous, i.e. t-1
def adam(theta,gradient,m_pre,v_pre,t,alpha=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
