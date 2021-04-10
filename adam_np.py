from .problem2_np import *

# pre detotes previous, i.e. t-1
def adam_optimizer(theta,gradient,m_pre,v_pre,t,alpha=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
    gradient_w,gradient_A,gradient_b = gradient
    theta_w,theta_A,theta_b = theta
    m_pre_w,m_pre_A,m_pre_b = m_pre
    v_pre_w,v_pre_A,v_pre_b = v_pre
    m_w = beta1*m_pre_w + (1-beta1)*gradient_w
    m_A = beta1*m_pre_A + (1-beta1)*gradient_A
    m_b = beta1*m_pre_b + (1-beta1)*gradient_b
    m = (m_w,m_A,m_b)
    v_w = beta2*v_pre_w + (1-beta2)*(gradient_w**2)
    v_A = beta2*v_pre_A + (1-beta2)*(gradient_A**2)
    v_b = beta2*v_pre_b + (1-beta2)*(gradient_b**2)
    v = (v_w,v_A,v_b)
    m_w_hat = m_w / (1.-beta1**t)
    m_A_hat = m_A / (1.-beta1**t)
    m_b_hat = m_b / (1.-beta1**t)
    v_w_hat = v_w / (1.-beta2**t)
    v_A_hat = v_A / (1.-beta2**t)
    v_b_hat = v_b / (1.-beta2**t)
    theta_w = theta_w - alpha*m_w_hat/(np.sqrt(v_w_hat)+epsilon)
    theta_A = theta_A - alpha*m_A_hat/(np.sqrt(v_A_hat)+epsilon)
    theta_b = theta_b - alpha*m_b_hat/(np.sqrt(v_b_hat)+epsilon)
    theta = (theta_w,theta_A,theta_b)
    return {"m":m,"v":v,"theta":theta}