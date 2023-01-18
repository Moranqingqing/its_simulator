
import numpy as np
import torch

def IDM(v,v0,headway,rel_speed):
    print('relspeed',rel_speed)
    # print(v,v0,headway,rel_speed)
    a=1,
    b=1.5,
    delta=4,
    s0=2,
    time_delay=0
    noise=0
    T=1
    x= v * T + v * rel_speed / 2.5
    if x<0:
        x=-0.00001*x
    print('x',x)
    s_star = s0 + x
    action=a * (1 - (v / v0)**delta - (s_star / headway)**2)
    if action<-3:
        action=-2.8
    if action>3:
        action=2.8
    return action

