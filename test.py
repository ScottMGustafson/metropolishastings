import numpy as np
from test_data import x,y,ye
import matplotlib.pyplot as plt
from metropolis_hastings import *

#true param values
a=1.2
b=4.3
c=7.

def generate_test_data(sig=2.2,a=1.2,b=4.3,c=7.):
    xx=np.arange(-10,10,0.1)
    y=test_fn(xx,a,b,c)+np.random.normal(scale=sig, size=np.shape(xx)[0])
    ye=np.fabs(np.random.normal(scale=sig/2., size=np.shape(xx)[0]))
    return xx, y, ye

def test_fn(x,a=1.2,b=4.3,c=7.):
    return a*x**2+b*x+c

def test_simple_sample(x):
    param_samples, out =simple_sample(10000, test_fn, x, a=[1.,2.], b=[3.8,5.4], c=[5.,8.])
    keys, cov=get_cov(param_samples)
    print(keys)
    print(cov)

def test_mh():
    param_samples, out =metropolis_hastings(10000, chi2, 
                                             test_fn, x,y,ye, 
                                             a=[1.,2.], b=[3.8,5.4], c=[5.,8.])

if __name__=="__main__":
    a_,b_,c_=[],[],[]
    chi2_=[]
    test_num=list(map(int, (10.**(np.arange(0.,3.,0.1))).tolist()))
    for num in test_num:
        param_samples, out=adaptive_mh(num, chi2, test_fn,x,y,ye, 
                                       chi2_lim=4., best=None, 
                                        a=[1.,2.8], b=[3.8,5.4], c=[5.,10.])
        a_.append((param_samples[0]['a']-a)/a)
        b_.append((param_samples[0]['b']-b)/b)
        c_.append((param_samples[0]['c']-c)/c)
        chi2_.append(np.amin(out))
        print( param_samples[0]['a'], param_samples[0]['b'],param_samples[0]['c'])
        print( (param_samples[0]['a']-a)/a, (param_samples[0]['b']-b)/b,(param_samples[0]['c']-c)/c)
        print('out',out[0])
    plt.plot(test_num, chi2_,'ro')
    plt.ylabel('chi2')
    plt.xlabel('num samples')
    plt.show()
    plt.plot(test_num, np.fabs(a_),'ro')
    plt.ylabel('deviation ratio from true a')
    plt.xlabel('num samples')
    plt.show()
