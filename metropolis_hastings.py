import numpy as np

step=0.1  #calibrate this to your liking
dstep=0.5 #fraction by which step is increased/decreased

def metropolis_hastings(num_samples, fn, *x, **param_ranges):
    """
    naive metropolis hastings with user-specified guess range
    """

    nparams=len(dict(param_ranges).keys())
    guess=get_random_samples(1, fn, **param_ranges)
    
    p = fn(*x,**guess)
    samples = [ [guess, p] ]
    for i in range(num_samples):
        new_guess={}
        for key, val in guess.items():
            step_size=step*(param_ranges[key][1]-param_ranges[key][0])
            new_guess[key] = guess[key] + np.random.normal()*step_size
        p_new = fn(*x, **new_guess)
        if p_new <= p:
            p = p_new
            guess = new_guess
            samples.insert( 0,[guess, p] )
        else:
            u = np.random.rand() #randomly accept a poor solution
            if u < p_new/p:
                p = p_new
                guess = new_guess
                samples.append( [guess, p] )

    samples=sorted(samples, key=lambda x: x[-1], reverse=False)
    param_samples=[{key:val[0] 
                    for key, val in samples[i][0].items()} 
                    for i in range(0,len(samples))]
    return np.array(param_samples), np.array([it[1] for it in samples])


def adaptive_mh(num_samples, fn, *x, **kwargs):
    """
    a simple adaptation to metropolis hastings which 
    (a) reduces the learning rate each time a beter fit is found
    and
    (b) allows the user to specify a best fit guess, 
    (c) applies _mh multiple times

    input:
    ------
    num_samples: the number of iterations, each of which is done to convergence
    fn: function to minimize (chi2, for example)
    best: specified best params.  default None
    *x: input args.  for chi2 will be *[x,y,y_error] 
    **kwargs:  
    **param_ranges: parameter input ranges
    
    """

    def _mh(fn, chi2_lim, best, *x, **param_ranges):

        nparams=len(dict(param_ranges).keys())
        if not best:
            guess=get_random_samples(1, fn, **param_ranges)
            best=guess
        else:
            guess=best
        p = fn(*x,**guess)
        p_new=p+chi2_lim+10000.  #initialize p_new to be much bigger than p
        samples = []  #needs to be empty in case best params specified.
        _step=step
        count=0
        while p_new>p+chi2_lim or count<min_steps:
            new_guess={}
            for key, val in guess.items():
                step_size=_step*(param_ranges[key][1]-param_ranges[key][0])
                new_guess[key] = guess[key] + np.random.normal()*step_size
            p_new = fn(*x, **new_guess)
            if p_new <= p:
                p = p_new
                guess = new_guess
                samples.insert( 0,[guess, p] )
                _step*=dstep  #each we find a better solution, reduce the learning rate
            else:
                _step*=1.0+dstep
                u = np.random.rand()
                if u < p_new/p:
                    p = p_new
                    guess = new_guess
                    samples.append( [guess, p] )
            count+=1
        return sorted(samples, key=lambda x: x[-1], reverse=False)


    best=kwargs.pop('best', None)
    chi2_lim=kwargs.pop('chi2_lim',4.)
    #to ensure each iteration has adequate time to converge to a local min
    min_steps=kwargs.pop('min_steps',500) 
    #at this point, all that should be left in kwargs is param ranges
    #initial run
    samples =_mh(fn, chi2_lim, None, *x, **kwargs)
    best=samples[0][0]
    for i in range(num_samples): #we run it many times because each run may only find a local min
        samples+=_mh(fn, chi2_lim, best, *x, **kwargs)
        samples=sorted(samples, key=lambda x: x[-1], reverse=False)
        best=samples[0][0]
        
    param_samples=[{key:val[0] 
                    for key, val in samples[j][0].items()} 
                    for j in range(len(samples))]
    return np.array(param_samples), np.array([it[1] for it in samples])

def get_random_samples(num_samples, fn, **param_ranges):
    param_samples={}
    for key, val in dict(param_ranges).items():
        param_samples[key]=(val[1]-val[0])*np.random.random_sample(num_samples)+val[0]
    return param_samples


def simple_sample(num_samples, fn, x,**param_ranges):
    """
    param_ranges: kwargs of param names with values [min,max]
    where min, max are allowed ranges, perhaps estimated 2 sigma limits if
    fn is a chi2 or likelihood function

    function must be defined as:
    def fn(s, **params):
        #...
        pass
    """
    param_samples=get_random_samples(num_samples, fn, **param_ranges)
    #turn param sample into list of len 1 dicts so each element maps 
    # param choices to an output value
    _param_samples=[{key:val[i] for key, val in param_samples.items()} for i in range(num_samples)]
    
    out=[]
    for i in range(num_samples):
        out.append(fn(x,**_param_samples[i]))
    return np.array(_param_samples), np.array(out)

def get_cov(param_samples):
    keys=sorted(param_samples[0].keys())
    arr=np.array([[it[key] for it in param_samples] for key in keys])
    return keys, np.cov(arr)

def chi2(fn, xx, y, ye, **params):
    model=fn(xx,**params)
    return np.sum((y-model)*(y-model)/(ye*ye))

