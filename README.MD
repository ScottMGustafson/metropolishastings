#My Little Implementation of Metropolis Hastings

For info on the algorithm see the [Wikipedia](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)

I wrote this little bit of code as a side-project.  Currently, it is written to 
minimize a function only, typically chi-square to fit a model to data.

##Usage

To run the vanilla metropolis hastings to minimize chi square (the function is 
surprisingly called `chi2`) for some function called `test_fn` with three 
parameters: `a`, `b`, and `c`, where the user has specified a guess range for 
each as keyword args.  

```python
param_samples, out =metropolis_hastings(iterations, chi2, 
                                         test_fn, x,y,ye, 
                                         a=[1.,2.], b=[3.8,5.4], c=[5.,8.])
```

Here, `iterations` is the number of iterations to take and x, y, ye are the x-values,
y-values and y-errors on the data we are using.

To run the less vanilla version, 

```python
param_samples, out=adaptive_mh(num_trials, chi2, test_fn,x,y,ye, 
                               chi2_lim=4., best=None, 
                               a=[1.,2.8], b=[3.8,5.4], c=[5.,10.])
```

Here, `num_trials`, is the number of runs to convergence, `chi2_lim` is our what we consider to be a sucessful model, namely a model such that: chi2(model)<min(chi2 of all models)+chi2_lim.  Additionally, the user can specify a guess best fit model parameters.

For both functions the output is a tuple of `param_samples` and `out` where `param_samples` is an array of dicts of the input params for the corresponding function output of `out`
