from scipy.stats import norm
import numpy as np
import pylab
from operator import mul

x = range(100)
cdf1 = [1-norm.cdf(i,50,5) for i in x]
print max(map(mul, x, cdf1))
print map(mul, x, cdf1).index(max(map(mul, x, cdf1)))
#cdf2 = [1-norm.cdf(i,47,5) for i in x]
pylab.plot(map(mul, x, cdf1), label='cdf1')
#pylab.plot(map(mul, x, cdf2), label='cdf2')
#pylab.plot(max(map(mul, x, cdf1),map(mul,x,cdf2)), label='max')
pylab.show()
