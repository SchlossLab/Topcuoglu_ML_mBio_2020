import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
x = np.array([1500,2000,2500,3500,4000,4500])
y = np.array([5.35,6.3,7.8,8.0,9,10])
gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
mn=np.min(x)
mx=np.max(x)
x1=np.linspace(mn,mx,500)
y1=gradient*x1+intercept
plt.plot(x,y,'ob')
plt.plot(x1,y1,'-r')
plt.xlabel('Lead Actor Facebook Likes')
plt.ylabel('IMDB score')
plt.savefig('Desktop/testplot.png', dpi=1000)
