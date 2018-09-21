import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
x = np.array([1500,2000,2500,3500,4000,4500])
y = np.array([5.35,6.3,7.8,8.0,9,10])
gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
plt.plot([1500, 4500], [8, 8], linestyle='-', color='red', alpha=.8)
plt.plot(x,y,'ob')
plt.xlabel('Lead Actor Facebook Likes')
plt.ylabel('IMDB score')
plt.show()
