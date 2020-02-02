import numpy as np
import statistics as stat

a=np.load('STOP.npy')
b=np.load('Not_STOP.npy')

P_red=len(a)/(len(a)+len(b))
mu_red=np.array([stat.mean(a[:,0]),stat.mean(a[:,1]),stat.mean(a[:,2])]).T
cov_red=np.cov(np.transpose(a))

# print(P_red)
# print(mu_red)
# print(cov_red)
np.save('mu_red',mu_red)
np.save('cov_red', cov_red)
np.save('P_red',P_red)

P_not_red=len(b)/(len(a)+len(b))
mu_not_red=np.array([stat.mean(b[:,0]),stat.mean(b[:,1]),stat.mean(b[:,2])]).T
cov_not_red=np.cov(np.transpose(b))

# print(P_not_red)
# print(mu_not_red)
# print(cov_not_red)
np.save('mu_not_red',mu_not_red)
np.save('cov_not_red', cov_not_red)
np.save('P_not_red',P_not_red)
