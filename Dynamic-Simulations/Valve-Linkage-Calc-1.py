import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

theta_min = -50
theta_max = 130
d_theta = 1

g = 21
a = 10.17
b = 20.78
h = 30.63

thetas = np.arange(theta_min, theta_max+d_theta, d_theta)

thetas = np.deg2rad(thetas)

def Solve_Angles(x, theta):
    phi, psi = x
    F = np.zeros(2)
    F[0] = a*np.cos(theta) - h*np.cos(phi) - b*np.cos(psi) + g
    F[1] = a*np.sin(theta) + h*np.sin(phi) - b*np.sin(psi)
    return F

phis = np.zeros(len(thetas))
psis = np.zeros(len(thetas))

for i in range(0, len(thetas)):
    solution = root(Solve_Angles, (np.pi/2,np.pi), args=(thetas[i]), method='lm', tol=1e-5)
    phis[i],psis[i] = solution.x

valve_precision = psis[1:-1] - psis[0:-2]

As = np.array([a*np.cos(thetas), a*np.sin(thetas)])
Bs = np.array([As[0]-h*np.cos(phis), As[1]+h*np.sin(phis)])
C = np.array([-g, 0])
O = np.array([0,0])


plt.plot([O[0], As[0,0]],[O[1], As[1,0]], color='r')
plt.plot([As[0,0], Bs[0,0]],[As[1,0], Bs[1,0]], color='g')
plt.plot([C[0], Bs[0,0]],[C[1], Bs[1,0]], color='b')

plt.scatter(As[0,0],As[1,0], color='k')
plt.scatter(Bs[0,0],Bs[1,0], color='k')
plt.scatter(C[0],C[1], color='k')
plt.scatter(O[0],O[1], color='k')

N = len(thetas)-1

plt.plot([O[0], As[0,N]],[O[1], As[1,N]], color='pink')
plt.plot([As[0,N], Bs[0,N]],[As[1,N], Bs[1,N]], color='olive')
plt.plot([C[0], Bs[0,N]],[C[1], Bs[1,N]], color='cyan')

plt.scatter(As[0,0],As[1,0], color='k')
plt.scatter(Bs[0,0],Bs[1,0], color='k')
plt.scatter(C[0],C[1], color='k')
plt.scatter(O[0],O[1], color='k')

plt.scatter(As[0,N],As[1,N], color='k')
plt.scatter(Bs[0,N],Bs[1,N], color='k')
plt.scatter(C[0],C[1], color='k')
plt.scatter(O[0],O[1], color='k')

#set axis aspect ratio
axes=plt.gca()
axes.set_aspect('equal','box')

plt.show()

plt.plot(np.rad2deg(thetas)-theta_min, np.rad2deg(psis-psis[0]))
plt.xlabel('Servo Angle [deg]')
plt.ylabel('Valve Angle [deg]')
plt.show()

plt.plot(np.rad2deg(thetas[0:-2])-theta_min, np.rad2deg(valve_precision))
plt.xlabel('Servo Angle [deg]')
plt.ylabel('Valve Precision [deg]')
plt.show()