import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

#theta_min = -50
#theta_max = 130

# Define start and end angles of rotation
theta_min = -20
theta_max = 135
d_theta = 1

#g = 21
#a = 10.17
#b = 20.78
#h = 30.63

# Define lengths of the different levers
g = 42 #Distance between pivots [mm]
a = 20 #Servo Horn Length [mm]
b = 38 #Valve Lever Length [mm]
h = 61 #Connecting linkage length [mm]

# Define the coordinate locations of fixed points (all dimensions in mm)
C = np.array([-g, 0]) #Valve rotation axis
O = np.array([0,0]) #Servo rotation axis
D = np.array([10,-40]) #Spring mounting point (for spring return)

k_spring = 0.00005 #Spring constant [N/mm]
x0_spring = 31 #Spring free length [mm]

servo_torque = 3.5 #Max. servo torque [Nm]

thetas = np.arange(theta_min, theta_max+d_theta, d_theta)

thetas = np.deg2rad(thetas)

# Function used for iteratively solving the 
def Solve_Angles(x, theta):
    phi, psi = x
    F = np.zeros(2)
    F[0] = a*np.cos(theta) - h*np.cos(phi) - b*np.cos(psi) + g
    F[1] = a*np.sin(theta) + h*np.sin(phi) - b*np.sin(psi)
    return F

def linkageForce(F_vect,r,B,D,O):
    M = np.cross(F_vect,r) #calculate moment on the joint due to the force
    F_unit = (1/np.sqrt((D[0]-B[0])**2 + (D[1]-B[1])**2))*(D - B) #calculate unit vector of actuator position
    r = D - O #calculate vector from pivot to actuator joint
    F_act = M/(np.cross(F_unit,r)) #calculate magnitude of actuator force in N
    return F_act, M/1000

phis = np.zeros(len(thetas))
psis = np.zeros(len(thetas))


for i in range(0, len(thetas)):
    solution = root(Solve_Angles, (np.pi/2,np.pi), args=(thetas[i]), method='lm', tol=1e-5)
    phis[i],psis[i] = solution.x

valve_precision = psis[1:-1] - psis[0:-2]

As = np.array([a*np.cos(thetas), a*np.sin(thetas)])
Bs = np.array([As[0]-h*np.cos(phis), As[1]+h*np.sin(phis)])

#print(np.rad2deg(psis[0]))

r_AB = Bs - As
r_AB_unit = r_AB/np.sqrt(r_AB[0]**2 + r_AB[1]**2)

F_linkage = np.zeros(len(thetas))
Valve_torque = np.zeros(len(thetas))
F_Spring = np.zeros(len(thetas))
Spring_torque = np.zeros(len(thetas))

for i in range(0,len(thetas)):
    r_OA = As[:,i] - O
    r_CB = Bs[:,i] - C

    spring_length = np.sqrt((As[0,i]-D[0])**2 + (As[1,i]-D[1])**2)
    F_Spring[i] = (spring_length - x0_spring)*k_spring
    r_DA = As[:,i] - D
    r_DA_unit = r_DA/np.sqrt(r_DA[0]**2 + r_DA[1]**2)
    Spring_torque[i] = np.cross(F_Spring[i]*r_DA_unit,r_OA)*1e-3

    F_linkage[i] = (servo_torque-Spring_torque[i])/(np.cross(r_AB_unit[:,i],r_OA))
    Valve_torque[i] = np.cross(F_linkage[i]*r_AB_unit[:,i],r_CB)





plt.plot([O[0], As[0,0]],[O[1], As[1,0]], color='r')
plt.plot([As[0,0], Bs[0,0]],[As[1,0], Bs[1,0]], color='g')
plt.plot([C[0], Bs[0,0]],[C[1], Bs[1,0]], color='b')
plt.plot([D[0], As[0,0]],[D[1], As[1,0]], color='k', linestyle='dashed')
'''
plt.scatter(As[0,0],As[1,0], color='k')
plt.scatter(Bs[0,0],Bs[1,0], color='k')
plt.scatter(C[0],C[1], color='k')
plt.scatter(O[0],O[1], color='k')
'''
N = len(thetas)-1

plt.plot([O[0], As[0,N]],[O[1], As[1,N]], color='pink')
plt.plot([As[0,N], Bs[0,N]],[As[1,N], Bs[1,N]], color='olive')
plt.plot([C[0], Bs[0,N]],[C[1], Bs[1,N]], color='cyan')
plt.plot([D[0], As[0,N]],[D[1], As[1,N]], color='k', linestyle='dashed')

plt.scatter(As[0,0],As[1,0], color='k')
plt.scatter(Bs[0,0],Bs[1,0], color='k')
plt.scatter(C[0],C[1], color='k')
plt.scatter(O[0],O[1], color='k')
plt.scatter(D[0],D[1], color='k')

plt.scatter(As[0,N],As[1,N], color='k')
plt.scatter(Bs[0,N],Bs[1,N], color='k')

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

plt.plot(np.rad2deg(psis-psis[0]), Valve_torque)
plt.xlabel('Valve Angle [deg]')
plt.ylabel('Valve Torque [Nm]')
plt.show()

plt.plot(np.rad2deg(psis-psis[0]), F_Spring)
plt.xlabel('Valve Angle [deg]')
plt.ylabel('Spring Force [N]')
plt.show()

plt.plot(np.rad2deg(psis-psis[0]), Spring_torque)
plt.xlabel('Valve Angle [deg]')
plt.ylabel('Spring Torque [Nm]')
plt.show()

