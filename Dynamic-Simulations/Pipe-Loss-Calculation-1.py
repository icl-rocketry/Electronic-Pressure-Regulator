import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

m_dot = 0.2 #Gas mass flow rate [kg/s]
P_1 = 40e5 #Gas Pressure [Pa]
T_1 = 250 #Gas Temp [K]

D = 7e-3 #Pipe Diameter [m]
L = 0.3 #Pipe Length [m]
k = 0.015e-3 #Pipe Roughness [m]

mu = PropsSI('V','T',T_1,'P',P_1,'Nitrogen')
rho = PropsSI('D','T',T_1,'P',P_1,'Nitrogen')

U = m_dot/(rho*(D**2)*np.pi/4)

Re_D = rho*U*D/mu

f = 0.25/(np.log10(k/(3.7*D) + 5.74/(Re_D**0.9))**2)

print('Mean Gas Velocity [m/s]: ',U)

print('N2 Density [kg/m^3]: ',rho)

print('N2 Viscosity: ', mu)

print('Darcy Friction Factor:',f)

dP = (m_dot**2)*8*f*L/(rho*(np.pi**2)*D**5)

print('Pressure Drop [bar]: ',dP*1e-5)
