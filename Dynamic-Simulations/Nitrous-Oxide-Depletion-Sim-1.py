import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# Uses the Equilibrium model described in the following paper:
# https://web.stanford.edu/~cantwell/Selected_Publications/Liquifying%20hybrid%20fuels,%20hybrid%20rocket%20design,%20small%20thrusters,%20propulsion%20designs%20for%20Mars/Review%20and%20evaluation%20of%20models%20for%20self-pressurizing%20propellant%20tank%20dynamics%20AIAA%202013-4045.pdf

#-------------------------------------------------------
#------------------- Constants -------------------------
#-------------------------------------------------------

V_tank = 10 #Tank volume [L]
Cd = 0.7 #Injector discharge coefficient
A = 10e-6 #Injector area [m^2]
P_2 = 101325 #Atmospheric pressure [Pa]


#-------------------------------------------------------
#-------------- Initial Conditions ---------------------
#-------------------------------------------------------

T_0 = 295 #Initial N2O temperature [K]
m_tot_0 = 7 #Initial N2O total mass [kg]

#------------------------------------------------------
#------------- Convert to SI units --------------------
#------------------------------------------------------

V_tank = V_tank*1e-3

#------------------------------------------------------
#-------------- Derive calculated ICs -----------------
#------------------------------------------------------

#Function to calculate various fluid properties of the N2O based on the N2O temperature and the total mass of N2O in the tank
def Fluid_Properties(T, m_tot):
    #u_liq = PropsSI('U', 'T', T, 'Q', 0, 'N2O')
    #u_vap = PropsSI('U', 'T', T, 'Q', 1, 'N2O')
    h_liq = PropsSI('H', 'T', T, 'Q', 0, 'N2O')
    #h_vap = PropsSI('H', 'T', T, 'Q', 1, 'N2O')
    #s_liq = PropsSI('S', 'T', T, 'Q', 0, 'N2O')
    #s_vap = PropsSI('S', 'T', T, 'Q', 1, 'N2O')
    rho_liq = PropsSI('D', 'T', T, 'Q', 0, 'N2O')
    rho_vap = PropsSI('D', 'T', T, 'Q', 1, 'N2O')
    P = PropsSI('P', 'T', T, 'Q', 1, 'N2O')

    #x = PropsSI('Q', 'T', T, 'D', (m_tot/V_tank), 'N2O')
    x = ((V_tank/m_tot)*rho_liq*rho_vap - rho_vap)/(rho_liq - rho_vap) #Calculate the vapour quality

    m_vap = m_tot*x #Calculate the mass of N2O vapour using the total mass and vapour quality
    m_liq = m_tot*(1-x) #Calculate the mass of N2O liquid using the total mass and vapour quality

    liquid_level = m_liq/(V_tank*rho_liq) #Calculate the fractional liquid level using the mass of liquid, the liquid density and the tank volume

    return rho_liq, rho_vap, P, x, m_vap, m_liq, h_liq, liquid_level

#Find initial fluid properties
rho_liq_0, rho_vap_0, P_0, x_0, m_vap_0, m_liq_0, h_liq_0, liq_level_0 = Fluid_Properties(T_0, m_tot_0)

#Check that the mass of N2O isn't too large (otherwise tank would be overfilled)
if x_0 < 0:
    print('Nitrous Tank is Overfilled. Reduce starting mass.')

#Calculate the initial specific internal energy within the tank using the initial temperature and vapour quality and multiply by total mass to get the total internal energy
U_tot_0 = PropsSI('U', 'T', T_0, 'Q', x_0, 'N2O')*m_tot_0

#Function used to iteratively calculate the N2O temperature
def Temperature_calc(T, U_tot, m_tot):
    if T > 309.5: #Check that temperature is not above critical point
        T = 309.5 #If it is, limit the maximum temperature to prevent issues with coolprop

    u = U_tot/m_tot #Calculate the specific internal energy from the bulk internal energy

    #Calculate the specific internal energy and density of the liquid and vapour phases at the given temperature
    u_liq = PropsSI('U', 'T', T, 'Q', 0, 'N2O')
    u_vap = PropsSI('U', 'T', T, 'Q', 1, 'N2O')
    rho_liq = PropsSI('D', 'T', T, 'Q', 0, 'N2O')
    rho_vap = PropsSI('D', 'T', T, 'Q', 1, 'N2O')

    # Calculate the vapour quality based on the specific internal energy of the gas/liquid mixture and the specific internal energy of each phase
    x = (u - u_liq)/(u_vap - u_liq)

    # Define an equation relating the sum of volume of liquid and vapour phases to the volume of the tank (correct temperature will make these equal)
    F = V_tank - (((1-x)/rho_liq) + (x/rho_vap))*m_tot
    return F

# Define the DAE (Differential-Algebraic Equations) system
def dae_system(t, z):
    m_tot, U_tot = z  # State variables: Total mass of N2O in the tank, Total internal energy of N2O in the tank
    
    #Algebraic equations to determine other state variables
    
    # Iteratively solve for the N2O temperature using the total internal energy and total mass of N2O in the tank
    solution = root(Temperature_calc, 270, args=(U_tot, m_tot), method='lm', tol=1e-10)
    T = solution.x

    #Calculate lots of fluid properties of the N2O using the temperature and total mass
    rho_liq, rho_vap, P, x, m_vap, m_liq, h_liq, liq_level = Fluid_Properties(T, m_tot)

    #Check if the liquid has run out. If it has, set the mass flow rate to zero to finish the simulation
    if m_liq <= 0:
        m_dot_outlet = 0
    else:
        m_dot_outlet = Cd*A*np.sqrt(2*rho_liq*(P - P_2)) #Mass flow vs dP equation (REPLACE WITH NHNE OR OTHER MULTIPHASE FLOW EQUATION)

    # Differential Equations
    m_tot_dot = -m_dot_outlet # negative of mass flow out of tank is equal to the change in mass of the N2O in the tank

    U_tot_dot = -m_dot_outlet*h_liq # rate of change of internal energy is equal to negative of the mass flow out of the tank times the specific enthalpy of the liquid exiting the tank

    return [m_tot_dot, U_tot_dot]

#------------------------------------------------------
#---------- Solve Differential Equations --------------
#------------------------------------------------------

# Define initial conditions
z0 = [m_tot_0, U_tot_0]

# Define the time span for integration [s]
t_span = (0, 12)

# Generate evaluation times
t_eval = np.linspace(*t_span, 100)

# Solve the DAE
sol = solve_ivp(dae_system, t_span, z0, method='LSODA', t_eval=t_eval)

# Access the solution
t = sol.t
m_tot, U_tot = sol.y

#Define arrays of further values to dervie from results (FIGURE OUT A CLEANER WAY OF DOING THIS)
T = np.zeros(len(t_eval))
P = np.zeros(len(t_eval))
m_liq = np.zeros(len(t_eval))
m_vap = np.zeros(len(t_eval))
rho_liq = np.zeros(len(t_eval))
rho_vap = np.zeros(len(t_eval))
x = np.zeros(len(t_eval))
h_liq = np.zeros(len(t_eval))
s_liq = np.zeros(len(t_eval))
liquid_level = np.zeros(len(t_eval))

#Calculate derived parameters using algebraic equations
for i in range(0,len(t_eval)):
    solution = root(Temperature_calc, 270, args=(U_tot[i], m_tot[i]), method='hybr', tol=1e-5)
    T[i] = solution.x
    rho_liq[i], rho_vap[i], P[i], x[i], m_vap[i], m_liq[i], h_liq[i], liquid_level[i] = Fluid_Properties(T[i], m_tot[i])

#------------------------------------------------------
#------------------ Plot Results ----------------------
#------------------------------------------------------

plt.plot(t,m_liq, label='Liquid')
plt.plot(t,m_vap, label='Vapour')
plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
plt.xlabel('Time [s]')
plt.ylabel('Mass [kg]')
plt.legend()
plt.show()

plt.plot(t,P*1e-5, label='N2O Tank')
plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
plt.xlabel('Time [s]')
plt.ylabel('Pressure [bar]')
plt.legend()
plt.show()

plt.plot(t,T, label='N2O Temp.')
plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')
plt.legend()
plt.show()

plt.plot(t,U_tot, label='U_total')
plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
plt.xlabel('Time [s]')
plt.ylabel('Internal Energy [J]')
plt.legend()
plt.show()

plt.plot(t,x, label='x')
plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
plt.xlabel('Time [s]')
plt.ylabel('Vapour Quality')
plt.legend()
plt.show()

plt.plot(t,liquid_level*100, label='Liquid Level')
plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
plt.xlabel('Time [s]')
plt.ylabel('Liquid Level [%]')
plt.legend()
plt.show()