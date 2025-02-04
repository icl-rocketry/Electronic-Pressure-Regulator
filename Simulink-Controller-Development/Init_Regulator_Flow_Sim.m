%-------------------------------------------------------
%----------------- System Constants --------------------
%-------------------------------------------------------

rho_L = 1000; %Liquid propellant density [kg/m^3]
R_N2 = 296; %Gas constant for nitrogen [J/kgK]
gamma_N2 = 1.4; %Specific Heat Ratio for nitrogen
P_atm = 0; %Atmospheric pressure (gauge) [Pa]
Tst = 273.15; %Standard Temperature [K]
Pst = 101325; %Standard Pressure [Pa]
rhost = Pst/(R_N2*Tst); %Standard Nitrogen Density [kg/m^3]

V_1 = 10; %HP Tank volume [L]
V_2 = 13; %Prop tank volume [L]
A_3 = 60*1e-6; %Injector Orifice Area [m^2]
Cd_3 = 0.7; %Injector Orifice Discharge Coefficient
K_v_4 = 0.5; %Flow coefficient of the check valve

Kv_1_max = 1; %Reg Valve Maximum Kv

%-------------------------------------------------------
%--------------- Controller Parameters -----------------
%-------------------------------------------------------

K_P = 1.5; %Proportional Constant
K_I = 3; %Integral Constant
K_D = 0; %Derivative Constant
N = 100; %Filter Coefficient
Feedforward = 25; %Reg Valve Feed Forward step in degrees
Servo_Speed = 180; %Forward and Reverse Speed of the Servo [deg/s]

%-------------------------------------------------------
%-------------- Initial Conditions ---------------------
%-------------------------------------------------------

T_0_N2 = 300; %Initial temperature of nitrogen in the HP tank and ullage [K]
P_1_0 = 230; %Initial pressure of nitrogen in the HP tank [bar]
P_2_0 = 40; %Initial pressure of nitrogen ullage in prop tank [bar]
P_3_0 = 0; %Initial injector upstream pressure [bar]
P_4_0 = 0; %Initial pressure of nitrogen ullage in prop tank [bar]
m_dot_N2_0 = 0; %Initial N2 mass flow rate [kg/s]
m_dot_L_0 = 0; %Initial Propellant mass flow rate [kg/s]
m_3_0 = 8; %Initial mass of liquid propellant [kg]

Kv_2 = 1; %Set water valve opening Kv for step change

%------------------------------------------------------
%------------- Convert to SI units --------------------
%------------------------------------------------------

V_1 = V_1*1e-3; %convert volumes to m^3
V_2 = V_2*1e-3; %convert volumes to m^3
P_1_0 = P_1_0*1e5; %convert pressure to Pa
P_2_0 = P_2_0*1e5; %convert pressure to Pa
P_4_0 = P_4_0*1e5; %convert pressure to Pa
%------------------------------------------------------
%-------------- Derive calculated ICs -----------------
%------------------------------------------------------

rho_1_0 = (P_1_0)/(R_N2*T_0_N2); %Initial density of nitrogen in the HP tank [kg/m^3]
rho_2_0 = (P_2_0)/(R_N2*T_0_N2); %Initial density of nitrogen in the prop tank [kg/m^3]
rho_4_0 = (P_4_0)/(R_N2*T_0_N2); %Initial density of nitrogen in the prop tank [kg/m^3]

m_2_0 = rho_2_0*(V_2-(m_3_0/rho_L)); %Get ullage gas initial mass from density and volume
m_1_0 = V_1*rho_1_0; %Get high pressure N2 initial mass from density and volume

%------------------------------------------------------
%------- Run Simulink Model and Plot Results ----------
%------------------------------------------------------

results = sim("Regulator_Flow_Sim_1.slx");

subplot(2,3,1)
plot(results.logsout.get("P_tank [bar]").Values)
hold on
plot(results.logsout.get("P_set [bar]").Values)
xlabel('Time [s]')
ylabel('Pressure [bar]')
ylim([0,45])
hold off

subplot(2,3,2)
plot(results.logsout.get("m_3").Values)
xlabel('Time [s]')
ylabel('Water Mass [kg]')

subplot(2,3,3)
plot(results.logsout.get("m_dot_L").Values)
xlabel('Time [s]')
ylabel('Water Mass Flow Rate [kg/s]')

subplot(2,3,4)
plot(results.logsout.get("Normalised Valve Area").Values)
xlabel('Time [s]')
ylabel('Fractional Valve Area')

subplot(2,3,5)
plot(results.logsout.get("servo_demand").Values)
xlabel('Time [s]')
ylabel('Servo Demand Angle [deg]')

subplot(2,3,6)
plot(results.logsout.get("Valve angle").Values)
xlabel('Time [s]')
ylabel('Valve Angle [deg]')


