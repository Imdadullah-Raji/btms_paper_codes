import numpy as np
from heat_eq_solver import OneDHeatEquation
from interpolate_EIT import interpolate_current, interpolate_temperature, interpolate_voltage
from interpolate_SOC_OCV import interpolate_soc_ocv, interpolate_soc_ocv_temperature_derivative

R = 8.314  # Universal gas constant (J/mol*K)
F = 96485  # Faraday's constant (C/mol)
I_1C = 2.9 # 1C current ( A )

applied_current = interpolate_current()
experimental_temperature = interpolate_temperature()
experimental_voltage = interpolate_voltage()
dEdt = interpolate_soc_ocv_temperature_derivative()



open_circuit_voltage = interpolate_soc_ocv()

def ohmic_overpotential(time, eta_ohm_1C_ref, Ea_ohmic,
                         T, I_app=applied_current , T_ref=298.15):
    """
    Calculate the ohmic overpotential at a given temperature.

    Parameters:
    eta_ohm_1C_ref (float): Ohmic overpotential at 
    1C rate and reference temperature (V)

    Ea_ohmic (float): Activation energy for ohmic overpotential (J/mol)
    I_app (callable): Applied current as a function of time (callable)
    T (float): Temperature in Kelvin
    T_ref (float): Reference temperature in Kelvin

    Returns:
    float: Ohmic overpotential at the given temperature (V)
    """
    
    # Calculate the ohmic overpotential using the Arrhenius equation
   
    eta_ohm1C = eta_ohm_1C_ref * np.exp((-Ea_ohmic / (R*T))) / (np.exp(-Ea_ohmic / (8.314 * T_ref)))
    eta_ohm = eta_ohm1C * (I_app(time) / I_1C)  # Scale the ohmic overpotential by the current density
    
    return eta_ohm
def j0(j0_ref, Ea_j, T, T_ref):
    
    # Calculate the activation overpotential using the Arrhenius equation
    exchange_current =  j0_ref* np.exp((-Ea_j / (R * T))) / (np.exp(-Ea_j / (R * T_ref)))
    
    return exchange_current
def activation_overpotential(time, j0,T,I_app= applied_current ):
    """
    Calculate the activation overpotential at a given current density.

    Parameters:
    j0 (float): Exchange current density (A/m^2)
    I_app (float): Applied current density (A/m^2)

    Returns:
    float: Activation overpotential for given j0, T at a forcing current (V)
    """
    
    # Calculate the activation overpotential using the Butler-Volmer equation
    eta_act = (2*R*T/F) * np.arcsinh(I_app(time) / (2 * j0*I_1C))
    
    return eta_act

def SOC(tau, I_app, SOC_0, Q_cell, num_points=100, time_step=0.01, total_time=1):
    """
    Calculate the state of charge (SOC) at a given time.

    Parameters:
    SOC_0 (float): Initial state of charge (0 to 1)
    Q_cell (float): Cell capacity (Ah)
    I_app (): Applied current (A)
    t (float): Time in seconds

    Returns:
    np.ndarray(dtype= 'float64'): surface and average SOC over time as two separate arrays
    """
    
    solver =OneDHeatEquation(
        length=1.0,
        num_points=num_points,
        alpha=1/tau,
        initial_condition=lambda x: np.ones_like(x) * SOC_0,
        boundary_condition_type='Neumann',
        left_bc=0.0,  # zero flux
        right_bc=lambda t: -tau*(I_app(t))/(3*Q_cell),  # flux proportional to current
        solver_type='crank-nicolson',
        time_step=time_step,
        total_time=total_time
    )

    soc_distribution = solver.solve()

    surface_soc = soc_distribution[:,-1]
    avg_soc = 3*np.mean(soc_distribution**2, axis=1)/num_points

    #final_soc_distribution = soc_distribution[-1]  # Get the SOC distribution at the final time step

    return surface_soc, avg_soc # return surface and avg soc

def diffusion_time_constant(tau0, Ea_tau, T, T_ref):
    return tau0 * np.exp((-Ea_tau / (R * T))) / (np.exp(-Ea_tau / (R * T_ref)))

def concentration_overpotential(time, tau0,  SOC_0, Q_cell, Ea_tau, 
                                I_app=applied_current, ocv=open_circuit_voltage):
    """
    
    :param time: Time in seconds
    :param tau0: Reference diffusion time constant (s)
    :param SOC_0: Initial state of charge (0 to 1)
    :param Q_cell: Battery cell capacity in Ah
    :param I_app: Applied current as a function of time (callable)
    :param ocv: Open-circuit voltage as a function of SOC (callable)

    concentration overpotential for entire time range.
    """
    tau = diffusion_time_constant(tau0, Ea_tau, 
                                  T=experimental_temperature(time), T_ref=298.15)
    surface_soc, avg_soc = SOC(tau, I_app, SOC_0, Q_cell, 
                               num_points=100, time_step=0.01,
                                 total_time=time)
    eta_conc = ocv(surface_soc) - ocv(avg_soc) 
    return eta_conc

def reversible_entropy_voltage(time, T, T_ref=298.15):
    return dEdt(time) * (T - T_ref)

def cell_voltage(time, tau0, SOC_0, Q_cell, 
                 eta_ohm_1C_ref, Ea_ohmic, 
                 j0_ref, Ea_j, Ea_tau, 
                 I_app=applied_current, ocv=open_circuit_voltage):
    """
    Calculate the cell voltage at a given time.

    Parameters:
    time (float): Time in seconds
    tau (float): Diffusion time constant (s)
    SOC_0 (float): Initial state of charge (0 to 1)
    Q_cell (float): Battery cell capacity in Ah
    eta_ohm_1C_ref (float): Ohmic overpotential at 1C rate and reference temperature (V)
    Ea_ohmic (float): Activation energy for ohmic overpotential (J/mol)
    j0_ref (float): Exchange current density at reference temperature (A/m^2)
    Ea_j (float): Activation energy for exchange current density (J/mol)

    Returns:
    float: Cell voltage at the given time (V)
    """
    
    T = experimental_temperature(time)  # Get the experimental temperature at the given time
   

    eta_ohm = ohmic_overpotential(time, eta_ohm_1C_ref, Ea_ohmic, T, I_app)  # Calculate ohmic overpotential
    j0_value = j0(j0_ref, Ea_j, T, 298.15)  # Calculate exchange current density at the given temperature
    eta_act = activation_overpotential(time, j0_value,T, I_app)  # Calculate activation overpotential
    eta_conc = concentration_overpotential(time, tau0,SOC_0,
                                            Q_cell,Ea_tau = Ea_tau,
                                              I_app = I_app, ocv = ocv)[-1]  # Calculate concentration overpotential
    entropy_voltage = reversible_entropy_voltage(time, T)  # Calculate reversible entropy voltage
    # Calculate cell voltage using OCV and overpotentials
    V_cell = ocv(SOC_0)+ entropy_voltage + eta_ohm + eta_act + eta_conc  
    
    return V_cell


if __name__ == "__main__":
    #Example parameters
    tau0 = 100  # Diffusion time constant (s) at reference temperature
    SOC_0 = 1  # Initial state of charge
    Q_cell = 2.9  # Cell capacity in Ah
    eta_ohm_1C_ref = 0.05  # Ohmic overpotential at 1C and reference temperature (V)
    Ea_ohmic = 5000  # Activation energy for ohmic overpotential (J/mol)
    j0_ref = 1e-5  # Exchange current density at reference temperature (A/m^2)
    Ea_j = -4000  # Activation energy for exchange current density (J/mol)
    Ea_tau = 5000 # Activation energy for diffusion time constant (J/mol)

    # tau0 = 10  # Diffusion time constant (s) at reference temperature
    # SOC_0 = 1.0  # Initial state of charge
    # Q_cell = 2.9  # Cell capacity in Ah
    # eta_ohm_1C_ref = 0.0511  # Ohmic overpotential at 1C and reference temperature (V)
    # Ea_ohmic = 0  # Activation energy for ohmic overpotential (J/mol)
    # j0_ref = 10  # Exchange current density at reference temperature (A/m^2)
    # Ea_j = -59041  # Activation energy for exchange current density (J/mol)
    # Ea_tau = 23619 # Activation energy for diffusion time constant (J/mol)

    time_test = 1000  # Test time value in seconds
    voltage = cell_voltage(time_test, tau0= tau0, SOC_0=SOC_0, Q_cell=Q_cell, 
                           eta_ohm_1C_ref=eta_ohm_1C_ref, Ea_ohmic=Ea_ohmic, 
                           j0_ref=j0_ref, Ea_j=Ea_j, Ea_tau=Ea_tau)
    print(f"Simulation cell voltage at time={time_test} s: {voltage} V")
    print(f"Experimental cell voltage at time={time_test} s: {experimental_voltage(time_test)} V")
#ohmic exchange tau
#24019	-59041	23619	0.051120	3.4901	8959.2	