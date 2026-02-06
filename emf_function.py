import numpy as np
from heat_eq_solver import OneDHeatEquation

R = 8.314  # Universal gas constant (J/mol*K)
F = 96485  # Faraday's constant (C/mol)
I_1C = 2.9 # 1C current ( A )

def I_app():
    pass


def eta_ohm(eta_ohm_1C_ref, Ea_ohmic, I_app, T, T_ref):
    """
    Calculate the ohmic overpotential at a given temperature.

    Parameters:
    eta_ohm_1C_ref (float): Ohmic overpotential at 1C rate and reference temperature (V)
    Ea_ohmic (float): Activation energy for ohmic overpotential (J/mol)
    I_app (float): Applied current density (A/m^2)  
    T (float): Temperature in Kelvin
    T_ref (float): Reference temperature in Kelvin

    Returns:
    float: Ohmic overpotential at the given temperature (V)
    """
    
    # Calculate the ohmic overpotential using the Arrhenius equation
    eta_ohm1C = eta_ohm_1C_ref * np.exp((-Ea_ohmic / (8.314 * T))) / (np.exp(-Ea_ohmic / (8.314 * T_ref)))
    eta_ohm = eta_ohm1C * (I_app / I_1C)  # Scale the ohmic overpotential by the current density
    
    return eta_ohm
def j0(j0_ref, Ea_j, T, T_ref):
    
    # Calculate the activation overpotential using the Arrhenius equation
    exchange_current =  j0_ref* np.exp((-Ea_j / (8.314 * T))) / (np.exp(-Ea_j / (8.314 * T_ref)))
    
    return exchange_current
def eta_act(j0,I_app,T ):
    """
    Calculate the activation overpotential at a given current density.

    Parameters:
    j0 (float): Exchange current density (A/m^2)
    I_app (float): Applied current density (A/m^2)

    Returns:
    float: Activation overpotential at the given current density (V)
    """
    
    # Calculate the activation overpotential using the Butler-Volmer equation
    eta_act = (2*R*T/F) * np.arcsinh(I_app / (2 * j0*I_1C))
    
    return eta_act

def SOC(SOC_0, Q_cell, I_app, t, tau, num_points=100, time_step=0.01, total_time=1):
    """
    Calculate the state of charge (SOC) at a given time.

    Parameters:
    SOC_0 (float): Initial state of charge (0 to 1)
    Q_cell (float): Cell capacity (Ah)
    I_app (): Applied current (A)
    t (float): Time in seconds

    Returns:
    np.ndarray(dtype= 'float64'): SOC distribution over the particle at time t (0 to 1)
    """
    
    solver =OneDHeatEquation(
        length=1.0,
        num_points=num_points,
        alpha=1/tau,
        initial_condition=lambda : SOC_0,
        boundary_condition_type='Neumann',
        left_bc=0.0,  # zero flux
        right_bc=lambda t: -tau*I_app/(3*Q_cell),  # flux proportional to current
        solver_type='Crank-Nicolson',
        time_step=time_step,
        total_time=total_time
    )
    










