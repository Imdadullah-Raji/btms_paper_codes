import numpy as np
from heat_eq_solver import OneDHeatEquation
from interpolate_SOC_OCV import interpolate_soc_ocv
from interpolate_EIT import interpolate_current, interpolate_temperature, interpolate_voltage
import matplotlib.pyplot as plt

applied_current = interpolate_current()
experimental_temperature = interpolate_temperature()
experimental_voltage = interpolate_voltage()
open_circuit_voltage = interpolate_soc_ocv()

R = 8.314 
F = 96485

def SOC(tau, I_app, SOC_0, Q_cell, num_points=100, time_step=0.01, total_time=5000):
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

    #print(soc_distribution[200:300, :])  # Print the SOC distribution for time steps 200 to 300

    surface_soc = soc_distribution[:,-1]
    avg_soc = 3*np.mean(soc_distribution**2, axis=1)/num_points

    #final_soc_distribution = soc_distribution[-1]  # Get the SOC distribution at the final time step

    return surface_soc, avg_soc # return surface and avg soc

def diffusion_time_constant(tau0, Ea_tau, T, T_ref):
    return tau0 * np.exp((-Ea_tau / R) * (1/T - 1/T_ref))

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
    #eta_conc = ocv(surface_soc) - ocv(avg_soc)
    eta_conc = ocv(surface_soc) - ocv(avg_soc)
    return eta_conc

# def get_concentration_overpotential(time_array, tau0, SOC_0, Q_cell, Ea_tau):
#     """
#     Calculate concentration overpotential for an array of time points.

#     Parameters:
#     time_array (array-like): Array of time points in seconds
#     tau0 (float): Reference diffusion time constant (s)
#     SOC_0 (float): Initial state of charge (0 to 1)
#     Q_cell (float): Battery cell capacity in Ah
#     Ea_tau (float): Activation energy for diffusion time constant (J/mol)

#     Returns:
#     np.ndarray: Array of concentration overpotentials corresponding to the input time points
#     """
#     eta_conc_array = np.zeros_like(time_array)
#     for i, t in enumerate(time_array):
#         eta_conc_array[i] = concentration_overpotential(t, tau0, SOC_0, Q_cell, Ea_tau)
#     return eta_conc_array

if __name__ == "__main__":
    time = 1000  
    time_points = np.linspace(0, time, num = int(time/0.01)+1)
    #print(time_points.shape)

    tau0 = 7000  # Example reference diffusion time constant (s)
    SOC0 = 0.9  # Example initial SOC
    Q_cell = 2.9*3600  # Example cell capacity (Ah)
    Ea_tau = 18000  # Example activation energy for diffusion time constant (J/mol)

 
    eta_conc = concentration_overpotential(time, tau0, SOC0, Q_cell, Ea_tau)

    #print(eta_conc.shape)
    print(eta_conc[-1])

    plt.plot(time_points, eta_conc)
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration Overpotential (V)')
    plt.title('Concentration Overpotential vs Time')
    plt.grid()
    plt.show()