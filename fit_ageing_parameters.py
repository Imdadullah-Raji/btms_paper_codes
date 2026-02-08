import numpy as np
from scipy.optimize import least_squares
from emf_function import cell_voltage, experimental_voltage, applied_current, experimental_temperature

def residual_function(params, time_points, Q_cell, SOC_0):
    """
    Calculate residuals between model and experimental voltage.
    
    Parameters:
    params: array of [j0_ref, tau0, eta_ohm_1C_ref, Ea_j, Ea_tau, Ea_ohmic]
    time_points: array of time values where voltage is measured
    Q_cell: cell capacity
    SOC_0: initial SOC
    
    Returns:
    residuals: difference between model and experimental voltage
    """
    j0_ref, tau0, eta_ohm_1C_ref, Ea_j, Ea_tau, Ea_ohmic = params
    
    residuals = np.zeros(len(time_points))
    
    for i, t in enumerate(time_points):
        V_model = cell_voltage(t, tau0, SOC_0, Q_cell, 
                              eta_ohm_1C_ref, Ea_ohmic, 
                              j0_ref, Ea_j, Ea_tau)
        V_exp = experimental_voltage(t)
        residuals[i] = V_model - V_exp
    
    return residuals

# Initial parameter guesses
params_initial = np.array([
    0.1,      # j0_ref
    1000,       # tau0
    0.05,      # eta_ohm_1C_ref
    -30000,     # Ea_j
    10000,      # Ea_tau
    10000       # Ea_ohmic
])

# Define bounds (lower, upper) for each parameter
#bounds_lower = [1e-7, 10, 0.001, 1000, 100, 100]
#bounds_upper = [1e-3, 1000, 0.5, 100000, 50000, 50000]

# Time points from your experimental data
# You'll need to get these from your load_cycle_data.csv
time_points = np.linspace(0, 5000, 10000)

# Fixed parameters
Q_cell = 2.9
SOC_0 = 0.5  # You might need to adjust this

# Run optimization
result = least_squares(
    residual_function,
    params_initial,
    args=(time_points, Q_cell, SOC_0),
    #bounds=(bounds_lower, bounds_upper),
    method='lm',  
    verbose=1,
    ftol=1e-8,
    xtol=1e-8,
    max_nfev=1000
)

# Extract optimized parameters
j0_ref_opt, tau0_opt, eta_ohm_1C_ref_opt, Ea_j_opt, Ea_tau_opt, Ea_ohmic_opt = result.x

print("\nOptimized parameters:")
print(f"j0_ref: {j0_ref_opt}")
print(f"tau0: {tau0_opt}")
print(f"eta_ohm_1C_ref: {eta_ohm_1C_ref_opt}")
print(f"Ea_j: {Ea_j_opt}")
print(f"Ea_tau: {Ea_tau_opt}")
print(f"Ea_ohmic: {Ea_ohmic_opt}")
print(f"\nFinal cost: {result.cost}")
print(f"Optimization success: {result.success}")