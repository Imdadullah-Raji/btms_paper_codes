import numpy as np
import pandas as pd

df1 = pd.read_csv('soc_ocv_data.csv') 
df2 = pd.read_csv('soc_ocv_temperature_derivative_data.csv')
#print(df1.head())
#print(df2.head())
#df2.columns = df2.columns.str.strip()

def interpolate_soc_ocv(soc_values= df1['SOC'].values, 
                        ocv_values= df1['OCV'].values):
    """
    Create an interpolation function for SOC vs OCV.

    Parameters:
    soc_values (array-like): Array of state of charge (SOC) values (0 to 1)
    ocv_values (array-like): Array of corresponding open-circuit voltage (OCV) values (V)

    Returns:
    function: A function that takes SOC as input and returns the corresponding OCV.
    """
    
    # Ensure the input arrays are numpy arrays
    soc_values = np.array(soc_values)
    ocv_values = np.array(ocv_values)
    
    # Create an interpolation function using numpy's interp
    def ocv_interpolation(soc):
        return np.interp(soc, soc_values, ocv_values)
    
    return ocv_interpolation
def interpolate_soc_ocv_temperature_derivative(soc_values= df2['SOC'].values,
                                                dEdT_values= df2['dEdt'].values):
    """
    Create an interpolation function for SOC vs temperature derivative of OCV.

    Parameters:
    soc_values (array-like): Array of state of charge (SOC) values (0 to 1)
    temperature_derivative_values (array-like): Array of corresponding temperature derivative of OCV values (V/K)

    Returns:
    function: A function that takes SOC as input and returns the corresponding temperature derivative of OCV.
    """
    
    # Ensure the input arrays are numpy arrays
    soc_values = np.array(soc_values)
    temperature_derivative_values = np.array(dEdT_values)
    
    # Create an interpolation function using numpy's interp
    def temperature_derivative_interpolation(soc):
        return np.interp(soc, soc_values, temperature_derivative_values)
    
    return temperature_derivative_interpolation

if __name__ == "__main__":
    data = pd.read_csv('soc_ocv_data.csv')  # Assuming the CSV file has columns 'SOC' and 'OCV'
    print(data.head())

    soc_values = data['SOC'].values
    ocv_values = data['OCV'].values

    ocv_function = interpolate_soc_ocv(soc_values, ocv_values)
    # Example usage:
    soc_test = 0.523  # Test SOC value
    ocv_test = ocv_function(soc_test)
    print(f"OCV at SOC={soc_test}: {ocv_test} V")

    data_dEdT = pd.read_csv('soc_ocv_temperature_derivative_data.csv')  # Assuming the CSV file has columns 'SOC' and 'dEdT'
    print(data_dEdT.head())
    data_dEdT.columns = data_dEdT.columns.str.strip()  # Remove any leading/trailing whitespace from column names
    #print(data_dEdT.columns)
    soc_values_dEdT = data_dEdT['SOC'].values
    dEdT_values = data_dEdT['dEdt'].values
    dEdT_function = interpolate_soc_ocv_temperature_derivative(soc_values_dEdT, dEdT_values)
    # Example usage:        
    dEdT_test = dEdT_function(soc_test)
    print(f"dEdT at SOC={soc_test}: {dEdT_test} mV/K")

