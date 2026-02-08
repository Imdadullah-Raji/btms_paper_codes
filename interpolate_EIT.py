import numpy as np
import pandas as pd


data = pd.read_csv('load_cycle_data.csv')
#print(data.head())

def interpolate_current():
    current_values = data['Current'].values
    time_values = data['Time'].values
    current_values = np.array(current_values)
    time_values = np.array(time_values)
    
    # Create an interpolation function using numpy's interp
    def current_interpolation(time):
        return np.interp(time, time_values, current_values)
    
    return current_interpolation
def interpolate_temperature():
    temperature_values = data['Temperature'].values
    time_values = data['Time'].values
    temperature_values = np.array(temperature_values)
    time_values = np.array(time_values)
    
    # Create an interpolation function using numpy's interp
    def temperature_interpolation(time):
        return np.interp(time, time_values, temperature_values)
    
    return temperature_interpolation
def interpolate_voltage():
    emf_values = data['Voltage'].values
    time_values = data['Time'].values
    emf_values = np.array(emf_values)
    time_values = np.array(time_values)
    
    # Create an interpolation function using numpy's interp
    def emf_interpolation(time):
        return np.interp(time, time_values, emf_values)
    
    return emf_interpolation

if __name__ == "__main__":
    current_function = interpolate_current()
    temperature_function = interpolate_temperature()
    emf_function = interpolate_voltage()

    print(type(current_function))
    print(type(temperature_function))
    print(type(emf_function))

    # Example usage:
    time_test = 500  # Test time value in seconds
    current_test = current_function(time_test)
    temperature_test = temperature_function(time_test)
    emf_test = emf_function(time_test)

    print(f"Current at time={time_test} s: {current_test} A")
    print(f"Temperature at time={time_test} s: {temperature_test} °C")
    print(f"Voltage at time={time_test} s: {emf_test} V")