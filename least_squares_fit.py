import pandas as pd
import numpy as np

# Read csv file
df = pd.read_csv("open_circles9groups.csv")

# Store the "Distance" and "Velocity" columns in separate arrays
x_data = df["Distance (megaparsecs)"].to_numpy() # distance
y_data = df["Velocity (km/s)"].to_numpy() # velocity

# Print the arrays
print("Distance array:", x_data)
print("Velocity array:", y_data)

def least_squares_fit(x, y):
    dataLength = len(x)

    if dataLength < 2:
        return {"Error": "Not enough data"}

    # Assuming equal uncertainty for all data points
    sigma = np.ones(dataLength)

    S = 0
    s_x = 0
    s_y = 0
    s_tt = 0
    b = 0

    for i in range(dataLength):
        sigma_i = sigma[i]
        S += 1.0 / (sigma_i ** 2)
        s_x += x[i] / (sigma_i ** 2)
        s_y += y[i] / (sigma_i ** 2)

    if abs(S) < 0.000001:
        return {"Error": "Small denominator S"}

    for i in range(dataLength):
        sigma_i = sigma[i]
        t_i = 1.0 / sigma_i * (x[i] - s_x / S)
        s_tt += t_i ** 2
        b += t_i * y[i] / sigma_i

    a = (s_y - s_x * b) / S
    b = b / s_tt
    sigma_a2 = (1 + s_x ** 2 / (S * s_tt)) / S
    sigma_b2 = 1.0 / s_tt

    sigma = np.sqrt(np.sum(((y - a - b * x) / sigma) ** 2) / (dataLength - 2))
    sigma_a = np.sqrt(sigma_a2)
    sigma_b = np.sqrt(sigma_b2)

   
    # a -intercept, 
    # b -slope, 
    # sigma -total uncertanity, 
    # sigma_a -uncertanity on intercept, 
    # sigma_b -uncertanity on slope
    result = [a, b, sigma, sigma_a, sigma_b]    
    return result

print("end")

res = least_squares_fit(x_data, y_data)
print("res", res)