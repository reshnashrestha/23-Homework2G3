import numpy as np

# Define the data points (replace with the actual data)
x_data = np.array([0.032, 0.034, 0.214, 0.263, 0.275, 0.275, 0.45, 0.5, 0.5])  # Distances
y_data = np.array([170, 290, -130, -70, -185, -220, 200, 290, 270])  # Velocities
sigma_data = np.array([20, 15, 90, 70, 100, 100, 70, 90, 85])  # Uncertainties

# Perform the least-squares fit with uncertainties
n = len(x_data)

if n < 2:
    print("Error! Not enough data!")
else:
    S = 0
    s_x = 0
    s_y = 0
    s_tt = 0
    b = 0
    chi2 = 0

    for i in range(n):
        sigma_i = sigma_data[i]
        if abs(sigma_i) < 0.00001:
            print("Error! Small uncertainty encountered.")
            break

        S += 1.0 / (sigma_i ** 2)
        s_x += x_data[i] / (sigma_i ** 2)
        s_y += y_data[i] / (sigma_i ** 2)

    if abs(S) < 0.000001:
        print("Error! Small denominator S")
    else:
        for i in range(n):
            sigma_i = sigma_data[i]
            t_i = 1.0 / sigma_i * (x_data[i] - s_x / S)
            s_tt += t_i ** 2
            b += t_i * y_data[i] / sigma_i

        a = (s_y - s_x * b) / S
        b = b / s_tt
        sigma_a2 = (1 + s_x ** 2 / (S * s_tt)) / S
        sigma_b2 = 1.0 / s_tt

        for i in range(n):
            chi2 += ((y_data[i] - a - b * x_data[i]) / sigma_data[i]) ** 2

        sigma_a = np.sqrt(sigma_a2)
        sigma_b = np.sqrt(sigma_b2)

        print("Intercept (a):", a)
        print("Slope (b):", b)
        print("Uncertainty in Intercept (sigma_a):", sigma_a)
        print("Uncertainty in Slope (sigma_b):", sigma_b)
        print("Chi-squared (chi2):", chi2)
