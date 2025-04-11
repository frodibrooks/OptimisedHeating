import numpy as np
import matplotlib.pyplot as plt

# Polynomial approximations for each pump's efficiency curve
eff_curves = {
    '25': np.poly1d([-0.00001, 0.00014, 0.01076, 0.35117]),  # Replace with actual coefficients
    '26': np.poly1d([-0.00001, 0.00014, 0.01076, 0.35117]),
    '27': np.poly1d([-0.00250, 0.12407, -1.11199]),

    '17': np.poly1d([1.8e-07, -0.0001395, 0.02017, -0.1193]),
    '10': np.poly1d([1.8e-07, -0.0001395, 0.02017, -0.1193]),


    # Add more pumps as needed
}


# # Choose a pump
# pump_id = '25'
# curve = eff_curves[pump_id]

# # Define speed range
# speeds = np.linspace(0, 90, 100)
# efficiencies = curve(speeds)
# efficiencies = np.clip(efficiencies, 0, 1)  # Optional: Clamp to [0, 1]

# # Plot
# plt.plot(speeds, efficiencies, label=f"Pump {pump_id} Efficiency Curve")
# plt.xlabel("Speed")
# plt.ylabel("Efficiency")
# plt.title(f"Efficiency Curve for Pump {pump_id}")
# plt.grid(True)
# plt.legend()
# plt.show()