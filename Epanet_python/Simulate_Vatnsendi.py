""" Runs the hydraulic analysis of a network.

    This example contains:
      Load a network.
      Set simulation time duration.
      Hydraulic analysis using ENepanet binary file.
      Hydraulic analysis using EN functions.
      Hydraulic analysis step-by-step.
      Unload library.
"""
# Run hydraulic analysis of a network
from epyt import epanet
import time

# Load a network.

inp_file = "C:\\Users\\frodi\\Documents\\OptimisedHeating\\Epanet\\currently_working\\Vatnsendi_epanet_pumpcurve.inp"
d = epanet(inp_file)

# Set simulation time duration.
hrs = 0
d.setTimeSimulationDuration(hrs * 3600)

# Hydraulic analysis using epanet2.exe binary file.
start_1 = time.time()
hyd_res_1 = d.getComputedTimeSeries_ENepanet()
stop_1 = time.time()
hyd_res_1.disp()

tstep, P, T_H, D, H, F = 1, [], [], [], [], []
while tstep > 0:
    t = d.runHydraulicAnalysis()
    P.append(d.getNodePressure())
    D.append(d.getNodeActualDemand())
    H.append(d.getNodeHydraulicHead())
    F.append(d.getLinkFlows())
    T_H.append(t)
    tstep = d.nextHydraulicAnalysisStep()
d.closeHydraulicAnalysis()
stop_4 = time.time()

print(f'Pressure: {P}')
print(f'Demand: {D}')
print(f'Hydraulic Head {H}')
print(f'Flow {F}')

# Unload library.
d.unload()

print(f'Elapsed time for the function `getComputedTimeSeries_ENepanet` is: {stop_1 - start_1:.8f}')
