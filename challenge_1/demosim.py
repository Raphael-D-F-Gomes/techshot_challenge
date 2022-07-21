from des_model import DesModel
from des_simulator import DesSimulator
import numpy as np
import matplotlib.pyplot as plt

# build model
d_port_terminal = np.array([[0, 40e3], [40e3, 0]])  # port-terminal distance (m)
d_port_port = np.array([[1e7, 680e3], [680e3, 1e7]])  # port-port distance (m)
unloading_time = np.array([[6 * 3600.0], [6 * 3600.0]])  # unloading time (s)
loading_time = np.array([[7 * 3600.0], [7 * 3600.0]])  # loading time (s)
v_loaded = np.array([34 / 3.6])  # Loaded train speed (m/s)
v_empty = np.array([40 / 3.6])  # Empty train speed (m/s)
train_load = np.array([[1e3]])  # train load (Volume)
ntmax = 2  # maximum number of trains
demand = np.array([0, 3.5e3])  # Demand in each terminal
storage = np.array([10e3, 0])  # Storage in each terminal

# simulate model
T = 8 * 24 * 3600  # time horizon (s)

for i in range(1, ntmax):
    # model
    nt = np.array([i], dtype=int)  # train count of each model
    model = DesModel(d_port_terminal, d_port_port, unloading_time, loading_time, nt, v_loaded, v_empty, train_load,
                     demand, storage)

    # simulation
    simulator = DesSimulator()
    simulator.simulate(model, T)
