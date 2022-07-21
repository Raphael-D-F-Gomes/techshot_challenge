from des_model import DesModel
from des_simulator import DesSimulator
import numpy as np

# build model (Desafio_2)
d_port_terminal = np.array([[0, 40e8, 40e8], [40e8, 0, 0]])  # port-terminal distance (m)
d_port_port = np.array([[1e9, 680e3], [680e3, 1e9]])  # port-port distance (m)
unloading_time = np.array([[6 * 3600.0], [10 * 3600.0]])  # unloading time (s)
loading_time = np.array([[7 * 3600.0], [7 * 3600.0], [7 * 3600.0]])  # loading time (s)
v_loaded = np.array([34 / 3.6])  # Loaded train speed (m/s)
v_empty = np.array([40 / 3.6])  # Empty train speed (m/s)
train_max_load = np.array([[1e3], [1e3]])  # train load (Volume)
ntmax = 3  # maximum number of trains
demand = np.array([0, 15e3, 4e3])  # Demand in each terminal
storage = np.array([20e3, 0, 0])  # Storage in each terminal
status_train = ['Loaded', 'Loaded']
train_load = np.array([[1e3], [1e3]])
terminal_initial_condition = [(1, 1), (2, 1)]
# simulate model
T = 5 * 24 * 3600  # time horizon (s)

for i in range(2, ntmax):
    # model
    nt = np.array([i], dtype=int)  # train count of each model
    model = DesModel(d_port_terminal, d_port_port, unloading_time, loading_time, nt, v_loaded, v_empty, train_max_load,
                     train_load, demand, storage, status_train, terminal_initial_condition)

    # simulation
    simulator = DesSimulator()
    simulator.simulate(model, T)
