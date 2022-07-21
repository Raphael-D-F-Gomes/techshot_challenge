import numpy as np


class DesModel:
    """ Discrete event system model.
    """

    def __init__(self, d, dp, tu, tl, nt, v_loaded, v_empty, train_max_load, train_load, demand, storage, status_train,
                 tic):
        """ Construct a discrete event system model.

        Args:
            d (numpy.array): distance from port to terminal.
            tu (numpy.array): unloading time.
            tl (numpy.array): loading time.
            nt (numpy.array): train count.
            v_empty (numpy.array): empty train speed.
            v_loaded (numpy.array): loaded train speed.
            train_max_load (numpy.array): train maximum load.
            train_load (numpy.array): train load.
            demand: demand on each terminal
            storage: storage in each terminal
            status_train: status of train load
            tic: terminal initial condition
        """

        # setup
        self.verbose = True
        self.distance = d
        self.distance_ptp = dp
        self.unloading_time = tu
        self.loading_time = tl
        self.train_count = nt
        self.train_speed_loaded = v_loaded
        self.train_speed_empty = v_empty
        self.train_load = train_load
        self.train_max_load = train_max_load
        self.status_train = status_train
        self.demand = demand
        self.storage = storage
        self.from_port = [True, True]
        self.tic = tic
        nt = self.distance.shape[1]  # Number of terminals
        n_ports = self.distance.shape[0]  # Number of ports
        self.terminal_queue = [[0] for _ in range(nt)]
        self.port_queue = [[0] for _ in range(n_ports)]
        self.terminal_queue_forecast = [[0] for _ in range(nt)]
        self.port_queue_forecast = [[0] for _ in range(n_ports)]
        self.clear()

    def clear(self):
        """ Clear model to start a new simulation.
        """

        # queue
        nt = self.distance.shape[1]  # Number of terminals
        n_ports = self.distance.shape[0]  # Number of ports
        self.terminal_queue = [[0] for _ in range(nt)]
        self.port_queue = [[0] for _ in range(n_ports)]
        self.terminal_queue_forecast = [[0] for _ in range(nt)]
        self.port_queue_forecast = [[0] for _ in range(n_ports)]

    def starting_events(self, simulator):
        """ Add starting events to simulator calendar.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
        """
        data = []
        # add starting events
        tid = 0
        for im in range(len(self.train_count)):
            for i in range(self.train_count[im]):
                it = self.tic[tid][0]
                ip = self.tic[tid][1]

                # Train speed is different constidering the load
                if self.status_train[tid] == 'Loaded':
                    train_speed = self.train_speed_loaded[im]
                else:
                    train_speed = self.train_speed_empty[im]

                t = simulator.time + self.distance[ip, it] / train_speed
                data.append([ip, it, im])
                simulator.add_event(t, self.sending_to_terminal_queue, data, tid)
                tid += 1

    def sending_to_terminal(self, simulator, data, tid):
        """ Callback function for finishing unloaded path.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
            data (list): port, terminal and train model indexes.
            tid (int): train index
        """

        # debug information
        if self.verbose:
            day = (simulator.time // 3600) // 24
            hour = (simulator.time // 3600) % 24
            minute = (simulator.time % 3600) // 60
            print('{:01.0f} {:02.0f}:{:02.0f} Train {} arrived at terminal {} ({})'.format(day, hour, minute,
                                                                                           tid + 1, data[tid][1] + 1,
                                                                                           self.status_train[tid]))

        # add new event
        it = data[tid][1]  # terminal index
        im = data[tid][2]  # train model index

        # If the train is loaded, it will be unloaded in the terminal, otherwise it will be loaded
        if self.status_train[tid] == 'Empty':
            t = max(simulator.time, self.terminal_queue[it][-1]) + self.loading_time[tid, im]  # terminal loading time
            self.status_train[tid] = 'Loaded'
            self.train_load[tid][im] = min(self.train_max_load[tid][im], self.storage[it])
            self.storage[it] += - self.train_load[tid][im]

        else:
            t = max(simulator.time, self.terminal_queue[it][-1]) + self.unloading_time[tid, im]  # port unloading time
            self.status_train[tid] = 'Empty'
            self.storage[it] += self.train_load[tid][im]
            self.train_load[tid][im] = 0

        self.terminal_queue[it].append(t)
        simulator.add_event(t, self.sending_from_terminal_to_port, data, tid)

    def sending_from_terminal_to_port(self, simulator, data, tid):
        """ Callback function for finishing loading.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
            data (list): port, terminal and train model indexes.
            tid (int): train index
        """
        self.from_port[tid] = False

        # debug information
        if self.verbose:
            day = (simulator.time // 3600) // 24
            hour = (simulator.time // 3600) % 24
            minute = (simulator.time % 3600) // 60
            print('{:01.0f} {:02.0f}:{:02.0f} Train {} going from terminal {} to port {}'.format(day, hour,
                                                                                                 minute, tid + 1,
                                                                                                 data[tid][1] + 1,
                                                                                                 data[tid][0] + 1))

        # add new event
        it = data[tid][1]  # terminal index
        im = data[tid][2]  # train model index
        ip = data[tid][0]  # port index

        # Train speed is different constidering the load
        if self.status_train[tid] == 'Loaded':
            train_speed = self.train_speed_loaded[im]
        else:
            train_speed = self.train_speed_empty[im]

        t = simulator.time + self.distance[ip, it] / train_speed
        simulator.add_event(t, self.sending_to_port, data, tid)

    def sending_to_port(self, simulator, data, tid):
        """ Callback function for finishing loaded path.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
            data (list): port, terminal and train model indexes.
            tid (int): train index
        """

        # debug information
        if self.verbose:
            day = (simulator.time // 3600) // 24
            hour = (simulator.time // 3600) % 24
            minute = (simulator.time % 3600) // 60
            print('{:01.0f} {:02.0f}:{:02.0f} Train {} arrived at port {} ({})'.format(day, hour, minute, tid + 1,
                                                                                       data[tid][0] + 1,
                                                                                       self.status_train[tid]))

        # add new event
        ip = data[tid][0]  # port index
        im = data[tid][2]  # train model index

        # In this scenario, the train might go to another port to transport load or to fetch load. The train also
        # might go to a terminal to loading or unloading
        if self.status_train[tid] == 'Loaded':
            if self.from_port[tid]:
                it = self.dispatch_to_terminal(simulator.time, ip, im, tid)
                data[tid][1] = it
                t = max(simulator.time, self.port_queue[ip][-1]) + self.distance[ip, it] / self.train_speed_loaded[im]
                self.port_queue[ip].append(t)
                simulator.add_event(t, self.sending_from_port_to_terminal, data, tid)
                self.from_port[tid] = True
            else:
                self.from_port[tid] = True
                ip2 = self.dispatch_to_port(simulator.time, ip, im, tid)  # port index
                data[tid][1] = ip2
                t = max(simulator.time, self.port_queue[ip][-1])
                simulator.add_event(t, self.transporting_from_port_to_port, data, tid)

        elif self.status_train[tid] == 'Empty':
            if self.from_port[tid]:
                it = self.dispatch_to_terminal(simulator.time, ip, im, tid)
                t = max(simulator.time, self.port_queue[ip][-1]) + self.distance[ip, it] / self.train_speed_empty[im]
                simulator.add_event(t, self.sending_from_port_to_terminal, data, tid)

            else:
                self.from_port[tid] = True
                ip2 = self.dispatch_to_port(simulator.time, ip, im, tid)  # port index
                data[tid][1] = ip2
                t = max(simulator.time, self.port_queue[ip][-1])
                simulator.add_event(t, self.transporting_from_port_to_port, data, tid)

    def sending_from_port_to_terminal(self, simulator, data, tid):
        """ Callback function for finishing loading.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
            data (list): port, terminal and train model indexes.
            tid (int): train index
        """

        # debug information
        if self.verbose:
            day = (simulator.time // 3600) // 24
            hour = (simulator.time // 3600) % 24
            minute = (simulator.time % 3600) // 60
            print('{:01.0f} {:02.0f}:{:02.0f} Train {} going from port {} to terminal {}'.format(day, hour, minute,
                                                                                                 tid + 1,
                                                                                                 data[tid][0] + 1,
                                                                                                 data[tid][1] + 1))

        # add new event
        it = data[tid][1]  # terminal index
        im = data[tid][2]  # train model index
        self.from_port[tid] = False
        ip = self.dispatch_to_port(simulator.time, it, im, tid)  # terminal index
        data[tid][0] = ip
        t = simulator.time
        simulator.add_event(t, self.sending_to_terminal_queue, data, tid)

    def sending_to_terminal_queue(self, simulator, data, tid):
        """ Callback function transport between ports.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
            data (list): port, terminal and train model indexes.
            tid (int): train index
        """
        # debug information

        if self.verbose:
            day = (simulator.time // 3600) // 24
            hour = (simulator.time // 3600) % 24
            minute = (simulator.time % 3600) // 60
            print('{:01.0f} {:02.0f}:{:02.0f} Train {} arrived at '
                  'terminal {} queue ({})'.format(day, hour, minute, tid + 1, data[tid][1] + 1, self.status_train[tid]))

        it = data[tid][1]
        t = max(simulator.time, self.terminal_queue[it][-1])
        simulator.add_event(t, self.sending_to_terminal, data, tid)

    def transporting_from_port_to_port(self, simulator, data, tid):
        """ Callback function transport between ports.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
            data (list): port, terminal and train model indexes.
            tid (int): train index
        """

        self.from_port[tid] = True

        # debug information
        if self.verbose:
            day = (simulator.time // 3600) // 24
            hour = (simulator.time // 3600) % 24
            minute = (simulator.time % 3600) // 60
            if self.status_train[tid] == 'Loaded':
                s = f'transporting {self.train_load[tid][0]} kg'
            else:
                s = 'returning'

            print('{:01.0f} {:02.0f}:{:02.0f} Train {} {} from port {} to port {}'.format(day, hour, minute,
                                                                                          tid + 1, s, data[tid][0] + 1,
                                                                                          data[tid][1] + 1))

        # add new event
        ip1 = data[tid][0]  # port index
        im = data[tid][2]  # train model index
        ip2 = data[tid][1]  # port2 index
        data[tid][0] = ip2

        if self.status_train[tid] == 'Loaded':
            train_speed = self.train_speed_loaded[im]
        else:
            train_speed = self.train_speed_empty[im]

        t = simulator.time + self.distance_ptp[ip1][ip2] / train_speed

        simulator.add_event(t, self.sending_to_port, data, tid)

    def dispatch_to_terminal(self, t, ip, im, tid):
        """ Route train to terminal.

        Args:
            t (int): simulation time.
            ip (int): port index
            im (int): train model index
            tid (int): train index
        """

        # dispatch to early terminal
        tq = np.array([q[-1] for q in self.terminal_queue_forecast])

        if self.status_train[tid] == 'Loaded':
            train_speed = self.train_speed_loaded[im]
            highest_demand = [self.demand[i] - self.storage[i] if self.distance[ip, i] == 0 else -1e6
                              for i in range(len(self.distance[ip, :]))]
            it = np.argmax(highest_demand)
            tt = self.distance[ip, :] / train_speed
            tl = self.loading_time[:, im]
            tf = np.maximum(tq, t + tt) + tl

        else:
            train_speed = self.train_speed_empty[im]
            tt = self.distance[ip, :] / train_speed
            tl = self.loading_time[:, im]
            tf = np.maximum(tq, t + tt) + tl
            it = np.argmin(tf)

        self.terminal_queue_forecast[it].append(tf[it])
        return it

    def dispatch_to_port(self, t, it, im, tid):
        """ Route train to port.

        Args:
            t (int): simulation time.
            it (int): terminal index
            im (int): train model index
            tid (int): train index
        """

        # dispatch to early port
        tq = np.array([q[-1] for q in self.port_queue_forecast])
        if self.status_train[tid] == 'Loaded':
            train_speed = self.train_speed_loaded[im]
        else:
            train_speed = self.train_speed_empty[im]

        if self.from_port[tid]:
            tt = self.distance_ptp[:, it] / train_speed
            tf = np.maximum(tq, t + tt)
        else:
            tt = self.distance[:, it] / train_speed
            tu = self.unloading_time[:, im]
            tf = np.maximum(tq, t + tt) + tu

        ip = np.argmin(tf)
        self.port_queue_forecast[ip].append(tf[ip])
        return ip
