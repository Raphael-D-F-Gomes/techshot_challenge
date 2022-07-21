import numpy as np


class DesModel:
    ''' Discrete event system model.
    '''

    def __init__(self, d, dp, tu, tl, nt, v_loaded, v_empty, L, demand, storage):
        ''' Construct a discrete event system model.

        Args:
            d (numpy.array): distance from port to terminal.
            tu (numpy.array): unloading time.
            tl (numpy.array): loading time.
            nt (numpy.array): train count.
            v_empty (numpy.array): empty train speed.
            v_loaded (numpy.array): loaded train speed.
            L (numpy.array): train load.
            demand: demand on each terminal
            storage: storage in each terminal
        '''

        # setup
        self.verbose = True
        self.distance = d
        self.distance_ptp = dp
        self.unloading_time = tu
        self.loading_time = tl
        self.train_count = nt
        self.train_speed_loaded = v_loaded
        self.train_speed_empty = v_empty
        self.train_load = [[0]]
        self.train_max_load = L
        self.status_train = 'Empty'
        self.demand = demand
        self.storage = storage
        self.from_port = True
        self.clear()

    def clear(self):
        ''' Clear model to start a new simulation.
        '''

        # queue
        nt = self.distance.shape[1]  # Number of terminals
        np = self.distance.shape[0]  # Number of ports
        self.terminal_queue = [[0] for _ in range(nt)]
        self.port_queue = [[0] for _ in range(np)]
        self.terminal_queue_forecast = [[0] for _ in range(nt)]
        self.port_queue_forecast = [[0] for _ in range(np)]

    def starting_events(self, simulator):
        ''' Add starting events to simulator calendar.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
        '''

        # add starting events
        id = 0
        for im in range(len(self.train_count)):
            for i in range(self.train_count[im]):
                ip = 0
                it = self.dispatch_to_terminal(simulator.time, ip, im)
                t = simulator.time + self.distance[ip, it] / self.train_speed_empty[im]
                data = [ip, it, im, id]
                simulator.add_event(t, self.sending_to_terminal, data)
                id += 1

    def sending_to_terminal(self, simulator, data):
        ''' Callback function for finishing unloaded path.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
            data (list): port, terminal and train model indexes.
        '''

        # debug information
        if self.verbose:
            day = (simulator.time // 3600) // 24
            hour = (simulator.time // 3600) % 24
            minute = (simulator.time % 3600) // 60
            print('{:01.0f} {:02.0f}:{:02.0f} Train {} arrived at terminal {} ({})'.format(day, hour, minute,
                                                                                           data[3] + 1, data[1] + 1,
                                                                                           self.status_train))

        # add new event
        it = data[1]  # terminal index
        im = data[2]  # train model index
        ip = data[0]  # Port index

        # If the train is loaded, it will be unloaded in the terminal, otherwise it will be loaded
        if self.status_train == 'Empty':
            t = max(simulator.time, self.terminal_queue[it][-1]) + self.loading_time[it, im]  # terminal loading time
            self.status_train = 'Loaded'
            self.train_load[im][0] = min(self.train_max_load[im][0], self.storage[it])
            self.storage[it] += - self.train_load[im][0]
        elif self.status_train == 'Loaded':
            t = max(simulator.time, self.terminal_queue[it][-1]) + self.unloading_time[it, im]  # port unloading time
            self.status_train = 'Empty'
            self.storage[it] += self.train_load[im][0]
            self.train_load[im][0] = 0

        self.terminal_queue[it].append(t)
        simulator.add_event(t, self.sending_from_terminal_to_port, data)

    def sending_from_terminal_to_port(self, simulator, data):
        ''' Callback function for finishing loading.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
            data (list): port, terminal and train model indexes.
        '''
        self.from_port = False

        # debug information
        if self.verbose:
            day = (simulator.time // 3600) // 24
            hour = (simulator.time // 3600) % 24
            minute = (simulator.time % 3600) // 60
            print('{:01.0f} {:02.0f}:{:02.0f} Train {} going from terminal {} to port {}'.format(day, hour,
                                                                                                 minute, data[3] + 1,
                                                                                                 data[1] + 1,
                                                                                                 data[0] + 1))

        # add new event
        it = data[1]  # terminal index
        im = data[2]  # train model index
        ip = self.dispatch_to_port(simulator.time, it, im)  # port index
        data[0] = ip

        # Train speed is different considering the load
        if self.status_train == 'Loaded':
            train_speed = self.train_speed_loaded[im]
        elif self.status_train == 'Empty':
            train_speed = self.train_speed_empty[im]

        t = simulator.time + self.distance[ip, it] / train_speed
        simulator.add_event(t, self.sending_to_port, data)

    def sending_to_port(self, simulator, data):
        ''' Callback function for finishing loaded path.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
            data (list): port, terminal and train model indexes.
        '''

        # debug information
        if self.verbose:
            day = (simulator.time // 3600) // 24
            hour = (simulator.time // 3600) % 24
            minute = (simulator.time % 3600) // 60
            print('{:01.0f} {:02.0f}:{:02.0f} Train {} arrived at port {} ({})'.format(day, hour, minute, data[3] + 1,
                                                                                   data[0] + 1, self.status_train))

        # add new event
        ip = data[0]  # port index
        im = data[2]  # train model index

        # I this scenario, the train might go to another port to transport load or to fetch load. The train also
        # might go to a terminal to loading or unloading
        if self.status_train == 'Loaded':
            if self.from_port:
                it = self.dispatch_to_terminal(simulator.time, ip, im)
                data[1] = it
                t = max(simulator.time, self.port_queue[ip][-1]) + self.distance[ip, it] / self.train_speed_loaded[im]
                self.port_queue[ip].append(t)
                simulator.add_event(t, self.sending_from_port_to_terminal, data)
                self.from_port = True
            else:
                ip2 = self.dispatch_to_port(simulator.time, ip, im)  # port index
                data[1] = ip2
                t = max(simulator.time, self.port_queue[ip][-1])
                simulator.add_event(t, self.transporting_from_port_to_port, data)

        elif self.status_train == 'Empty':
            if self.from_port:
                it = self.dispatch_to_terminal(simulator.time, ip, im)
                t = max(simulator.time, self.port_queue[ip][-1]) + self.distance[ip, it] / self.train_speed_empty[im]
                # data = [ip, it, im, id]
                simulator.add_event(t, self.sending_from_port_to_terminal, data)
                self.from_port = True
            else:
                ip2 = self.dispatch_to_port(simulator.time, ip, im)  # port index
                data[1] = ip2
                t = max(simulator.time, self.port_queue[ip][-1])
                simulator.add_event(t, self.transporting_from_port_to_port, data)

    def sending_from_port_to_terminal(self, simulator, data):
        ''' Callback function for finishing loading.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
            data (list): port, terminal and train model indexes.
        '''

        # debug information
        if self.verbose:
            day = (simulator.time // 3600) // 24
            hour = (simulator.time // 3600) % 24
            minute = (simulator.time % 3600) // 60
            print('{:01.0f} {:02.0f}:{:02.0f} Train {} going from port {} to terminal {}'.format(day, hour, minute,
                                                                                                 data[3] + 1,
                                                                                                 data[0] + 1,
                                                                                                 data[1] + 1))

        # add new event
        it = data[1]  # terminal index
        im = data[2]  # train model index

        self.from_port = False
        ip = self.dispatch_to_port(simulator.time, it, im)  # terminal index
        data[0] = ip

        if self.status_train == 'Loaded':
            train_speed = self.train_speed_loaded[im]
        elif self.status_train == 'Empty':
            train_speed = self.train_speed_empty[im]

        t = simulator.time
        simulator.add_event(t, self.sending_to_terminal, data)

    def transporting_from_port_to_port(self, simulator, data):
        ''' Callback function transport between ports.

        Args:
            simulator (:obj:DES_simulator): DES simulator.
            data (list): port, terminal and train model indexes.
        '''
        self.from_port = True
        # debug information
        if self.verbose:
            day = (simulator.time // 3600) // 24
            hour = (simulator.time // 3600) % 24
            minute = (simulator.time % 3600) // 60
            if self.status_train == 'Loaded':
                s = f'transporting {self.train_load[data[3]][0]} kg'
            elif self.status_train == 'Empty':
                s = 'returning'

            print('{:01.0f} {:02.0f}:{:02.0f} Train {} {} from port {} to port {}'.format(day, hour, minute,
                                                                                          data[3] + 1, s, data[0] + 1,
                                                                                          data[1] + 1))

        # add new event
        ip1 = data[0]  # port index
        im = data[2]  # train model index
        ip2 = data[1]  # port2 index
        data[0] = ip2

        if self.status_train == 'Loaded':
            train_speed = self.train_speed_loaded[im]
        elif self.status_train == 'Empty':
            train_speed = self.train_speed_empty[im]

        t = simulator.time + self.distance_ptp[ip1][ip2] / train_speed

        simulator.add_event(t, self.sending_to_port, data)

    def dispatch_to_terminal(self, t, ip, im):
        ''' Route train to terminal.
        '''

        # dispatch to early terminal
        tq = np.array([q[-1] for q in self.terminal_queue_forecast])

        if self.status_train == 'Loaded':
            train_speed = self.train_speed_loaded[im]
        elif self.status_train == 'Empty':
            train_speed = self.train_speed_empty[im]

        tt = self.distance[ip, :] / train_speed
        tl = self.loading_time[:, im]
        tf = np.maximum(tq, t + tt) + tl
        it = np.argmin(tf)
        self.terminal_queue_forecast[it].append(tf[it])
        return it

    def dispatch_to_port(self, t, it, im):
        ''' Route train to port.
        '''

        # dispatch to early port
        tq = np.array([q[-1] for q in self.port_queue_forecast])
        if self.status_train == 'Loaded':
            train_speed = self.train_speed_loaded[im]
        elif self.status_train == 'Empty':
            train_speed = self.train_speed_empty[im]

        if self.from_port:
            tt = self.distance_ptp[:, it] / train_speed
        else:
            tt = self.distance[:, it] / train_speed

        tu = self.unloading_time[:, im]
        tf = np.maximum(tq, t + tt) + tu
        ip = np.argmin(tf)
        self.port_queue_forecast[ip].append(tf[ip])
        return ip
