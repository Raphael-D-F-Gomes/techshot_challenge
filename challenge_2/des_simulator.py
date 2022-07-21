import desafio_2.event_calendar as ec


class DesSimulator:
    ''' Discrete event system simulator.
    '''

    def __init__(self):
        ''' Construct an event system simulator.
        '''

        # setup
        self.time = 0
        self.calendar = ec.EventCalendar()

    def add_event(self, t, f, data, id):
        ''' Add event to calendar.

        Args:
            t (float): fire time.
            f (function): callback function.
            data: custom callback data.
        '''
        self.calendar.push(t, f, data, id)

    def simulate(self, model, T=24 * 3600):
        ''' Simulate discret event system.

        Args:
            model (:obj:DES_model): discrete event system model.
            T (float): time horizon.
        '''

        # discrete event simulator
        model.clear()
        model.starting_events(self)
        while (not self.calendar.is_empty()):

            if (self.time > T):
                if min(model.storage - model.demand) < 0:
                    print('Demand was not fulfilled in time')

                break

            if min(model.storage - model.demand) >= 0:
                print('Demand was fulfilled ')
                break

            self.time, f, data, id = self.calendar.pop()  # get next event
            f(self, data, id)  # callback function
