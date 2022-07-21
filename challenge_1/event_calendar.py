import math
from desafio_1.des_model import DesModel


class EventCalendar:
    ''' Class for handling an event calendar for discrete event system
    simulation.
    '''

    def __init__(self):
        ''' Constructs an event calendar.
        '''

        # initialize attributes
        self.calendar = []

    def push(self, t, f, data):
        ''' Add event to calendar.

        Args:
            t (float): fire time.
            f (function): callback function.
            data: custom callback data.
        '''

        # binary search
        i = [0, len(self.calendar)]  # insertion calender index
        while i[0] != i[1]:
            im = math.floor((i[0] + i[1]) / 2)
            if t >= self.calendar[im][0]:
                i[0] = im + 1
            else:
                i[1] = im

        # insert new event in calendar
        self.calendar.insert(i[0], (t, f, data))

    def pop(self, model):
        ''' Get the nearest time event.

        Returns:
            (float): fire time
            (function): callback function.
            (any type): callback data.
        '''

        return self.calendar.pop(0)

    def is_empty(self):
        ''' Check whether calendar is empty.

        Returns:
            (logical): true if calendar is empty.
        '''

        return len(self.calendar) == 0
