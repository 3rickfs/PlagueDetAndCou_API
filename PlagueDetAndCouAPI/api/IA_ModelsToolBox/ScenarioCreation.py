"""
    Module for creating scenarios to simulate objetcs motions 
    and make them collide each other for creating new forms 
    of objects
"""
from abc import ABC, abstractmethod

class Scenario(ABC):
    """
        Abtract class for creating scenarios to simulate objts motion
    """

    @abstractmethod
    def create(self):
        """
            Function to create the scenario considering all the properties
            it needs to perform well under object simulation
        """
        pass

class BasicScenario(Scenario):
    """
        Basic scenario based on a rectangle
    """

    def __init__(self, width, height, time_vector):
        self.width = width
        self.height = height
        self.scenario_tuples = None
        self.environment_time = time_vector
        self.centroid = [int(width/2), int(height/2)]
        self.objs_number = 0
        self.objs_list = []

    def add_an_img_obj(self, obj):
        """
            To have a count about how many objects exist in the scenario
        """
        self.objs_number += 1
        self.objs_list.extend([obj])

    def get_scenario_tuples(self):
        """
            Get tuples in a set that represent the environment where the
            objects will interact with others
        """

        tuplelist = []
        for w in range(self.width):
            for h in range(self.height):
                tuplelist.append((w,h))

        return set(tuplelist)

    def get_current_time(self, i):
        """
            Get the scenario time for simulation process
        """

        return self.environment_time[i]

    def create(self):
        self.scenario_tuples = self.get_scenario_tuples()


