from controller import Supervisor
import math


class Controller(Supervisor):

    def __init__(self, timestep):
        super(Controller, self).__init__()
        self.timeStep = timestep
    
    def run(self):
        self.root_node = self.getRoot()
        root_children_field = self.root_node.getField('children')
        n = root_children_field.getCount()
        
        for i in range(n):
            node = root_children_field.getMFNode(i)
            print(f'-> {node.getTypeName()}')
        print()