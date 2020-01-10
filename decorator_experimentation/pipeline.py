
"""
pipeline.py

Script to test the different decorator on the metaflow framework
"""

import random
from metaflow import FlowSpec, step, Parameter, conda, conda_base

@conda_base(disabled = False ,python="3.7.4", libraries={"pandas" : "0.25.2"})
class ExampleFlow(FlowSpec):
    @step
    def start(self):
        """
        Step to launch the flow
        """
        import pandas as pd
        import sys

        print("Go")
        self.pandasversion = pd.__version__
        self.pythonversion = sys.version
        print("Python version : ", self.pythonversion)
        print("Pandas version : ", self.pandasversion)
        
        self.next(self.check_version)

    @conda(python = "3.6.8", libraries={"pandas": random.choice(["0.25.3","0.23.3"])})
    @step
    def check_version(self):
        """
        Step to get the current version of python and pandas start a 3 branches transition
        """
        print("Let's have a look to my pandas version")
        import pandas as pd
        import sys

        self.pandasversion = pd.__version__
        self.pythonversion = sys.version
        print("Python version : ", self.pythonversion)
        print("Pandas version : ", self.pandasversion)
        self.next(self.give_number, self.give_letter, self.give_something)

    @step
    def give_number(self):
        """
        Step to generate a random number
        """
        self.number_produced = random.choice([1,2,3])
        print("This is your number : ", self.number_produced)
        self.next(self.recap)

    @step
    def give_letter(self):
        """
        Step to generate a random letter
        """
        self.letter_produced = random.choice(["a","b","c"])
        print("This is your letter : ", self.letter_produced)
        self.next(self.recap)

    @step
    def give_something(self):
        """
        Step to generate a random number or letter
        """
        self.something_produced = random.choice([1,2,3,"a","b","c"])
        print("This is your something : ", self.something_produced)
        self.next(self.recap)
    
    @step
    def recap(self, inputs):
        """
        Step to collect alla th results of the branch
        """
        print("This is what we get :", 
        inputs.give_number.number_produced, 
        inputs.give_letter.letter_produced,
        inputs.give_something.something_produced)
        self.next(self.end)

    @step
    def end(self):
        """
        Step to end the flow
        """
        import pandas as pd
        import sys

        self.pandasversion = pd.__version__
        self.pythonversion = sys.version
        print("Python version : ", self.pythonversion)
        print("Pandas version : ", self.pandasversion)

        print("Done")

if __name__ == '__main__':
    ExampleFlow()




    
    

    

    


