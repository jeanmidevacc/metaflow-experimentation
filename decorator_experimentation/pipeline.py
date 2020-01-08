
import random
from metaflow import FlowSpec, step, Parameter, conda, conda_base

@conda_base(python="3.6.9", libraries={'pandas' : "0.25.2"})
class ExampleFlow(FlowSpec):
    @step
    def start(self):
        import pandas as pd
        import sys

        print("Go")
        self.pandasversion = pd.__version__
        self.pythonversion = sys.version
        print("Python version : ", self.pandasversion)
        print("Pandas version : ", self.pythonversion)
        
        self.next(self.check_pandas)

    #@conda(python = "3.6.8", libraries={"pandas": random.choice(["0.25.3","0.23.3"])})
    @step
    def check_pandas(self):
        print("Let's have a look to my pandas version")
        import pandas as pd
        import sys

        self.pandasversion = pd.__version__
        self.pythonversion = sys.version
        print("Python version : ", self.pandasversion)
        print("Pandas version : ", self.pythonversion)
        self.next(self.give_number, self.give_letter, self.give_something)

    @step
    def give_number(self):
        self.number_produced = random.choice([1,2,3])
        print("This is your number : ", self.number_produced)
        self.next(self.recap)

    @step
    def give_letter(self):
        self.letter_produced = random.choice(["a","b","c"])
        print("This is your letter : ", self.letter_produced)
        self.next(self.recap)

    @step
    def give_something(self):
        self.something_produced = random.choice([1,2,3,"a","b","c"])
        print("This is your something : ", self.something_produced)
        self.next(self.recap)
    
    @step
    def recap(self, inputs):
        print("This is what we get :", 
        inputs.give_number.number_produced, 
        inputs.give_letter.letter_produced,
        inputs.give_something.something_produced)
        self.next(self.end)
    @step
    def end(self):
        import pandas as pd
        import sys

        self.pandasversion = pd.__version__
        self.pythonversion = sys.version
        print("Python version : ", self.pandasversion)
        print("Pandas version : ", self.pythonversion)

        print("Done")

if __name__ == '__main__':
    ExampleFlow()




    
    

    

    


