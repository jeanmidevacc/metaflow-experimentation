# Load the libraries
import ast
import random
import itertools
import time

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sklearn

import external as ext

# Define the possible parameters for the random forest
parameters_randomforest = {
    "n_estimators" : [100,200,400],
    "criterion" : ["gini","entropy"],
    "max_depth" : [None,2,4,8,16,32]
}

# Build all the combinations of the parameters for the HPO of the random forest
combinations_parameters_randomforest = [dict(zip(parameters_randomforest.keys(), elt)) for elt in itertools.product(*parameters_randomforest.values())]







