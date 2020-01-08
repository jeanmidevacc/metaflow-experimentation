"""
pipeline_nodecorator.py


Script to produce
* a model to predict the archetype of an unknown deck
* prediction of the archetype for the unknow deck in the sample

Date : 2020-12-28
"""

# Load the libraries
from metaflow import FlowSpec, step, Parameter
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


# Definition of the possible parameters for the random forest
parameters_randomforest = {
    "n_estimators" : [100,200,400],
    "criterion" : ["gini","entropy"],
    "max_depth" : [None,2,4,8]
}
combinations_parameters_randomforest = [dict(zip(parameters_randomforest.keys(), elt)) for elt in itertools.product(*parameters_randomforest.values())]

# Function to build the card features based on the card selected as representative
def build_cardfeatures(deck,all_topcards):
    features = []
    for card in all_topcards:
        if card in deck:
            features.append(1)
        else:
            features.append(0)
    return features

def get_model_output(model,classes,array):
    predictions = list(model.predict(array))
    probabilities = list(model.predict_proba(array))
    confidences = []
    for idx,prediction in enumerate(predictions):
        probability = probabilities[idx]
        idx_classes = list(classes).index(prediction)
        confidences.append(probability[idx_classes])
    return predictions, probabilities, confidences

# Defintion of the Flow
class ArchetypeEstimator(FlowSpec):
    """
    A Flow to estimate in the case of an unknown archetype for a deck, what could be the close4st archetype that can be associated
    """
    
    # Define an input parameter for the Flow (number of top cards to keep to define the )
    limittopcards =  Parameter('topcards', help = 'Top cards to choose', default = random.choice(list(range(1,40))))
    
    @step
    def start(self):
        """
        Launch the Flow
        """
        print("Let's go")
        limittopcards = self.limittopcards
        
        # Launch a join operation to collect the decks and cards for the Flow
        self.next(self.collect_decks)
    
    @step
    def collect_decks(self):
        """
        Step to collect and process the decks used for the Flow
        """
        import pandas as pd
        
        # Collect the decks
        file_decks = "../decks_sample.csv"
        df_decks = pd.read_csv(file_decks)

        # Do some operations (cleaning, formatting)
        df_decks = df_decks[df_decks["is_gooddeck"] == 1]
        df_decks["cards"] = df_decks["cards"].apply(lambda elt: ast.literal_eval(elt))
        df_decks["createddate"] = pd.to_datetime(df_decks["createddate"])
        df_decks["individual_cards"] = df_decks["cards"].apply(lambda elt: list(dict.fromkeys(elt)))
        df_decks["deckid"] = pd.to_numeric(df_decks["deckid"], downcast = 'integer')
        # Save the decks
        self.df_decks_tmp = df_decks
        self.next(self.store_data)
        
    @step 
    def store_data(self,inputs):
        """
        Step to join the result of the previous steps
        """
        #Associated the results of the previous steps
        self.df_decks = inputs.collect_decks.df_decks_tmp

        # Operation to keep the variable that have been produced before the join
        self.merge_artifacts(inputs)
        
        # Launc hte segmentation of the data between training testing and scoring datasets
        self.next(self.segment_decks)
        
    @step 
    def segment_decks(self):
        """
        Step to do the segmentation of the data between the different datasets (train, test and score)
        """
        # Select the data to score
        self.df_decks_toscore = self.df_decks[self.df_decks["archetype"] == "Unknown"]
        
        # Build the training and testing set
        df_decks_training = self.df_decks[self.df_decks["archetype"] != "Unknown"]
        df_train, df_test = train_test_split(df_decks_training, train_size = 0.8)
        self.df_decks_totrain = df_train
        self.df_decks_totest = df_test
        
        self.next(self.collect_archetype)
        
    @step
    def collect_archetype(self):
        """
        Step to estimate the archetypes that will be predicted
        """
        # Rank the archetype by their presence in the training data
        stats_deckarchetype = self.df_decks_totrain.groupby(["archetype"]).size().sort_values(ascending = False)
        
        # Store the archetype in a list
        all_archetypes = list(stats_deckarchetype.index)
        print(all_archetypes)
        
        # store it in the Flow
        self.archetypes = all_archetypes
        
        self.next(self.trigger_collect_topcards)
    
    @step 
    def trigger_collect_topcards(self):
        """
        Step to trigger a for each one operation to collect the cards that are more present in each archetype that could be use for the prediction
        """
        self.next(self.collect_topcards, foreach = 'archetypes')
    
    @step
    def collect_topcards(self):
        """
        Step to estimate the top cards for each archetype
        """
        # Select the right decks (the one associated to the deck)
        df_decks_archetype = self.df_decks[self.df_decks["archetype"] == self.input]
        
        # Collect all the cards played with this archetype
        all_cards_archetype = df_decks_archetype["individual_cards"].explode()
        
        # Compute the calculationm of the occurency
        df_countcards_archetype = all_cards_archetype.to_frame(name = "cardid").groupby(["cardid"]).size().reset_index()
        df_countcards_archetype.columns = ["cardid","occurency"]
        df_countcards_archetype.sort_values(["occurency"], ascending = False, inplace = True)
        
        # Select the top cards for the archetype
        self.topcards = df_countcards_archetype["cardid"].head(limittopcards).to_list()
        self.next(self.build_features)
    
    @step
    def build_features(self, inputs):
        """
        Step to compute the features for the model (occurency of the cards in the deck/archetype)
        """
        # Get all the top cards
        features_cardid = [input.topcards for input in inputs]
        # Join all the top cards
        features_cardid = list(itertools.chain.from_iterable(features_cardid))
        # Drop the duplicates
        features_cardid = list(dict.fromkeys(features_cardid))
        self.features = features_cardid
        
        # Get the variable from the previous steps (exclude just the top cards)
        self.merge_artifacts(inputs, exclude = ["topcards"])
        
        self.next(self.trigger_prepare_data)
        
    @step
    def trigger_prepare_data(self):
        """
        Step to trigger the pareparation of the data with a branch
        """
        self.next(self.prepare_datatraining, self.prepare_datascoring)
        
    @step
    def prepare_datatraining(self):
        """
        Step to build the training and testing set
        """
        df_train = self.df_decks_totrain.copy()
        df_test = self.df_decks_totest.copy()

        df_train["features"] = df_train["cards"].apply(lambda deck: build_cardfeatures(deck, self.features))
        df_test["features"] = df_test["cards"].apply(lambda deck: build_cardfeatures(deck, self.features))
        
        self.df_totrain = df_train
        self.df_totest = df_test
        
        self.next(self.prepare_mlmagic)
        
    @step
    def prepare_datascoring(self):
        """
        Step to build the scoring set
        """
        df_score = self.df_decks_toscore.copy()
        df_score["features"] = df_score["cards"].apply(lambda deck: build_cardfeatures(deck, self.features))
        
        self.df_toscore = df_score
        
        self.next(self.prepare_mlmagic)
    
    @step  
    def prepare_mlmagic(self,inputs):
        """
        Step to prepare he data for the ml part and the parameters to test for the HPO
        """
        # Works on the training set
        df_tmp = inputs.prepare_datatraining.df_totrain
        self.array_features_totrain = np.array(df_tmp['features'].tolist())
        self.array_labels_totrain = np.array(df_tmp['archetype'].tolist())
        
        # Works on the testing set
        df_tmp = inputs.prepare_datatraining.df_totest
        self.array_features_totest = np.array(df_tmp['features'].tolist())
        self.array_labels_totest = np.array(df_tmp['archetype'].tolist())
        self.df_totest = df_tmp
        
        # Works on the scoring set
        df_tmp = inputs.prepare_datascoring.df_toscore
        self.array_features_toscore = np.array(df_tmp['features'].tolist())
        self.df_toscore = df_tmp
    
        # Define the possible parameters to test for the model (pick randomly 5 combinations)
        self.parameters_model = random.choices(combinations_parameters_randomforest, k = 5)
        
        self.next(self.trigger_build_model)
        
    @step
    def trigger_build_model(self):
        """
        Step to trigger the HPO that will build a model for each parameters_model selected in the previous step
        """
        self.next(self.build_model, foreach = 'parameters_model')
    
    @step
    def build_model(self):
        """
        Step to compute the model 
        """
        tic = time.time()
        parameters = self.input
        model = RandomForestClassifier(n_estimators = parameters["n_estimators"],
                                       criterion = parameters["criterion"],
                                       max_depth = parameters["max_depth"],
                                       random_state=0)
        
        model.fit(self.array_features_totrain, self.array_labels_totrain)
        time_training = time.time() - tic
        
        tic = time.time()
        array_labels_predictions =  model.predict(self.array_features_totest)
        time_testing = time.time() - tic
        
        self.model_parameters = parameters
        self.model_object = model
        
        # Store the accuracy on the training set
        self.accuracy = accuracy_score(self.array_labels_totest, array_labels_predictions)
        self.classes = model.classes_
        
        self.time_training = time_training
        self.time_testing = time_testing    

        self.next(self.select_and_score)
    
    @step
    def select_and_score(self, inputs):
        """
        Step to select the right model and do the scoring 
        """
        #Get the best model based on the accuracy metric
        accuracy_reference = 0
        for input in inputs:
            if input.accuracy > accuracy_reference :
                self.model = input.model_object
                self.parameters = input.model_parameters
                self.classes = input.classes
                self.accuracy = input.accuracy
                self.time_training = input.time_training
                
                accuracy_reference = input.accuracy
                
        # Get the articfacts from the previous steps (and exclude all the model thingy from the hpo)
        self.merge_artifacts(inputs, exclude = ["model_object","model_parameters","accuracy","classes","time_training","time_testing"])
        
        # Collect the useful data
        predictions, probabilities, confidences = get_model_output(self.model,self.classes,self.array_features_toscore)

        # Build the dataframe of element scored
        df_scored = self.df_toscore
        df_scored["prediction"] = predictions
        df_scored["confidence"] = confidences
        df_scored["probabilities"] = probabilities
        
        # Wrok on the test set
        predictions, probabilities, confidences = get_model_output(self.model,self.classes,self.array_features_totest)
        
        df_tested = self.df_totest
        df_tested["prediction"] = predictions
        df_tested["confidence"] = confidences
        df_tested["probabilities"] = probabilities
        df_tested["is_goodprediction"] = df_tested.apply(lambda row: row["prediction"] == row["archetype"], axis = 1)
        
        # Store the dataframe in an artifact
        self.df_scored = df_scored
        self.df_tested = df_tested
        self.next(self.end)
        
    @step 
    def end(self):
        """
        Step to conclude the Flow
        """
        print("Done")
        pass
        
if __name__ == '__main__':
    ArchetypeEstimator()
    