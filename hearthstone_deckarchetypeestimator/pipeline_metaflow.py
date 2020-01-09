"""
pipeline_nodecorator.py


Script to produce
* a model to predict the archetype of an unknown deck
* prediction of the archetype for the unknow deck in the sample

Date : 2020-12-28
"""

# Load the libraries
from metaflow import FlowSpec, step, Parameter, current, conda_base
import ast
import ast
import random
import itertools
import time

import external as ext


# Definition of the possible parameters for the random forest
parameters_randomforest = {
    "n_estimators" : [100,200,400],
    "criterion" : ["gini","entropy"],
    "max_depth" : [None,2,4,8,16,32]
}

# Build all the combinations of the parameters for the HPO of the random forest
combinations_parameters_randomforest = [dict(zip(parameters_randomforest.keys(), elt)) for elt in itertools.product(*parameters_randomforest.values())]

# Defintion of the Flow
@conda_base(disabled = True, python="3.7.4", libraries={"pandas" : "0.25.2", "numpy" : "1.17.0", "scikit-learn" : "0.22.1"})
class ArchetypeEstimator(FlowSpec):
    """
    A Flow to estimate in the case of an unknown archetype for a deck, what could be the close4st archetype that can be associated
    """
    
    # Define an input parameter for the Flow (number of top cards to keep to define the )
    nbrcards =  Parameter("topcards", help = "Top cards to choose", default = random.choice(list(range(1,40))))
    
    @step
    def start(self):
        """
        Launch the Flow
        """
        self.tags_script = "nolayer"
        self.limittopcards = self.nbrcards
        print("Let's go !!\n I know I could have done something here like loading the decks but I wanted to have a dedicated step for that :-)")
        self.next(self.collect_decks)
    
    @step
    def collect_decks(self):
        """
        Step to collect and process the decks used for the Flow
        """
        import pandas as pd
        
        # Collect the decks
        file_decks = "../data/hearthstone_deckarchetypeestimator/decks_sample.csv"
        df_decks = pd.read_csv(file_decks)

        # Do some operations (cleaning, formatting) on the dataframe
        df_decks = df_decks[df_decks["is_gooddeck"] == 1]
        df_decks["cards"] = df_decks["cards"].apply(lambda elt: ast.literal_eval(elt))
        df_decks["createddate"] = pd.to_datetime(df_decks["createddate"])
        df_decks["individual_cards"] = df_decks["cards"].apply(lambda elt: list(dict.fromkeys(elt)))
        df_decks["deckid"] = pd.to_numeric(df_decks["deckid"], downcast = "integer")
        
        # Save the decks
        self.df_decks = df_decks
        self.next(self.segment_decks)
        
    @step 
    def segment_decks(self):
        """
        Step to do the segmentation of the data between the different datasets (train, test and score)
        """
        from sklearn.model_selection import train_test_split

        # Select the data to score
        self.df_decks_toscore = self.df_decks[self.df_decks["archetype"] == "Unknown"]
        
        # Build the training and testing set
        df_decks_training = self.df_decks[self.df_decks["archetype"] != "Unknown"]
        self.df_decks_totrain, self.df_decks_totest = train_test_split(df_decks_training, train_size = 0.8)

        self.next(self.collect_archetypes)
        
    @step
    def collect_archetypes(self):
        """
        Step to estimate the archetypes that will be predicted
        """
        # Rank the archetype by their presence in the training data
        stats_deckarchetype = self.df_decks_totrain.groupby(["archetype"]).size().sort_values(ascending = False)
        
        # Store the archetypes in a list
        all_archetypes = list(stats_deckarchetype.index)
        print("Collect the archetypes possible",all_archetypes)
        
        # Save the archetypes
        self.archetypes = all_archetypes[:5] # It's just the top5 decks for quick execution but you can drop this limit 
        #(you will need to add the attribute --max-num-splits to your run command with a value > 100 like 150 for example)
        print("Collect the archetypes for the mlmagic", self.archetypes)

        self.next(self.collect_topcards, foreach = "archetypes")
    
    @step
    def collect_topcards(self):
        """
        Step to estimate the top cards for each archetype
        """
        # Select the right decks (the one associated to the deck)
        df_decks_archetype = self.df_decks_totrain[self.df_decks_totrain["archetype"] == self.input]

        # Collect all the cards played with this archetype
        all_cards_archetype = df_decks_archetype["individual_cards"].explode()
        
        # Compute and sort the calculation of the occurency of the card usage
        df_countcards_archetype = all_cards_archetype.to_frame(name = "cardid").groupby(["cardid"]).size().reset_index()
        df_countcards_archetype.columns = ["cardid","occurency"]
        df_countcards_archetype.sort_values(["occurency"], ascending = False, inplace = True)
        
        # Select the top cards for the archetype based on the occurency of the card usage in the decks
        self.archetype = self.input
        self.topcards = df_countcards_archetype["cardid"].head(self.nbrcards).to_list()
        
        self.next(self.build_features)
    
    @step
    def build_features(self, inputs):
        """
        Step to compute the features for the model (occurency of the cards in the deck/archetype)
        """
        # Get all the top cards
        features_cardid = [input.topcards for input in inputs]
        informations_topcards = {}
        for input in inputs:
            features_cardid.append(input.topcards)
            informations_topcards[input.archetype] = input.topcards

        self.informations_topcards = informations_topcards

        # Join all the top cards
        features_cardid = list(itertools.chain.from_iterable(features_cardid))
        # Drop the duplicates (some cards can be on the top cards of multiple archetypes)
        features_cardid = list(dict.fromkeys(features_cardid))
        self.features = features_cardid
        
        # Get the artifacts from the previous steps (exclude just the top cards and archetypes artifacts)
        self.merge_artifacts(inputs, exclude = ["topcards","archetype"])
        
        self.next(self.trigger_prepare_datas)
        
    @step
    def trigger_prepare_datas(self):
        """
        Step to trigger the pareparation of the data with a branch
        """
        self.next(self.prepare_datatraining, self.prepare_datatesting, self.prepare_datascoring)

    @step
    def prepare_datatraining(self):
        """
        Step to build the training and testing set
        """
        self.array_features_totrain, self.array_labels_totrain , self.df_totrain = ext.prepare_data(self.df_decks_totrain, self.features)

        self.next(self.prepare_mlmagic)
        
    @step
    def prepare_datatesting(self):
        """
        Step to build the scoring set
        """
        self.array_features_totest, self.array_labels_totest , self.df_totest = ext.prepare_data(self.df_decks_totest, self.features)
        
        self.next(self.prepare_mlmagic)
    
    @step
    def prepare_datascoring(self):
        """
        Step to build the scoring set
        """
        self.array_features_toscore, self.array_labels_toscore, self.df_toscore = ext.prepare_data(self.df_decks_toscore, self.features)
        
        self.next(self.prepare_mlmagic)

    @step  
    def prepare_mlmagic(self, inputs):
        """
        Step to prepare he data for the ml part and the parameters to test for the HPO
        """
        print("Data are ready")
        self.parameters_model = random.choices(combinations_parameters_randomforest, k = 5)
        self.merge_artifacts(inputs, exclude = ["df_decks_toscore","df_decks_totest", "df_decks_totrain"])

        self.next(self.trigger_build_model)
        
    @step
    def trigger_build_model(self):
        """
        Step to trigger the HPO that will build a model for each parameters_model selected in the previous step
        """
        self.next(self.build_model, foreach = "parameters_model")
    
    @step
    def build_model(self):
        """
        Step to compute the model with specific parameters
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        tic = time.time()

        # Prepare the model for the training
        parameters = self.input
        model = RandomForestClassifier(n_estimators = parameters["n_estimators"],
                                       criterion = parameters["criterion"],
                                       max_depth = parameters["max_depth"],
                                       random_state=0)
        
        # Fit the model
        model.fit(self.array_features_totrain, self.array_labels_totrain)
        time_training = time.time() - tic
        
        # Make the preidctions on the testing set
        tic = time.time()
        array_labels_predictions =  model.predict(self.array_features_totest)
        time_testing = time.time() - tic
        
        # Storing time
        # Store the model details
        self.model_parameters = parameters
        self.model_object = model
        self.classes = model.classes_
        
        # Store the metrics
        self.accuracy = accuracy_score(self.array_labels_totest, array_labels_predictions)
        self.time_training = time_training
        self.time_testing = time_testing   

        self.next(self.select_and_score)
    
    @step
    def select_and_score(self, inputs):
        """
        Step to select the right model and do the scoring 
        """
        # Get the best model based on the accuracy metric
        accuracy_reference = 0
        for input in inputs:
            if input.accuracy > accuracy_reference :
                self.model = input.model_object
                self.parameters = input.model_parameters
                self.classes = input.classes
                self.accuracy = input.accuracy
                self.time_training = input.time_training
                
                accuracy_reference = input.accuracy

        # Time to brag on the best model
        print(f"The best RF has the following : {self.parameters}")
        print(f"With an accuracy of {round(self.accuracy,2)} for a training time of {round(self.time_training,1)} seconds")
                
        # Get the artifacts from the previous steps (and exclude all the model thingy from the hpo)
        self.merge_artifacts(inputs, exclude = ["model_object","model_parameters","accuracy","classes","time_training","time_testing"])
        
        # Build the final version for the testing and scoring dataframe (computing the predictions)
        df_tested = ext.update_df_withpredictions(self.model, self.classes, self.array_features_totest, self.df_totest)
        df_scored = ext.update_df_withpredictions(self.model, self.classes, self.array_features_toscore, self.df_toscore)

        # Get a flag on the testing set if it was a good prediction
        df_tested["is_goodprediction"] = df_tested.apply(lambda row: row["prediction"] == row["archetype"], axis = 1)

        # Store the dataframe
        self.df_tested = df_tested
        self.df_scored = df_scored
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
    