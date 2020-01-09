# Load the libraries
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

limittopcards = 10
"""
start step
"""
print("Let's go !!\n(I know I could have done something here like loading the decks but I wanted to have a dedicated step for that :-)")

"""
collect_decks step
"""
# Collect_decks step
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

"""
segment_decks step
"""
from sklearn.model_selection import train_test_split

# Select the data to score
df_decks_toscore = df_decks[df_decks["archetype"] == "Unknown"]

# Build the training and testing set
df_decks_training = df_decks[df_decks["archetype"] != "Unknown"]
df_decks_totrain, df_decks_totest = train_test_split(df_decks_training, train_size = 0.8)


"""
collect_archetypes step
"""
# Rank the archetype by their presence in the training data
stats_deckarchetype = df_decks_totrain.groupby(["archetype"]).size().sort_values(ascending = False)

# Store the archetypes in a list
all_archetypes = list(stats_deckarchetype.index)
print("Collect the archetypes possible",all_archetypes)

# Save the archetypes
archetypes = all_archetypes[:5] 
print("Collect the archetypes for the mlmagic", archetypes)

"""
collect_topcards step
"""
features_cardid = []
for archetype in archetypes:
    df_decks_archetype = df_decks_totrain[df_decks_totrain["archetype"] == archetype]

    # Collect all the cards played with this archetype
    all_cards_archetype = df_decks_archetype["individual_cards"].explode()
    
    # Compute and sort the calculation of the occurency of the card usage
    df_countcards_archetype = all_cards_archetype.to_frame(name = "cardid").groupby(["cardid"]).size().reset_index()
    df_countcards_archetype.columns = ["cardid","occurency"]
    df_countcards_archetype.sort_values(["occurency"], ascending = False, inplace = True)
    
    # Select the top cards for the archetype based on the occurency of the card usage in the decks
    topcards = df_countcards_archetype["cardid"].head(limittopcards).to_list()
    features_cardid.append(topcards)

"""
build_features step
"""
informations_topcards = {}
print(len(archetypes), len(features_cardid))
for idx, features in enumerate(features_cardid[:-1]):
    features_cardid.append(features)
    informations_topcards[archetypes[idx]] = features

features = list(itertools.chain.from_iterable(features_cardid))
features = list(dict.fromkeys(features))

dict_df_decks_dataset = {
    "train" : df_decks_totrain,
    "test" : df_decks_totest,
    "score" : df_decks_toscore,
}

dict_df_decks_dataset_rtu = {}
for dataset in dict_df_decks_dataset:
    array_features, array_labels, df = ext.prepare_data(dict_df_decks_dataset[dataset], features)

    dict_df_decks_dataset_rtu[dataset] = {
        "array_features" : array_features,
        "array_labels" : array_labels,
        "df" : df
    }

parameters_model = random.choices(combinations_parameters_randomforest, k = 5)

"""
trigger_build_model + build_model steps
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

accuracy_ref = 0
for parameters in parameters_model:
    tic = time.time()
    model = RandomForestClassifier(n_estimators = parameters["n_estimators"],
                                       criterion = parameters["criterion"],
                                       max_depth = parameters["max_depth"],
                                       random_state=0)
        
    # Fit the model
    model.fit(dict_df_decks_dataset_rtu["train"]["array_features"], dict_df_decks_dataset_rtu["train"]["array_labels"])
    time_training = time.time() - tic
    
    # Make the preidctions on the testing set
    tic = time.time()
    array_labels_predictions =  model.predict(dict_df_decks_dataset_rtu["test"]["array_features"])
    time_testing = time.time() - tic

    accuracy_test = accuracy_score(dict_df_decks_dataset_rtu["test"]["array_labels"], array_labels_predictions)
    if accuracy_test > accuracy_ref:
        accuracy_ref = accuracy_test
        model_ref = model
        classes_ref = model.classes_
        parameters_ref = parameters
        time_training_ref = time_training
    
print(f"The best RF has the following : {parameters_ref}")
print(f"With an accuracy of {round(accuracy_ref,2)} for a training time of {round(time_training_ref,1)} seconds")
 
"""
select_and_score step
"""
df_tested = ext.update_df_withpredictions(model_ref, classes_ref, dict_df_decks_dataset_rtu["test"]["array_features"], dict_df_decks_dataset_rtu["test"]["df"])
df_scored = ext.update_df_withpredictions(model_ref, classes_ref, dict_df_decks_dataset_rtu["score"]["array_features"], dict_df_decks_dataset_rtu["score"]["df"])
       
df_tested["is_goodprediction"] = df_tested.apply(lambda row: row["prediction"] == row["archetype"], axis = 1)

"""
end step
"""
print("Done")