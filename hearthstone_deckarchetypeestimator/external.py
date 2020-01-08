# Function to build the card features based on the card selected as representative
def build_cardfeatures(deck, all_topcards):
    features = []
    for card in all_topcards:
        if card in deck:
            features.append(1)
        else:
            features.append(0)
    return features

# Function to build the dataset for the mlmagic part
def build_array_features_labels(df):
    import numpy as np

    array_features = np.array(df["features"].tolist())
    array_labels = np.array(df["archetype"].tolist())

    return array_features, array_labels


def prepare_data(df, features):

    df["features"] = df["cards"].apply(lambda deck: build_cardfeatures(deck, features))
    array_features, array_labels = build_array_features_labels(df)
    
    return array_features, array_labels, df

# Function to compute preictions of the model
def get_model_output(model, classes, array):
    predictions = list(model.predict(array))
    probabilities = list(model.predict_proba(array))
    confidences = []
    for idx,prediction in enumerate(predictions):
        probability = probabilities[idx]
        idx_classes = list(classes).index(prediction)
        confidences.append(probability[idx_classes])
    return predictions, probabilities, confidences

def update_df_withpredictions(model, classes, array_features, df):
    predictions, probabilities, confidences = get_model_output(model, classes, array_features)
    df["prediction"] = predictions
    df["confidence"] = confidences
    df["probabilities"] = probabilities
    return df





