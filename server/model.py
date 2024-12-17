"""
Imports
"""
import pandas as pd
import numpy as np

# data preprocessing and tuning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

# for webapp
from flask import Flask, request, jsonify

# helper functions
import sys
sys.path.insert(1, "../notebooks/")
import helper

"""
Constants
"""
# map each product group to a number
product_group_mapper = {
    "Garment Upper body": 1,
    "Garment Lower body": 2,
    "Garment Full body": 3,
    "Accessories": 4,
    "Swimwear": 5,
    "Bags": 6
}

# we also want to map the popularity index to numbers
popularity_mapper = {
    "low": 0,
    "medium": 1,
    "high": 2
}


"""
Preprocessing the data
"""
def preprocess_data(df):
    # drop rows of irrelevant product groups
    processed_df = df[df["product_group_name"].isin(["Garment Upper body", "Garment Lower body", "Garment Full body", "Accessories", "Swimwear", "Bags"])]

    # replace the product_group_name with a number
    processed_df["product_group_name"] = processed_df["product_group_name"].replace(product_group_mapper)

    # replace the popularity with a number
    processed_df["popularity"] = processed_df["popularity"].replace(popularity_mapper)

    # encode the strings
    helper.encode_strings(processed_df)

    return processed_df

"""
Preparing the data for the model
"""
def prepare_data(df, feature_scaler):
    # split the product codes, popularity, and features of the model
    product_codes = modeldata_df["product_code"].values
    popularity = modeldata_df["popularity"].values
    features = modeldata_df.drop(["popularity", "product_code"], axis=1).values
    
    # scale the features
    features = feature_scaler.fit_transform(features)

    return (product_codes, popularity, features)

"""
function to recommend outfit based on user preferences
"""
def recommend_outfit_for_user(df, articles_df, feature_scaler, recommender, user_outfit_type, user_preference):
    # Filter Data Based on User Input
    filtered_df = df[df["product_group_name"] == user_outfit_type]

    # get filtered product codes, popularity, and features based on user input
    filtered_product_codes = filtered_df["product_code"].values
    filtered_popularity = filtered_df["popularity"].values
    filtered_features = feature_scaler.transform(filtered_df.drop(["popularity", "product_code"], axis=1).values)

    # get the recommended outfit
    recommended_outfit = recommender.recommend_outfit(filtered_product_codes, filtered_features, filtered_df, preference=user_preference)

    # print the recommended product
    recommended_product_name = articles_df[articles_df["product_code"] == recommended_outfit["product_code"]]["prod_name"].values[0]
    # print(f"Recommended Outfit Based on Your Preference:\nProduct Name: {recommended_product_name}, Product Code: {recommended_outfit['product_code']}, Popularity: {recommended_outfit['popularity']}")

    # return recommended product name and popularity
    return (recommended_product_name, recommended_outfit['popularity'])

"""
Model Class
"""

# Generalized Matrix Factorization (GMF)
# GMF-Based Outfit Recommender
class OutfitRecommenderGMF:
    def __init__(self, num_products, num_features, embedding_dim=8, learning_rate=0.01):
        self.num_products = num_products
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.model = self.initialize_model()
        
    def initialize_model(self):
        # Inputs
        product_input = Input(shape=(1,), name="Product_Input")
        feature_input = Input(shape=(self.num_features,), name="Feature_Input")

        # Embedding layer for product
        product_embedding = Embedding(input_dim=self.num_products, output_dim=self.embedding_dim, name="Product_Embedding")(product_input)

        product_embedding = Flatten()(product_embedding)

        # Transform feature input to match embedding size
        transformed_features = Dense(self.embedding_dim, activation='relu')(feature_input)

        # Dot product of the two embeddings
        dot_product = Dot(axes=1)([product_embedding, transformed_features])

        # Dense layer for prediction
        output = Dense(1, activation="linear", name="Output")(dot_product)

        # Compile the model
        model = Model(inputs=[product_input, feature_input], outputs=output)

        # print the model summary
        model.summary()

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse", metrics=["accuracy"])
        return model

    def train(self, product_codes, features, popularity, epochs=100, batch_size=32, verbose=True):
        # fit the model
        self.model.fit([product_codes, features], popularity, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.2)

    def predict(self, product_codes, features):
        # predict the model using the given product codes and features
        return self.model.predict([product_codes, features])

    def recommend_outfit(self, product_codes, features, df, preference):
        # return the recommended outfit based on the predicted values
        # of the given product codes and features
        predictions = self.predict(product_codes, features)
        
        if preference == 0: # low popularity
            recommended_idx = np.argmin(predictions)
        elif preference == 2: # high popularity
            recommended_idx = np.argmax(predictions)
        else: # medium popularity
            recommended_idx = np.argmax(np.mean(predictions))
        return df.iloc[recommended_idx]

"""
APP
"""
# use Flask to create an API endpoint
app = Flask(__name__)

# Route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    outfit_type = data['outfitType']
    preference = data['preference']

    # Filter dataset based on user input
    name, popularity = recommend_outfit_for_user(modeldata_df, articles_df, feature_scaler, recommender, user_outfit_type=outfit_type, user_preference=preference)
    
    response = {
        "name": name,
        "popularity": popularity
    }
    return jsonify(response)

"""
MAIN FUNCTION
"""

# loading in the article data
articles_df = pd.read_csv("../data/articles.csv")

# load the modeldata from previous notebook
modeldata_df = pd.read_csv("../data/modeldata_df.csv")

modeldata_df = preprocess_data(modeldata_df)

feature_scaler = StandardScaler()
product_codes, popularity, features = prepare_data(modeldata_df, feature_scaler)

# train test split the product codes, popularity, and features
prod_train, prod_test, feat_train, feat_test, pop_train, pop_test = train_test_split(
    product_codes, features, popularity, test_size=0.2, random_state=42)

# Create and Train the Model

# num_products: max value of product_code
# num_features: number of features in the model
# embedding_dim: want to recommend 1 outfit
recommender = OutfitRecommenderGMF(num_products=modeldata_df["product_code"].max() + 1, num_features=features.shape[1], embedding_dim=1, learning_rate=0.01)

# train the model
recommender.train(prod_train, feat_train, pop_train, epochs=50, batch_size=32)

print("recommender is trained")

# run the app
app.run(port=8000)