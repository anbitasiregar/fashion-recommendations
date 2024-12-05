"""
This python file has helper methods that will help throughout the machine learning pipeline.
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, explained_variance_score
import seaborn as sns

"""
reset scores for model performance tests for different X_train variations
"""
def reset_model_scores(models):
  for name in models:
    models[name]["All_Scores"] = list()
    models[name]["Top_Score"] = float()
    models[name]["Mean_Score"] = float()
    models[name]["Std_Score"] = float()

  return models



"""
helper function to test multiple model performances using cross_val_score
"""
def test_models_performance(models, x_train, y_train, isRegressor, num_folds = 10):

  # reset the performance scores first using function above
  reset_model_scores(models)

  # set scoring type based on model type
  scoring = "neg_mean_squared_error" if isRegressor else "accuracy"

  # get the performance scores for each model and add them to the
  # corresponding result list
  for name in models:

    folds = KFold(n_splits=num_folds) if isRegressor else StratifiedKFold(n_splits=num_folds)

    results = cross_val_score(estimator=models[name]["Estimator"],
                              X=x_train,
                              y=y_train,
                              cv=folds,
                              scoring=scoring)
    models[name]["Top_Score"] = results.max()
    models[name]["Mean_Score"] = results.mean()
    models[name]["Std_Score"] = results.std()

    for result in results:
      models[name]["All_Scores"].append(result)

  # print the results
  for name in models:
    print("\n[MODEL TYPE: {}]\n".format(name))
    print(">>>> Top Performance: \t\t{:.4f}".format(models[name]["Top_Score"]))
    print(">>>> Average Performance: \t{:.4f}".format(models[name]["Mean_Score"]))
    print(">>>> Spread of Performance: \t{:.4f}".format(models[name]["Std_Score"]))



"""
printing accuracy scores
"""
def print_accuracy(y_test, y_pred, isRegressor):

  if isRegressor:
    accuracy = 100 * explained_variance_score(y_test, y_pred)
  else:
    accuracy = 100 * accuracy_score(y_true=y_test,
                                y_pred=y_pred)

  print("> ACCURACY: \t{:.2f}%".format(accuracy))



"""
helper function to fit and predict a model
prints the accuracy and returns the predicted y values
"""
def fit_predict(model, X_train, y_train, X_test, y_test, isRegressor):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print_accuracy(y_test, y_pred, isRegressor=isRegressor)

    return y_pred



"""
helper function to use LabelEncoder on string objects
in a dataframe
"""
def encode_strings(df):
  # Apply LabelEncoder to each text column in the DataFrame
  for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])


"""
helper function to print confusion matrix and heat map
"""
def print_confusion_matrix_details(y_true, y_pred):
  conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

  # print the confusion matrix
  print(f"confusion matrix:\n {conf_matrix}\n")

  # graph confusion matrix heatmap
  sns.heatmap(conf_matrix, annot=True)

  # print classification report for further details
  print(classification_report(y_true=y_true, y_pred=y_pred))


"""
helper function to get random sample of data with
same distribution of target data

How to apply:
red_articles_df = helper.sample_data(articles_df, bin_column="popularity", frac=0.75)
"""
def sample_data(df, bin_column, frac=0.5):
    reduced_df = df.groupby(bin_column).apply(lambda x: x.sample(frac=frac, random_state=42)).reset_index(drop=True)

    # Calculate proportions in the original data
    original_count = df[bin_column].count()
    original_proportions = df[bin_column].value_counts(normalize=True)
    
    # Calculate proportions in the reduced data
    reduced_count = df[bin_column].count()
    reduced_proportions = df[bin_column].value_counts(normalize=True)
    
    # Compare proportions
    print(f"Original Count: {original_count}, Proportions: {original_proportions}\n")
    print(f"Reduced Count: {reduced_count}, Proportions: {reduced_proportions}\n")

    return reduced_df