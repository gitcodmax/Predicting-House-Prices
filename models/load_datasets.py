from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_log_error
import pandas 
import numpy as np

# Gets the training input and target and test set
def load_train_test_sets():
  X_train = pandas.read_csv(r'..\processed_data\X_train.csv')
  X_test = pandas.read_csv(r'..\processed_data\X_test.csv')

  train_target = pandas.read_csv(r'..\processed_data\y_train.csv')

  return X_train, X_test, train_target


# Splits the training data using K-Fold cross validation, 
# trains the model and evaluates the results after prediction.
# Returns the model, Avg Training and Validation RMSLE
def split_train_evaluate(model, train_input, train_target):
  # Split the data into 4 folds
  kf = KFold(n_splits=4, shuffle=True, random_state=42)

  training_loss = np.array([])
  validation_loss = np.array([])

  for train_index, val_index in kf.split(train_input):
    X_train = train_input.iloc[train_index]
    X_val = train_input.iloc[val_index]

    y_train = train_target.iloc[train_index]
    y_val = train_target.iloc[val_index]

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    train_loss = root_mean_squared_log_error(y_train, train_preds)
    val_loss = root_mean_squared_log_error(y_val, val_preds)

    training_loss = np.append(training_loss, train_loss)
    validation_loss = np.append(validation_loss, val_loss)

  avg_train_loss = training_loss.mean()
  avg_val_loss = validation_loss.mean()

  print(f'Training Loss: {avg_train_loss}')
  print(f'Validation Loss: {avg_val_loss}')
  return model, avg_train_loss, avg_val_loss