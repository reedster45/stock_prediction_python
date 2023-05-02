import tensorflow as tf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import random
from sklearn import preprocessing


class StockPricePredictor():

  def __init__(self, filepath) -> None:
    # scaler for preprocessing data
    self.scaler = preprocessing.MinMaxScaler()

    # filepath to dataset
    self.filepath = filepath

    # timesteps used for windowing
    self.timesteps = 12

    # number of features used in training
    self.num_features = 8

  # DataGenerator for StockPricePredictor
  class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset_filepath, use_rows, batch_size, scaler, timesteps):
      use_rows.insert(0,0)
      self.batch_size = batch_size
      self.scaler = scaler
      self.timesteps = timesteps

      self.feature_data = pd.read_excel(dataset_filepath, usecols=('C,D,E,F,H,I,J,K'), skiprows=lambda x: x not in use_rows)
      self.label_data = pd.read_excel(dataset_filepath, usecols=('B'), skiprows=lambda x: x not in use_rows)
      
      self.preprocess_data()
      self.prepare_dataset()

    def preprocess_data(self):
      # convert data to numpy arrays
      self.feature_data = self.feature_data.to_numpy()
      self.label_data = self.label_data.to_numpy()

      # preprocess data
      self.scaler.fit(self.feature_data)
      self.feature_data = self.scaler.transform(self.feature_data)
      self.scaler.fit(self.label_data)
      self.label_data = self.scaler.transform(self.label_data)

    def prepare_dataset(self):
      train_len = len(self.feature_data)
    
      self.x_train = []
      self.y_train = []
      for i in range(self.timesteps, train_len):
        self.x_train.append(self.feature_data[i-self.timesteps:i,:])
        self.y_train.append(self.label_data[i, 0])
      self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)

    def __len__(self):
      #batches_per_epoch is the total number of batches used for one epoch
      batches_per_epoch = int(len(self.x_train) / self.batch_size)
      return batches_per_epoch

    def __getitem__(self, index):
      start = index
      end = index + self.batch_size

      x = self.x_train[start:end]
      y = self.y_train[start:end]
      
      return x, y
  ######### (end of DataGenerator) ###########



  # randomly generate (within a range) adn return two lists of rows for training and validation set
  def train_validation_split(self):
    lis = list(range(1, 3000))
    train = random.sample(lis, 1500)
    validate = list(set(lis) - set(train))
    
    train.sort()
    validate.sort()

    return train, validate

  # generate and return list of rows for test set
  def test_set(self):
    test = list(range(3000, 3762))
    return test

  # genereate and return list of rows for prediction set
  def predict_set(self):
    predict = list(range(3000, 3762))
    return predict

  # return 'date' column from specified rows in dataset
  def get_dates(self, use_rows):
    dates = pd.read_excel(self.filepath, usecols=('A'), skiprows=lambda x: x not in use_rows)
    return dates

  # return 'date' and 'close' columns from specified rows in dataset
  def get_real_prices(self, use_rows):
    real_prices = pd.read_excel(self.filepath, usecols=('A,B'), skiprows=lambda x: x not in use_rows)
    return real_prices



  # build/compile model and return training and validation performance
  def build_train_model(self):
    # create data generator
    batch_size = 32
    train_rows, validate_rows = self.train_validation_split()
    training_data_gen = self.DataGenerator(self.filepath, train_rows, batch_size, self.scaler, self.timesteps)
    validation_data_gen = self.DataGenerator(self.filepath, validate_rows, batch_size, self.scaler, self.timesteps)

    # Assemble model
    neurons = self.timesteps * self.num_features
    self.model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(self.timesteps, self.num_features)),
    tf.keras.layers.LSTM(units=neurons, return_sequences=True),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(units=neurons, return_sequences=True),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(units=64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(units=32),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(1)
    ])

    # compile model - SWITCHED TO ADAM********
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    self.model.compile(optimizer='adam', loss='mean_squared_error')

    history = self.model.fit(x=training_data_gen, epochs=25, validation_data=validation_data_gen, verbose=1)

    training_performance = history.history['loss'][-1]
    validation_performance = history.history['val_loss'][-1]

    self.plot_loss(history.history['loss'], history.history['val_loss'])

    # return training and validation performance (model is self contained)
    return training_performance, validation_performance
  
  def plot_loss(self, training, validation):
    plt.plot(training, c='black', label='train loss')
    plt.plot(validation, c='red', label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

  # test the model against the held out test set and return performance
  def test_model(self):
    batch_size = 32
    test_rows = self.test_set()
    test_data_gen = self.DataGenerator(self.filepath, test_rows, batch_size, self.scaler, self.timesteps)
    performance = self.model.evaluate(x=test_data_gen)

    return performance
  
  # evalute performance of model predictions
  def evaluate_predictions(self, real, predicted):
    real = real['close'].to_numpy()
    predicted = predicted['close'].to_numpy()

    # Mean Absolute Error (MAE)
    MAE = np.mean(abs(predicted - real))
    print('Mean Absolute Error (MAE): ' + str(np.round(MAE, 2)))

    # Median Absolute Error (MedAE)
    MEDAE = np.median(abs(predicted - real))
    print('Median Absolute Error (MedAE): ' + str(np.round(MEDAE, 2)))

    # Mean Squared Error (MSE)
    MSE = np.square(np.subtract(predicted, real)).mean()
    print('Mean Squared Error (MSE): ' + str(np.round(MSE, 2)))

    # Root Mean Squarred Error (RMSE) 
    RMSE = np.sqrt(np.mean(np.square(predicted - real)))
    print('Root Mean Squared Error (RMSE): ' + str(np.round(RMSE, 2)))

    # Mean Absolute Percentage Error (MAPE)
    MAPE = np.mean((np.abs(np.subtract(real, predicted)/ real))) * 100
    print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE, 2)) + ' %')

    # Median Absolute Percentage Error (MDAPE)
    MDAPE = np.median((np.abs(np.subtract(real, predicted)/ real))) * 100
    print('Median Absolute Percentage Error (MDAPE): ' + str(np.round(MDAPE, 2)) + ' %')
     

  # plot real prices against predicted prices
  def plot_graph(self, real, predicted):
    plt.plot(real["close"], c='black', label='Actual Price')
    plt.plot(predicted["close"], c='red', label="Predicted Price")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
  # takes predictions as input and formats them into pandas dataframe with corresponding dates
  # gets corresponding real prices in pandas dataframe
  def format_prediction(self, predictions, rows):
    # rescale predictions to normal scale
    predictions = self.scaler.inverse_transform(predictions)

    # get the dates associated with the predictions and format into pd dataframe
    predicted_prices = self.get_dates(rows)
    predicted_prices = predicted_prices.tail(predicted_prices.shape[0]-self.timesteps)
    predicted_prices['close'] = np.squeeze(predictions).tolist()

    # get the real prices associated with these dates so we can compare with the predicted prices
    real_prices = self.get_real_prices(rows)
    real_prices = real_prices.tail(real_prices.shape[0]-self.timesteps)

    # convert date columns to datetime64[ns]
    predicted_prices['date'] = pd.to_datetime(predicted_prices['date'])
    real_prices['date'] = pd.to_datetime(real_prices['date'])
    predicted_prices.set_index('date', inplace=True)
    real_prices.set_index('date', inplace=True)

    # measure performance
    self.evaluate_predictions(real_prices, predicted_prices)

    # plot the real_prices against the predicted prices
    self.plot_graph(real_prices, predicted_prices)

  # predict specified amount of closing prices
  def predict_model(self):
    batch_size = 1
    predict_rows = self.predict_set()
    predict_data_gen = self.DataGenerator(self.filepath, predict_rows, batch_size, self.scaler, self.timesteps)
    predictions = self.model.predict(x=predict_data_gen)

    self.format_prediction(predictions, predict_rows)

    return predictions



def main():
  # data_gen = DataGenerator('data/complete_data.xlsx', [1,2,3,4,5,6,7,8,9,10], 2)
  stock_predictor = StockPricePredictor('data/complete_data.xlsx')
  training_performance, validation_performance = stock_predictor.build_train_model()
  print(training_performance, validation_performance)

  performance = stock_predictor.test_model()
  print(performance)

  stock_predictor.predict_model()

main()