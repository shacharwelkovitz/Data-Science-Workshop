# lasso regressor
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def split_dataset_to_data_and_labels(dataset):
  labels = dataset.score
  dataset_without_labels = dataset.drop('score',axis=1)
  return dataset_without_labels, labels


def plotModelResults(model, X_train, y_train, X_test, y_test, plot_intervals=False, plot_anomalies=False, scale=1.96):
    """
        Plots modelled vs fact values, prediction intervals and anomalies
    
    """
    tscv = TimeSeriesSplit(n_splits = 20)

    prediction = model.predict(X_test)
    
    plt.figure(figsize=(12, 6))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train, cv = tscv,
                                    scoring="neg_mean_absolute_error")
        #mae = cv.mean() * (-1)
        deviation = np.sqrt(cv.std())
        
        lower = prediction - (scale * deviation)
        upper = prediction + (scale * deviation)
        
        plt.plot(lower, "r--", label="upper bound / lower bound", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    plt.title("Actual VS Prediction", fontsize=TITLE_SIZE)
    plt.xlabel("sample", fontsize=XLABEL_SIZE)
    plt.ylabel("score", fontsize=YLABEL_SIZE)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);

def plotCoefficients(model, x_train):
    """
        Plots sorted coefficient values of the model
    """

    coefs = pd.DataFrame(model.coef_, x_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    plt.figure(figsize=(12, 6))
    plt.title("features coefficients", fontsize=TITLE_SIZE)
    plt.xlabel("feature", fontsize=XLABEL_SIZE)
    plt.ylabel("coefficient", fontsize=YLABEL_SIZE)
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');

def cross_validation_lasso(X, y):  
  tscv = TimeSeriesSplit(n_splits=10)
  for train_index, test_index in tscv.split(X):
      X_train, X_test = X.iloc[:train_index[len(train_index) - 1]], X.iloc[test_index[0]:test_index[len(test_index) - 1]]
      y_train, y_test = y.iloc[:train_index[len(train_index) - 1]], y.iloc[test_index[0]:test_index[len(test_index) - 1]]
      prediction = lasso_regressor(X_train, y_train).predict(X_test)
      error = mean_absolute_percentage_error(y_test, prediction)
      print("Train size: {}, Test size: {}, Mean Absolute Percentage Error: {}%".format(X_train.shape[0], X_test.shape[0],error))

# ---------------------------------------------------------------------------------------------------------------------------------

# ARIMA

def fit_ARIMA(p1,d1,q1, p2,d2,q2):
  model = ARIMA(np.log(dfw.score).dropna(), (p1,d1,q1))
  res_0 = model.fit()
  print(res_0.summary())

  model = ARIMA(np.log(dfw.score).dropna(), (p2,d2,q2))
  res_1 = model.fit()
  print(res_1.summary())
  
  fig, ax = plt.subplots(1, 2, sharey=True, figsize=FIG_SIZE)
  ax[0].plot(res_0.resid.values, alpha=0.7, label='variance={:.3f}'.format(np.std(res_0.resid.values)));
  ax[0].hlines(0, xmin=0, xmax=350, color='r');
  ax[0].set_title("ARIMA ({}, {}, {}) Residuals".format(p1,d1,q1));
  ax[0].set_ylabel("residuals");
  ax[0].set_xlabel("example number");
  ax[0].legend();
  ax[1].plot(res_1.resid.values, alpha=0.7, label='variance={:.3f}'.format(np.std(res_1.resid.values)));
  ax[1].hlines(0, xmin=0, xmax=350, color='r');
  ax[1].set_title("ARIMA ({}, {}, {}) Residuals".format(p2,d2,q2));
  ax[1].set_ylabel("residuals");
  ax[1].set_xlabel("example number");
  ax[1].legend();


def cross_validation_arima(X, y):
  tscv = TimeSeriesSplit(n_splits=10)

  for train_index, test_index in tscv.split(X):
      X_train, X_test = X.iloc[:train_index[len(train_index) - 1]], X.iloc[test_index[0]:test_index[len(test_index) - 1]]
      y_train, y_test = y.iloc[:train_index[len(train_index) - 1]], y.iloc[test_index[0]:test_index[len(test_index) - 1]]
      # Build the ARIMA Model.
      model = ARIMA(y_train, order=(1, 1, 0))  
      fitted = model.fit(disp=-1) 
      # Forecast
      fc, se, conf = fitted.forecast(X_test.shape[0], alpha=0.05)  # 95% conf
      # Make as pandas series.
      prediction = pd.Series(fc, index=y_test.index)
      # calculate the error.
      error = mean_absolute_percentage_error(y_test, prediction)
      print("Train size: {}, Test size: {}, Mean Absolute Percentage Error: {}%".format(X_train.shape[0], X_test.shape[0],error))

def run_arima_and_plot_results():
  X_train, X_test = X.iloc[:726], X.iloc[726:805]
  y_train, y_test = y.iloc[:726], y.iloc[726:805]
  # Build Model
  model = ARIMA(y_train, order=(1, 1, 0))  
  fitted = model.fit(disp=-1)
  # Forecast
  fc, se, conf = fitted.forecast(X_test.shape[0], alpha=0.05)  # 95% conf
  # Make as pandas series
  fc_series = pd.Series(fc, index=y_test.index)
  lower_series = pd.Series(conf[:, 0], index=y_test.index)
  upper_series = pd.Series(conf[:, 1], index=y_test.index)

  error = mean_absolute_percentage_error(y_test, fc_series)
  print("Train size: {}, Test size: {}, Mean Absolute Percentage Error: {}%".format(X_train.shape[0], X_test.shape[0],error))

  # Plot
  plt.figure(figsize=FIG_SIZE, dpi=100)
  plt.plot(y_train, label='training')
  plt.plot(y_test, label='actual')
  plt.plot(fc_series, label='forecast')
  plt.fill_between(lower_series.index, lower_series, upper_series, 
                  color='k', alpha=.15)
  plt.title('Forecast vs Actuals', fontsize=TITLE_SIZE)
  plt.xlabel('Year', fontsize=XLABEL_SIZE)
  plt.ylabel('average score', fontsize=YLABEL_SIZE)
  plt.legend(loc='upper left', fontsize=8)
  plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------

# Logistic Regression

def convert_reg_score_to_categories(input_labels):
  """
  bins:
  [0,1) - 0, no drought
  [1,2) - D0
  [2,3) - D1
  [3,4) - D2
  [4,5) - D3
  [5, ++++) - D4
  """
  categorical_scores = []
  for i in input_labels:
    if i < 1:
      i = 0
    elif i >= 1 and i < 2:
      i = 1
    elif i >= 2 and i < 3:
      i = 2
    elif i >= 3 and i < 4:
      i = 3
    elif i >= 4 and i < 5:
      i = 4
    elif i >= 5:
      i = 5
    else:
      continue
    categorical_scores.append(i)
  return categorical_scores

def plotRoc(fpr, tpr, auc):
    plt.figure(figsize=FIG_SIZE)
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=XLABEL_SIZE)
    plt.ylabel('True Positive Rate', fontsize=YLABEL_SIZE)
    plt.title('Receiver Operating Characteristic', fontsize=TITLE_SIZE)
    plt.legend(loc="lower right")
    plt.show()
