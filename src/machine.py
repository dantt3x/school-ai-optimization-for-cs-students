import pandas

from sklearn.model_selection import train_test_split as trainTestSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor as randomForestR, AdaBoostRegressor as ABRegressor, StackingRegressor as StackR
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error, r2_score, mean_absolute_error
from matplotlib import pyplot

# Sets up the dataFrame from the provided data.
dataFrame = pandas.read_csv('data\dataset.csv')
dataFrame = dataFrame.drop(
    columns=[
    'datetimeEpoch', 
    'feelslikemax',
    'feelslikemin',
    'feelslike',
    'visibility',
    'sunriseEpoch',
    'sunsetEpoch',
    'moonphase',
    ]
)
dataFrame = dataFrame.drop_duplicates()
'''
paramGrid = {
    'n_estimators': [10, 100],
    'max_depth': [5, 10],
    'min_samples_leaf': [3, 5],
} 
'''
# Find the best suit for the regulated parameters
'''
search = GridSearchCV(estimator=forest, param_grid=paramGrid, cv=10)
search.fit(xTrain, yTrain)
'''
#
# best_params_ = {'max_depth': 10, 'min_samples_leaf': 3, 'n_estimators': 100}
#
'''
print(best_params_)
'''

# Creates a randomForestRegressor with 10 trees
# Update: Now uses the regulated best params from GridSearchCV
# Update: Now implements the new ensemble techniques: Boosting and Stacking
forest = randomForestR(n_estimators=60, max_depth=10, min_samples_leaf=5)
boostedForest = ABRegressor(estimator=forest, n_estimators=40)
stackedForest = StackR(estimators=[('boostedForest', boostedForest), ('forest', forest)])

# Takes the training data w/o the target (partial selection of data set)
# Takes the training data with only the target
# Outputs the training data it used and the training data it didn't use
xTrain, xTest, yTrain, yTest = trainTestSplit(
    dataFrame.drop(columns='healthRiskScore'), 
    dataFrame['healthRiskScore'],
)

# Build forest on training data
stackedForest.fit(xTrain, yTrain)
# Gets the predictions of the training data
forecast = stackedForest.predict(xTest)

# Gets metrics from forecast
MAPE = mean_absolute_percentage_error(yTest, forecast)
RMSE = root_mean_squared_error(yTest, forecast)
R2 = r2_score(yTest, forecast)
MAE = mean_absolute_error(yTest, forecast)

# Prints the metrics
print("#########################################")
print("-> The predictions error percentile: " + str(MAPE * 100) + "%")
print("-> The predictions deviation from original values: +-" + str(RMSE) + " units")
print("-> The predictions variance score: " + str(R2 * 100) + "%")
print("-> The predictions average deviation: +-" + str(MAE) + " units")
print("#########################################")

# Plots the forecast
pyplot.figure(figsize=(10,10))
pyplot.plot(forecast[:25])
pyplot.xlabel('Forecast Data -> Weather Conditions + Polution')
pyplot.ylabel('Health risk')
pyplot.title('Forecast')
pyplot.show()
