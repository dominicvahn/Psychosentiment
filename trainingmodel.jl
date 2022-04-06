# Load required packages
using ScikitLearn, RDatasets, DataFrames, CSV

# Import modules
@sk_import linear_model: LogisticRegression
@sk_import model_selection: train_test_split
@sk_import metrics: accuracy_score

# Assign model and load data
trainer = LogisticRegression(fit_intercept=true)

news = CSV.read(joinpath(pwd(),"news50.csv"), DataFrame)

psycho = CSV.read(joinpath(pwd(),"psycho50.csv"), DataFrame)

# Add targets to DataFrames
news[!, :target] .= "normal"
psycho[!, :target] .= "psychopath"

# Combine DataFrames
data = vcat(news, psycho)

# Generate arrays for learning and split data into training and test sets
X = Array(data[!,Not(:target)])
y = Array(data[!, :target])
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2)

# Fit the model
fit!(trainer, X_train, Y_train)

# Run prediction for model
y_predict_train = predict(trainer, X_train)
y_predict_test = predict(trainer, X_test)

# Assess training accuracy
training_score = accuracy_score(Y_train, y_predict_train)
test_score = accuracy_score(Y_test, y_predict_test)