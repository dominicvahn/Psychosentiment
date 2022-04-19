# Load required packages
using ScikitLearn, RDatasets, DataFrames, CSV, PyPlot
using ScikitLearn.GridSearch: GridSearchCV

# Import modules
@sk_import linear_model: LogisticRegression
@sk_import model_selection: train_test_split
@sk_import metrics: accuracy_score
@sk_import neural_network: MLPClassifier;

# Assign model and load data
trainer = LogisticRegression(fit_intercept=true)
simpleneuralnetwork = MLPClassifier(hidden_layer_sizes=(5));
gridsearch_logistic = GridSearchCV(LogisticRegression(),
        Dict(:solver => ["newton-cg", "lbfgs", "liblinear"],
        :C => [0.01, 0.1, 0.5, 0.9]))
news = CSV.read("C:\\Users\\Dom\\Desktop\\School\\PsychoModel\\news50.csv", DataFrame)
psycho = CSV.read("C:\\Users\\Dom\\Desktop\\School\\PsychoModel\\psycho50.csv", DataFrame)

# Add targets to DataFrames
news[!, :target] .= "normal"
psycho[!, :target] .= "psychopath"

# Combine DataFrames
data = vcat(news, psycho)

# Generate arrays for learning and split data into training and test sets
X = Array(data[!,Not(:target)])
y = Array(data[!, :target])
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2)

# Fit the models
fit!(trainer, X_train, Y_train)
fit!(simpleneuralnetwork, X_train, Y_train);
fit!(gridsearch_logistic, X_train, Y_train);

# Run prediction for models
y_predict_train = predict(trainer, X_train)
y_predict_test = predict(trainer, X_test)
Y_pred_train = predict(simpleneuralnetwork,X_train)
# Assess training accuracy
print(accuracy_score(Y_train, y_predict_train))
print(accuracy_score(Y_test, y_predict_test))
print(accuracy_score(Y_train,Y_pred_train))

# GridSearch
gridsearch_logistic_results = DataFrame(gridsearch_logistic.grid_scores_);
        hcat(DataFrame(gridsearch_logistic_results.parameters),
        gridsearch_logistic_results)[!,Not(:parameters)]
best_logistic_model = gridsearch_logistic.best_estimator_

gridsearch_neuralnet = GridSearchCV(MLPClassifier(),
Dict(:solver => ["sgd", "lbfgs", "adam"],
:hidden_layer_sizes => [(2), (20), (1,5,10), (10,10), (10,20,10)]))
fit!(gridsearch_neuralnet, X_train, Y_train);

gridsearch_neuralnet_results = DataFrame(gridsearch_neuralnet.grid_scores_);
hcat(DataFrame(gridsearch_neuralnet_results.parameters),
gridsearch_neuralnet_results)[!,Not(:parameters)]

best_neuralnetwork_model = gridsearch_neuralnet.best_estimator_

Y_pred_train_logistic = predict(best_logistic_model, X_train)

@sk_import metrics: classification_report
print(classification_report(Y_pred_train_logistic, Y_train))

Y_pred_test_logistic = predict(best_logistic_model, X_test)
print(classification_report(Y_pred_test_logistic, Y_test))

Y_pred_train_neural = predict(best_neuralnetwork_model, X_train)
print(classification_report(Y_pred_train_neural, Y_train))

Y_pred_test_neural = predict(best_neuralnetwork_model, X_test)
print(classification_report(Y_pred_test_neural, Y_test))

@sk_import metrics: confusion_matrix
cf = confusion_matrix(Y_test,Y_pred_test_neural,normalize="true")

figure()
@sk_import metrics: ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cf,
    display_labels=best_neuralnetwork_model.classes_)
gcf()
