# For reading, visualizing, and preprocessing data
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import traProject.basicFuns.DataProcessingFuns as dfun
import traProject.utils as tu
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import pickle


# INPATH='./datasets1/'
# OUTPATH='./results1/'

class PipelineRFE(Pipeline):
    # Source: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_
        return self
    
def myfun(INDEX,DATASET_LIST,INPATH,OUTPATH,RANDOM_STATE=1229,vlist=[20,30,40,50,60,70,80]):
    DATASET_NAME=DATASET_LIST[INDEX].split('.csv')[0]
    tu.pathCheck(OUTPATH+'%s/'%DATASET_NAME)
    data=pd.read_csv(INPATH+'%s.csv'%DATASET_NAME)
    data=dfun.trans_data(data,onehot_x=False,minmax_x=False,onehot_y=False,minN=1)
    data=data[data.maxspeed.isin(vlist)]
    y=data['maxspeed']
    yunique=list(y.unique())
    yunique.sort()
    y=y.apply(lambda x: yunique.index(x))
    X=data.drop(['maxspeed','Sdis','Xdis'],axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,
                                                    random_state = RANDOM_STATE)
    classifiers = {}
    classifiers.update({"LDA": LinearDiscriminantAnalysis()})
    classifiers.update({"QDA": QuadraticDiscriminantAnalysis()})
    classifiers.update({"AdaBoost": AdaBoostClassifier()})
    classifiers.update({"Bagging": BaggingClassifier()})
    classifiers.update({"Extra Trees Ensemble": ExtraTreesClassifier()})
    classifiers.update({"Gradient Boosting": GradientBoostingClassifier()})
    classifiers.update({"Random Forest": RandomForestClassifier()})
    # classifiers.update({"Ridge": RidgeClassifier()})
    # classifiers.update({"SGD": SGDClassifier()})
    classifiers.update({"BNB": BernoulliNB()})
    classifiers.update({"GNB": GaussianNB()})
    classifiers.update({"KNN": KNeighborsClassifier()})
    classifiers.update({"MLP": MLPClassifier()})
    # classifiers.update({"LSVC": LinearSVC()})
    # classifiers.update({"NuSVC": NuSVC()})
    # classifiers.update({"SVC": SVC()})
    classifiers.update({"DTC": DecisionTreeClassifier()})
    classifiers.update({"ETC": ExtraTreeClassifier()})

    # Create dict of decision function labels
    # DECISION_FUNCTIONS = {"Ridge", "SGD", "LSVC", "NuSVC", "SVC"}
    DECISION_FUNCTIONS = {"Ridge", "SGD"}

    # Create dict for classifiers with feature_importances_ attribute
    FEATURE_IMPORTANCE = {"Gradient Boosting", "Extra Trees Ensemble", "Random Forest"}
    
    # Initiate parameter grid
    parameters = {}
    # Update dict with LDA
    parameters.update({"LDA": {"classifier__solver": ["svd"], 
                                            }})

    # Update dict with QDA
    parameters.update({"QDA": {"classifier__reg_param":[0.09], 
                                            }})
    # Update dict with AdaBoost
    parameters.update({"AdaBoost": { 
                                    "classifier__base_estimator": [DecisionTreeClassifier(max_depth = 3)],
                                    "classifier__n_estimators": [200],
                                    "classifier__learning_rate": [0.001]
                                    }})

    # Update dict with Bagging
    parameters.update({"Bagging": { 
                                    "classifier__base_estimator": [DecisionTreeClassifier(max_depth = 5)],
                                    "classifier__n_estimators": [200],
                                    "classifier__max_features": [0.9],
                                    "classifier__n_jobs": [-1]
                                    }})

    # Update dict with Gradient Boosting
    parameters.update({"Gradient Boosting": { 
                                            "classifier__learning_rate":[0.05], 
                                            "classifier__n_estimators": [200],
                                            "classifier__max_depth": [6],
                                            "classifier__min_samples_split": [0.10],
                                            "classifier__min_samples_leaf": [0.01],
                                            "classifier__max_features": ["sqrt"],
                                            "classifier__subsample": [1]
                                            }})


    # Update dict with Extra Trees
    parameters.update({"Extra Trees Ensemble": { 
                                                "classifier__n_estimators": [200],
                                                "classifier__class_weight": [None],
                                                "classifier__max_features": ["sqrt"],
                                                "classifier__max_depth" : [8],
                                                "classifier__min_samples_split": [0.01],
                                                "classifier__min_samples_leaf": [0.005],
                                                "classifier__criterion" :["gini", "entropy"]     ,
                                                "classifier__n_jobs": [-1]
                                                }})


    # Update dict with Random Forest Parameters
    parameters.update({"Random Forest": { 
                                        "classifier__n_estimators": [200],
                                        "classifier__class_weight": ["balanced"],
                                        "classifier__max_features": ["log2"],
                                        "classifier__max_depth" : [7],
                                        "classifier__min_samples_split": [0.005],
                                        "classifier__min_samples_leaf": [0.005],
                                        "classifier__criterion" :["entropy"],
                                        "classifier__n_jobs": [-1]
                                        }})

    # # Update dict with Ridge
    # parameters.update({"Ridge": { 
    #                             "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
    #                             }})

    # # Update dict with SGD Classifier
    # parameters.update({"SGD": { 
    #                             "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
    #                             "classifier__penalty": ["l1", "l2"],
    #                             "classifier__n_jobs": [-1]
    #                             }})


    # Update dict with BernoulliNB Classifier
    parameters.update({"BNB": { 
                                "classifier__alpha": [1e-7]
                                }})

    # Update dict with GaussianNB Classifier
    parameters.update({"GNB": { 
                                # "classifier__var_smoothing": []
                                }})

    # Update dict with K Nearest Neighbors Classifier
    parameters.update({"KNN": { 
                                "classifier__n_neighbors": [3],
                                "classifier__p": [1, 2, 3, 4, 5],
                                "classifier__leaf_size": [5],
                                "classifier__n_jobs": [-1]
                                }})

    # Update dict with MLPClassifier
    parameters.update({"MLP": { 
                                "classifier__hidden_layer_sizes": [(128,128)],
                                "classifier__activation": ["tanh"],
                                "classifier__learning_rate": ["constant", "invscaling", "adaptive"],
                                "classifier__max_iter": [10000,20000],
                                "classifier__alpha": [0.01],
                                }})

    # parameters.update({"LSVC": { 
    #                             "classifier__penalty": ["l2"],
    #                             "classifier__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
    #                             'multi_class':['crammer_singer']
    #                             }})

    # parameters.update({"NuSVC": { 
    #                             "classifier__nu": [0.25, 0.50, 0.75],
    #                             "classifier__kernel": ["linear", "rbf", "poly"],
    #                             "classifier__degree": [1,2,3,4,5,6],
    #                             }})

    # parameters.update({"SVC": { 
    #                             "classifier__kernel": ["linear", "rbf", "poly"],
    #                             "classifier__gamma": ["auto"],
    #                             "classifier__C": [0.1, 0.5, 1, 5, 10, 50, 100],
    #                             "classifier__degree": [1, 2, 3, 4, 5, 6]
    #                             }})


    # Update dict with Decision Tree Classifier
    parameters.update({"DTC": { 
                                "classifier__criterion" :["entropy"],
                                "classifier__splitter": ["best", "random"],
                                "classifier__class_weight": [None, "balanced"],
                                "classifier__max_features": ["sqrt"],
                                "classifier__max_depth" : [6],
                                "classifier__min_samples_split": [0.005],
                                "classifier__min_samples_leaf": [0.05],
                                }})

    # Update dict with Extra Tree Classifier
    parameters.update({"ETC": { 
                                "classifier__criterion" :["entropy"],
                                "classifier__splitter": ["best"],
                                "classifier__class_weight": [None, "balanced"],
                                "classifier__max_features": ["auto"],
                                "classifier__max_depth" : [5],
                                "classifier__min_samples_split": [0.01],
                                "classifier__min_samples_leaf": [0.05],
                                }})
    # # Update dict with LDA
    # parameters.update({"LDA": {"classifier__solver": ["svd"], 
    #                                         }})

    # # Update dict with QDA
    # parameters.update({"QDA": {"classifier__reg_param":[0.01*ii for ii in range(0, 101)], 
    #                                         }})
    # # Update dict with AdaBoost
    # parameters.update({"AdaBoost": { 
    #                                 "classifier__base_estimator": [DecisionTreeClassifier(max_depth = 3)],
    #                                 "classifier__n_estimators": [200],
    #                                 "classifier__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]
    #                                 }})

    # # Update dict with Bagging
    # parameters.update({"Bagging": { 
    #                                 "classifier__base_estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
    #                                 "classifier__n_estimators": [200],
    #                                 "classifier__max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #                                 "classifier__n_jobs": [-1]
    #                                 }})

    # # Update dict with Gradient Boosting
    # parameters.update({"Gradient Boosting": { 
    #                                         "classifier__learning_rate":[0.05,0.01,0.005,0.001], 
    #                                         "classifier__n_estimators": [200],
    #                                         "classifier__max_depth": [2,3,4,5,6],
    #                                         "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
    #                                         "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
    #                                         "classifier__max_features": ["sqrt"],
    #                                         "classifier__subsample": [0.8, 0.9, 1]
    #                                         }})


    # # Update dict with Extra Trees
    # parameters.update({"Extra Trees Ensemble": { 
    #                                             "classifier__n_estimators": [200],
    #                                             "classifier__class_weight": [None, "balanced"],
    #                                             "classifier__max_features": ["sqrt"],
    #                                             "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
    #                                             "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
    #                                             "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
    #                                             "classifier__criterion" :["gini", "entropy"]     ,
    #                                             "classifier__n_jobs": [-1]
    #                                             }})


    # # Update dict with Random Forest Parameters
    # parameters.update({"Random Forest": { 
    #                                     "classifier__n_estimators": [200],
    #                                     "classifier__class_weight": [None, "balanced"],
    #                                     "classifier__max_features": ["auto", "sqrt", "log2"],
    #                                     "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
    #                                     "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
    #                                     "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
    #                                     "classifier__criterion" :["gini", "entropy"],
    #                                     "classifier__n_jobs": [-1]
    #                                     }})

    # # Update dict with Ridge
    # parameters.update({"Ridge": { 
    #                             "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
    #                             }})

    # # Update dict with SGD Classifier
    # parameters.update({"SGD": { 
    #                             "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
    #                             "classifier__penalty": ["l1", "l2"],
    #                             "classifier__n_jobs": [-1]
    #                             }})


    # # Update dict with BernoulliNB Classifier
    # parameters.update({"BNB": { 
    #                             "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
    #                             }})

    # # Update dict with GaussianNB Classifier
    # parameters.update({"GNB": { 
    #                             "classifier__var_smoothing": [1e-9, 1e-8,1e-7, 1e-6, 1e-5]
    #                             }})

    # # Update dict with K Nearest Neighbors Classifier
    # parameters.update({"KNN": { 
    #                             "classifier__n_neighbors": list(range(1,5)),
    #                             "classifier__p": [1, 2, 3, 4, 5],
    #                             "classifier__leaf_size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    #                             "classifier__n_jobs": [-1]
    #                             }})

    # # Update dict with MLPClassifier
    # parameters.update({"MLP": { 
    #                             "classifier__hidden_layer_sizes": [(128,128)],
    #                             "classifier__activation": ["identity", "logistic", "tanh", "relu"],
    #                             "classifier__learning_rate": ["constant", "invscaling", "adaptive"],
    #                             "classifier__max_iter": [2000,10000],
    #                             "classifier__alpha": list(10.0 ** -np.arange(1, 10)),
    #                             }})

    # # parameters.update({"LSVC": { 
    # #                             "classifier__penalty": ["l2"],
    # #                             "classifier__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
    # #                             'multi_class':['crammer_singer']
    # #                             }})

    # # parameters.update({"NuSVC": { 
    # #                             "classifier__nu": [0.25, 0.50, 0.75],
    # #                             "classifier__kernel": ["linear", "rbf", "poly"],
    # #                             "classifier__degree": [1,2,3,4,5,6],
    # #                             }})

    # # parameters.update({"SVC": { 
    # #                             "classifier__kernel": ["linear", "rbf", "poly"],
    # #                             "classifier__gamma": ["auto"],
    # #                             "classifier__C": [0.1, 0.5, 1, 5, 10, 50, 100],
    # #                             "classifier__degree": [1, 2, 3, 4, 5, 6]
    # #                             }})


    # # Update dict with Decision Tree Classifier
    # parameters.update({"DTC": { 
    #                             "classifier__criterion" :["gini", "entropy"],
    #                             "classifier__splitter": ["best", "random"],
    #                             "classifier__class_weight": [None, "balanced"],
    #                             "classifier__max_features": ["auto", "sqrt", "log2"],
    #                             "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
    #                             "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
    #                             "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
    #                             }})

    # # Update dict with Extra Tree Classifier
    # parameters.update({"ETC": { 
    #                             "classifier__criterion" :["gini", "entropy"],
    #                             "classifier__splitter": ["best", "random"],
    #                             "classifier__class_weight": [None, "balanced"],
    #                             "classifier__max_features": ["auto"],
    #                             "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
    #                             "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
    #                             "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
    #                             }})
    # Filter Method: Spearman's Cross Correlation > 0.95
    # Make correlation matrix
    corr_matrix = X_train.corr(method = "spearman").abs()

    # Draw the heatmap
    sns.set(font_scale = 1.0)
    f, ax = plt.subplots(figsize=(13, 11),dpi=300)
    sns.heatmap(corr_matrix, cmap= "YlGnBu", square=True, ax = ax,annot=True)
    f.tight_layout()
    plt.savefig(OUTPATH+"%s/correlation_matrix.png"%DATASET_NAME, dpi = 1080)
    plt.close()
    # Select upper triangle of matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    # Drop features
    # X_train = X_train.drop(to_drop, axis = 1)
    # X_test = X_test.drop(to_drop, axis = 1)
    
    # Define classifier to use as the base of the recursive feature elimination algorithm
    selected_classifier = "Random Forest"
    classifier = classifiers[selected_classifier]

    # Tune classifier (Took = 4.8 minutes)
        
    # Scale features via Z-score normalization
    scaler = StandardScaler()

    # Define steps in pipeline
    steps = [("scaler", scaler), ("classifier", classifier)]

    # Initialize Pipeline object
    pipeline = Pipeline(steps = steps)
    
    # Define parameter grid
    param_grid = parameters[selected_classifier]

    # Initialize GridSearch object
    gscv = GridSearchCV(pipeline, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = "accuracy")
                    
    # Fit gscv
    print(f"Now tuning {selected_classifier}. Go grab a beer or something.")
    gscv.fit(X_train, np.ravel(y_train))  

    # Get best parameters and score
    best_params = gscv.best_params_
    best_score = gscv.best_score_
            
    # Update classifier parameters
    tuned_params = {item[12:]: best_params[item] for item in best_params}
    classifier.set_params(**tuned_params)
    
    # Define pipeline for RFECV
    steps = [("scaler", scaler), ("classifier", classifier)]
    pipe = PipelineRFE(steps = steps)

    # Initialize RFECV object
    feature_selector = RFECV(pipe, cv = 5, step = 1, scoring = "accuracy", verbose = 1)

    # Fit RFECV
    feature_selector.fit(X_train, np.ravel(y_train))

    # Get selected features
    feature_names = X_train.columns
    selected_features = feature_names[feature_selector.support_].tolist()
    
    # Get Performance Data
    performance_curve = {"Number of Features": list(range(1, len(feature_names) + 1)),
                        "Accuracy": feature_selector.grid_scores_}
    performance_curve = pd.DataFrame(performance_curve)
    performance_curve.to_csv(OUTPATH+'%s/performance_curve.csv'%DATASET_NAME,index=0)
    fnum=performance_curve[performance_curve.Accuracy==performance_curve.Accuracy.max()]['Number of Features'].values[0]
    # Performance vs Number of Features
    # Set graph style
    sns.set(font_scale = 1.75)
    sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
                "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
                'ytick.color': '0.4'})
    colors = sns.color_palette("RdYlGn", 20)
    line_color = colors[3]
    marker_colors = colors[-1]

    # Plot
    f, ax = plt.subplots(figsize=(13, 6.5))
    sns.lineplot(x = "Number of Features", y = "Accuracy", data = performance_curve,
                color = line_color, lw = 4, ax = ax)
    sns.regplot(x = performance_curve["Number of Features"], y = performance_curve["Accuracy"],
                color = marker_colors, fit_reg = False, scatter_kws = {"s": 200}, ax = ax)

    # Axes limits
    plt.xlim(0.5, len(feature_names)+0.5)
    # plt.ylim(0.60, 0.925)

    # Generate a bolded horizontal line at y = 0
    # ax.axhline(y = 0.625, color = 'black', linewidth = 1.3, alpha = .7)

    # Turn frame off
    ax.set_frame_on(False)

    # Tight layout
    plt.tight_layout()

    # Save Figure
    plt.savefig(OUTPATH+'%s/performance_curve.png'%DATASET_NAME,dpi=1080)
    plt.close()
    # Define pipeline for RFECV
    steps = [("scaler", scaler), ("classifier", classifier)]
    pipe = PipelineRFE(steps = steps)

    # Initialize RFE object
    feature_selector = RFE(pipe, n_features_to_select = fnum, step = 1, verbose = 1)

    # Fit RFE
    feature_selector.fit(X_train, np.ravel(y_train))

    # Get selected features labels
    feature_names = X_train.columns
    selected_features = feature_names[feature_selector.support_].tolist()
    
    # Get selected features data set
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # Train classifier
    classifier.fit(X_train, np.ravel(y_train))

    # Get feature importance
    feature_importance = pd.DataFrame(selected_features, columns = ["Feature Label"])
    feature_importance["Feature Importance"] = classifier.feature_importances_

    # Sort by feature importance
    feature_importance = feature_importance.sort_values(by="Feature Importance", ascending=False)
    feature_importance.to_csv(OUTPATH+"%s/feature_importance.csv"%DATASET_NAME,index=0)
    # Set graph style
    sns.set(font_scale = 1.75)
    sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
                "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
                'ytick.color': '0.4'})

    # Set figure size and create barplot
    f, ax = plt.subplots(figsize=(12, 9))
    sns.barplot(x = "Feature Importance", y = "Feature Label",
                palette = reversed(sns.color_palette('YlOrRd', 15)),  data = feature_importance)

    # Generate a bolded horizontal line at y = 0
    ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)

    # Turn frame off
    ax.set_frame_on(False)

    # Tight layout
    plt.tight_layout()

    # Save Figure
    plt.savefig(OUTPATH+"%s/feature_importance.png"%DATASET_NAME, dpi = 1080)
    plt.close()
    
    # Initialize dictionary to store results
    results = {}

    # Tune and evaluate classifiers
    for classifier_label, classifier in classifiers.items():
        try:
            # Print message to user
            print(f"Now tuning {classifier_label}.")
            
            # Scale features via Z-score normalization
            scaler = StandardScaler()
            
            # Define steps in pipeline
            steps = [("scaler", scaler), ("classifier", classifier)]
            
            # Initialize Pipeline object
            pipeline = Pipeline(steps = steps)
            
            # Define parameter grid
            param_grid = parameters[classifier_label]
            
            # Initialize GridSearch object
            gscv = GridSearchCV(pipeline, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = "accuracy")
                            
            # Fit gscv
            gscv.fit(X_train, np.ravel(y_train))  
            
            # Get best parameters and score
            best_params = gscv.best_params_
            best_score = gscv.best_score_
            
            # Update classifier parameters and define new pipeline with tuned classifier
            tuned_params = {item[12:]: best_params[item] for item in best_params}
            classifier.set_params(**tuned_params)
                    
            # Make predictions
            if classifier_label in DECISION_FUNCTIONS:
                y_pred = gscv.decision_function(X_test)
            else:
                y_pred = gscv.predict(X_test)
            
            # Evaluate model
            auc = metrics.accuracy_score(y_test.values.astype(np.float32), y_pred)
            
            # Save results
            result = {"Classifier": gscv,
                    "Best Parameters": best_params,
                    "Training Accuracy": best_score,
                    "Test Accuracy": auc}
            
            results.update({classifier_label: result})
        except:
            pass
    
    # Initialize auc_score dictionary
    auc_scores = {
                "Classifier": [],
                "Accuracy": [],
                "Accuracy Type": []
                }

    # Get AUC scores into dictionary
    for classifier_label in results:
        auc_scores.update({"Classifier": [classifier_label] + auc_scores["Classifier"],
                        "Accuracy": [results[classifier_label]["Training Accuracy"]] + auc_scores["Accuracy"],
                        "Accuracy Type": ["Training"] + auc_scores["Accuracy Type"]})
        
        auc_scores.update({"Classifier": [classifier_label] + auc_scores["Classifier"],
                        "Accuracy": [results[classifier_label]["Test Accuracy"]] + auc_scores["Accuracy"],
                        "Accuracy Type": ["Test"] + auc_scores["Accuracy Type"]})

    # Dictionary to PandasDataFrame
    auc_scores = pd.DataFrame(auc_scores)
    auc_scores.to_csv(OUTPATH+'%s/results.csv'%DATASET_NAME,index=0)
    # Set graph style
    sns.set(font_scale = 1.75)
    sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
                "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
                'ytick.color': '0.4'})

        
    # Colors
    training_color = sns.color_palette("RdYlBu", 10)[1]
    test_color = sns.color_palette("RdYlBu", 10)[-2]
    colors = [training_color, test_color]

    # Set figure size and create barplot
    f, ax = plt.subplots(figsize=(12, 9))

    sns.barplot(x="Accuracy", y="Classifier", hue="Accuracy Type", palette = colors,
                data=auc_scores)

    # Generate a bolded horizontal line at y = 0
    ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)

    # Turn frame off
    ax.set_frame_on(False)

    # Tight layout
    plt.tight_layout()

    # Save Figure
    plt.savefig(OUTPATH+"%s/Accuracy Scores.png"%DATASET_NAME, dpi = 1080)
    
    name=OUTPATH+'%s/results'%DATASET_NAME 
    with open(name + '.pkl','wb') as f:
        pickle.dump(results,f,pickle.HIGHEST_PROTOCOL)
        
def main(INPATH,OUTPATH,vlist=[20,30,40,50,60,70,80]):
    OUTPATH=tu.pathCheck(OUTPATH)
    DATASET_LIST=os.listdir(INPATH)
    for i in range(len(DATASET_LIST)):
        myfun(i,DATASET_LIST,INPATH,OUTPATH,vlist=vlist)
# test        
# INPATH='./dataset'
# OUTPATH='./results1'
# DATASET_LIST=os.listdir(INPATH)
# myfun(0,DATASET_LIST,INPATH,OUTPATH)