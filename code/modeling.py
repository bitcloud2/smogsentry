# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import cPickle as pickle


def fit():
    '''
    Fits a Gradient Boosted Random Forest Classifier model to dataset and
    then saves it to a pickle file.
    '''

    # Open combo manfill file
    with open('../data/df_combo_manfill_final.pkl', 'r') as fp6:
        df_combo_slim = pickle.load(fp6)

    # Creating y variables and dropping them from feature set
    y_airpollution = df_combo_slim['air_pollution_score']
    df_combo_norm = df_combo_slim.drop(df_combo_slim[
                    ['air_pollution_score', 'greenhouse_gas_score']], axis=1)

    # Selecting columns for model
    df_select = df_combo_norm[['displ'
                               'cert_region'
                               'trans_speed'
                               'weight'
                               'torque'
                               'torque_rpm'
                               'horsepower'
                               'msrp'
                               'city_mpg'
                               'hwy_mpg'
                               'cmb_mpg']]

    # Test/Train split
    X_airpollution_train, X_airpollution_test, \
        y_airpollution_train, y_airpollution_test = \
        train_test_split(df_select, y_airpollution, random_state=42)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc_air = rfc.fit(X_airpollution_train, y_airpollution_train)

    print 'Air Pollution Score:', \
        rfc_air.score(X_airpollution_test, y_airpollution_test)
    # print 'Air importances:', rfc_air.feature_importances_
    air_pred = rfc_air.predict(X_airpollution_test)
    print 'Air Pollution Precision:', \
        precision_score(y_airpollution_test, air_pred)
    print 'Air Pollution Recall:', \
        recall_score(y_airpollution_test, air_pred)
    print 'Air Pollution f1(macro):', \
        f1_score(y_airpollution_test, air_pred, average='macro')

    # Grid search for best gradient boosted parameters
    gradient_boost_grid = {'max_depth': [1, 3, 5],
                           'max_features': ['sqrt', 'log2', None],
                           'min_samples_split': [1, 3, 5],
                           'min_samples_leaf': [1, 3, 5],
                           'n_estimators': [50, 500, 1500],
                           'random_state': [1]}

    gdbr_gridsearch = GridSearchCV(GradientBoostingClassifier(),
                                   gradient_boost_grid,
                                   n_jobs=-1,
                                   verbose=True)
    gdbr_gridsearch.fit(X_airpollution_train, y_airpollution_train)

    print "best parameters:", gdbr_gridsearch.best_params_

    # K-fold validation
    y_airpollution.index = range(1987)
    df_select.index = range(1987)
    y_airpollution_k = y_airpollution.reshape(1987, 1)

    kf = KFold(1987, n_folds=5, shuffle=True)
    air_acc_lst = []
    air_prec_lst = []
    air_rec_lst = []
    for train_index, test_index in kf:
        X_airpollution_train, X_airpollution_test = \
            df_select.loc[train_index], df_select.loc[test_index]
        y_airpollution_train, y_airpollution_test = \
            y_airpollution_k[train_index], y_airpollution_k[test_index]

        # GradientBoostingClassifier
        gradc = GradientBoostingClassifier(min_samples_leaf=3,
                                           n_estimators=1300,
                                           min_samples_split=1,
                                           random_state=1,
                                           max_features='sqrt',
                                           max_depth=3)
        gradc_air = gradc.fit(X_airpollution_train, y_airpollution_train)

        air_acc_lst.append(gradc_air.score(X_airpollution_test,
                                           y_airpollution_test))
        air_pred = gradc_air.predict(X_airpollution_test)
        air_prec_lst.append(precision_score(y_airpollution_test, air_pred))
        air_rec_lst.append(recall_score(y_airpollution_test, air_pred))

    print 'air acc:', np.mean(air_acc_lst)
    print 'air prec:', np.mean(air_prec_lst)
    print 'air rec:', np.mean(air_rec_lst)

    # GradientBoostingClassifier
    gradbc = GradientBoostingClassifier(min_samples_leaf=3,
                                        n_estimators=1300,
                                        min_samples_split=1,
                                        random_state=1,
                                        max_features='sqrt',
                                        max_depth=3)
    gradbc_air = gradbc.fit(df_select, y_airpollution)

    # Save model in pickle file
    with open('data/model_gradboost_air15_final.pkl', 'w') as f:
            pickle.dump(gradbc_air, f)
