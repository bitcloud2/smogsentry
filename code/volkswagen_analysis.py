# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import ttest_rel
import cPickle as pickle


def gas_predict():
    '''
    Predicts the scores for the gasoline vehicles that have a diesel
    counterpart.
    '''

    # Open combo manfill file
    with open('../data/df_combo_manfill_final.pkl', 'r') as fp6:
        df_combo_slim = pickle.load(fp6)

    # Full Gas counterpart list
    df_volk = df_combo_slim[df_combo_slim['model'].isin([
        'AUDI A3', 'AUDI A3 Cabriolet', 'AUDI A6', 'AUDI A7', 'AUDI A8',
        'AUDI A8 L', 'AUDI Q5', 'AUDI Q5 Hybrid', 'AUDI Q7', 'BMW 328i',
        'BMW 328i Gran Turismo', 'BMW 328i Sports Wagon', 'BMW 535i',
        'BMW 535i Gran Turismo', 'BMW 740Li', 'BMW 740i', 'BMW X3 sDrive28i',
        'BMW X3 xDrive28i', 'BMW X5', 'BMW X5 M', 'CHEVROLET Cruze',
        'JEEP Grand Cherokee SRT8', 'PORSCHE Cayenne S',
        'PORSCHE Cayenne Turbo', 'VOLKSWAGEN Beetle',
        'VOLKSWAGEN Beetle Convertible', 'VOLKSWAGEN Golf',
        'VOLKSWAGEN Golf R', 'VOLKSWAGEN Golf SportWagen', 'VOLKSWAGEN Jetta',
        'VOLKSWAGEN Jetta Hybrid', 'VOLKSWAGEN Passat', 'VOLKSWAGEN Touareg',
        'VOLKSWAGEN Touareg Hybrid'])]

    # Select only gasoline
    df_vgas = df_volk[df_volk['fuel'] == 0]
    df_vgas = df_vgas[df_vgas['cert_region'] == 0]

    df_vgas_train = df_combo_slim.drop(df_vgas.index)

    # Creating y variables and dropping them from test set
    y_vgas_air = df_vgas['air_pollution_score']
    df_vgas.drop(df_vgas[['air_pollution_score', 'greenhouse_gas_score']
                         ], axis=1, inplace=True)

    # Creating y variables and dropping them from training set
    y_airpollution = df_vgas_train['air_pollution_score']
    df_vgas_train.drop(df_vgas_train[['air_pollution_score',
                                      'greenhouse_gas_score']
                                     ], axis=1, inplace=True)

    # Selecting columns for test set
    df_vgas_select = df_vgas[['displ',
                              'cert_region',
                              'trans_speed',
                              'weight',
                              'torque',
                              'torque_rpm',
                              'horsepower',
                              'msrp',
                              'city_mpg',
                              'hwy_mpg',
                              'cmb_mpg']]

    # Selecting columns for training set
    df_select = df_vgas_train[['displ',
                               'cert_region',
                               'trans_speed',
                               'weight',
                               'torque',
                               'torque_rpm',
                               'horsepower',
                               'msrp',
                               'city_mpg',
                               'hwy_mpg',
                               'cmb_mpg']]

    # GradientBoostingClassifier
    # Tuning parameters for full model
    gradc_vgas = GradientBoostingClassifier(min_samples_leaf= 3,
                                            n_estimators= 1300,
                                            min_samples_split= 1,
                                            random_state= 1,
                                            max_features= 'sqrt',
                                            max_depth= 3)
    gradc_vgas_air = gradc_vgas.fit(df_select, y_airpollution)

    air_pred = gradc_vgas_air.predict(df_vgas_select)
    print 'Volkswagon air prediction:', air_pred
    print 'Volkswagon air actual:', y_vgas_air.values

    # Pair-wise ttest checking if predictions are different from actual
    print 'Air t-stat, p-value:', ttest_rel(air_pred, y_vgas_air.values)

    return df_combo_slim


def maker_diff(df_combo_slim):
    '''
    Tests if there is a difference between the car makers.
    '''

    # Full Gas counterpart list
    df_makers = df_combo_slim[df_combo_slim['model'].isin([
          'AUDI A3', 'AUDI A3 Cabriolet', 'AUDI A6', 'AUDI A7', 'AUDI A8',
          'AUDI A8 L', 'AUDI Q5', 'AUDI Q5 Hybrid', 'AUDI Q7', 'BMW 328i',
          'BMW 328i Gran Turismo', 'BMW 328i Sports Wagon', 'BMW 535i',
          'BMW 535i Gran Turismo', 'BMW 740Li', 'BMW 740i', 'BMW X3 sDrive28i',
          'BMW X3 xDrive28i', 'BMW X5', 'BMW X5 M', 'CHEVROLET Cruze',
          'JEEP Grand Cherokee SRT8', 'PORSCHE Cayenne S',
          'PORSCHE Cayenne Turbo', 'VOLKSWAGEN Beetle',
          'VOLKSWAGEN Beetle Convertible', 'VOLKSWAGEN Golf',
          'VOLKSWAGEN Golf R', 'VOLKSWAGEN Golf SportWagen',
          'VOLKSWAGEN Jetta', 'VOLKSWAGEN Jetta Hybrid', 'VOLKSWAGEN Passat',
          'VOLKSWAGEN Touareg', 'VOLKSWAGEN Touareg Hybrid'])]

    # Select only gasoline
    df_makers = df_makers[df_makers['fuel'] == 0]
    df_makers = df_makers[df_makers['cert_region'] == 0]

    # Saving arrays from the predictions
    maker_pred = np.array([
          5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
          5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
          5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
          5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
          5.,  5.,  5.,  9.,  5.,  5.,  5.,  5.,  5.])
    maker_actual = np.array([
          9.,  9.,  9.,  9.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
          5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
          5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  7.,  7.,  5.,  5.,
          5.,  5.,  5.,  5.,  5.,  5.,  9.,  9.,  5.,  9.,  9.,  5.,  5.,  5.,
          5.,  5.,  5.,  7.,  5.,  5.,  5.,  5.,  5.])
    maker_dif = maker_pred - maker_actual

    # Adding difference in scores as feature
    df_makers['delta_score'] = maker_dif

    # Seperating makers into their own dataframes
    df_audi = df_makers[df_makers['model'].isin([
      'AUDI A3', 'AUDI A3 Cabriolet', 'AUDI A6', 'AUDI A7', 'AUDI A8',
      'AUDI A8 L', 'AUDI Q5', 'AUDI Q5 Hybrid', 'AUDI Q7'])]
    df_bmw = df_makers[df_makers['model'].isin([
         'BMW 328i', 'BMW 328i Gran Turismo', 'BMW 328i Sports Wagon',
         'BMW 535i', 'BMW 535i Gran Turismo', 'BMW 740Li', 'BMW 740i',
         'BMW X3 sDrive28i', 'BMW X3 xDrive28i', 'BMW X5', 'BMW X5 M'])]
    df_other = df_makers[df_makers['model'].isin([
         'CHEVROLET Cruze', 'JEEP Grand Cherokee SRT8', 'PORSCHE Cayenne S',
         'PORSCHE Cayenne Turbo'])]
    df_volkswagen = df_makers[df_makers['model'].isin([
         'VOLKSWAGEN Beetle', 'VOLKSWAGEN Beetle Convertible',
         'VOLKSWAGEN Golf', 'VOLKSWAGEN Golf R',
         'VOLKSWAGEN Golf SportWagen', 'VOLKSWAGEN Jetta',
         'VOLKSWAGEN Jetta Hybrid', 'VOLKSWAGEN Passat', 'VOLKSWAGEN Touareg',
         'VOLKSWAGEN Touareg Hybrid'])]

    # Extracting each maker's delta score as numpy array
    audi_array = df_audi['delta_score']
    audi_array = np.array(audi_array)
    bmw_array = df_bmw['delta_score']
    bmw_array = np.array(bmw_array)
    other_array = df_other['delta_score']
    other_array = np.array(other_array)
    volkswagen_array = df_volkswagen['delta_score']
    volkswagen_array = np.array(volkswagen_array)

    # One-way ANOVA to check for differences between makers populations
    print f_oneway(volkswagen_array, audi_array, bmw_array, other_array)
