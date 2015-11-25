# Import necessary modules
import numpy as np
import pandas as pd
import requests
import bs4
import json
import datetime as dt
import time
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import ttest_ind, ttest_rel
import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns
%pylab inline


def create_newdb():
    '''
    Loads in the EPA dataset, performs initial cleaning, and returns database
    in pandas dataframe.
    '''
    # Import EPA database
    df = pd.read_csv('../data/all_alpha_15.txt', sep='\t')
    df.columns = ['model', 'displ', 'cyl', 'trans', 'drive', 'fuel', 'cert_region',
           'stnd', 'stnd_description', 'underhood_id', 'veh_class',
           'air_pollution_score', 'city_mpg', 'hwy_mpg', 'cmb_mpg',
           'greenhouse_gas_score', 'smartway', 'comb_co2']

    # Removing cars with 0 emissions
    df = df[df['fuel'] != 'Electricity']
    df = df[df['fuel'] != 'Gasoline/Electricity']
    df = df[df['fuel'] != 'Hydrogen']

    # Seperate by type of fuel
    df_eth_dirty = df[df['fuel'] == 'Ethanol/Gas']
    df_gas_dirty = df[df['fuel'] == 'Ethanol/Gas']
    df_cng_dirty = df[df['fuel'] == 'CNG/Gasoline']
    df_gasoline_dirty = df[df['fuel'] == 'CNG/Gasoline']

    df_eth_dirty.loc[:, 'fuel'] = 'Ethanol'
    df_gas_dirty.loc[:, 'fuel'] = 'eGas'
    df_cng_dirty.loc[:, 'fuel'] = 'CNG'
    df_gasoline_dirty.loc[:, 'fuel'] = 'Gasoline'

    # Converting NaN's to 0
    df_eth_dirty[df_eth_dirty.loc[:, 'city_mpg'].isnull()] = df_eth_dirty[df_eth_dirty.loc[:, 'city_mpg'].isnull()].fillna('0')
    df_gas_dirty[df_gas_dirty.loc[:, 'city_mpg'].isnull()] = df_gas_dirty[df_gas_dirty.loc[:, 'city_mpg'].isnull()].fillna('0')
    df_eth_dirty[df_eth_dirty.loc[:, 'hwy_mpg'].isnull()] = df_eth_dirty[df_eth_dirty.loc[:, 'hwy_mpg'].isnull()].fillna('0')
    df_gas_dirty[df_gas_dirty.loc[:, 'hwy_mpg'].isnull()] = df_gas_dirty[df_gas_dirty.loc[:, 'hwy_mpg'].isnull()].fillna('0')
    df_eth_dirty[df_eth_dirty.loc[:, 'cmb_mpg'].isnull()] = df_eth_dirty[df_eth_dirty.loc[:, 'cmb_mpg'].isnull()].fillna('0')
    df_gas_dirty[df_gas_dirty.loc[:, 'cmb_mpg'].isnull()] = df_gas_dirty[df_gas_dirty.loc[:, 'cmb_mpg'].isnull()].fillna('0')
    df_eth_dirty[df_eth_dirty.loc[:, 'comb_co2'].isnull()] = df_eth_dirty[df_eth_dirty.loc[:, 'comb_co2'].isnull()].fillna('0')
    df_gas_dirty[df_gas_dirty.loc[:, 'comb_co2'].isnull()] = df_gas_dirty[df_gas_dirty.loc[:, 'comb_co2'].isnull()].fillna('0')

    df_cng_dirty[df_cng_dirty.loc[:, 'city_mpg'].isnull()] = df_cng_dirty[df_cng_dirty.loc[:, 'city_mpg'].isnull()].fillna('0')
    df_gasoline_dirty[df_gasoline_dirty.loc[:, 'city_mpg'].isnull()] = df_gasoline_dirty[df_gasoline_dirty.loc[:, 'city_mpg'].isnull()].fillna('0')
    df_cng_dirty[df_cng_dirty.loc[:, 'hwy_mpg'].isnull()] = df_cng_dirty[df_cng_dirty.loc[:, 'hwy_mpg'].isnull()].fillna('0')
    df_gasoline_dirty[df_gasoline_dirty.loc[:, 'hwy_mpg'].isnull()] = df_gasoline_dirty[df_gasoline_dirty.loc[:, 'hwy_mpg'].isnull()].fillna('0')
    df_cng_dirty[df_cng_dirty.loc[:, 'cmb_mpg'].isnull()] = df_cng_dirty[df_cng_dirty.loc[:, 'cmb_mpg'].isnull()].fillna('0')
    df_gasoline_dirty[df_gasoline_dirty.loc[:, 'cmb_mpg'].isnull()] = df_gasoline_dirty[df_gasoline_dirty.loc[:, 'cmb_mpg'].isnull()].fillna('0')
    df_cng_dirty[df_cng_dirty.loc[:, 'comb_co2'].isnull()] = df_cng_dirty[df_cng_dirty.loc[:, 'comb_co2'].isnull()].fillna('0')
    df_gasoline_dirty[df_gasoline_dirty.loc[:, 'comb_co2'].isnull()] = df_gasoline_dirty[df_gasoline_dirty.loc[:, 'comb_co2'].isnull()].fillna('0')

    # Seperate Ethanol and gas values
    df_eth_dirty.loc[:, 'city_mpg'] = df_eth_dirty.loc[:, 'city_mpg'].str.split('/')
    df_eth_dirty.loc[:, 'hwy_mpg'] = df_eth_dirty.loc[:, 'hwy_mpg'].str.split('/')
    df_eth_dirty.loc[:, 'cmb_mpg'] = df_eth_dirty.loc[:, 'cmb_mpg'].str.split('/')
    df_eth_dirty.loc[:, 'comb_co2'] = df_eth_dirty.loc[:, 'comb_co2'].str.split('/')
    df_gas_dirty.loc[:, 'city_mpg'] = df_gas_dirty.loc[:, 'city_mpg'].str.split('/')
    df_gas_dirty.loc[:, 'hwy_mpg'] = df_gas_dirty.loc[:, 'hwy_mpg'].str.split('/')
    df_gas_dirty.loc[:, 'cmb_mpg'] = df_gas_dirty.loc[:, 'cmb_mpg'].str.split('/')
    df_gas_dirty.loc[:, 'comb_co2'] = df_gas_dirty.loc[:, 'comb_co2'].str.split('/')

    # Seperate CNG and Gasoline values
    df_cng_dirty.loc[:, 'city_mpg'] = df_cng_dirty.loc[:, 'city_mpg'].str.split('/')
    df_cng_dirty.loc[:, 'hwy_mpg'] = df_cng_dirty.loc[:, 'hwy_mpg'].str.split('/')
    df_cng_dirty.loc[:, 'cmb_mpg'] = df_cng_dirty.loc[:, 'cmb_mpg'].str.split('/')
    df_cng_dirty.loc[:, 'comb_co2'] = df_cng_dirty.loc[:, 'comb_co2'].str.split('/')
    df_gasoline_dirty.loc[:, 'city_mpg'] = df_gasoline_dirty.loc[:, 'city_mpg'].str.split('/')
    df_gasoline_dirty.loc[:, 'hwy_mpg'] = df_gasoline_dirty.loc[:, 'hwy_mpg'].str.split('/')
    df_gasoline_dirty.loc[:, 'cmb_mpg'] = df_gasoline_dirty.loc[:, 'cmb_mpg'].str.split('/')
    df_gasoline_dirty.loc[:, 'comb_co2'] = df_gasoline_dirty.loc[:, 'comb_co2'].str.split('/')

    # Convert non-helpful values to 0
    # Good --------- Changed gas[eth] to gas[gas]
    df_eth_dirty.ix[df_eth_dirty['city_mpg'].map(lambda x: len(x) < 2), 'city_mpg'] = df_eth_dirty[df_eth_dirty['city_mpg'].map(lambda x: len(x) < 2)]['city_mpg'].apply(lambda x: [0])
    df_eth_dirty.ix[df_eth_dirty['hwy_mpg'].map(lambda x: len(x) < 2), 'hwy_mpg'] = df_eth_dirty[df_eth_dirty['hwy_mpg'].map(lambda x: len(x) < 2)]['hwy_mpg'].apply(lambda x: [0])
    df_eth_dirty.ix[df_eth_dirty['cmb_mpg'].map(lambda x: len(x) < 2), 'cmb_mpg'] = df_eth_dirty[df_eth_dirty['cmb_mpg'].map(lambda x: len(x) < 2)]['cmb_mpg'].apply(lambda x: [0])
    df_eth_dirty.ix[df_eth_dirty['comb_co2'].map(lambda x: len(x) < 2), 'comb_co2'] = df_eth_dirty[df_eth_dirty['comb_co2'].map(lambda x: len(x) < 2)]['comb_co2'].apply(lambda x: [0])
    df_gas_dirty.ix[df_gas_dirty['city_mpg'].map(lambda x: len(x) < 2), 'city_mpg'] = df_gas_dirty[df_gas_dirty['city_mpg'].map(lambda x: len(x) < 2)]['city_mpg'].apply(lambda x: [0])
    df_gas_dirty.ix[df_gas_dirty['hwy_mpg'].map(lambda x: len(x) < 2), 'hwy_mpg'] = df_gas_dirty[df_gas_dirty['hwy_mpg'].map(lambda x: len(x) < 2)]['hwy_mpg'].apply(lambda x: [0])
    df_gas_dirty.ix[df_gas_dirty['cmb_mpg'].map(lambda x: len(x) < 2), 'cmb_mpg'] = df_gas_dirty[df_gas_dirty['cmb_mpg'].map(lambda x: len(x) < 2)]['cmb_mpg'].apply(lambda x: [0])
    df_gas_dirty.ix[df_gas_dirty['comb_co2'].map(lambda x: len(x) < 2), 'comb_co2'] = df_gas_dirty[df_gas_dirty['comb_co2'].map(lambda x: len(x) < 2)]['comb_co2'].apply(lambda x: [0])

    df_cng_dirty.ix[df_cng_dirty['city_mpg'].map(lambda x: len(x) < 2), 'city_mpg'] = df_cng_dirty[df_cng_dirty['city_mpg'].map(lambda x: len(x) < 2)]['city_mpg'].apply(lambda x: [0])
    df_cng_dirty.ix[df_cng_dirty['hwy_mpg'].map(lambda x: len(x) < 2), 'hwy_mpg'] = df_cng_dirty[df_cng_dirty['hwy_mpg'].map(lambda x: len(x) < 2)]['hwy_mpg'].apply(lambda x: [0])
    df_cng_dirty.ix[df_cng_dirty['cmb_mpg'].map(lambda x: len(x) < 2), 'cmb_mpg'] = df_cng_dirty[df_cng_dirty['cmb_mpg'].map(lambda x: len(x) < 2)]['cmb_mpg'].apply(lambda x: [0])
    df_cng_dirty.ix[df_cng_dirty['comb_co2'].map(lambda x: len(x) < 2), 'comb_co2'] = df_cng_dirty[df_cng_dirty['comb_co2'].map(lambda x: len(x) < 2)]['comb_co2'].apply(lambda x: [0])
    df_gasoline_dirty.ix[df_gasoline_dirty['city_mpg'].map(lambda x: len(x) < 2), 'city_mpg'] = df_gasoline_dirty[df_gasoline_dirty['city_mpg'].map(lambda x: len(x) < 2)]['city_mpg'].apply(lambda x: [0])
    df_gasoline_dirty.ix[df_gasoline_dirty['hwy_mpg'].map(lambda x: len(x) < 2), 'hwy_mpg'] = df_gasoline_dirty[df_gasoline_dirty['hwy_mpg'].map(lambda x: len(x) < 2)]['hwy_mpg'].apply(lambda x: [0])
    df_gasoline_dirty.ix[df_gasoline_dirty['cmb_mpg'].map(lambda x: len(x) < 2), 'cmb_mpg'] = df_gasoline_dirty[df_gasoline_dirty['cmb_mpg'].map(lambda x: len(x) < 2)]['cmb_mpg'].apply(lambda x: [0])
    df_gasoline_dirty.ix[df_gasoline_dirty['comb_co2'].map(lambda x: len(x) < 2), 'comb_co2'] = df_gasoline_dirty[df_gasoline_dirty['comb_co2'].map(lambda x: len(x) < 2)]['comb_co2'].apply(lambda x: [0])

    # Grab correct value for ethanol or gas for respective tables
    df_eth_dirty.loc[:, 'city_mpg'] = df_eth_dirty.loc[:, 'city_mpg'].apply(lambda x: [x[0]])
    df_eth_dirty.loc[:, 'hwy_mpg'] = df_eth_dirty.loc[:, 'hwy_mpg'].apply(lambda x: [x[0]])
    df_eth_dirty.loc[:, 'cmb_mpg'] = df_eth_dirty.loc[:, 'cmb_mpg'].apply(lambda x: [x[0]])
    df_eth_dirty.loc[:, 'comb_co2'] = df_eth_dirty.loc[:, 'comb_co2'].apply(lambda x: [x[0]])
    df_gas_dirty.ix[df_gas_dirty['city_mpg'].map(lambda x: len(x) > 1), 'city_mpg'] = df_gas_dirty.ix[df_gas_dirty['city_mpg'].map(lambda x: len(x) > 1), 'city_mpg'].apply(lambda x: [x[1]])
    df_gas_dirty.ix[df_gas_dirty['hwy_mpg'].map(lambda x: len(x) > 1), 'hwy_mpg'] = df_gas_dirty.ix[df_gas_dirty['hwy_mpg'].map(lambda x: len(x) > 1), 'hwy_mpg'].apply(lambda x: [x[1]])
    df_gas_dirty.ix[df_gas_dirty['cmb_mpg'].map(lambda x: len(x) > 1), 'cmb_mpg'] = df_gas_dirty.ix[df_gas_dirty['cmb_mpg'].map(lambda x: len(x) > 1), 'cmb_mpg'].apply(lambda x: [x[1]])
    df_gas_dirty.ix[df_gas_dirty['comb_co2'].map(lambda x: len(x) > 1), 'comb_co2'] = df_gas_dirty.ix[df_gas_dirty['comb_co2'].map(lambda x: len(x) > 1), 'comb_co2'].apply(lambda x: [x[1]])

    # Grab correct value for CNG and Gasoline for respective tables
    df_cng_dirty.ix[df_cng_dirty['city_mpg'].map(lambda x: len(x) > 1), 'city_mpg'] = df_gas_dirty.ix[df_gas_dirty['city_mpg'].map(lambda x: len(x) > 1), 'city_mpg'].apply(lambda x: [x[0]])
    df_cng_dirty.ix[df_cng_dirty['hwy_mpg'].map(lambda x: len(x) > 1), 'hwy_mpg'] = df_gas_dirty.ix[df_gas_dirty['hwy_mpg'].map(lambda x: len(x) > 1), 'hwy_mpg'].apply(lambda x: [x[0]])
    df_cng_dirty.ix[df_cng_dirty['cmb_mpg'].map(lambda x: len(x) > 1), 'cmb_mpg'] = df_gas_dirty.ix[df_gas_dirty['cmb_mpg'].map(lambda x: len(x) > 1), 'cmb_mpg'].apply(lambda x: [x[0]])
    df_cng_dirty.ix[df_cng_dirty['comb_co2'].map(lambda x: len(x) > 1), 'comb_co2'] = df_gas_dirty.ix[df_gas_dirty['comb_co2'].map(lambda x: len(x) > 1), 'comb_co2'].apply(lambda x: [x[0]])
    df_gasoline_dirty.ix[df_gasoline_dirty['city_mpg'].map(lambda x: len(x) > 1), 'city_mpg'] = df_gasoline_dirty.ix[df_gasoline_dirty['city_mpg'].map(lambda x: len(x) > 1), 'city_mpg'].apply(lambda x: [x[1]])
    df_gasoline_dirty.ix[df_gasoline_dirty['hwy_mpg'].map(lambda x: len(x) > 1), 'hwy_mpg'] = df_gasoline_dirty.ix[df_gasoline_dirty['hwy_mpg'].map(lambda x: len(x) > 1), 'hwy_mpg'].apply(lambda x: [x[1]])
    df_gasoline_dirty.ix[df_gasoline_dirty['cmb_mpg'].map(lambda x: len(x) > 1), 'cmb_mpg'] = df_gasoline_dirty.ix[df_gasoline_dirty['cmb_mpg'].map(lambda x: len(x) > 1), 'cmb_mpg'].apply(lambda x: [x[1]])
    df_gasoline_dirty.ix[df_gasoline_dirty['comb_co2'].map(lambda x: len(x) > 1), 'comb_co2'] = df_gasoline_dirty.ix[df_gasoline_dirty['comb_co2'].map(lambda x: len(x) > 1), 'comb_co2'].apply(lambda x: [x[1]])

    # Making ints from lists
    df_eth_dirty.loc[:, 'city_mpg'] = df_eth_dirty.loc[:, 'city_mpg'].apply(lambda x: x[0])
    df_eth_dirty.loc[:, 'hwy_mpg'] = df_eth_dirty.loc[:, 'hwy_mpg'].apply(lambda x: x[0])
    df_eth_dirty.loc[:, 'cmb_mpg'] = df_eth_dirty.loc[:, 'cmb_mpg'].apply(lambda x: x[0])
    df_eth_dirty.loc[:, 'comb_co2'] = df_eth_dirty.loc[:, 'comb_co2'].apply(lambda x: x[0])
    df_gas_dirty.loc[:, 'city_mpg'] = df_gas_dirty.loc[:, 'city_mpg'].apply(lambda x: x[0])
    df_gas_dirty.loc[:, 'hwy_mpg'] = df_gas_dirty.loc[:, 'hwy_mpg'].apply(lambda x: x[0])
    df_gas_dirty.loc[:, 'cmb_mpg'] = df_gas_dirty.loc[:, 'cmb_mpg'].apply(lambda x: x[0])
    df_gas_dirty.loc[:, 'comb_co2'] = df_gas_dirty.loc[:, 'comb_co2'].apply(lambda x: x[0])

    df_cng_dirty.loc[:, 'city_mpg'] = df_cng_dirty.loc[:, 'city_mpg'].apply(lambda x: x[0])
    df_cng_dirty.loc[:, 'hwy_mpg'] = df_cng_dirty.loc[:, 'hwy_mpg'].apply(lambda x: x[0])
    df_cng_dirty.loc[:, 'cmb_mpg'] = df_cng_dirty.loc[:, 'cmb_mpg'].apply(lambda x: x[0])
    df_cng_dirty.loc[:, 'comb_co2'] = df_cng_dirty.loc[:, 'comb_co2'].apply(lambda x: x[0])
    df_gasoline_dirty.loc[:, 'city_mpg'] = df_gasoline_dirty.loc[:, 'city_mpg'].apply(lambda x: x[0])
    df_gasoline_dirty.loc[:, 'hwy_mpg'] = df_gasoline_dirty.loc[:, 'hwy_mpg'].apply(lambda x: x[0])
    df_gasoline_dirty.loc[:, 'cmb_mpg'] = df_gasoline_dirty.loc[:, 'cmb_mpg'].apply(lambda x: x[0])
    df_gasoline_dirty.loc[:, 'comb_co2'] = df_gasoline_dirty.loc[:, 'comb_co2'].apply(lambda x: x[0])

    # Convert to correct type
    df_eth_dirty[['city_mpg', 'hwy_mpg', 'cmb_mpg', 'comb_co2']] = df_eth_dirty[['city_mpg', 'hwy_mpg', 'cmb_mpg', 'comb_co2']].astype(int64)
    df_gas_dirty[['city_mpg', 'hwy_mpg', 'cmb_mpg', 'comb_co2']] = df_gas_dirty[['city_mpg', 'hwy_mpg', 'cmb_mpg', 'comb_co2']].astype(int64)
    df_cng_dirty[['city_mpg', 'hwy_mpg', 'cmb_mpg', 'comb_co2']] = df_cng_dirty[['city_mpg', 'hwy_mpg', 'cmb_mpg', 'comb_co2']].astype(int64)
    df_gasoline_dirty[['city_mpg', 'hwy_mpg', 'cmb_mpg', 'comb_co2']] = df_gasoline_dirty[['city_mpg', 'hwy_mpg', 'cmb_mpg', 'comb_co2']].astype(int64)

    # Combine all df, drop fuel type, and sort newly formed df
    df_dirty = pd.concat([df, df_eth_dirty, df_gas_dirty, df_cng_dirty, df_gasoline_dirty])
    df_clean_temp = df_dirty[df_dirty['fuel'] != 'Ethanol/Gas']
    df_clean = df_clean_temp[df_clean_temp['fuel'] != 'CNG/Gasoline']
    df_clean.sort(['model', 'displ', 'cyl', 'trans', 'drive', 'fuel'], inplace=True)
    df_clean.dropna(inplace=True)

    # Setting new index
    df_clean.index = range(2561)

    return df_clean



def cat_conv():
    '''
    Converts several of the dataframe's features into categorical variables a model can handle.
    '''

    # Convert 'smartway' feature to int categorical
    smartway_conv = {'No': 0, 'Yes': 1, 'Elite': 2}
    df_clean.loc[:, 'smartway'] = df_clean.loc[:, 'smartway'].map(smartway_conv)

    # Convert 'trans' feature to int categorical
    # Automatic: 0, Manual: 1, CVT: 2
    trans_conv = {'AMS-6': 0
                  , 'AMS-7': 0
                  , 'AMS-8': 0
                  , 'Auto-4': 0
                  , 'Auto-5': 0
                  , 'Auto-6': 0
                  , 'Auto-7': 0
                  , 'Auto-8': 0
                  , 'Auto-9': 0
                  , 'AutoMan-6': 0
                  , 'AutoMan-7': 0
                  , 'AutoMan-8': 0
                  , 'CVT': 2
                  , 'Man-5': 1
                  , 'Man-6': 1
                  , 'Man-7': 1
                  , 'SCV-6': 0
                  , 'SCV-7': 0
                  , 'SCV-8': 0
                  , 'SemiAuto-5': 0
                  , 'SemiAuto-6': 0
                  , 'SemiAuto-7': 0
                  , 'SemiAuto-8': 0
                  , 'SemiAuto-9': 0}
    trans_speed_conv = {'AMS-6': 6
                      , 'AMS-7': 7
                      , 'AMS-8': 8
                      , 'Auto-4': 4
                      , 'Auto-5': 5
                      , 'Auto-6': 6
                      , 'Auto-7': 7
                      , 'Auto-8': 8
                      , 'Auto-9': 9
                      , 'AutoMan-6': 6
                      , 'AutoMan-7': 7
                      , 'AutoMan-8': 8
                      , 'CVT': 0
                      , 'Man-5': 5
                      , 'Man-6': 6
                      , 'Man-7': 7
                      , 'SCV-6': 6
                      , 'SCV-7': 7
                      , 'SCV-8': 8
                      , 'SemiAuto-5': 5
                      , 'SemiAuto-6': 6
                      , 'SemiAuto-7': 7
                      , 'SemiAuto-8': 8
                      , 'SemiAuto-9': 9}
    df_clean.loc[:, 'trans_speed'] = df_clean.loc[:, 'trans'].map(trans_speed_conv)
    df_clean.loc[:, 'trans'] = df_clean.loc[:, 'trans'].map(trans_conv)

    # Convert 'drive' feature to int categorical
    drive_conv = {'2WD': 0, '4WD': 1}
    df_clean.loc[:, 'drive'] = df_clean.loc[:, 'drive'].map(drive_conv)

    # Convert 'fuel' feature to int categorical
    fuel_conv = {'Gasoline': 0, 'eGas': 0, 'Diesel': 1, 'Ethanol': 2, 'CNG': 3}
    df_clean.loc[:, 'fuel'] = df_clean.loc[:, 'fuel'].map(fuel_conv)

    # Convert 'cert_region' feature to int categorical
    cert_region_conv = {'FA': 0, 'CA': 1}
    df_clean.loc[:, 'cert_region'] = df_clean.loc[:, 'cert_region'].map(cert_region_conv)

    # Convert 'stnd' feature to int categorical
    stnd_conv = {'B2': 0
                 , 'B3': 1
                 , 'B4': 2
                 , 'B5': 3
                 , 'B6': 4
                 , 'B8': 5
                 , 'L2': 6
                 , 'L2ULEV125': 7
                 , 'L3LEV160': 8
                 , 'L3SULEV30': 9
                 , 'L3SULEV30/PZEV': 10
                 , 'L3ULEV125': 11
                 , 'L3ULEV70': 12
                 , 'PZEV': 13
                 , 'S2': 14
                 , 'T3B110': 15
                 , 'T3B125': 16
                 , 'T3B30': 17
                 , 'T3B70': 18
                 , 'T3B85': 19
                 , 'U2': 20}
    df_clean.loc[:, 'stnd'] = df_clean.loc[:, 'stnd'].map(stnd_conv)

    # Convert 'veh_class' feature to int categorical
    veh_class_conv = {'small car': 0
               , 'small SUV': 1
               , 'midsize car': 2
               , 'large car': 3
               , 'standard SUV': 4
               , 'station wagon': 5
               , 'special purpose': 6
               , 'pickup': 7
               , 'van': 8
               , 'minivan': 9}
    df_clean.loc[:, 'veh_class'] = df_clean.loc[:, 'veh_class'].map(veh_class_conv)

    # Setting new index
    df_clean.dropna(inplace=True)
    df_clean.index = range(2124)

    # Converting to correct types
    df_clean[['greenhouse_gas_score',
              'city_mpg',
              'hwy_mpg', 
              'cmb_mpg', 
              'comb_co2']] = \
        df_clean[['greenhouse_gas_score', 
                  'city_mpg', 
                  'hwy_mpg', 
                  'cmb_mpg', 
                  'comb_co2']].astype(float64)

    df_clean[['trans', 
              'trans_speed', 
              'drive', 
              'fuel', 
              'cert_region', 
              'stnd', 
              'veh_class']] = \
      df_clean[['trans', 
                'trans_speed', 
                'drive', 
                'fuel', 
                'cert_region', 
                'stnd', 
                'veh_class']].astype(int64)


def merge_auto_scrape(saved_copy=False):
    '''
    Merges new features from an automatic scrape of the Motortrend website.
    
    Parameters:
    save_copy : boolean, optional (default=False)
        Whether to save dataframe locally as a pickle file.
    
    New features are: 
        'msrp': Manufacture's suggested retail price, 
        'fuel_type': Specific type of fuel the car uses
                    {'Unleaded Regular': 0,
                    'Unleaded Midgrade': 1,
                    'Unleaded Premium': 2,
                    'Diesel': 3}
        'weight': Curb weight of the car in pounds, 
        'torque': Maximum torque of the car, 
        'torque_rpm': Engine RPM at maximum torque, 
        'horsepower': Engine horsepower.
    '''
    
    # Reading in json of scraped data
    with open('data/motortrend_specs_2015.json', 'r') as fp:
        s_temp = json.load(fp)
    
    # Creating df of new features
    user_ids = []
    frames = []
    
    for user_id, d in s_temp.iteritems():
        user_ids.append(user_id)
        frames.append(pd.DataFrame.from_dict(d, orient='index'))
    
    s_temp = pd.concat(frames, keys=user_ids)
    s_temp['model'] = zip(s_temp.index.get_level_values(0), s_temp.index.get_level_values(1))
    s_temp['model'] = s_temp['model'].apply(lambda x: x[0] + ' ' + x[1])
    s_temp = s_temp.reset_index(level=1, drop=True)
    
    # Left_Outer join of df_clean and new features
    df_combo = df_clean.merge(s_temp, how='left', left_on='model', right_on='model')

    # Set correct values
    df_combo['weight'] = df_combo[df_combo['weight'].notnull()]['weight'].apply(lambda x: x[:-5])
    df_combo[df_combo['weight'] == ''] = np.nan
    df_combo['msrp'] = df_combo[df_combo['msrp'].notnull()]['msrp'].apply(lambda x: int(x.replace(',', '')))
    
    # Convert 'fuel_type' feature to int categorical
    fuel_type_conv = {'Unleaded Regular': 0
               , 'Unleaded Midgrade': 1
               , 'Unleaded Premium': 2
               , 'Diesel': 3}
    df_combo.loc[:, 'fuel_type'] = df_combo.loc[:, 'fuel_type'].map(fuel_type_conv)
    
    # Filling in dropped row.
    df_combo.ix[1090:1091] = \
       df_clean.ix[1090:1091]
   
   # Filling in missing values from the automatic scrape with data from
   # scrape of links collected manually.
   # Open link dict locally
    with open('data/motortrend_links.json', 'r') as fp3:
        linked_dict = json.load(fp3)
    
    # Open link json locally
    with open('data/motortrend_specs_2015_leftovers_v2.json', 'r') as fp4:
        whole_fill = json.load(fp4)
        
        
    
    for key2 in linked_dict.keys():
        new_link2 = linked_dict[key2]
    
        if new_link2[-1] == '/':
            link_complete2 = new_link2 + '2015/specifications/'
        else:
            link_complete2 = new_link2
        
    
        if whole_fill[key2][link_complete2] == 'Error':
                df_combo.loc[df_combo['model'] == key2, 'msrp'] = np.nan
                df_combo.loc[df_combo['model'] == key2, 'fuel_type'] = np.nan
                df_combo.loc[df_combo['model'] == key2, 'weight'] = np.nan
                df_combo.loc[df_combo['model'] == key2, 'torque'] = np.nan
                df_combo.loc[df_combo['model'] == key2, 'torque_rpm'] = np.nan
                df_combo.loc[df_combo['model'] == key2, 'horsepower'] = np.nan
        else:
            soup2 = bs4.BeautifulSoup(whole_fill[key2][link_complete2],\
                                      'html.parser')
    
            lines_price = soup2.find_all('span')
            for line in lines_price:
                if line.get('itemprop') != None:
                    if line.get('itemprop') == 'price':
                        df_combo.loc[df_combo['model'] == key2, 'msrp'] = \
                            str(line.string)
                    if line.get('itemprop') == 'fuelType':
                        df_combo.loc[df_combo['model'] == key2, 'fuel_type'] = \
                            str(line.string)
            
            lines_weight = soup2.find_all('div', attrs={'class': 'key'})
            for line in lines_weight:
                if line.string == 'Curb Weight':
                    df_combo.loc[df_combo['model'] == key2, 'weight'] = \
                        str(line.next.next.string)
                if line.string == 'Torque':
                    df_combo.loc[df_combo['model'] == key2, 'torque'] =  \
                        str(line.next.next.string)
                if line.string == 'Torque (rpm)':
                    df_combo.loc[df_combo['model'] == key2, 'torque_rpm'] =  \
                        str(line.next.next.string)
                if line.string == 'Horsepower':
                    if '@' not in line.next.next.string:
                        df_combo.loc[df_combo['model'] == key2, 'horsepower'] = \
                            str(line.next.next.string)
    
    # Cleaning up newly added data                  
    # Setting correct Lamborghini Huracan weight
    df_combo.loc[1090:1091, 'weight'] = '3135'
    
    # Correcting cleaning weight's strings
    df_combo.loc[:, 'weight'] = \
        df_combo.loc[:, 'weight'].apply(lambda x: str(x)[:4])
    
    # Dropping fuel_type
    df_combo.drop('fuel_type', axis=1, inplace=True)
    
    # Dropping NaNs
    df_combo.dropna(inplace=True)
    
    # Cleaning artifacts from msrp's price
    df_combo['msrp'] = df_combo[df_combo['msrp'].notnull()]['msrp'].apply( \
        lambda x: str(x).replace(',', ''))
    df_combo['msrp'] = df_combo[df_combo['msrp'].notnull()]['msrp'].apply( \
        lambda x: int(str(x).replace('.0', '')))
    
    # Convert new values to floats
    df_combo[['weight', 'torque', 'torque_rpm', 'horsepower', 'msrp']] = \
        df_combo[['weight', 'torque', 'torque_rpm', 'horsepower', 'msrp'] \
            ].astype(float64) 
    

    # Has duplicate stnd:
    '''
    FORD Edge [(579, 580), (582, 583)] STND 3 or 2 / Bin 5 or 4
    FORD Focus [635, 637, 641 vs 639] STND 1 or 3 / Bin 3 or 5
    FORD Fusion [649, 650, 653 vs 654] STND 3 or 1 / Bin 5 or 3
    HONDA Accord [707, 709] STND 1 or 0 / Bin 3 or 2
    HONDA Accord [713, 715] STND 1 or 0 / Bin 3 or 2
    HONDA Accord [717, 723 vs 719, 725] STND 3 or 0 / Bin 5 or 2
    HONDA Civic [737, 739] STND 3 or 0 / Bin 5 or 2
    HONDA Civic [743, 745] STND 3 or 0 / Bin 5 or 2
    HONDA Civic HF [749, 751] STND 3 or 0 / Bin 5 or 2
    MERCEDES-BENZ GLK350 [1395, 1397] STND 2 or 3 / Bin 4 or 5
    MERCEDES-BENZ GLK350 4Matic [1399, 1401] STND 2 or 3 / Bin 4 or 5
    SUBARU Legacy [1847, 1848] STND 2 or 3 / Bin 4 or 5
    SUBARU Outback [1850, 1851] STND 2 or 3 / Bin 4 or 5
    VOLKSWAGEN Beetle [1971, 1972] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Beetle [1974, 1975] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Beetle [1979, 1980] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Beetle [1984, 1985] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Beetle Convertible [1987, 1988] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Beetle Convertible [1992, 1993] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Beetle Convertible [1997, 1998] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Beetle CC [2000, 2002] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Beetle CC [2004, 2006] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Jetta [2034, 2035] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Jetta [2037, 2038] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Jetta [2042, 2052 vs 2043] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Jetta [2049, 2050] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Passat [2056, 2057] STND 3 or 1 / Bin 5 or 3
    VOLKSWAGEN Passat [2059, 2060] STND 3 or 1 / Bin 5 or 3
    '''
  
    # Indexes of doubles opposite of 3s indexes
    op_doubles = [580, 583, 635, 637, 641, 653, 654, 707, 709, 713, 715, 719,
                  725, 739, 745, 1395, 1399, 1847, 1850, 1971, 1972, 1974, 
                  1975, 1980, 1985, 1988, 1993, 1998, 2002, 2006, 2035, 2038,
                  2043, 2050, 2057, 2060]
    
    df_combo_slim = df_combo_combo.drop(op_doubles)


        
    # Save filled df to local pickle
    if saved_copy == True:
        with open('data/df_combo_manfill_final.pkl', 'w') as fin:
            pickle.dump(df_combo_slim, fin)
        
    
    
                      
                      
        