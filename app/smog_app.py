from flask import Flask, render_template, request
import requests
import cPickle as pickle
from collections import OrderedDict
import numpy as np
import pandas as pd

app = Flask(__name__)
with open('../data/model_gradboost_final_air15.pkl', 'r') as f:
    model = pickle.load(f)
def dict_to_html(d):
    return '<br>'.join('{0}: {1}'.format(k, d[k]) for k in sorted(d))

# # home page
# @app.route('/')
# def index():
#     return render_template('jumbotron.html', title='Hello!')

# home page
@app.route('/')
def index():
    feature = {
    # 'adrop1':{'drop':[('Federal', '0'), ('California', '1')], 'title': 'Certification Region', 'nick': 'Item_7', 'button': False},
               'adrop2':{'drop':[('4', '4'), ('5', '5'), ('6', '6'), ('7', '7'), ('8', '8'), ('9', '9'), ('0', '0')], 'title': 'Transmission Speed', 'nick': 'Item_8', 'button': False},
               'box1':{'boxfill':'2.0', 'title': 'Liters of Cylinder', 'nick': 'Item_1', 'button': False},
               'box2':{'boxfill':'4652', 'title': 'Weight in lbs.', 'nick': 'Item_9', 'button': False},
               'box3':{'boxfill':'406', 'title': 'Torque', 'nick': 'Item_10', 'button': False},
               'box4':{'boxfill':'1750', 'title': 'Torque RPM', 'nick': 'Item_11', 'button': False},
               'box5':{'boxfill':'240', 'title': 'Horsepower', 'nick': 'Item_12', 'button': False},
               'box6':{'boxfill':'61700', 'title': 'MSRP', 'nick': 'Item_13', 'button': False},
               'box7':{'boxfill':'20', 'title': 'City MPG', 'nick': 'Item_14', 'button': False},
               'box8':{'boxfill':'29', 'title': 'Highway MPG', 'nick': 'Item_15', 'button': True},
    }
    feature_temp = []
    for k, v in feature.iteritems():
        feature_temp.append([k,v])
    ofeature = OrderedDict(sorted(feature_temp))
    return render_template('pollution_template.html', feature=ofeature, title='Hello!')

@app.route('/more/')
def more():
    return render_template('starter_template.html')

@app.route('/logo.jpg')
def get_logo():

    with open('../images/gooogle-old-3.jpg') as googleLogo:
        return googleLogo.read(), 200, {'Content-Type': 'image/jpg'}

@app.route('/prediction', methods=['POST'] )
def get_search():

    i1 = float(request.form['Item_1'])
    # i2 = int(request.form['Item_2'])
    # i3 = int(request.form['Item_3'])
    # i4 = int(request.form['Item_4'])
    # i5 = int(request.form['Item_5'])
    # i6 = int(request.form['Item_6'])
    # i7 = int(request.form['Item_7'])
    i8 = int(request.form['Item_8'])
    i9 = int(request.form['Item_9'])
    i10 = int(request.form['Item_10'])
    i11 = int(request.form['Item_11'])
    i12 = int(request.form['Item_12'])
    i13 = int(request.form['Item_13'])
    i14 = int(request.form['Item_14'])
    i15 = int(request.form['Item_15'])
    i16 = (float(i14) + i15)/2
    # Calculation for Federal
    nums_f = [i1, 0, i8, i9, i10, i11, i12, i13, i14, i15, i16]
    nums_array_f = np.array(nums_f).reshape(1,11)
    prediction_f = model.predict(nums_array_f)
    result_f = ' ' + (str(prediction_f[0])[0])
    # Calculation for California
    nums_c = [i1, 1, i8, i9, i10, i11, i12, i13, i14, i15, i16]
    nums_array_c = np.array(nums_c).reshape(1,11)
    prediction_c = model.predict(nums_array_c)
    result_c = ' ' + (str(prediction_c[0])[0])

    named_nums = [('Transmission Speed', i8),
                  ('Liters of Cylinder', i1),
                  ('Weight in lbs.', i9),
                  ('Torque', i10),
                  ('Torque RPM', i11),
                  ('Horsepower', i12),
                  ('MSRP', i13),
                  ('City MPG', i14),
                  ('Highway MPG', i15)]
    return render_template('pollution_prediction.html', named_nums=named_nums, result_f=result_f, result_c=result_c)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
