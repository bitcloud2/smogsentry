# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:58:39 2015

@author: Bit
"""


def auto_scrape(save_copy=False):
  '''
  
  '''
  
  # Creating make/model df
  carz = df_clean[df_clean['cert_region'] == 0]['model'].unique()
  carz = list(carz)
  
  make_temp = []
  model_temp = []
  
  for car in carz:
      car_temp = car.split()
      make_temp.append(car_temp[0])
      model_temp.append(car_temp[1:][0])
  
  df_carz = pd.DataFrame(make_temp, columns = ['make'])
  df_carz['model'] = model_temp
  df_carz.drop_duplicates(keep='first', inplace=True)
  df_carz.index = range(288)
  
  # Loop and make links
  whole = {}
  
  specs = {}
  for m in df_carz['make'].unique():
      specs[m] = {}
  
  for i in xrange(df_carz.shape[0]):
      make = df_carz['make'][i]
      model = df_carz['model'][i]
      specs[make][model] = {}
      link = 'http://www.motortrend.com/cars/'
      link_complete = link + make + '/' + model + '/2015/specifications/'
  
      # Go to the link and get the html as a string
      html = requests.get(link_complete)
      if html.status_code != 200:
          specs[make][model]['msrp'] = np.nan
          specs[make][model]['fuel_type'] = np.nan
          specs[make][model]['weight'] = np.nan
          specs[make][model]['torque'] = np.nan
          specs[make][model]['torque_rpm'] = np.nan
          specs[make][model]['horsepower'] = np.nan
          whole[link_complete] = 'Error'
      else:
          soup2 = bs4.BeautifulSoup(html.content, 'html.parser')
          whole[link_complete] = html.content
          time.sleep(2)
  
          lines_price = soup2.find_all('span')
          for line in lines_price:
              if line.get('itemprop') != None:
                  if line.get('itemprop') == 'price':
                      specs[make][model]['msrp'] = line.string
                  if line.get('itemprop') == 'fuelType':
                      specs[make][model]['fuel_type'] = line.string
          
          lines_weight = soup2.find_all('div', attrs={'class': 'key'})
          for line in lines_weight:
              if line.string == 'Curb Weight':
                  specs[make][model]['weight'] = line.next.next.string
              if line.string == 'Torque':
                  specs[make][model]['torque'] =  line.next.next.string
              if line.string == 'Torque (rpm)':
                  specs[make][model]['torque_rpm'] =  line.next.next.string
              if line.string == 'Horsepower':
                  if '@' not in line.next.next.string:
                      specs[make][model]['horsepower'] = line.next.next.string
                    
    if save_copy=True:
      # Save scrape locally
      with open('data/motortrend_scrape_2015.json', 'w') as fp1:
          json.dump(whole, fp1)
      
      # Save scrape specifics locally
      with open('data/motortrend_specs_2015.json', 'w') as fp2:
          json.dump(specs, fp2)
          

def manual_scrape():
  '''
  
  '''
  
   # Open manually collected links dict locally
   with open('data/motortrend_links.json', 'r') as fp3:
       linked_dict = json.load(fp3)
  
   whole_nan = {}
   for key in linked_dict.keys():
       new_link = linked_dict[key]
       whole_nan[key] = {}
      
       if new_link[-1] == '/':
           link_complete = new_link + '2015/specifications/'
       else:
           link_complete = new_link
  
       # Go to the link and get the html as a string
       html = requests.get(link_complete)
       if html.status_code != 200:
           df_combo.loc[df_combo['model'] == key, 'msrp'] = np.nan
           df_combo.loc[df_combo['model'] == key, 'fuel_type'] = np.nan
           df_combo.loc[df_combo['model'] == key, 'weight'] = np.nan
           df_combo.loc[df_combo['model'] == key, 'torque'] = np.nan
           df_combo.loc[df_combo['model'] == key, 'torque_rpm'] = np.nan
           df_combo.loc[df_combo['model'] == key, 'horsepower'] = np.nan
           whole_nan[key][link_complete] = 'Error'
       else:
           soup2 = bs4.BeautifulSoup(html.content, 'html.parser')
           whole_nan[key][link_complete] = html.content
           time.sleep(2)
  
           lines_price = soup2.find_all('span')
           for line in lines_price:
               if line.get('itemprop') != None:
                   if line.get('itemprop') == 'price':
                       df_combo.loc[df_combo['model'] == key, 'msrp'] = \
                           line.string
                   if line.get('itemprop') == 'fuelType':
                       df_combo.loc[df_combo['model'] == key, 'fuel_type'] = \
                           line.string
          
           lines_weight = soup2.find_all('div', attrs={'class': 'key'})
           for line in lines_weight:
               if line.string == 'Curb Weight':
                   df_combo.loc[df_combo['model'] == key, 'weight'] = \
                       line.next.next.string
               if line.string == 'Torque':
                   df_combo.loc[df_combo['model'] == key, 'torque'] = \
                       line.next.next.string
               if line.string == 'Torque (rpm)':
                   df_combo.loc[df_combo['model'] == key, 'torque_rpm'] = \
                       line.next.next.string
               if line.string == 'Horsepower':
                   if '@' not in line.next.next.string:
                       df_combo.loc[df_combo['model'] == key, 'horsepower'] = \
                           line.next.next.string
                           
                           
    # Save link dict locally
    with open('data/motortrend_specs_2015_leftovers_v2.json', 'w') as fp3:
        json.dump(whole_nan, fp3)