from functools import reduce
import numpy as np
import pandas as pd
import re

xl = pd.ExcelFile('DataDownload.xls')
sheets = xl.sheet_names[2:] # First two sheets are meta data

# sheets that we will merge/join using the FIPS column
merge_sheets = [
     'Supplemental Data - County',
     'ACCESS',
     'STORES',
     'RESTAURANTS',
     'ASSISTANCE',
     'INSECURITY',
     'PRICES_TAXES',
     'LOCAL',
     'HEALTH',
     'SOCIOECONOMIC'
]
# The state sheet only has one row per state, will merge on State column
state_only = 'Supplemental Data - State'

dataframes = (pd.read_excel(xl,i).rename({'FIPS ':'FIPS'},axis=1) if i == 'Supplemental Data - County' else
                pd.read_excel(xl,i).drop(['State','County'],axis=1) for i in merge_sheets)
df = reduce(lambda x,y: x.merge(y,on='FIPS',how='left'), dataframes)
df['State'] = df['State'].apply(lambda x: x.strip()) # strip whitespace from the States column to join
states = pd.read_excel(xl,sheet_name=state_only).query("State==State") # NaN's are not equal to anything, so make sure there is a value for states because of that one stupid row
df = df.merge(states,on='State',how='left')

convert_columns = [
    '2010 Census Population',
    'Population Estimate, 2011', 
    'Population Estimate, 2012',
    'Population Estimate, 2013', 
    'Population Estimate, 2014',
    'Population Estimate, 2015', 
    'Population Estimate, 2016',
    'School Breakfast Program participants FY 2011',
    'School Breakfast Program participants, FY 2012',
]

def converter(x):
    if x is np.nan:
        return x
    if type(x) != str:
        return x
    no_comma = x.replace(',','')
    return pd.to_numeric(no_comma, downcast='integer',errors='coerce')

for col in convert_columns:
    df[col] = df[col].apply(converter)

drop_thresh_col = df.shape[0]*.9
df = df.dropna(thresh=drop_thresh_col, how='all', axis='columns').copy()

drop_thresh_row = df.shape[1]*.78
df = df.dropna(thresh=drop_thresh_row, how='all', axis='index').copy()


num_features = df.select_dtypes(exclude='object')

df_num = num_features.fillna(num_features.median())

def standardize_name(cname):
    cname = re.sub(',','', cname)
    cname = cname.strip().lower()
    cname = re.sub(r'\s+','_', cname)
    return cname

df_num.columns = df_num.columns.map(standardize_name)

df_num = df_num.drop('pct_obese_adults08', axis=1)

df_num.to_csv('USDA-0.2.csv')