from functools import reduce
import numpy as np
import pandas as pd

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