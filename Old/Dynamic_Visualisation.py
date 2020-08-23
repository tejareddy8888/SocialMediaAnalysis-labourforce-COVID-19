#%%
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_file, save, curdoc
#from bokeh.layouts import layout
from bokeh.layouts import column, row, layout,widgetbox
from bokeh.models import Slider, Select, Button, CustomJS
from bokeh.models import ColumnDataSource, HoverTool,DatetimeTickFormatter
from bokeh.models import BoxAnnotation
from math import pi
import os
from datetime import datetime as dt
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
import matplotlib as matl
import re
from  tqdm import tqdm
from langdetect import detect
import string
import nltk.data
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
import datetime
!pip install empath
from empath import Empath
!pip install textblob
import textblob as tb

#%%

def load_data_fromfile(file):
    data = pd.read_csv(file,sep=',',encoding='utf8',engine='python')
    return data

def fetch_file_fromdir(directory):
    df = pd.DataFrame({'tweet' : []})
    for file_path in tqdm(os.listdir(directory),desc=os.path.basename(directory)):
        df1 = load_data_fromfile(os.path.join(directory, file_path))
        df = pd.concat([df,df1],sort=True).reset_index(drop=True,)
    return df

def load_dataset(directory):
    train_df = fetch_file_fromdir(os.path.join(os.getcwd(),directory))
    return train_df

def Dateformatter(df):
    for index,row in df.iterrows():
        print(index)
        try:
            df.loc[index,'Date'] = pd.to_datetime(row['Date'])
        except:
            df.drop(index,inplace=True)
    return df 

#%%
# 
tweet_df = load_dataset('new_train_data')
tweet_df.drop('Unnamed: 0',axis=1,inplace=True) 
tweet_df = Dateformatter(tweet_df)
#%%

__file__ = 'englishtweetM9_M15.csv'
my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
df = pd.read_csv(my_absolute_dirpath+'\\'+__file__ )
df.head(10)
df.Date = df.Date.map(lambda x: str(x).split(' ')[0])
df.Date = pd.to_datetime(df.Date,format ="%Y-%m-%d %H:%M:%S")
country_df = df[df['location']=='switzerland']
groupdf = country_df.groupby('Date')
Mean = groupdf['sentiment'].mean()
mysource = ColumnDataSource(data={'Date': Mean.index.values ,'sentiment' :Mean.values})

events_df = pd.read_csv(r'..\2020_05_10\Summary_stats_all_locs.csv')
events_df = events_df[['location_name','travel_limit_start_date', 'travel_limit_end_date',\
       'stay_home_start_date', 'stay_home_end_date',\
       'educational_fac_start_date', 'educational_fac_end_date',\
       'any_gathering_restrict_start_date', 'any_gathering_restrict_end_date',\
       'any_business_start_date', 'any_business_end_date',\
       'all_non-ess_business_start_date', 'all_non-ess_business_end_date']]
countryeventdf = events_df[events_df['location_name'].str.lower()=='switzerland']

#%%
def change_country(input):
    countryname = input.lower()
    country_df = df[df['location']==countryname]
    groupdf = country_df.groupby('Date')
    Mean = groupdf['sentiment'].mean()
    mysource1 = ColumnDataSource(data={'Date': Mean.index.values ,'sentiment' :Mean.values})
    mysource.data.update(mysource1.data)

def select_country(attr, old, new):
    change_country(Country_select.value)
    reset(Country_select.value)  

def reset(input):
    fig.title.text = input+"""'s Mean Semntiment"""

Country_select = Select(options=['United States', 'United Kingdom', 'India','Switzerland','Nigeria'], value='Switzerland', title='Country')

Country_select.on_change('value',select_country)

hover = HoverTool()
hover.tooltips=[
    ('Date', '@Date'),
    ('sentiment', '@sentiment'),
]

#%%

fig = figure(plot_width=1240, plot_height=400,x_axis_type='datetime' ,title="Tweet Sentiment of individual countries",x_axis_label = 'Year', y_axis_label = 'Sentiment')
fig.add_tools(hover)
fig.line(x='Date', y='sentiment',source=mysource,color="blue")
fig.xaxis.formatter=DatetimeTickFormatter(\
        seconds=["%Y-%m-%d"],\
        minutes=["%Y-%m-%d"],\
        hours=["%Y-%m-%d"],\
        days=["%Y-%m-%d"],\
        months=["%Y-%m-%d"],\
        years=["%Y-%m-%d"],\
    )
fig.title.text = 'Switzerland Mean Sentiment'

low_box = BoxAnnotation(bottom=-1,top=1,left=pd.to_datetime(countryeventdf['all_non-ess_business_start_date'].values[0]),right=pd.to_datetime('2020-05-14'),fill_alpha=0.1, fill_color='red')
fig.add_layout(low_box)
#dashboard = layout(Plot1)

#%%
widget1 = widgetbox(Country_select,sizing_mode='scale_width')

column1 = column(fig)
row1 = row(widget1)
row2 = row(column1)


dashboard = layout(row1,row2)
show(dashboard)
#%%

curdoc().add_root(dashboard)

curdoc().title = 'COVID_Tweet'

output_file("COVID_Tweet.html")

# %%
