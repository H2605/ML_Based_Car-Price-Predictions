import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor 
import pandas as pd
import joblib
import pickle
from datetime import date
import streamlit as st
#import pickle as pkle
import os.path
st.title("Welcome to SellMyCar!")
 
manufacturer_list=['acura','alfa-romeo','audi','bmw','buick','cadillac','chevrolet','chrysler','dodge','ford','gmc','honda', 'hyundai','infiniti','jaguar','jeep','kia','lexus','lincoln','mazda','mercedes-benz','mercury','mini','mitsubishi','nissan',
'pontiac','porsche','ram','rover','saturn','subaru','toyota','volkswagen','volvo']

ordinal_columns= ['condition', 'fuel', 'title_status', 'transmission', 'drive', 'size', 'paint_color']
ord_condition_list = ['salvage', 'fair', 'good', 'excellent', 'like new', 'new']
ord_fuel_list = ['other', 'diesel', 'gas', 'hybrid', 'electric']
ord_title_status_list = ['salvage', 'missing', 'parts only', 'rebuilt', 'lien', 'clean']
ord_tranmission_list = ['other', 'manual', 'automatic']
ord_drive_list= ['fwd', 'rwd', '4wd']
long_drive=["Front-Wheel-Drive","Rear-Wheel-Drive","4-Wheel-Drive"]
ord_size_list = ['compact','mid-size','full-size', 'sub-compact' ]
ord_paint_color = ['white', 'blue', 'red', 'black', 'silver', 'grey', 'none', 'brown',
                   'yellow', 'orange', 'green', 'custom', 'purple']s
types=['pickup', 'truck', 'other', 'coupe', 'SUV', 'hatchback','mini-van', 'sedan', 'offroad', 'bus', 'van', 'convertible','wagon']

colu1, colu2 = st.columns(2)
# welcome_img = Image.open('race-car-309123_960_720.png')
with colu1:
    st.image("https://cdn.pixabay.com/photo/2018/09/06/10/02/car-icon-3657902_960_720.png",width=300, clamp=True)
        
with colu2:
    st.image('https://cdn.pixabay.com/photo/2014/04/03/00/41/race-car-309123_960_720.png' , width = 300)

#st.write("""
#Audio Transkrption:

##### You want to sell your car? But have no idea how much it is worth? 
##### Then you've come to the right place! Sell My Car! 
##### All you have to do? 
#Just enter the necessary information of your car and Sell! My! Car! will calculate the optimal selling price for you! You don't know certain information about your car? No problem! Just enter your VIN and Sell My Car will do the Rest!
#""")

mmc_=pd.read_csv("/Users/huyduc/Documents/Notebooks/Final Project/WebApp/Sources/car_models.csv")
#info = st.selectbox("Do you need some help?",("","yes","no"))
def list_transformer(Series):
    list_=Series
    list_a=list(list_)
    return list_a 
def cylinder_tansformer(cylinders):
    cyl_df=12312
    if cylinders=="other":
        cyl_df=0
    elif drive=="2 cylinders":
        cyl_df=2
    elif drive=="3 cylinders":
        cyl_df=3
    elif drive=="4 cylinders":
        cyl_df=4
    elif drive=="5 cylinders":
        cyl_df=5
    elif drive=="6 cylinders":
        cyl_df=6
    elif drive=="8 cylinders":
        cyl_df=8
    elif drive=="10 cylinders":
        cyl_df=10
    elif drive=="12 cylinders":
        cyl_df=12
    return cyl_df
def drive_tansformer(drive):
    #"transform" drive columns
    if drive=="Front-Wheel-Drive":
        drive="fwd"
    elif drive=="Rear-Wheel-Drive":
        drive="rwd"
    elif drive=="4-Wheel-Drive":
        drive="4wd"
    return drive

today = date.today()
today_int=int(today.strftime("%Y%m%d"))
st.write("Posting Date ",today)

col1, col2, col3 = st.columns(3)

with col1:
    year = st.selectbox("1. Production year",(range(1918,2023)), key="str")
    odometer=st.number_input("2. Odometer reading in Miles",value=10000.00)
    color_paint=st.selectbox("3. Color of your car",(ord_paint_color))    
    title_stat=st.selectbox("4.Status",(ord_title_status_list))
    condi=st.selectbox("5. Condition",(ord_condition_list))

with col2:
    manufac_models=[]
    car_man = st.selectbox("6. Car Manufacturer",(manufacturer_list))
    ser_mod=mmc_.loc[(mmc_['manufacturer']==car_man), ['manufacturer','model']]
    list_mod=list_transformer(ser_mod["model"].unique())
    car_mod=st.selectbox("7. Car Model",(list_mod))

    ser_fuel=mmc_.loc[(mmc_['manufacturer']==car_man) &(mmc_['model']==car_mod), ['model','fuel']]
    list_fuel=list((ser_fuel["fuel"].unique()))
    fuel=st.selectbox("8. Fuel",(list_fuel))

    ser_trans=mmc_.loc[(mmc_['manufacturer']==car_man) & (mmc_['model']==car_mod) & (mmc_['fuel']==fuel), ['model','transmission']]
    list_trans=ser_trans["transmission"].unique()        
    transimission=st.selectbox("9. Transmission",(list_trans) )


with col3:
    ser_drive=mmc_.loc[(mmc_['manufacturer']==car_man) & (mmc_['model']==car_mod) & (mmc_['fuel']==fuel)& (mmc_['transmission']==transimission), ['model','drive']]
    list_drive=list((ser_drive["drive"].unique()))
    drive=st.selectbox("10. Drive",(list_drive))
        #long_drive))#('fwd', 'rwd', '4wd'))

    ser_type=mmc_.loc[(mmc_['manufacturer']==car_man) & (mmc_['model']==car_mod) & (mmc_['fuel']==fuel)& (mmc_['transmission']==transimission)& (mmc_['drive']==drive), ['model','type']]
    list_type=list((ser_type["type"].unique()))
    type_=st.selectbox("11. Type of Car",(list_type))
        #types))

    ser_size=mmc_.loc[(mmc_['manufacturer']==car_man) & (mmc_['model']==car_mod) & (mmc_['fuel']==fuel)& (mmc_['transmission']==transimission)& (mmc_['drive']==drive)& (mmc_['type']==type_), ['model','size']]
    list_size=list((ser_size["size"].unique()))
    car_size=st.selectbox('12. Car Size',(list_size))
#          ord_size_list))

    ser_cyl=mmc_.loc[(mmc_['manufacturer']==car_man) & (mmc_['model']==car_mod) & (mmc_['fuel']==fuel)& (mmc_['transmission']==transimission)& (mmc_['drive']==drive)& (mmc_['type']==type_)& (mmc_['size']==car_size), ['model','cylinders']]
    list_cyl=list((ser_cyl["cylinders"].unique()))
    cylinders=st.selectbox("13. Number of Cylinder",(list_cyl))

    #"transform" drive columns
if drive=="Front-Wheel-Drive":
    drive="fwd"
elif drive=="Rear-Wheel-Drive":
    drive="rwd"
elif drive=="4-Wheel-Drive":
    drive="4wd"

cylinder=cylinder_tansformer(cylinders)
selling_car = pd.DataFrame({
    'year':[year],
    'manufacturer':[car_man], 
    'model':[car_mod], 
    'cylinders':[cylinders],
    'odometer':[odometer],
    'type':[type_], 
    'posting_date':[today_int], 
    'condition':[condi],
    'fuel':[fuel], 
    'title_status':[title_stat],
    'transmission':[transimission], 
    'drive':[drive_tansformer(drive)],
    'size':[car_size], 
    'paint_color':[color_paint]

})

ordinal_lists =[ord_condition_list,ord_fuel_list,ord_title_status_list,ord_tranmission_list,ord_drive_list,ord_size_list,ord_paint_color]
ct_ordinal= ColumnTransformer([("ordinal", OrdinalEncoder(categories=ordinal_lists),ordinal_columns)])
ct_ordinal.fit(selling_car)

ordinal_df = pd.DataFrame(ct_ordinal.transform(selling_car), columns= ordinal_columns)

data_v7 = selling_car.copy()
data_v7.reset_index(inplace=True, drop=True)
data_v7.drop(columns=ordinal_columns, inplace= True) 
data_v7[ordinal_columns]= ordinal_df[ordinal_columns]
#final_data = pd.get_dummies(data_v7)
#X = data_v7.drop(columns=['price'])
#y = data_v10['price']
model = pickle.load(open('/Users/huyduc/Documents/GitHub/used_car_price_predication/catboost_gscv.sav', 'rb'))
#model=joblib.load("/Users/huyduc/Documents/Notebooks/Final Project/Models/rf_model.joblib")
car_prize=model.predict(data_v7)
#pred=model.predict(ot)
st.write("You should sell your car at a price of ",car_prize[0].round(2),"USD")



#next = st.button('Go to next page')


#posting_date=#todays date
