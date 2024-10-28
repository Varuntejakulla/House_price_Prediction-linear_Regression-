import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv('house_price.csv') #reafing the input csv file 

#Implementing the linear regression model
reg = linear_model.LinearRegression()
reg.fit(data[['year']],data[['per capita income (US$)']])

new_data = data[['year']].copy()
new_data.to_csv('year.csv',index=False)

#were predict based on year 
predict = reg.predict(new_data)

new_data['prices'] = predict
new_data.to_csv('year_price.csv',index=False)

