import pandas as pd
from sklearn.model_selection import train_test_split
import sqlite3
from sklearn.preprocessing import StandardScaler

#call the data

def vendor_data(file):
    conn=sqlite3.connect(file)

    data=pd.read_sql_query(f'select * from vendor_invoice',conn)
    conn.close()
    return data

def prepare_features(data):
    x=data[['Dollars']]
    y=data['Freight']

    return x,y

def t_t_s(x,y):
    std=StandardScaler()
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

    return x_train,x_test,y_train,y_test

