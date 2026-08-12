import joblib
import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind

def load_invoice_data():
    conn=sqlite3.connect('../inventory.db')


    query= """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(
                julianday(p.ReceivingDate) - julianday(p.PODate)
            ) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )

    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,

        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay

    FROM vendor_invoice vi

    LEFT JOIN purchase_agg pa
        ON vi.PONumber = pa.PONumber
    """

    df=pd.read_sql_query(query,conn)
    conn.close()
    return df

def create_invoice_risk_labeling(row):
    if abs(row['total_item_dollars']-row['invoice_dollars'])>5:
        return 1
    if row['avg_receiving_delay']>10:
        return 1

    else:
        return 0
def apply_labels(df):
    df['flagging_invoice']=df.apply(create_invoice_risk_labeling,axis=1)
    return df

def split_data(df,features,target):
    x=df[features]
    y=df[target]

    x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.20,random_state=42
    )
    return x_train,x_test,y_train,y_test

def scale_features(x_train,x_test):
    scaler=StandardScaler()
    x_scaled_train=scaler.fit_transform(x_train)
    x_scaled_test=scaler.transform(x_test)

    joblib.dump(scaler,'../models/scaler.pkl')
    return x_scaled_train,x_scaled_test

