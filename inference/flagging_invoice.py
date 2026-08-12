import pandas
import joblib
import pandas as pd
import pathlib
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
model_path=BASE_DIR/'models'/'invoice_classification.pkl'

def load_model():
    print("MODEL PATH:", model_path)
    print("EXISTS:", model_path.exists())
    with open(model_path,'rb') as file:
        model=joblib.load(file)
    return model


def prediction_flagging(input_data):
    model=load_model()
    input_df=pd.DataFrame(input_data)

    input_df['flagging_pred']=model.predict(input_df).round()
    return input_df

if __name__=='__main__':
    sampledata = {
        'invoice_quantity': [10],
        'invoice_dollars': [2500],
        'Freight': [180],
        'total_item_quantity': [35],
        'total_item_dollars': [8500]
    }
    prediction=prediction_flagging(sampledata)
    print(prediction['flagging_pred'])


