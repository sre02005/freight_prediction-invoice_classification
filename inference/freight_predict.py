import pickle
import pandas as pd
import joblib


model_path='../freight_cost/models/predict_freight_model.pkl'
def load_model():
    with open(model_path,'rb') as file:
        model=joblib.load(file)
    return model


def pred_freight_cost(input_data):
    model=load_model()
    input_df=pd.DataFrame(input_data)

    input_df['freigh_predicted']=model.predict(input_df).round()
    return input_df

if __name__=='__main__':
    sample_data={
        'Dollars':[18500,9000],


    }

    prediction=pred_freight_cost(sample_data)
    print(prediction)
