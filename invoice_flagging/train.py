from invoice_flagging.data_preprocessing import load_invoice_data,create_invoice_risk_labeling,split_data,scale_features
from invoice_flagging.data_preprocessing import apply_labels
from model_evaluation import train_random_forest,model_evaluate
import os
import joblib

features=['invoice_quantity',
        'invoice_dollars',
        'Freight',

        'total_item_quantity',
        'total_item_dollars']
target='flagging_invoice'

def main():
    data=load_invoice_data()
    data=apply_labels(data)




    x_train,x_test,y_train,y_test=split_data(data,features,target)
    x_scaled_train,x_scaled_test=scale_features(x_train,x_test)

    grid_search=train_random_forest(x_scaled_train,y_train)

    model_evaluate(grid_search,x_scaled_test,y_test,'GRID SEARCH_CV')

    joblib.dump(grid_search.best_estimator_,'.. /models/invoice_classification.pkl')

if __name__=='__main__':
    main()