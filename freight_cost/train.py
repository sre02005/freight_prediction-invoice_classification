import pathlib

import joblib
import pathlib

from freight_cost.data_preprocessing import vendor_data,prepare_features,t_t_s

from freight_cost.model_eval import linear_regression,random_forest,metrics_calculation,decision_tree

def main():
    file_loc='../inventory.db'
    model_dir=pathlib.Path('models')
    model_dir.mkdir(exist_ok=True)
    df=vendor_data(file_loc)

    x,y=prepare_features(df)

    x_train,x_test,y_train,y_test=t_t_s(x,y)

    lr_model=linear_regression(x_train,y_train)
    dt_model=decision_tree(x_train,y_train)
    rf_model=random_forest(x_train,y_train)

    result=[]

    result.append(metrics_calculation(lr_model,x_test,y_test,'LIENAR_REGRESSION'))
    result.append(metrics_calculation(dt_model, x_test, y_test, 'DECISION_TREE'))
    result.append(metrics_calculation(rf_model, x_test, y_test, 'RANDOM_FOREST'))

    print(result)
    best_model_info=min(result,key=lambda  x: x['mae'])
    best_model_name=best_model_info['model_name']

    best_model={
        'LIENAR_REGRESSION':lr_model
        ,'DECISION_TREE':dt_model,
        'RANDOM_FOREST':rf_model
    }[best_model_name]

    model_path=model_dir/'predict_freight_model.pkl'
    joblib.dump(best_model,model_path)

    print(f'\nBest model saved :{model_path}')
    print(f'model name:{best_model}')






if __name__=='__main__':
    main()