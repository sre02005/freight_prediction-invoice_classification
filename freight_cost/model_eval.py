from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def linear_regression(x_train,y_train):
    model=LinearRegression()
    model.fit(x_train,y_train)
    return model


def decision_tree(x_train,y_train):
    model=DecisionTreeRegressor(random_state=42)
    model.fit(x_train,y_train)
    return model

def random_forest(x_train,y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(x_train, y_train)
    return model

def metrics_calculation(model,x_test,y_test,model_name):
    y_predict=model.predict(x_test)

    r2=r2_score(y_test,y_predict)
    mae=mean_absolute_error(y_test,y_predict)
    mse=mean_squared_error(y_test,y_predict)

    print(f'model_name:{model_name}\n'
          f'mse:{mse}\n'
          f'mae:{mae}\n'
          f'r2_score:{r2}')
    return {
        'model_name':model_name,
        'mse':mse,
        'mae':mae,
        'r2_score':r2,

    }