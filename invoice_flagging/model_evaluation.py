from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from scipy.stats import ttest_ind
from sklearn.metrics import classification_report,accuracy_score,f1_score,make_scorer
from sklearn.model_selection import GridSearchCV

def train_random_forest(x_train,y_train):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {

        "n_estimators": [100, 200, 300],

        "max_depth": [None, 4, 5, 6],

        "min_samples_split": [2, 3, 5],

        "min_samples_leaf": [1, 2, 5],

        "criterion": ['gini', 'entropy']

    }
    # %%
    scorer = make_scorer(f1_score)

    gridcv = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    gridcv.fit(x_train, y_train)
    return gridcv
def model_evaluate(model,x_test,y_test,model_name):
    y_predict=model.predict(x_test)

    accuracy=accuracy_score(y_test,y_predict)
    report =classification_report(y_test,y_predict)

    print(f'accuracy is {accuracy:.2f}')
    print(report)

