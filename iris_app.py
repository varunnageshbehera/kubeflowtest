from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import mlflow
import mlflow.xgboost
from sklearn import datasets
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score, log_loss, accuracy_score
import uvicorn

app = FastAPI()

@app.get("/train_iris")
async def train_iris():
    try:
        mlflow.xgboost.autolog()

        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        #Splitting the arrays into random train and test subsets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        #Creating the Xgboost DMatrix data format (from the arrays already obtained)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        with mlflow.start_run():
            # Setting some parameters
            parameters = {
                'eta': 0.3,  
                'objective': 'multi:softprob',  # error evaluation for multiclass tasks
                'num_class': 3,  # number of classes to predic
                'max_depth': 3  # depth of the trees in the boosting process
                }  
            num_round = 20  # the number of training iterations

            #training the model
            bst = xgb.train(parameters, dtrain, num_round)
            #
            #result
            preds = bst.predict(dtest)
            
            best_preds = np.asarray([np.argmax(line) for line in preds])

            #calculating the precision
            precision = precision_score(y_test, best_preds, average='macro')
            loss = log_loss(y_test, preds)
            acc = accuracy_score(y_test, best_preds)
            
            mlflow.log_metrics({"log_loss": loss, "accuracy": acc, "precision": precision})

        output = {"log_loss": loss, "accuracy": acc, "precision": precision}
        
    except Exception as e:
        print(str(e))
        output = {"Error":str(e)}
     
    return output

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="XGBoost - Iris",
        version="1.0",
        description="This API used to train xgBoost on iris dataset",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)