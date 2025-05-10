from src.data_loading import load_file
from src.data_cleaning import data_clean
from src.data_transformation import encoding
from src.splitting import splitting_fun
from src.data_transformation import scaller
from pipline.train import linear_model
from pipline.utils import model_save,load_model
from predict import pred
from sklearn.metrics import r2_score

print(f"loading of file has been started")
data = load_file("artifacts/raw.csv")
print(f"Data set loaded successfully\n")


print(f"Model cleaning..")
cleaned = data_clean(data)
print(cleaned)
print(f"Cleaned successfully")


print(f"Applying encoding....")
encoded = encoding(cleaned)
print(encoded)
print(f"encoding apply successfully")


print(f"Applying spllinting....")
x_train,x_test,y_train,y_test = splitting_fun(encoded)
print(f"applied spliting successfully..")


print(f"Applying scaller....")
scalled_data = scaller(x_train,x_test)
print(scalled_data)
print(f"scalled successfully")


print(f"Training....")
model = linear_model(x_train,y_train)
print(f"training successfully")


print(f"saving model....")
saved_model = model_save(model)
print(f"saved successfully")



print(f"model prediction....")
prediction = pred(x_test)
print(f"successfully {prediction}")


print("calc accuracyy")
score = r2_score(y_test,prediction)
print(f"Accuracy is {score}%")


