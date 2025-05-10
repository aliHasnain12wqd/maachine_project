from pipline.utils import load_model
from sklearn.linear_model import LinearRegression


loaded_model = load_model("models/model.pkl")

def pred(input):
    return loaded_model.predict(input)


