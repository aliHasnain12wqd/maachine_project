import pickle

def model_save(model):
    with open("models/model.pkl","wb") as f:
        pickle.dump(model,f)
    


def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model