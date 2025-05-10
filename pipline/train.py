from sklearn.linear_model import LinearRegression

def linear_model(x,y):
    model = LinearRegression()
    model.fit(x,y)
    return model