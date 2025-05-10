from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler


def encoding(data):

    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_columns = ['Company', 'TypeName', 'Cpu', 'Gpu', 'OpSys']

    encoded = encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded.toarray(), columns = encoder.get_feature_names_out(categorical_columns))

    data = data.reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    data = pd.concat([data, encoded_df], axis=1).drop(columns=categorical_columns)

    return data



def scaller(x_train,x_test):

    numeric_cols = x_train.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = x_test.select_dtypes(include=['float64', 'int64']).columns

# # Initialize and apply StandardScaler
    scaler = StandardScaler()
    x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
    x_test[numeric_cols] = scaler.fit_transform(x_test[numeric_cols])

    return x_train,x_test

