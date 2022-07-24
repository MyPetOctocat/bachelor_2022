# Takes in a Dataframe and returns a new Dataframe with prediction values

from predictor import prediction_server as ps

def predict_batch(df, model):
    for index, row in df.iterrows():
        entry = row.to_dict()  # Dictionary representation of row for predictor
        pred = ps(model, entry)  # Rating Prediction
        df.at[index, 'pred_rating'] = pred  # set predicted rating

        # Convert float to integer for absolute ratings (conditions to keep rating interval [1;5])
        int_pred = int(round(pred))

        if int_pred < 1:
            int_pred = 1
        if int_pred > 5:
            int_pred = 5

        df.at[index, 'pred_int_rating'] = round(int_pred)
    return df

