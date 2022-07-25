# Takes in a model and a dataset entry (in form of a dictionary) and returns the predicted value of the model

def prediction_server(model, entry):
    score = model({'user_id': [[entry['user_id']]], 'movie_id': [[entry['movie_id']]], 'user_gender': [[entry['user_gender']]], 'user_occupation': [[entry['user_occupation']]], 'user_age_cohort': [[entry['user_age_cohort']]]}).numpy()
    return score[0][0][0]