# Detects CD in the dataset based on threshold parameters

# Scans Dataframe for threshold violation
def cd_detection(rmse_df, threshold):
    for index, elem in rmse_df.iterrows():
        if (elem > threshold).all():
            return False, index  # If CD detected return Bool + Date window of detection
    return True, None  # If no CD detection found, return Bool and None as placeholder

# Receives DF of RMSE values and returns a Bool: True --> Data is OK | False --> Datadrift detected
def cd_detector(rmse_df, delta_threshold, absolute_threshold):
    """"
    delta_threshold: Acceptable RMSE increase from one window to another
    absolute_threshold: Threshold RMSE value that applies to every value
    """
    rmse_delta = rmse_df.rolling(2).apply(
        lambda x: x.iloc[1] - x.iloc[0])  # Calculate delta between 2 values within a rolling window

    # Checks whether RMSE thresholds have been surpassed.
    delta_rmse_ok = cd_detection(rmse_delta, delta_threshold)
    absolute_rmse_ok = cd_detection(rmse_df, absolute_threshold)

    # Returns tuple: 1. index is bool, 2. index is list of CD timestamp for delta and absolute threshold
    # Info to bool: True if both absolute and delta are within threshold, else False is passed
    return delta_rmse_ok[0] and absolute_rmse_ok[0], [delta_rmse_ok[1], absolute_rmse_ok[1]]
