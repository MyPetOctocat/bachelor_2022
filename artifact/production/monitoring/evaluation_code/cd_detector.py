# Detects CD in the dataset based on threshold parameters

# Receives DF of RMSE values and returns a Bool: True --> Data is OK | False --> Datadrift detected
def cd_detector(rmse_df, delta_threshold, absolute_threshold):
    """"
    delta_threshold: Acceptable RMSE increase from one window to another
    absolute_threshold: Threshold RMSE value that applies to every value
    """
    rmse_delta = rmse_df.rolling(2).apply(
        lambda x: x.iloc[1] - x.iloc[0])  # Calculate delta between 2 values within a rolling window

    # Checks whether RMSE thresholds have been surpassed. If not return True
    delta_rmse_ok = (rmse_delta < delta_threshold).any().all()
    absolute_rmse_ok = (rmse_df < absolute_threshold).any().all()

    # Returns bool: True if both absolute and delta are within threshold, else False is passed
    return delta_rmse_ok and absolute_rmse_ok
