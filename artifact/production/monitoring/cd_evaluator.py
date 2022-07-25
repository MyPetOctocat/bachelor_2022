import pandas as pd
from time import time
from rmse_calculator import rmse_calc
from cd_detector import cd_detector
from rmse_plot_generator import plot_gen as rpg


df = pd.read_csv('DCN-iterate_1658684509_predictions.csv')  # Load prediction score dataset


df['date'] = pd.to_datetime(df['timestamp'], unit='s')

# Group by date and calculate the rmse for each month/week/year & join them to one dataset
group_by = "Y"

rmse_float = rmse_calc(df[['date', 'user_rating', 'pred_rating']], group_by)  # RMSE of native predictions
rmse_int = rmse_calc(df[['date', 'user_rating', 'pred_int_rating']], group_by)  # RMSE of rounded predictions
rmse_df = pd.DataFrame(dict(rmse_float=rmse_float, rmse_int=rmse_int))


# CD detection (determine whether CD occurred in the data)
no_cd = cd_detector(rmse_df, delta_threshold=0.1, absolute_threshold=1.5)

# Write file of the CD evaluation result (prerequisite to CD adaptation: File can be read by Airflow an initiate TFX retraining)
if no_cd:
    with open('OK.txt', 'w') as file:
        file.write(str(time()))
else:
    with open('CD_DETECTED.txt', 'w') as file:
        file.write(str(time()))


# CD understanding (give visual feedback about CD)
group_by = "M"  # Group by month for better visualization

vis_rmse_float = rmse_calc(df[['date', 'user_rating', 'pred_rating']], group_by)  # RMSE of native predictions
vis_rmse_int = rmse_calc(df[['date', 'user_rating', 'pred_int_rating']], group_by)  # RMSE of rounded predictions
vis_rmse_df = pd.DataFrame(dict(rmse_float=vis_rmse_float, rmse_int=vis_rmse_int))

rpg(vis_rmse_df, "rmse_trend.png")


