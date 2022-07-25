# Generates and saves plot from RMSE DF (part of CD understanding)

import seaborn as sns


def plot_gen(rmse_df, location):
    rmse_trend = sns.lineplot(data=rmse_df)

    # Save plot
    fig = rmse_trend.get_figure()
    fig.savefig(location)
    return fig