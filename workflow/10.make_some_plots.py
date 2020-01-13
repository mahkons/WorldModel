import plotly as plt
import plotly.graph_objects as go

import torch
import numpy as np
from statistics import mean

paths = [
        #  'plots/classic_1e-2.torch', 
        #  'plots/classic_1e-3.torch', 
        #  'plots/classic_1e-4.torch', 
        #  'plots/prioritized_1e-2.torch',
        #  'plots/prioritized_1e-3.torch',
        #  'plots/prioritized_1e-4.torch',
        #  'plots/prioritized_no_clamp_1e-3.torch',
        #  'plots/classic_1e-3_long.torch', 
        #  'plots/prioritized_1e-3_long.torch',
        #  'plots/prioritized_1e-3_long_v2.torch',
        #  'plots/prioritized_no_clamp_1e-3_long.torch',
        #  'plots/error_sum_1e-3_long.torch',
        #  'plots/error_prod_1e-3_long.torch',
        #  'plots/error_p-0.5_long.torch',
        #  'plots/error_p-1_long.torch',
        #  'plots/error_p-2_long.torch',
        #  'plots/error_p0.5_long.torch',
        #  'plots/error_p2_long.torch',
        #  'plots/error_p-0.5_long_v2.torch',
        #  'plots/error_p-1_long_v2.torch',
        #  'plots/error_p-0.25_long.torch',
        #  'plots/error_p-0.75_long.torch',
        #  'plots/error_p-1_wy0.5_long.torch',
        #  'plots/error_p-1_wy2_long_.torch',
        #  'plots/error_p1_wy0.5_long.torch',
        #  'plots/error_p1_wy0.5_long_v2.torch',
        #  'plots/error_p1_wy0.5_long_v3.torch',
        #  'plots/error_p1_wy0.25_long.torch',
        #  'plots/error_p1_wy0.5_long_v4.torch',
        #  'plots/error_p1_wy3_long.torch',
        #  'plots/car_classic_1e-3.torch',
    

        'new_plots/classic_1e-3.torch',
        'new_plots/classic_1e-3_v2.torch',
        'new_plots/prioritized_1e-3.torch',
        'new_plots/prioritized_1e-3_v2.torch',
        'new_plots/error_1e-3.torch',
        'new_plots/error_1e-3_v2.torch',
    ]


def add_trace(plot, x, y, name):
    plot.add_trace(go.Scatter(x=x, y=y, name=name))


def add_avg_trace(plot, x, y, name, avg_epochs=100):
    y = [mean(y[max(0, i - avg_epochs):i]) for i in range(1, len(y) + 1)]
    add_trace(plot, x, y, name)


if __name__ == "__main__":
    data = [torch.load(path) for path in paths]
    plot = go.Figure()


    for (plot_data, path) in zip(data, paths):
        x, y = zip(*plot_data)
        x, y = np.array(x), np.array(y)
        for i in range(1, len(x)):
            x[i] += x[i - 1]
        #  x = np.arange(len(x))
        add_avg_trace(plot, x, y, path)

    plot.show()
