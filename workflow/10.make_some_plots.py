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
        'plots/prioritized_1e-3_long.torch',
        'plots/prioritized_1e-3_long_v2.torch',
        'plots/prioritized_no_clamp_1e-3_long.torch',
        ]


def add_trace(plot, plot_data, name):
    plot.add_trace(go.Scatter(x=np.arange(len(plot_data)), y=np.array(plot_data), name=name))


def add_avg_trace(plot, plot_data, name, avg_epochs=50):
    y = [mean(plot_data[max(0, i - avg_epochs):i]) for i in range(1, len(plot_data) + 1)]
    add_trace(plot, y, name)


if __name__ == "__main__":
    data = [torch.load(path) for path in paths]
    plot = go.Figure()

    for (plot_data, path) in zip(data, paths):
        add_avg_trace(plot, plot_data, path)

    plot.show()
