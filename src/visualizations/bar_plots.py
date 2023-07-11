import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from src.analysis import layer_wise_analysis


# code to plot a bar plot using plotly library
def plot_bar_plot(x, y, title, x_title, y_title, color, save_path):

    data = [go.Bar(x=x, y=y, marker_color=color, marker=dict(line=dict(
                color='MediumPurple',
                width=2
            )))]

    # for bar in data:
    #     bar.marker.line.color = '#1f77b4'
    #     bar.marker.line.width = [0, 4, 0, 0, 4, 4, 0]

    fig = go.Figure(data=data)

    # lighten colors and remove borders
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        font_color="black",
        bargap=0.4,
        bargroupgap=0.1
    )

    # save svg image
    fig.write_image(save_path)


# code to plot a bar plot using plotly library
def plot_stacked_bar_plot(x, y1, y2, title, x_title, y_title, color, save_path):
    data = [
        go.Bar(x=x, y=y1, marker_color='#2ca02c', marker=dict(line=dict(
            color='MediumPurple',
            width=2
        )),
               name="Helpful",
               opacity=0.8),
        go.Bar(x=x, y=y2, marker_color='#d62728', marker=dict(line=dict(
            color='MediumPurple',
            width=2
        )),
               name="Unhelpful",
               opacity=0.8)

    ]

    # for bar in data:
    #     bar.marker.line.color = '#1f77b4'
    #     bar.marker.line.width = [0, 4, 0, 0, 4, 4, 0]

    fig = go.Figure(data=data)

    # lighten colors and remove borders
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        font_color="black",
        bargap=0.4,
        bargroupgap=0.1,
        barmode='stack'
    )

    # save svg image
    fig.write_image(save_path)


if __name__ == '__main__':
    # counts = []
    # labels = []
    # # load the data
    # data_path = "../data/prolific/pilot_results_eco_v0/"
    # for file in os.listdir(data_path):
    #     if file.endswith(".csv"):
    #         df = pd.read_csv(data_path + file, sep="\t")
    #         num_annotations = df.shape[0]
    #         aspect = file.split("_")[-1].split(".")[0]
    #         counts.append(num_annotations)
    #         labels.append(aspect)

    path = "../data/prolific/pilot_results_eco_v0/lfqa_pilot_complete.csv"
    # labels, counts, _ = layer_wise_analysis(path, "factuality")
    labels, counts, ref_human_help, ref_model_help = layer_wise_analysis(path, "hard")

    # move first element in ref_human_help and ref_model_help to separate lists
    # ref_help = [ref_human_help[0], ref_model_help[0]]
    # ref_no_help = [ref_human_help[1], ref_model_help[1]]

    # df = pd.read_csv("data/annotations.csv")
    # plot the bar plot
    plot_bar_plot(x=labels, y=counts, title="Hard to understand - Economics", x_title="Answer", y_title="Count",
                  color="rgb(255, 127, 14)", save_path="../data/plots/hard_eco.svg")
    # plot_stacked_bar_plot(x=labels, y1=ref_help, y2=ref_no_help,
    #                       title="Reference - Economics", x_title="Answer", y_title="Helpful count",
    #                       color="rgb(255, 127, 14)", save_path="../data/plots/ref_eco_help.svg")
