import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List
import matplotlib.pyplot as plt

import plotly.io as pio
pio.kaleido.scope.mathjax = None

from src.annotation.analysis import layer_wise_analysis
from src.annotation.agreement import calculate_answer_stats


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


def length_plot(category: List = [], num_annotators: int = 3):

    dataframes = []
    for i in range(len(category)):
        if category[i] == "economics":
            num_annotators = 2
        data = calculate_answer_stats(
            category=category[i],
            num_annotators=num_annotators
        )
        data["category"] = [category[i].capitalize()]*len(data)
        dataframes.append(data)

    df = pd.concat(dataframes, ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Violin(y=df["category"],
                            x=df["human_ans_len"],
                            legendgroup='Human', scalegroup='Human', name='Human',
                            side='positive',
                            line_color='blue',
                            orientation="h"
                            )
                  )
    fig.add_trace(go.Violin(y=df["category"],
                            x=df["model_ans_len"],
                            legendgroup='Model', scalegroup='Model', name='Model',
                            side='negative',
                            line_color='orange',
                            orientation="h"
                            )
                  )
    fig.update_traces(meanline_visible=True)
    fig.update_layout(
        violingap=0,
        violinmode='overlay'
       )
    # fig.show()

    fig.update_layout(
        font=dict(family='Times New Roman', size=12, color='black'),
        plot_bgcolor='white',  # Set plot background color
        # showlegend=True,
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        xaxis=dict(
            title_font=dict(size=22, color='black'),
            ticks="outside", mirror=True, showline=True,
            linewidth=1.5,
            linecolor='black',
            range=[0, 600]
        ),  # Optional: Move x-axis ticks outside
        yaxis=dict(
            title_font=dict(size=22, color='black'),
            ticks="outside", mirror=True, showline=True,
            linewidth=1.5,
            linecolor='black',
        ),  # Optional: Move y-axis ticks outside
        xaxis_title="Answer length",
    )
    fig.update_layout(
        legend=dict(
            orientation="v", yanchor="bottom", y=0.68, xanchor="center", x=0.85, font=dict(size=14), itemsizing="trace")
        )
    fig.update_layout(width=500, height=400, template="ggplot2", margin=dict(t=10, b=10, r=10), )
    pio.write_image(fig, "./src/data/plots/ans_length_stats_2.pdf")

    # fig.show()


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

    # path = "../data/prolific/pilot_results_eco_v0/lfqa_pilot_complete.csv"
    # # labels, counts, _ = layer_wise_analysis(path, "factuality")
    # labels, counts, ref_human_help, ref_model_help = layer_wise_analysis(path, "hard")

    # move first element in ref_human_help and ref_model_help to separate lists
    # ref_help = [ref_human_help[0], ref_model_help[0]]
    # ref_no_help = [ref_human_help[1], ref_model_help[1]]

    # df = pd.read_csv("data/annotations.csv")
    # plot the bar plot
    # plot_bar_plot(x=labels, y=counts, title="Hard to understand - Economics", x_title="Answer", y_title="Count",
    #               color="rgb(255, 127, 14)", save_path="../data/plots/hard_eco.svg")
    # plot_stacked_bar_plot(x=labels, y1=ref_help, y2=ref_no_help,
    #                       title="Reference - Economics", x_title="Answer", y_title="Helpful count",
    #                       color="rgb(255, 127, 14)", save_path="../data/plots/ref_eco_help.svg")

    length_plot(category=["biology", "technology", "economics"], num_annotators=3)
