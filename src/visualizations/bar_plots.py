import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List
import os
# import matplotlib.pyplot as plt
import plotly.io as pio
pio.kaleido.scope.mathjax = None

from src.annotation.analysis import layer_wise_analysis
from src.annotation.agreement import calculate_answer_stats
from src.analysis.scorer import ResponseScorer

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

sns.set_theme()
# sns.set_style("white")
sns.set_context("talk")
# plt.style.use("seaborn-deep")
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
sns.set_context("paper")
# from matplotlib import rc
# rc('font', family='serif')
plt.rcParams["text.color"] = "black"

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


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


def plot_score_bar_plot():
    # plot the bar plot
    aspects = ["reference_example", "ans_preference", "factuality", "irrelevance", "incomplete_ans"]
    base_path = "src/data/annotated_data"
    # category = None  # or any particular category
    categories = ["biology", "chemistry", "economics", "history", "law", "physics", "technology"]
    model_scores, human_scores = [], []
    for category in categories:
        scorer = ResponseScorer(
            data_path=base_path,
            category=category,
        )
        model_score, human_score = scorer.scorer(aspects=aspects)
        model_scores.append(model_score)
        human_scores.append(human_score)

    # plot stacked bar plot with human and model scores with go lib in plotly
    data = pd.DataFrame({
        "category": categories,
        "human": human_scores,
        "model": model_scores
    })
    # data = data.melt(id_vars="category", var_name="score_type", value_name="score")
    fig = go.Figure(data=[
        go.Bar(name='Human', x=data["category"], y=data["human"], text=data["human"], marker_color='#1a76ff'),
        go.Bar(name='Model', x=data["category"], y=data["model"], text=data["model"], marker_color='#ff7f0e')
    ])
    fig.update_layout(barmode='group')

    fig.update_layout(
        font=dict(family='Times New Roman', size=12, color='black'),
        plot_bgcolor='white',  # Set plot background color
        showlegend=True,
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
        ),  # Optional: Move x-axis ticks outside
        yaxis=dict(
            title_font=dict(size=22, color='black'),
            ticks="outside", mirror=True, showline=True,
            linewidth=1.5,
            linecolor='black',
            range=[0, 5]
        ),  # Optional: Move y-axis ticks outside
        xaxis_title="Category",
        yaxis_title="Score",
    )
    # put legend on  right corner
    fig.update_layout(
        legend=dict(
            orientation="v", y=0.97, x=0.85, font=dict(size=12), traceorder="normal", itemsizing="trace")
    )

    fig.update_layout(width=800, height=600, template="ggplot2", margin=dict(t=10, b=10, r=10), )
    pio.write_image(fig, "./src/data/plots/human_model_score.png")


def plot_avg_aspect_score():
    # plot the bar plot
    aspects = ["factuality", "relevance", "completeness", "helpful references", "preference"]

    df = pd.read_csv(f"src/data/annotated_data/complete_data_scores.csv", sep="\t")
    df = df[df["category"] == "history"]

    columns = [
        "factuality_model_score",
        "factuality_human_score",
        "relevance_model_score",
        "relevance_human_score",
        "completeness_model_score",
        "completeness_human_score",
        "reference_model_score",
        "reference_human_score",
        "ans_preference_model_score",
        "ans_preference_human_score",
        "overall_model_score",
        "overall_human_score"
    ]

    df = df[columns]

    # aspect level scores for human and model
    model_scores = df[["factuality_model_score", "relevance_model_score", "completeness_model_score",
                        "reference_model_score", "ans_preference_model_score"]].mean(axis=0).values.tolist()
    human_scores = df[["factuality_human_score", "relevance_human_score", "completeness_human_score",
                        "reference_human_score", "ans_preference_human_score"]].mean(axis=0).values.tolist()

    # round the scores
    model_scores = [round(score, 2) for score in model_scores]
    human_scores = [round(score, 2) for score in human_scores]

    # plot stacked bar plot with human and model scores with go lib in plotly
    data = pd.DataFrame({
        "aspect": aspects,
        "human": human_scores,
        "model": model_scores
    })
    # data = data.melt(id_vars="aspect", var_name="score_type", value_name="score")
    # print(data.head())
    fig = go.Figure(data=[
        go.Bar(name='Human', x=data["aspect"], y=data["human"], text=data["human"], marker_color='#1a76ff'),
        go.Bar(name='Model', x=data["aspect"], y=data["model"], text=data["model"], marker_color='#ff7f0e')
    ])
    fig.update_layout(barmode='group')

    fig.update_layout(
        font=dict(family='Times New Roman', size=16, color='black'),
        plot_bgcolor='white',  # Set plot background color
        showlegend=True,
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        xaxis=dict(
            # title_font=dict(size=22, color='black'),
            ticks="outside", mirror=True, showline=True,
            linewidth=1.5,
            linecolor='black',
        ),  # Optional: Move x-axis ticks outside
        yaxis=dict(
            # title_font=dict(size=22, color='black'),
            ticks="outside", mirror=True, showline=True,
            linewidth=1.5,
            linecolor='black',
            range=[0, 1]
        ),  # Optional: Move y-axis ticks outside
        xaxis_title="Evaluation criteria",
        yaxis_title="Average score",
        xaxis_tickfont=dict(family="Computer Modern", size=16, color='black'),
        yaxis_tickfont=dict(family="Computer Modern", size=16, color='black'),
        xaxis_title_font=dict(family="Computer Modern", size=18, color='black'),
        yaxis_title_font=dict(family="Computer Modern", size=18, color='black'),
    )
    # put legend on  right corner
    fig.update_layout(
        legend=dict(
            orientation="v", y=0.97, x=0.65, font=dict(size=12), traceorder="normal", itemsizing="trace")
    )

    fig.update_layout(width=650, height=300, template="ggplot2", margin=dict(t=10, b=10, r=10), )
    pio.write_image(fig, "./src/data/plots/aspect_human_model_score_test.pdf")


def plot_aspect_score_charts():
    # plot the bar plot
    aspects = ["Q. misc.", "Factuality", "Relevance", "Completeness", "References", "Preference"]
    fig = make_subplots(rows=2, cols=4,
                        subplot_titles=("Physics", "Chemistry", "Biology", "Technology", "Economics", "History", "Law",
                                        "Overall"),
                        # shared_yaxes=True,
                        y_title="Aspect score",
                        )

    subjects = ["physics", "chemistry", "biology", "technology", "economics", "history", "law", "all"]
    df = pd.read_csv(f"src/data/annotated_data/complete_data_scores1.csv", sep="\t")


    for i, subject in enumerate(subjects):
        if i == 7:
            showlegend = True
        else:
            showlegend = False
        row = 1
        col = i
        if i > 3:
            row = 2
            col = i - 4

        df = pd.read_csv(f"src/data/annotated_data/complete_data_scores1.csv", sep="\t")
        if subject != "all":
            df = df[df["category"] == subjects[i]]

        columns = [
            "ques_misconception_model_score",
            "ques_misconception_human_score",
            "factuality_model_score",
            "factuality_human_score",
            "relevance_model_score",
            "relevance_human_score",
            "completeness_model_score",
            "completeness_human_score",
            "reference_model_score",
            "reference_human_score",
            "ans_preference_model_score",
            "ans_preference_human_score",
            "overall_model_score",
            "overall_human_score"
        ]

        df = df[columns]

        # aspect level scores for human and model
        model_scores = df[["ques_misconception_model_score",
                           "factuality_model_score", "relevance_model_score", "completeness_model_score",
                           "reference_model_score", "ans_preference_model_score"]].mean(axis=0).values.tolist()
        human_scores = df[["ques_misconception_human_score",
                           "factuality_human_score", "relevance_human_score", "completeness_human_score",
                           "reference_human_score", "ans_preference_human_score"]].mean(axis=0).values.tolist()

        # print(df.head())

        # round the scores
        model_scores = [round(score, 2) for score in model_scores]
        human_scores = [round(score, 2) for score in human_scores]

        # plot stacked bar plot with human and model scores with go lib in plotly
        data = pd.DataFrame({
            "aspect": aspects,
            "human": human_scores,
            "model": model_scores
        })
        # data = data.melt(id_vars="aspect", var_name="score_type", value_name="score")

        # print all df rows
        print(subject)
        pd.set_option('display.max_rows', None)
        print(data.head(10))

        fig.add_trace(
            go.Bar(name='Human', x=data["aspect"], y=data["human"],
                   # text=data["human"],
                   showlegend=showlegend,
                   # textposition="outside",
                   # textfont=dict(color="black", size=12),
                   marker=dict(
                       line=dict(width=1),
                       color='rgba(242, 143, 29, 1)',
                       pattern_shape="\\",
                       # pattern_fillmode="replace",
                   ),
                   ),
            row=row, col=col + 1)

        fig.add_trace(
            go.Bar(name='Model', x=data["aspect"], y=data["model"],
                   # text=data["model"],
                   showlegend=showlegend,
                   # textposition="outside",
                   # textfont=dict(color="black", size=12),
                   marker=dict(
                        # line=dict(),
                        color='rgba(43, 96, 69, 1)',
                        pattern_shape="x",
                        # size=10,
                     ),
                ),
            row=row, col=col + 1)
        fig.update_layout(barmode='group', bargap=0.2)
        tickangle = -30
        tickfont = dict(family="Computer Modern", size=14, color='black')

        fig.update_layout(
            font=dict(family='Times New Roman', size=16, color='black'),
            plot_bgcolor='white',  # Set plot background color
            showlegend=showlegend,
            legend=dict(
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            ),
            xaxis=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1.5,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1.5,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Move y-axis ticks outside
            xaxis1=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1.5,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis1=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1.5,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Mov
            xaxis2=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1.5,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis2=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1.5,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Mov
            xaxis3=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1.5,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis3=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1.5,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Mov
            xaxis4=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1.5,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis4=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1.5,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Mov

            xaxis5=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1.5,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis5=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1.5,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Mov
            xaxis6=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1.5,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis6=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1.5,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Mov
            xaxis7=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1.5,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis7=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1.5,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Mov
            xaxis8=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1.5,
                linecolor='red',
            ),  # Optional: Move x-axis ticks outside
            yaxis8=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1.5,
                linecolor='red',
                range=[0, 1.05]
            ),  # Optional: Mov
        # xaxis_title="Evaluation criteria",
        #     yaxis_title="Aspect score",
        #     yaxis5_title="Aspect score",
            xaxis_tickfont=tickfont,
            yaxis_tickfont=tickfont,
            xaxis1_tickfont=tickfont,
            yaxis1_tickfont=tickfont,
            xaxis2_tickfont=tickfont,
            yaxis2_tickfont=tickfont,
            xaxis3_tickfont=tickfont,
            yaxis3_tickfont=tickfont,
            xaxis4_tickfont=tickfont,
            yaxis4_tickfont=tickfont,
            xaxis5_tickfont=tickfont,
            yaxis5_tickfont=tickfont,
            xaxis6_tickfont=tickfont,
            yaxis6_tickfont=tickfont,
            xaxis7_tickfont=tickfont,
            yaxis7_tickfont=tickfont,
            xaxis8_tickfont=tickfont,
            yaxis8_tickfont=tickfont,
            # xaxis7_title_font=dict(family="Computer Modern", size=12, color='black'),
            # yaxis7_title_font=dict(family="Computer Modern", size=12, color='black'),
        )

    # put legend on  right corner
    fig.update_layout(
        legend=dict(
            orientation="h", y=1.15, x=0.4, font=dict(size=14), traceorder="normal", itemsizing="trace")
    )

    fig.update_layout(width=1000, height=500, template="ggplot2", margin=dict(t=10, b=10,), )
    pio.write_image(fig, "./src/data/plots/aspect_scores2.pdf", scale=5)


def plot_aspect_scores():
    # plot the bar plot
    aspects = ["ques_misconception", "factuality", "relevance", "completeness", "reference", "overall"]
    subjects = ["physics", "chemistry", "biology", "technology", "economics", "history", "law"]
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=("Question misconception", "Factuality", "Relevance", "Completeness",
                                        "References", "Overall"),
                        # shared_yaxes=True,
                        y_title="Aspect score",
                        vertical_spacing=0.3,
                        )

    # subjects = ["physics", "chemistry", "biology", "technology", "economics", "history", "law", "all"]
    df = pd.read_csv(f"src/data/annotated_data/complete_data_scores1.csv", sep="\t")

    for i, aspect in enumerate(aspects):
        if i == 5:
            showlegend = True
        else:
            showlegend = False
        row = 1
        col = i
        if i > 2:
            row = 2
            col = i - 3

        # print(row, col)
        # df = pd.read_csv(f"src/data/annotated_data/complete_data_scores1.csv", sep="\t")
        # print(df.head())
        # print(df.columns)
        # if aspect != "all":
        #     df = df[df["category"] == subjects[i]]

        if aspect == "overall":
            columns = [
                f"{aspect}_model_score",
                f"{aspect}_human_score",
                "category"
            ]
        else:
            columns = [
                f"{aspect}_model_score",
                f"{aspect}_human_score",
                "category"
            ]

        filtered_df = df[columns]
        # filter by subject
        model_scores, human_scores = [], []
        for subject in subjects:
            # if aspect == "all":
            #     cat_filtered_df = filtered_df
            # else:
            cat_filtered_df = filtered_df[filtered_df["category"] == subject]
            # print(cat_filtered_df.head(20))
            # aspect level scores for human and model
            model_scores.append(cat_filtered_df[[f"{aspect}_model_score"]].mean(axis=0).values.tolist()[0])
            human_scores.append(cat_filtered_df[[f"{aspect}_human_score"]].mean(axis=0).values.tolist()[0])

        # print(model_scores)
        # print(human_scores)
        # print(df.head())
        # break

        if aspect == "overall":
            model_scores = [round(score/5, 2) for score in model_scores]
            human_scores = [round(score/5, 2) for score in human_scores]
        # round the scores
        model_scores = [round(score, 2) for score in model_scores]
        human_scores = [round(score, 2) for score in human_scores]

        # plot stacked bar plot with human and model scores with go lib in plotly
        data = pd.DataFrame({
            "subject": subjects,
            "human": human_scores,
            "model": model_scores
        })
        # data = data.melt(id_vars="aspect", var_name="score_type", value_name="score")

        # print all df rows
        # print(subject)
        pd.set_option('display.max_rows', None)
        # print(data.head(10))

        fig.add_trace(
            go.Bar(name='Human', x=data["subject"], y=data["human"],
                   # text=data["human"],
                   showlegend=showlegend,
                   # textposition="outside",
                   # textfont=dict(color="black", size=12),
                   marker=dict(
                       line=dict(width=1),
                       color='rgba(242, 143, 29, 1)',
                       pattern_shape="\\",
                       # pattern_fillmode="replace",
                   ),
                   ),
            row=row, col=col + 1)

        fig.add_trace(
            go.Bar(name='Model', x=data["subject"], y=data["model"],
                   # text=data["model"],
                   showlegend=showlegend,
                   # textposition="outside",
                   # textfont=dict(color="black", size=12),
                   marker=dict(
                        # line=dict(),
                        color='rgba(43, 96, 69, 1)',
                        pattern_shape="x",
                        # size=10,
                     ),
                ),
            row=row, col=col + 1)
        fig.update_layout(barmode='group', bargap=0.2)
        tickangle = -30
        tickfont = dict(family="Computer Modern", size=12, color='black')

        fig.update_layout(
            font=dict(family='Times New Roman', size=14, color='black'),
            plot_bgcolor='white',  # Set plot background color
            showlegend=showlegend,
            legend=dict(
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            ),
            xaxis=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Move y-axis ticks outside
            xaxis1=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis1=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Mov
            xaxis2=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis2=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Mov
            xaxis3=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis3=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Mov
            xaxis4=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis4=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Mov

            xaxis5=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1,
                linecolor='black',
            ),  # Optional: Move x-axis ticks outside
            yaxis5=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1,
                linecolor='black',
                range=[0, 1.05]
            ),  # Optional: Mov
            xaxis6=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True, tickangle=tickangle,
                linewidth=1.5,
                linecolor='red',
            ),  # Optional: Move x-axis ticks outside
            yaxis6=dict(
                # title_font=dict(size=22, color='black'),
                ticks="outside", mirror=True, showline=True,
                linewidth=1.5,
                linecolor='red',
                range=[0, 1.05],
                # dtick=1
            ),  # Optional: Mov
        # xaxis_title="Evaluation criteria",
        #     yaxis_title="Aspect score",
        #     yaxis5_title="Aspect score",
            xaxis_tickfont=tickfont,
            yaxis_tickfont=tickfont,
            xaxis1_tickfont=tickfont,
            yaxis1_tickfont=tickfont,
            xaxis2_tickfont=tickfont,
            yaxis2_tickfont=tickfont,
            xaxis3_tickfont=tickfont,
            yaxis3_tickfont=tickfont,
            xaxis4_tickfont=tickfont,
            yaxis4_tickfont=tickfont,
            xaxis5_tickfont=tickfont,
            yaxis5_tickfont=tickfont,
            xaxis6_tickfont=tickfont,
            yaxis6_tickfont=tickfont,

        )

    # put legend on  right corner  x=0.4 y=1.15
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=1.1, xanchor="left", x=0.35, font=dict(size=12),
            traceorder="normal", itemsizing="trace", itemwidth=30)
    )

    fig.update_layout(width=700, height=350, template="ggplot2", margin=dict(t=5, b=5), )
    pio.write_image(fig, "./src/data/plots/aspect_scores4.pdf", scale=5)


def length_plot(category: List = [], num_annotators: int = 3):

    dataframes = []
    for i in range(len(category)):
        # if category[i] == "economics":
        #     num_annotators = 2
        data = calculate_answer_stats(
            category=category[i],
            num_annotators=num_annotators
        )
        data["category"] = [category[i].capitalize()]*len(data)
        dataframes.append(data)

    df = pd.concat(dataframes, ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Violin(x=df["category"],
                            y=df["human_ans_len"],
                            legendgroup='Human', scalegroup='Human', name='Human',
                            side='positive',
                            line_color='blue',
                            orientation="v"
                            )
                  )
    fig.add_trace(go.Violin(x=df["category"],
                            y=df["model_ans_len"],
                            legendgroup='Model', scalegroup='Model', name='Model',
                            side='negative',
                            line_color='orange',
                            orientation="v"
                            )
                  )
    fig.update_traces(meanline_visible=True)
    fig.update_layout(
        violingap=0,
        violinmode='overlay'
       )
    # fig.show()

    fig.update_layout(
        font=dict(family='Computer Modern', size=18, color='black'),
        plot_bgcolor='white',  # Set plot background color
        # showlegend=True,
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        xaxis=dict(
            # title_font=dict(family="Computer Modern", size=30, color='black'),
            ticks="outside", mirror=True, showline=True,
            linewidth=1.5,
            linecolor='black',

        ),  # Optional: Move x-axis ticks outside
        yaxis=dict(
            # title_font=dict(family="Computer Modern", size=30, color='black'),
            ticks="outside", mirror=True, showline=True,
            linewidth=1.5,
            linecolor='black',
            range=[0, 600]
        ),  # Optional: Move y-axis ticks outside
        # make y ticks tilted
        # yaxis_tickangle=-45,
        yaxis_title="Answer length",
        xaxis_title="Category",
        xaxis_tickfont=dict(family="Computer Modern", size=16, color='black'),
        yaxis_tickfont=dict(family="Computer Modern", size=16, color='black'),
        xaxis_title_font=dict(family="Computer Modern", size=18, color='black'),
        yaxis_title_font=dict(family="Computer Modern", size=18, color='black'),
    )
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="middle", y=1.15, xanchor="center", x=0.48, font=dict(size=16), itemsizing="trace")
        )
    fig.update_layout(width=600, height=350, template="ggplot2", margin=dict(t=10, b=10, r=10), )
    pio.write_image(fig, "./src/data/plots/ans_length_stats.pdf", scale=5)

    # fig.show()


def create_pie_chart():
    samples = [110, 96, 110, 92, 86, 94, 110]
    categories = ["biology", "chemistry", "economics", "history", "law", "physics", "technology"]

    fig = go.Figure(data=[go.Pie(labels=categories, values=samples, hole=.3)])
    fig.update_layout(
        font=dict(family='Times New Roman', size=12, color='black'),
        plot_bgcolor='white',  # Set plot background color
        # showlegend=True,
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        xaxis=dict( # Optional: Move x-axis ticks outside
            title_font=dict(size=22, color='black'),
            ticks="outside", mirror=True, showline=True,
            linewidth=1.5,
            linecolor='black',
        ),
        yaxis=dict( # Optional: Move y-axis ticks outside
            title_font=dict(size=22, color='black'),
            ticks="outside", mirror=True, showline=True,
            linewidth=1.5,
            linecolor='black',
        ),
        xaxis_title="No. of samples",
        yaxis_title="Category",
    )
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="middle", y=1.1, xanchor="center", x=0.5, font=dict(size=14), itemsizing="trace")
        )
    fig.update_layout(width=500, height=500, template="ggplot2", margin=dict(t=10, b=10, r=10, l=10), )
    pio.write_image(fig, "./src/data/plots/pie_chart.png")


def create_aspect_importance_chart():
    from src.analysis.aspect_importance import AspectImportance

    aspects = ["ques_misconception", "factuality", "irrelevance", "incomplete_ans", "reference_example"]
    categories = ["physics", "chemistry", "biology", "technology", "economics", "history", "law"]
    base_path = "src/data/annotated_data"
    files = os.listdir(base_path)

    importances = {}
    for category in categories:
        result = {"human": {}, "model": {}}  # store the results for each aspect
        data_path = f"{base_path}/{category}"
        imp = AspectImportance(
            data_path=os.path.join(data_path, "processed_data.csv"),
            aspects=aspects,
        )
        human_imp, model_imp = imp.analyze_fine_grained_importance()
        # print(human)

        for aspect, imp in human_imp.items():
            if aspect in result:
                result["human"][aspect] += imp
            else:
                result["human"][aspect] = imp
        for aspect, imp in model_imp.items():
            if aspect in result:
                result["model"][aspect] += imp
            else:
                result["model"][aspect] = imp
        # make all result values to 2 decimal places
        for key, value in result.items():
            for aspect, imp in value.items():
                result[key][aspect] = round(imp, 2)
        importances[category] = result

    print(importances)

    df = pd.DataFrame.from_dict(importances, orient='index')

    # Convert 'human' and 'model' dictionaries to separate DataFrames
    human_df = pd.DataFrame(df['human'].to_dict()).T
    model_df = pd.DataFrame(df['model'].to_dict()).T

    # Concatenate DataFrames along columns with a multi-level column index
    result = pd.concat([human_df, model_df], axis=1, keys=['human', 'model'])
    # print(result.head())

    fig = go.Figure(
        layout=go.Layout(
            height=400,
            width=1000,
            barmode="relative",
            yaxis_showticklabels=True,
            yaxis_showgrid=False,
            yaxis_range=[0, result.groupby(axis=1, level=0).sum().max().max() * 1.7],
            # Secondary y-axis overlayed on the primary one and not visible
            yaxis2=go.layout.YAxis(
                visible=False,
                matches="y",
                overlaying="y",
                anchor="x",
            ),
            template="ggplot2",
            # font=dict(size=24),
            legend_x=1,
            legend_y=1,
            legend_orientation="v",
            hovermode="x",
            margin=dict(b=0, t=10, l=0, r=10)
        )
    )

    col_names = {
        "ques_misconception": "question misconception",
        "factuality": "factuality",
        "irrelevance": "relevance",
        "incomplete_ans": "completeness",
        "reference_example": "references"
    }
    # rename result columns
    result = result.rename(columns=col_names)
    # print(result.head())

    colors = {
        "human": {
            # "question misconception": "rgba(222,45,38,1.0)",
            # "factuality": "rgba(222,45,38,0.8)",
            # "relevance": "rgba(222,45,38,0.6)",
            # "completeness": "rgba(222,45,38,0.4)",
            # "references": "rgba(222,45,38,0.2)",
            "question misconception": "rgba(242, 143, 29, 1)",
            "factuality": "rgba(242, 143, 29, 0.8)",
            "relevance": "rgba(242, 143, 29, 0.6)",
            "completeness": "rgba(242, 143, 29, 0.4)",
            "references": "rgba(242, 143, 29, 0.2)",
        },
        "model": {
            "question misconception": "rgba(43, 96, 69, 1)",
            "factuality": "rgba(43, 96, 69, 0.8)",
            "relevance": "rgba(43, 96, 69, 0.6)",
            "completeness": "rgba(43, 96, 69, 0.4)",  #"#F28F1D",
            "references": "rgba(43, 96, 69, 0.2)",  #"#F6C619",
        }
    }

    # Add the traces
    for i, t in enumerate(colors):
        for j, col in enumerate(result[t].columns):
            if (result[t][col] == 0).all():
                continue
            fig.add_bar(
                x=result.index,
                y=result[t][col],
                # Set the right yaxis depending on the selected product (from enumerate)
                yaxis=f"y{i + 1}",
                # Offset the bar trace, offset needs to match the width
                # For categorical traces, each category is spaced by 1
                offsetgroup=str(i),
                offset=(i - 1) * 1 / 3,
                width=1 / 3,
                legendgroup=t,
                legendgrouptitle_text=t,
                name=col,
                marker_color=colors[t][col],
                marker_line=dict(width=1, color="#333"),
                hovertemplate="%{y}<extra></extra>"
            )

    fig.update_layout(
        font=dict(family='Computer Modern', size=16, color='black'),
        plot_bgcolor='white',  # Set plot background color
        # showlegend=True,
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        xaxis=dict(
            # title_font=dict(family="Computer Modern", size=30, color='black'),
            ticks="outside", mirror=True, showline=True,
            linewidth=1.5,
            linecolor='black',

        ),  # Optional: Move x-axis ticks outside
        yaxis=dict(
            # title_font=dict(family="Computer Modern", size=30, color='black'),
            ticks="outside", mirror=True, showline=True,
            tick0=0,
            linewidth=1.5,
            linecolor='black',
            range=[0, 105]
        ),  # Optional: Move y-axis ticks outside
        # make y ticks tilted
        # yaxis_tickangle=-45,
        yaxis_title="Aspect annotations (%)",
        xaxis_title="Category",
        xaxis_tickfont=dict(family="Computer Modern", size=16, color='black'),
        yaxis_tickfont=dict(family="Computer Modern", size=16, color='black'),
        xaxis_title_font=dict(family="Computer Modern", size=18, color='black'),
        yaxis_title_font=dict(family="Computer Modern", size=18, color='black'),
    )

    pio.write_image(fig, "./src/data/plots/percent_hallucinations.pdf")
    # fig.show()


def plotly_wordcloud():
    import random
    random.seed(22)
    from src.annotation.analysis import get_top_k_words

    path = "src/data/annotated_data/complete_data_scores.csv"
    word_freq = get_top_k_words(path, "ans_preference", 50)
    print(word_freq)

    # word frequency to dataframe
    df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
    # remove duplicates
    df = df.drop_duplicates(subset=['Word'])
    words_to_remove = ['energy', 'however']

    # Filter the DataFrame to remove entries with the specified words
    df = df[~df['Word'].isin(words_to_remove)]
    num_words = df.shape[0]

    default_color = "rgba(0, 181, 204, 1.0)"
    aspects_new_color = "rgba(250, 190, 88, 1.0)"   #"rgba(242, 143, 29, 0.8)"
    aspects_studied_color = "rgba(210, 238, 130, 1)"

    word_color_map = {
        'comprehensive': aspects_studied_color,
        'concise': aspects_new_color,
        'understand': aspects_new_color,
        'easier': aspects_new_color,
        'clearly': aspects_new_color,
        'complete': aspects_studied_color,
        'relevant': aspects_studied_color,
        'factually': aspects_studied_color,
        'analogy': aspects_studied_color,
        'example': aspects_studied_color,
    }

    # Function to map words to colors
    def map_word_to_color(word):
        return word_color_map.get(word, default_color)

    # Add a new 'Color' column based on the mapping
    df['Color'] = df['Word'].apply(map_word_to_color)

    # make random position list for num words
    position_list = [random.randint(0, 50) for _ in range(num_words)]
    print(position_list)
    position_list = [8, 17, 0, 39, 28, 11, 34, 6, 47, 41, 22, 50, 5, 14, 25, 3, 20, 38, 11, 35, 43, 46,
                     27, 45, 3, 36, 1, 40, 16, 19, 26, 12, 15, 7, 37, 33, 46, 49, 36]

    trace1 = {
        "mode": "markers+text",
        "name": "Answer Preference",
        "type": "scatter",
        "x": df["Word"],
        "y": position_list,
        "marker": {
            "symbol": "circle",
            "size": df["Frequency"]*3,
            "sizemode": "diameter",
            "color": df["Color"]
        },
        "text": df["Word"].tolist(),
        "textposition": "middle center",
    }
    fig = go.Figure()
    fig.add_trace(trace=trace1)
    fig.update_layout(
        font=dict(family='Times New Roman', size=16, color='black'),
        plot_bgcolor='white',  # Set plot background color
        # showlegend=showlegend,
        legend=dict(
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        xaxis=dict(
            # title_font=dict(size=22, color='black'),
            ticks="",
            showticklabels=False,
            mirror=True,
            showline=True,
            linewidth=1.5,
            linecolor='black',

        ),  # Optional: Move x-axis ticks outside
        yaxis=dict(
            # title_font=dict(size=22, color='black'),
            ticks="",
            showticklabels=False,
            mirror=True,
            showline=True,
            linewidth=1.5,
            linecolor='black',

        )
    )
    fig.update_layout(width=1000, height=400, template="ggplot2", margin=dict(t=10, b=10, ))
    pio.write_image(fig, "./src/data/plots/ans_reason_frequency.pdf", scale=5)


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

    # categories = ["physics", "chemistry", "biology", "technology", "economics", "history", "law"]
    # length_plot(
    #     category=categories,
    #     num_annotators=3,
    # )
    # plot_score_bar_plot()
    # create_pie_chart()
    # plot_aspect_score_charts()
    # plot_avg_aspect_score()
    # plot_aspect_scores()
    # create_aspect_importance_chart()
    plotly_wordcloud()
