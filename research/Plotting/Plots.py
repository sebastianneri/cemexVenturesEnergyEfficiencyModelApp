import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
from itertools import combinations

def saveFigType(fig, type, data_type, name):
    fig.saveFigType(f"./Plots/{type}/{data_type}/{name}.jpeg")

def getCBoxPlots(df, number):
    for i in df.columns:
        bp = sns.boxplot(df[i])
        bp.get_figure().savefig(f"./Plots/{i}{number}.png")
        bp.get_figure().clf()

def getTableOfScatters(df, name=None):
    fig, axs = plt.subplots(len(df.columns), len(df.columns), figsize = (25, 25))
    c = -1
    p = -1
    for i in df.columns:
        c += 1
        for k in df.columns:
            p += 1
            axs[p, c].scatter(df[k], df[i])
            axs[p, c].set_xlabel(k)
            axs[p, c].set_ylabel(i)
        p = -1
    if name != None:
        fig.savefig(f"./Plots/Scatters/{name}.jpeg")

def removeOutliers(columns, df):
    for i in columns:
        q1_q = df[i].quantile(0.25)
        q3_q = df[i].quantile(0.75)
        iqr_q = q3_q - q1_q
        df = df[(df[i] >= q1_q - 1.5*iqr_q) & (df[i] <= q3_q + 1.5*iqr_q)]
    return df

def getHeatMap(df, corr_type):
    plt.figure(figsize=(30, 25))
    if corr_type == '':
        sns.heatmap(df, annot=True)
    else:
        sns.heatmap(df.corr(corr_type), annot=True)


def get3DChart(df, rows, cols, html_file_name):
    specs = []
    for i in range(rows):
        rows_specs = []
        for k in range(cols):
            rows_specs.append({'type':'scatter3d'})
        specs.append(rows_specs)
        
    df_combinations = list(combinations(df.columns, 3))
    titles = []
    for i in df_combinations:
        title = ''
        for k in i:
            if len(title) > 0:
                title = title + ' vs ' + k
            else:
                title = k
        titles.append(title)

    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=titles, horizontal_spacing = 0.06, vertical_spacing = 0.007)
    plot_num = -1
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            plot_num += 1
            x = df[df_combinations[plot_num][0]]
            y = df[df_combinations[plot_num][1]]
            z = df[df_combinations[plot_num][2]]
            
            meta = [titles[plot_num], df_combinations[plot_num][0], df_combinations[plot_num][1], df_combinations[plot_num][2]]
            
            if df_combinations[plot_num][1] == "Dureza":
                color = y
            elif df_combinations[plot_num][2] == "Dureza":
                color = z
            else:
                color = x

            scatter = go.Scatter3d(x=x,y=y,z=z, 
                                   mode='markers', 
                                   marker=dict(size=12,color=color, colorscale='Plasma'))
            
            fig.add_trace(scatter, row=i, col=j)
            fig.update_scenes(xaxis = dict(title_text = df_combinations[plot_num][0]),
                                  yaxis = dict(title_text = df_combinations[plot_num][1]),
                                  zaxis = dict(title_text = df_combinations[plot_num][2]), row=i, col=j)
            

    fig.update_layout(
    title_text=f'{html_file_name}',
    height=820*rows,
    width=1300,
    title_x=0.5,
    titlefont=dict(size=32)
    )
    url = 'file://' + os.getcwd() + '/3DCharts/' + html_file_name + ".html"
    fig.write_html(f"./3DCharts/{html_file_name}.html")
    print(f"Archivo en: {url}")

