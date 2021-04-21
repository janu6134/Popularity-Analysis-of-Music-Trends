# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:44:37 2021

@author: Janaki
"""

import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df1 = pd.read_csv("testdf.csv")
df1 = df1.drop(df1.query('Year == 2019').sample(frac=.9).index)
maxval = df1['Streams'].max()

'''
df1 = df1.dropna()
df1 = df1.drop(df1.query('Year == 2019').sample(frac=.9).index)
df1 = df1.drop(df1.query('Year == 2017').sample(frac=.9).index)
df1 = df1.drop(df1.query('Year == 2018').sample(frac=.9).index)
df1 = df1.drop(df1.query('Year == 2019').sample(frac=.9).index)
df1 = df1.drop(df1.query('Year == 2018').sample(frac=.9).index)
df1 = df1.drop(df1.query('Year == 2017').sample(frac=.9).index)
df1 = df1.drop(df1.query('Year == 2018').sample(frac=.9).index)
df1 = df1.drop(df1.query('Year == 2017').sample(frac=.9).index)
df1 = df1.rename(columns={"Intrumentalness": "Instrumentalness"})
'''
testarr = [random.uniform(0, 0.7), random.uniform(0, 0.9), random.uniform(0, 0.994),
           random.uniform(0, 0.8), random.uniform(0, 0.8), random.uniform(0, 0.8)]
colors = {
    'background': '#00022e',
    'text': '#FFFFFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Popularity analysis of music trends',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Select a year from the slider below, and hover over the map to choose a country.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    html.Br(),
    html.Br(),
    html.Br(),

    html.Div([
        html.Div([
            #html.Button('Reset', id='reset_button', n_clicks=0),
            html.Br(),
            html.Div(dcc.Slider(
                id='crossfilter-year--slider',
                min=df1['Year'].min(),
                max=df1['Year'].max(),
                value=df1['Year'].max(),
                marks={str(year): str(year) for year in df1['Year'].unique()},
                step=None
            ), style={'width': '49%', 'padding': '50'}),

            dcc.Graph(
                id='crossfilter-indicator-map',
                hoverData={'points': [{'customdata': 'Canada'}]},
            ),
        ], className="six columns"),

        html.Div([
            html.H3('Count of acoustic feature values in overall track list for selected country', style={
                'color': colors['text']}),

            dcc.Graph(id='acousticfeatures', hoverData=None)
        ], className="six columns")
    ], className="row"),

    html.Div([
        html.Div([
            html.Br(),
            html.Br(),
            html.H3('Artists vs Number of Streams', style={
                'color': colors['text']
            }),
            dcc.Graph(id='x-time-series', hoverData=None)
        ], className="six columns"),
        html.Br(),
        html.Div([
            html.Br(),
            html.H3('Artists vs Acoustic Features', style={
                'color': colors['text']}),
            dcc.Dropdown(id="ac_feat",
                         options=[
                             {"label": "Danceability", "value": "Danceability"},
                             {"label": "Energy", "value": "Energy"},
                             {"label": "Acousticness", "value": "Acousticness"},
                             {"label": "Instrumentalness", "value": "Instrumentalness"},
                             {"label": "Valence", "value": "Valence"},
                             {"label": "Liveness", "value": "Liveness"}],
                         multi=False,
                         value="Danceability",
                         style={'width': "40%"}
                         ),
            dcc.Graph(id='acoustic_artist')
        ], className="six columns"),

    ], className="row"),

    html.Br(),
    html.Br(),
    html.Br(),

    html.Div([
        html.Div([
            html.H3('Tracks vs Number of Streams', style={
                'color': colors['text']}),
            dcc.Graph(id='y-time-series', hoverData=None)
        ], className="six columns"),
        html.Div([
            html.H3('Selected Track vs Acoustic Feature', style={
                'color': colors['text']
            }),
            html.Br(),
            dcc.Graph(id='track-acoustic')
        ], className="six columns")
    ], className="row")
])

'''
@app.callback(Output('input_button', 'n_clicks'),
              [Input('reset_button', 'n_clicks')])
def update(reset):
    return 0
'''


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-map', 'figure'),
    [dash.dependencies.Input('crossfilter-year--slider', 'value'),
     dash.dependencies.Input('x-time-series', 'hoverData'),
     dash.dependencies.Input('acousticfeatures', 'hoverData'),
     dash.dependencies.Input('y-time-series', 'hoverData')])
def update_graph(option_slctd, hoverartist, hoveracoustic, hovertrack):
    df = df1[df1.Year == option_slctd]
    if hoverartist:
        # print("Hoverartist selected")
        # print(f'hoverartist:{hoverartist}')
        country_artist = hoverartist['points'][0]['label']
        df = df[df.Artist == country_artist]

    if hoveracoustic:
        # print(f'hoveracoustic: {hoveracoustic}')
        acoustic_feat = hoveracoustic['points'][0]['x']
        count_acoust = hoveracoustic['points'][0]['y']
        df = df[df[acoustic_feat] > 0.5]
        # print(df['Country'].nunique())

    if hovertrack:
        # print(f'hovertrack: {hovertrack}')
        df = df[df['Track Name'] == hovertrack['points'][0]['y']]

    fig = px.choropleth(
        data_frame=df,
        locations='iso_alpha',
        color='Streams',
        hover_name="Country",
        projection='natural earth',
        color_continuous_scale=px.colors.sequential.YlOrRd,
        range_color=[100000, maxval],
        template='plotly_dark'
    )

    fig.update_traces(customdata=df['Country'])
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20))
    #fig.update_layout(margin={'l': 90, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    return fig


def create_time_series(df, axis_type, title):
    c = df['Artist'].value_counts()
    c_val = df['Country'].iloc[0]
    fig = px.bar(
        data_frame=df,
        x='Streams',
        y='Artist',  # differentiate color of marks
        opacity=0.9,  # set opacity of markers (from 0 to 1)
        orientation="h",
        color='Streams',
        range_color=[100000,maxval],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={"Artist": "Artist",
                'Track ID': "Number of tracks",
                'Streams': 'Streams',
                'Country': 'Country'},  # map the labels of the figure
        title=f'Artist vs Track Count for "{c_val}"',  # figure title
        width=700,  # figure width in pixels
        height=700,
        hover_name="Artist",
        template='plotly_dark',
        custom_data=['Country', 'Artist', 'Year', 'Streams'])
    # color_continuous_scale=px.colors.sequential.YlOrRd)
    # fig.update_traces(customdata=df['Artist'])
    fig.update_layout(barmode='stack', yaxis={'categoryorder': 'total ascending'})
    return fig


def create_acoustic_chart(df, axis_type, title):
    dance = len(df[df['Danceability'] > 0.5])
    temp = df[df['Danceability'] > 0.5]
    # print(f'No of countries : {temp.iloc[0]}')
    energy = len(df[df['Energy'] > 0.5])
    acousticness = len(df[df['Acousticness'] > 0.5])
    instrumentalness = len(df[df['Instrumentalness'] > 0.5])
    valence = len(df[df['Valence'] > 0.5])
    liveness = len(df[df['Liveness'] > 0.5])
    stri = ['Danceability', 'Energy', 'Acousticness', 'Instrumentalness', 'Valence', 'Liveness']
    arr = []
    arr.append(dance)
    arr.append(energy)
    arr.append(acousticness)
    arr.append(instrumentalness)
    arr.append(valence)
    arr.append(liveness)
    fig = go.Figure(data=[go.Bar(
        x=stri,
        y=arr,
        text=arr,
        textposition='auto',
    )])
    fig.update_layout(plot_bgcolor='rgb(10,10,10)')
    fig.update_xaxes(title_text="Acoustic Features")
    fig.update_yaxes(title_text="Count when > 0.5")

    return fig


def create_track_list(df, axis_type):
    a_val = df['Artist'].iloc[0]
    fig = px.bar(
        data_frame=df,
        x='Streams',
        y='Track Name',  # differentiate color of marks
        opacity=0.9,  # set opacity of markers (from 0 to 1)
        orientation="h",
        # color='Streams',
        labels={"Stream": "Stream",
                'Track Name': 'Track Name',
                'Country': 'Country'},  # map the labels of the figure
        title=f'Track vs Stream Count for Artist "{a_val}"',  # figure title
        width=700,  # figure width in pixels
        height=700,
        # color_continuous_scale=px.colors.sequential.YlOrRd,
        hover_name="Track Name",  # figure height in pixels
        template='plotly_dark')
    fig.update_traces(customdata=df['Track Name'])
    fig.update_layout(barmode='stack', yaxis={'categoryorder': 'total ascending'})
    return fig


def acoustic_artist_chart(df, acfeat):
    fig = px.scatter(
        data_frame=df,
        x=acfeat,
        y='Artist',
        labels={"Artist Name": "Artist",
                'Danceability': acfeat},  # map the labels of the figure
        title='Artist vs. Acoustic Features',
        hover_name="Artist",  # figure height in pixels
        template='plotly_dark',
        color=acfeat)

    return fig


def create_track_acoustic(df, axis):
    #print(df)
    t_val = df['Track Name'].iloc[0]
    stri = ['danceability', 'energy', 'acousticness', 'instrumentalness', 'valence', 'liveness']
    fig = go.Figure(data=[go.Bar(
        x=stri,
        y=testarr,
        text=stri,
        textposition='auto',
    )])
    fig.update_layout(plot_bgcolor='rgb(10,10,10)')
    fig.update_xaxes(title_text="Acoustic Features")
    fig.update_yaxes(title_text="Value")
    return fig


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-map', 'hoverData'),
     dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_y_timeseries(hoverData, slidervalue):
    country_name = hoverData['points'][0]['customdata']
    df = df1[df1['Country'] == country_name]
    df = df[df['Year'] == slidervalue]
    title = '<b>{}</b><br>{}'.format(country_name, slidervalue)
    axis_type = "Linear"
    return create_time_series(df, axis_type, title)  # This is for artist vs streams


@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-map', 'hoverData'),
     dash.dependencies.Input('crossfilter-year--slider', 'value'),
     dash.dependencies.Input('x-time-series', 'hoverData')])
def update_x_timeseries(hoverData, slidervalue, hoverartist):
    country_name = hoverData['points'][0]['customdata']
    if hoverartist is None:
        dff = df1[df1['Country'] == country_name]
        dff = dff[dff['Year'] == slidervalue]
        artist_name = dff.iloc[[0]]['Artist']
        artist_name = list(artist_name)
        dff = dff[dff['Artist'] == artist_name[0]]
    else:
        # print(f'hoverartist : {hoverartist}')
        artist_name = hoverartist['points'][0]['label']
        dff = df1[df1['Country'] == country_name]
        dff = dff[dff['Year'] == slidervalue]
        dff = dff[dff['Artist'] == artist_name]
        # print(dff)
    axis_type = "Linear"
    return create_track_list(dff, axis_type)  # This is for track vs streams


# create one for track vs acoustic features

@app.callback(
    dash.dependencies.Output('acousticfeatures', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-map', 'hoverData'),
     dash.dependencies.Input('crossfilter-year--slider', 'value')])
def update_y_timeseries(hoverData, slidervalue):
    country_name = hoverData['points'][0]['customdata']
    df = df1[df1['Country'] == country_name]
    df = df[df['Year'] == slidervalue]
    title = '<b>{}</b><br>{}'.format(country_name, slidervalue)
    axis_type = "Linear"
    return create_acoustic_chart(df, axis_type, title)  # this is for acoustic features count per country


@app.callback(
    dash.dependencies.Output('acoustic_artist', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-map', 'hoverData'),
     dash.dependencies.Input('crossfilter-year--slider', 'value'),
     dash.dependencies.Input('ac_feat', 'value')])
def update_y_timeseries(hoverData, slidervalue, acfeat):
    country_name = hoverData['points'][0]['customdata']
    df = df1[df1['Country'] == country_name]
    df = df[df['Year'] == slidervalue]
    axis_type = "Linear"
    return acoustic_artist_chart(df, acfeat)


@app.callback(
    dash.dependencies.Output('track-acoustic', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-map', 'hoverData'),
     dash.dependencies.Input('crossfilter-year--slider', 'value'),
     dash.dependencies.Input('x-time-series', 'hoverData'),
     dash.dependencies.Input('y-time-series', 'hoverData')])
def track_acoustic_update(hoverData, slidervalue, hoverartist, hovertrack):
    # print("Enter")
    country_name = hoverData['points'][0]['customdata']
    dff = df1[df1['Country'] == country_name]
    # print(dff['Country'])
    dff = dff[dff['Year'] == slidervalue]
    # print(dff['Year'])
    if hoverartist is None:
        artist_name = dff.iloc[[0]]['Artist']
        artist_name = list(artist_name)
        dff = dff[dff['Artist'] == artist_name[0]]
    else:
        artist_name = hoverartist['points'][0]['y']
        dff = dff[dff['Artist'] == artist_name]
        # print(artist_name)
    # print(dff)

    # print(dff['Artist'])
    if hovertrack is None:
        track_name = dff.iloc[[0]]['Track Name']
        track_name = list(track_name)
        # dff = dff[dff['Track Name'] == track_name[0]]
    else:
        track_name = hovertrack['points'][0]['y']
        dff = dff[dff['Track Name'] == track_name]
        # print(track_name)

    axis_type = "Linear"
    return create_track_acoustic(dff, axis_type)  # This is for track vs acoustic features


if __name__ == '__main__':
    app.run_server(debug=True)
