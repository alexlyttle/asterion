import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def model(x, a, b):
    return np.sin(a*x + b)

# df = pd.read_csv('../examples/data/modes.csv')
a0 = 0.
b0 = 0.


df = pd.read_csv('../examples/data/modes.csv')
n = np.linspace(1, 40, 40, dtype=int)

df['y'] = model(df['x'], a0, b0)


app.layout = html.Div([
        html.Div([
                dcc.Graph(id='graph'),
                dcc.Slider(
                    id='a',
                    min=-10.0,
                    max=10.0,
                    value=a0,
                    step=1.0,
                    updatemode='drag',
                ),
                dcc.Slider(
                    id='b',
                    min=-10.0,
                    max=10.0,
                    value=a0,
                    step=1.0,
                    updatemode='drag',
                ),
            ],
            className='six columns',
        ),
        html.Div([
                dcc.Graph(id='graph-2'),
                dcc.Slider(
                    id='a-2',
                    min=-10.0,
                    max=10.0,
                    value=a0,
                    step=1.0,
                    updatemode='drag',
                ),
                dcc.Slider(
                    id='b-2',
                    min=-10.0,
                    max=10.0,
                    value=a0,
                    step=1.0,
                    updatemode='drag',
                ),
            ],
            className='six columns',
        ),
    ],
    className='row'
)


@app.callback(
    Output('graph', 'figure'),
    Input('a', 'value'),
    Input('b', 'value'),
)
def update_figure(a, b):

    df['y'] = model(df['x'], a, b)

    fig = px.line(df, x='x', y='y')
    fig.update_layout(
        # transition_duration=500
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
