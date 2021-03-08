import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

from glitch_fit import asy_fit, he_amplitude, he_glitch

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('data/modes.csv')
star = df.loc[500]
nu_cols = [f'nu_0_{i+1}' for i in range(40)]
# nu = star[nu_cols]
n = np.linspace(1, 40, 40, dtype=int)

idx_max = np.argmin(np.abs(star[nu_cols] - star['nu_max']))

def polynomial(x, a0, a1, a2, a3, a4):
    return a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4

def get_idx(n_modes):
    idx = np.arange(idx_max - np.floor(n_modes/2), idx_max + np.ceil(n_modes/2), dtype=int)
    return idx

def make_figures(n_modes, a0, a1, a2, a3, a4, b0, b1, tau, phi):
    idx = get_idx(n_modes)
    nu = star[nu_cols][idx]
    nu_asy = polynomial(n[idx], a0, a1, a2, a3, a4)
    nu_fit = np.linspace(nu[0], nu[-1], 200)
    n_fit = np.linspace(n[idx][0], n[idx][-1], 200)
    # dnu = he_glitch(nu_fit, b0, b1, tau, phi)
    dnu = he_glitch(n_fit*star['delta_nu_fit'], b0, b1, tau, phi)

    fig0 = go.Figure()
    # fig1 = px.scatter(x=star[nu_cols], y=star[nu_cols]%delta_nu)
    fig0.add_trace(go.Scatter(x=nu, y=nu%delta_nu, mode='markers', name='data'))
    fig0.add_trace(go.Scatter(x=nu_asy, y=nu_asy%delta_nu, mode='lines', name='asy fit'))
    # px.scatter()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=nu, y=nu-nu_asy, mode='markers', name='data'))   
    fig1.add_trace(go.Scatter(x=nu_fit, y=dnu, mode='lines', name='glitch fit'))
    return fig0, fig1

n_modes = 20
delta_nu = star['delta_nu_fit']
nu_max = star['nu_max']
epsilon = 0.3 * np.log10(nu_max) + 0.4
alpha = nu_max**(-0.9)
n_max = nu_max / delta_nu + epsilon

a0 = delta_nu * (epsilon + 0.5*alpha*n_max**2)
a1 = delta_nu * (1 - alpha)
a2 = 0.5 * delta_nu * alpha
a3 = 0.0
a4 = 0.0

b0 = 5e-3
b1 = 5e-7
tau = nu_max**(-0.9)
phi = 0.

fig1, fig2 = make_figures(n_modes, a0, a1, a2, a3, a4, b0, b1, tau, phi)

app.layout = html.Div([
        html.Div([
                html.Div([
                        dcc.Graph(id='graph1', figure=fig1),
                        html.Div(f'a0 = {a0:.3e} μHz', id='a0-output'),
                        dcc.Slider(
                            id='a0',
                            min=10,
                            max=1000,
                            value=2e2,
                            step=0.1,
                            # marks={
                            #     0: '1',
                            #     1: '10',
                            #     2: '100',
                            #     3: '1000',
                            # },
                            updatemode='drag',
                        ),
                        html.Div(f'a1 = {a1:.3e} μHz', id='a1-output'),
                        dcc.Slider(
                            id='a1',
                            min=10,
                            max=1000,
                            value=1e2,
                            step=0.01,
                            # marks={
                            #     2: '100',
                            #     3: '1000',
                            #     4: '10,000',
                            # },
                            updatemode='drag',
                        ),
                        html.Div(f'a2 = {a2:.3e} μHz', id='a2-output'),
                        dcc.Slider(
                            id='a2',
                            min=1e-2,
                            max=1e0,
                            value=1e-1,
                            step=1e-4,
                            # marks={
                            #     0.5: '0.5',
                            #     1.0: '1.0',
                            #     1.5: '1.5',
                            # },
                            updatemode='drag',
                        ),
                        html.Div(f'a3 = {a3:.3e} μHz', id='a3-output'),
                        dcc.Slider(
                            id='a3',
                            min=-1e-3,
                            max=1e-3,
                            value=a3,
                            step=1e-6,
                            # marks={
                            #     -3: '0.001',
                            #     -2: '0.01',
                            #     -1: '0.1',
                            # },
                            updatemode='drag',
                        ),
                        html.Div(f'a4 = {a4:.3e} μHz', id='a4-output'),
                        dcc.Slider(
                            id='a4',
                            min=-1e-4,
                            max=1e-4,
                            value=a4,
                            step=1e-6,
                            # marks={
                            #     -3: '0.001',
                            #     -2: '0.01',
                            #     -1: '0.1',
                            # },
                            updatemode='drag',
                        ),
                    ],
                    className='six columns',
                ),
                html.Div([
                        dcc.Graph(id='graph2', figure=fig2),
                        html.Div(f'b0 = {b0:.3e}', id='log_b0-output'),
                        dcc.Slider(
                            id='log_b0',
                            min=-3,
                            max=-1,
                            value=np.log10(b0),
                            step=0.001,
                            marks={
                                -3: '0.001',
                                -2: '0.01',
                                -1: '0.1',
                            },
                            updatemode='drag',
                        ),
                        html.Div(f'b1 = {b1:.3e}', id='log_b1-output'),
                        dcc.Slider(
                            id='log_b1',
                            min=-8,
                            max=-6,
                            value=np.log10(b1),
                            step=0.001,
                            marks={
                                -8: '1e-8',
                                -7: '1e-7',
                                -6: '1e-6',
                            },
                            updatemode='drag',
                        ),    
                        html.Div(f'tau = {tau:.3e}', id='log_tau-output'),
                        dcc.Slider(
                            id='log_tau',
                            min=-4,
                            max=-2,
                            value=np.log10(tau),
                            step=0.001,
                            marks={
                                -4: '0.0001',
                                -3: '0.001',
                                -2: '0.01',
                            },
                            updatemode='drag',
                        ),    
                        html.Div(f'phi = {phi:.3e}', id='phi-output'),
                        dcc.Slider(
                            id='phi',
                            min=-np.pi,
                            max=np.pi,
                            value=phi,
                            step=0.01,
                            marks={
                                -np.pi: '-π',
                                0: '0',
                                np.pi: 'π',
                            },
                            updatemode='drag',
                        ),    
                    ],
                    className='six columns',
                ),
            ],
            className='row'
        ),
        html.Div(f'n_modes = {n_modes}', id='n_modes-output'),
        dcc.Slider(
            id='n_modes',
            min=5,
            max=40,
            value=n_modes,
            step=1,
            marks={
                10: '10',
                20: '20',
                30: '30',
                40: '40',
            },
            updatemode='drag',
        ),
    ]
)

@app.callback(
    Output('graph1', 'figure'),
    Output('graph2', 'figure'),
    Output('n_modes-output', 'children'),
    Output('a0-output', 'children'),
    Output('a1-output', 'children'),
    Output('a2-output', 'children'),
    Output('a3-output', 'children'),
    Output('a4-output', 'children'),
    Output('log_b0-output', 'children'),
    Output('log_b1-output', 'children'),
    Output('log_tau-output', 'children'),
    Output('phi-output', 'children'),
    Input('n_modes', 'value'),
    Input('a0', 'value'),
    Input('a1', 'value'),
    Input('a2', 'value'),
    Input('a3', 'value'),
    Input('a4', 'value'),
    Input('log_b0', 'value'),
    Input('log_b1', 'value'),
    Input('log_tau', 'value'),
    Input('phi', 'value'),
)
def update(n_modes, a0, a1, a2, a3, a4, log_b0, log_b1, log_tau, phi):
    b0 = 10**log_b0
    b1 = 10**log_b1
    tau = 10**log_tau

    fig0, fig1 = make_figures(n_modes, a0, a1, a2, a3, a4, b0, b1, tau, phi)
    s0 = f'n_modes = {n_modes}'
    s1 = f'a0 = {a0:.3e} μHz'
    s2 = f'a1 = {a1:.3e} μHz'
    s3 = f'a2 = {a2:.3e} μHz'
    s4 = f'a3 = {a3:.3e} μHz'
    s5 = f'a4 = {a4:.3e} μHz'
    s6 = f'b0 = {b0:.3e} μHz'
    s7 = f'b1 = {b1:.3e} Ms2'
    s8 = f'tau = {tau:.3e} Ms'
    s9 = f'phi = {phi:.2f}'
    return fig0, fig1, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9



if __name__ == '__main__':
    app.run_server(debug=True)
