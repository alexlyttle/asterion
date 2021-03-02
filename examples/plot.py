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

def get_idx(n_modes):
    idx = np.arange(idx_max - np.floor(n_modes/2), idx_max + np.ceil(n_modes/2), dtype=int)
    return idx

def make_figures(n_modes, delta_nu, nu_max, epsilon, alpha, a, b, tau, phi):
    idx = get_idx(n_modes)
    nu = star[nu_cols][idx]
    nu_asy = asy_fit(n[idx], delta_nu, nu_max, epsilon, alpha)
    nu_fit = np.linspace(nu[0], nu[-1], 200)
    dnu = he_glitch(nu_fit, a, b, tau, phi)

    fig0 = go.Figure()
    # fig1 = px.scatter(x=star[nu_cols], y=star[nu_cols]%delta_nu)
    fig0.add_trace(go.Scatter(x=nu, y=nu%delta_nu, mode='markers', name='data'))
    fig0.add_trace(go.Scatter(x=nu_asy, y=nu_asy%delta_nu, mode='lines', name='asy fit'))
    # px.scatter()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=nu, y=nu-nu_asy, mode='markers', name='data'))   
    fig1.add_trace(go.Scatter(x=nu_fit, y=dnu, mode='lines', name='glitch fit'))
    return fig0, fig1

n_modes = 18
delta_nu = star['delta_nu_fit']
nu_max = star['nu_max']
epsilon = 0.3 * np.log10(nu_max) + 0.4
alpha = nu_max**(-0.9)
a = 5e-3
b = 5e-7
tau = nu_max**(-0.9)
phi = 0.

fig1, fig2 = make_figures(n_modes, delta_nu, nu_max, epsilon, alpha, a, b, tau, phi)

# fig1 = go.Figure()
# # fig1 = px.scatter(x=star[nu_cols], y=star[nu_cols]%delta_nu)
# fig1.add_trace(go.Scatter(x=star[nu_cols], y=star[nu_cols]%delta_nu, mode='markers'))
# fig1.add_trace(go.Scatter(x=nu_asy, y=nu_asy%delta_nu, mode='lines'))
# # px.scatter()

# fig2 = go.Figure()
# fig2.add_trace(go.Scatter(x=star[nu_cols], y=star[nu_cols]-nu_asy, mode='markers'))


# df['y'] = model(df['x'], a0, b0)

app.layout = html.Div([
        html.Div([
                html.Div([
                        dcc.Graph(id='graph1', figure=fig1),
                        html.Div(f'delta_nu = {delta_nu:.3f} μHz', id='log_delta_nu-output'),
                        dcc.Slider(
                            id='log_delta_nu',
                            min=0,
                            max=3,
                            value=np.log10(delta_nu),
                            step=0.00001,
                            marks={
                                0: '1',
                                1: '10',
                                2: '100',
                                3: '1000',
                            },
                            updatemode='drag',
                        ),
                        html.Div(f'nu_max = {nu_max:.1f} μHz', id='log_nu_max-output'),
                        dcc.Slider(
                            id='log_nu_max',
                            min=2,
                            max=4,
                            value=np.log10(nu_max),
                            step=0.0001,
                            marks={
                                2: '100',
                                3: '1000',
                                4: '10,000',
                            },
                            updatemode='drag',
                        ),
                        html.Div(f'epsilon = {epsilon:.2f}', id='epsilon-output'),
                        dcc.Slider(
                            id='epsilon',
                            min=0.0,
                            max=2.0,
                            value=epsilon,
                            step=0.001,
                            marks={
                                0.5: '0.5',
                                1.0: '1.0',
                                1.5: '1.5',
                            },
                            updatemode='drag',
                        ),
                        html.Div(f'alpha = {alpha:.3e}', id='log_alpha-output'),
                        dcc.Slider(
                            id='log_alpha',
                            min=-3.5,
                            max=-1,
                            value=np.log10(alpha),
                            step=0.01,
                            marks={
                                -3: '0.001',
                                -2: '0.01',
                                -1: '0.1',
                            },
                            updatemode='drag',
                        ),
                    ],
                    className='six columns',
                ),
                html.Div([
                        dcc.Graph(id='graph2', figure=fig2),
                        html.Div(f'a = {a:.3e}', id='log_a-output'),
                        dcc.Slider(
                            id='log_a',
                            min=-3,
                            max=-1,
                            value=np.log10(a),
                            step=0.001,
                            marks={
                                -3: '0.001',
                                -2: '0.01',
                                -1: '0.1',
                            },
                            updatemode='drag',
                        ),
                        html.Div(f'b = {b:.3e}', id='log_b-output'),
                        dcc.Slider(
                            id='log_b',
                            min=-8,
                            max=-6,
                            value=np.log10(b),
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
    Output('log_delta_nu-output', 'children'),
    Output('log_nu_max-output', 'children'),
    Output('epsilon-output', 'children'),
    Output('log_alpha-output', 'children'),
    Output('log_a-output', 'children'),
    Output('log_b-output', 'children'),
    Output('log_tau-output', 'children'),
    Output('phi-output', 'children'),
    Input('n_modes', 'value'),
    Input('log_delta_nu', 'value'),
    Input('log_nu_max', 'value'),
    Input('epsilon', 'value'),
    Input('log_alpha', 'value'),
    Input('log_a', 'value'),
    Input('log_b', 'value'),
    Input('log_tau', 'value'),
    Input('phi', 'value'),
)
def update(n_modes, log_delta_nu, log_nu_max, epsilon, log_alpha, log_a, log_b, log_tau, phi):
    delta_nu = 10**log_delta_nu
    nu_max = 10**log_nu_max
    alpha = 10**log_alpha
    a = 10**log_a
    b = 10**log_b
    tau = 10**log_tau

    fig0, fig1 = make_figures(n_modes, delta_nu, nu_max, epsilon, alpha, a, b, tau, phi)
    s0 = f'n_modes = {n_modes}'
    s1 = f'delta_nu = {delta_nu:.3f} μHz'
    s2 = f'nu_max = {nu_max:.1f} μHz'
    s3 = f'epsilon = {epsilon:.3f}'
    s4 = f'alpha = {alpha:.3e}'
    s5 = f'a = {a:.3e}'
    s6 = f'b = {b:.3e} Ms2'
    s7 = f'tau = {tau:.3e} Ms'
    s8 = f'phi = {phi:.2f}'
    return fig0, fig1, s0, s1, s2, s3, s4, s5, s6, s7, s8



if __name__ == '__main__':
    app.run_server(debug=True)
