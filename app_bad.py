import os
import sys
import logging
import multiprocessing as mp

import dash
from dash import html
from dash import dcc
from plotly import express as px
from dash.long_callback import DiskcacheLongCallbackManager 
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import diskcache

#import phenix
import process

from cjwutils.misc.simrpathutils import path_to_linux

logging.basicConfig(filename='phenixstitching.log',
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %I:%M:%S %p',
                    level=logging.INFO)

logging.info("Started app")


def main():
    cache = diskcache.Cache("./cache")
    lcm = DiskcacheLongCallbackManager(cache)
    app = dash.Dash(__name__,
                    long_callback_manager=lcm,
                    external_stylesheets=[dbc.themes.BOOTSTRAP])

    proj_list = ['None', 'MAX', 'SUM']

    proj_choices = dcc.Dropdown(proj_list, 'MAX', id='projection',
                                style={'width': '10em'})
    app.layout = html.Div(
        [   html.H5("Enter the location of the Images folder"),
            html.Div(dcc.Input(id='input-image-file', type='text', size='128')),
            html.P(""),
            html.H5("Folder to save the result"),
            html.Div(dcc.Input(id='save-image-file', type='text', size='28')),
            html.P(""),
            html.H5("Select a projection"),
            proj_choices,
            html.P(""),
            html.Button('Submit', id='submit-file', n_clicks=0),
            html.Div(id='button-info',
                     children='Click submit to convert to tif files'),
            html.Div([
            dbc.Progress(id='pbar', value=0, label="Progress",
                         style={'height': '25px', 'width': '40%'},
            ),
            dcc.Interval(id='interval', interval=1000, n_intervals=0),
            ]),
    ], style={'margin-left': '15px', 'margin-top':'15px', 'width':'75%'})

    
    @app.callback(
        output=[Output('pbar', 'value'),
                Output('pbar', 'label')],
        inputs=Input('interval', 'n_intervals'),
        prevent_initial_call=True)
    def progress_callback(n_intervals):
 
        try:
            with open('progress.txt', 'r') as pf:
                raw = pf.read()
            line = raw.split("\n")[-1]
            percent = float(line.split('%')[0])
        except:
            percent = 0
        finally:
            res = f"{percent:.0f}%"
            return percent, res

    app.run_server(host='0.0.0.0', port=10000, debug=True)

if __name__ == '__main__':
    main()