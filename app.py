import os
import sys
import logging
import tempfile

import dash
from dash import html
from dash import dcc
from dash.long_callback import DiskcacheLongCallbackManager 
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import diskcache

import process

from process import checkbox

from cjwutils.misc.simrpathutils import path_to_linux

logging.basicConfig(filename='phenixstitching.log',
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %I:%M:%S %p',
                    level=logging.INFO)

logging.info("Started app")


def main(port=10000):
    cache = diskcache.Cache("./cache")
    lcm = DiskcacheLongCallbackManager(cache)
    app = dash.Dash(__name__,
                    long_callback_manager=lcm,
                    external_stylesheets=[dbc.themes.BOOTSTRAP])

    proj_list = ['None', 'MAX', 'MEAN', 'SUM']

    proj_choices = dcc.Dropdown(proj_list, 'MAX', id='projection',
                                style={'width': '10em'})
    
    interval = dcc.Interval(id='interval', interval=500*1000, n_intervals=0)
    app.layout = html.Div(
        [   html.H5("Enter the location of the Images folder"),
            html.Div(dcc.Input(id='input-image-file', type='text', size='128')),
            html.P(""),
            html.H5("Folder to save the result"),
            html.Div(dcc.Input(id='save-image-file', type='text', size='28')),

            html.P("", style={"height":25}),
            html.H5("Select a projection"),
            proj_choices,
            html.P("", style={"height":25}),
            html.H5("Options"),
            dcc.Checklist(options=checkbox['options'],
                          value=checkbox['values'],
                          id='options',
                          labelStyle={'display':'block'},
                          style={'width':200}),

            html.P("", style={"height":25}),
            html.Button('Submit', id='submit-file', n_clicks=0),
            html.Div(id='button-info',
                     children='Click submit to convert to tif files'),
            html.Div([
            dbc.Progress(id='pbar', value=0, label="Progress",
                         style={'height': '25px', 'width': '40%'},
            ),
            interval,
            #dcc.Interval(id='interval', interval=500*1000, n_intervals=0),
            ]),
    ], style={'margin-left': '15px', 'margin-top':'15px', 'width':'75%'})

    @app.long_callback(
       output=Output('button-info', 'children'),
       inputs=( Input('submit-file', 'n_clicks'),
                State('input-image-file', 'value'),
                State('save-image-file', 'value'),
                State('projection', 'value'),
                State('options', 'value')),
       prevent_initial_call=True
    )
    def start_process(n_clicks, value, saveto, projection, options):
        if n_clicks < 1:
            return "Enter image file and press submit"
        npath = path_to_linux(value)
        npath = npath.strip('" ')
        spath = os.path.join(os.path.split(npath)[0], saveto)

        if not os.path.exists(npath):
            return value + " path does not exist"
            
        if not os.path.exists(spath):
            try:
                os.makedirs(spath)
            except:
                return "can't save to " + saveto
        else:
            if not os.path.isdir(spath):
                return saveto + " exists but isn't a directory"

        print(options)
        if os.path.exists(npath):
            logging.info("Starting process")
            orig_err = sys.stderr   
            serr = open("progress.txt", "w")
            #with tempfile.NamedTemporaryFile(".") as serr:
            sys.stderr = serr 
            process.convert(npath, spath, projection, options) 
            #serr.close()
            sys.stderr = orig_err
            return f"Converted files are in {value}/{saveto}"

        else:
            return value + " path does not exist"
        return "huh"
    
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

    app.run_server(host='0.0.0.0', port=port, debug=True)

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 10000

    main(port=port)