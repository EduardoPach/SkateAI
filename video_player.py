import dash_player
from dash import html, dcc 
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app 
from globals import *

controls = dbc.Col(
    [
        dcc.Dropdown(
            id="dd-my_videos",
            options=[{"label": f"Video_{idx}", "value": j} for idx, j in enumerate(VIDEO_PATH)],
            value=VIDEO_PATH[0],
            style={"margin-top": "10px"},
            placeholder="Select Video"
        ),
        dbc.Collapse(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            dbc.Input(
                                id="input-cut-name",
                                placeholder="Cut name",
                                type="text"
                            )
                        ]
                    )
                ]
            ),
            is_open=True

        )
    ]
)
