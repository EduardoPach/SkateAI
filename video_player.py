import dash_player
from dash import html, dcc 
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app 
from globals import *


video_drop_down = dcc.Dropdown(
    id="dd-my_videos",
    options=[{"label": f"Video_{idx}", "value": j} for idx, j in enumerate(VIDEO_PATH)],
    value=VIDEO_PATH[0],
    style={"margin-top": "10px"},
    placeholder="Select Video"
)

cut_name = dbc.Input(
    id="input-cut-name",
    placeholder="Cut name",
    type="text"
)

landed_status_radio = dcc.RadioItems(
    id="cut-landed",
    options=[{"label": "True", "value": True}, {"label": "False", "value": False}],
    value=True,
    labelStyle={"display": "inline-block", "margin": "10px"}
)

stance_dropdown = dcc.Dropdown(
    id="cut-stance",
    options=[{"label": stance.title(), "value":stance} for stance in STANCES],
    value=STANCES[0],
    style={"width": "110px"},
    placeholder="Select the stance"
)

body_rot_type = dcc.Dropdown(
    id="cut-body-rotation-type",
    options=[{"label": stance.title(), "value":stance} for stance in BODY_ROTATION_TYPE],
    value=BODY_ROTATION_TYPE[0],
    style={"width": "130px"},
    placeholder="Body Rotation Type",
)



controls = dbc.Col(
    [
        video_drop_down,
        dbc.Collapse(
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            cut_name,
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("Status"), landed_status_radio]),
                                    dbc.Col([dbc.Label("Stance"), stance_dropdown]),
                                    dbc.Col([dbc.Label("BRT"), body_rot_type]),
                                    dbc.Col([dbc.Label("# BRT")]),
                                    dbc.Col([dbc.Label("SIT")]),
                                    dbc.Col([dbc.Label("# SI")]),
                                ]
                            )
                        ]
                    )
                ]
            ),
            is_open=True

        )
    ]
)
