import dash_player
from dash import html, dcc 
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app 
from globals import *


video_drop_down = dcc.Dropdown(
    id="dd-my-videos",
    options=[{"label": title, "value": url} for title, url in VIDEO_PATH.items()],
    value=None,
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
    style={"width": "130px"},
    placeholder="Select the stance"
)

body_rot_type = dcc.Dropdown(
    id="cut-body-rotation-type",
    options=[{"label": stance.title(), "value":stance} for stance in ROTATION_TYPE],
    value=ROTATION_TYPE[0],
    style={"width": "130px"},
    placeholder="Body Rotation Type",
)

body_num_rot = dcc.Dropdown(
    id="cut-body-rotation-number",
    options=[{"label":quantity.title(), "value":idx} for idx, quantity in enumerate(NUMBER_ROTATION)],
    value=0,
    placeholder="# Body Rot",
    style={"width": "130px"}
)

shov_it_type = dcc.Dropdown(
    id="cut-shov-it-type",
    options=[{"label": stance.title(), "value":stance} for stance in ROTATION_TYPE],
    value=ROTATION_TYPE[0],
    style={"width": "130px"},
    placeholder="Shov-it Type",
) 

shov_it_num = dcc.Dropdown(
    id="cut-shov-it-number",
    options=[{"label":quantity.title(), "value":idx} for idx, quantity in enumerate(NUMBER_ROTATION)],
    value=0,
    placeholder="# Shov-it",
    style={"width": "130px"}
)
flip_type = dcc.Dropdown(
    id="cut-flip-type",
    options=[{"label": stance.title(), "value":stance} for stance in FLIP_TYPE],
    value=FLIP_TYPE[0],
    style={"width": "130px"},
    placeholder="Flip Type",
) 

flip_num = dcc.Dropdown(
    id="cut-flip-number",
    options=[{"label": "None", "value": 0}, {"label": "Once", "value": 1}, {"label": "Twice", "value": 2}],
    value=0,
    placeholder="# Flip",
    style={"width": "130px"}
)

start_btn = dbc.Button(
    "Start: 0", 
    color="secondary",
    id="btn-set-start",
    size="lg",
    style={"width": "100%", "display": "inline-block"}
)

end_btn = dbc.Button(
    "End: 10", 
    color="secondary",
    id="btn-set-end",
    size="lg",
    style={"width": "100%", "display": "inline-block"}
)

make_cut_btn = dbc.Button(
    "Make Cut", 
    color="success",
    id="btn-create-cut",
    size="lg",
    style={"width": "100%", "display": "inline-block"}
)

cut_panel_btn = dbc.Button(
    "Cut Panel",
    color="info",
    id="btn-collapse",
    size="lg",
    style={"width": "100%", "display": "inline-block"}
)

delete_cut_btn = dbc.Button(
    "Delete Cut", 
    color="danger",
    id="delete-cut",
    size="lg",
    style={"width": "100%", "display": "inline-block"}
)

cut_dropdown = dcc.Dropdown(
    id="dd-cut",
    style={"margin-top": "1px", "height": "47px"},
    placeholder="Select cut"
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
                                    dbc.Col([dbc.Label("# BRT"), body_num_rot]),
            
                                ],
                                style={"padding": "15px"}
                            ),
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("SIT"), shov_it_type]),
                                    dbc.Col([dbc.Label("# SI"), shov_it_num]),
                                    dbc.Col([dbc.Label("Flip Type"), flip_type]),
                                    dbc.Col([dbc.Label("# Flip"), flip_num]),

                                ],
                                style={"padding": "15px"}
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(start_btn),
                                    dbc.Col(end_btn),
                                    dbc.Col(make_cut_btn),
                                ]
                            )
                        ]
                    )
                ],
                color="dark",
                outline=True
            ),
            is_open=True,
            id="collapse",
            style={"margin-top": "25px", "margin-bottom": "25px"}
        ),
        dbc.Row(
            [
                dbc.Col(cut_panel_btn, md=2),
                dbc.Col(delete_cut_btn, md=2),
                dbc.Col(cut_dropdown, md=8)
            ],
            style={"margin-top": "25px"}
        ),
        dbc.Row(
            [
                dash_player.DashPlayer(
                    id="video-player",
                    width="100%",
                    height="600px",
                    intervalSecondsLoaded=200,
                    style={"margint-top": "20px"}
                )
            ],
            style={"margin-top": "25px"}
        )
    ]
)



@app.callback(
    Output("collapse", "is_open"),
    [Input("btn-collapse", "n_clicks"), State("collapse", "is_open")]
)
def toggle_collapse(n_clicks, is_open):
    if n_clicks is not None:
        return not is_open


@app.callback(
    Output("video-player", "url"),
    [Input("dd-my-videos", "value")]
)
def select_video(value):
    return value