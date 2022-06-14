import json

import dash_player
from dash import html, dcc
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app 
from globals import *


video_drop_down = dcc.Dropdown(
    id="dd-my-videos",
    options=[{"label": title, "value": url} for title, url in VIDEO_PATH.items()],
    value=list(VIDEO_PATH.values())[0],
    style={"margin-top": "10px"},
    placeholder="Select Video"
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

cut_name = dbc.Input(
    id="cut-name",
    placeholder="Cut Name",
    type="text"
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
                    height="500px",
                    intervalSecondsLoaded=200,
                    controls=True,
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
def toggle_collapse(n_clicks: int, is_open: bool) -> bool:
    if n_clicks is not None:
        return not is_open


@app.callback(
    Output("video-player", "url"),
    [Input("dd-my-videos", "value")]
)
def select_video(value: str) -> str:
    return value

@app.callback(
    Output("btn-set-start", "children"),
    [Input("btn-set-start", "n_clicks"), State("video-player", "currentTime")]
)
def update_start(n_clicks: int, start_time: float) -> str:
    value = 0 if start_time is None else start_time
    return f"Start: {value:.1f} "

@app.callback(
    Output("btn-set-end", "children"),
    [Input("btn-set-end", "n_clicks"), State("video-player", "currentTime")]
)
def update_end(n_clicks: int, end_time: float) -> str:
    value = 10 if end_time is None else end_time
    return f"End: {value:.1f} "

@app.callback(
    Output("dd-cut", "options"),
    [
        Input("btn-create-cut", "n_clicks"),
        Input("delete-cut", "n_clicks"),
        Input("video-player", "url")
    ],
    [
        State("btn-set-start", "children"), 
        State("btn-set-end", "children"),
        State("cut-name", "value"), 
        State("cut-landed", "value"),
        State("cut-stance", "value"),
        State("cut-body-rotation-type", "value"),
        State("cut-body-rotation-number", "value"),
        State("cut-shov-it-type", "value"),
        State("cut-shov-it-number", "value"),
        State("cut-flip-type", "value"),
        State("cut-flip-number", "value"),
        State("dd-cut", "value")
    ]
)
def make_cut(
    create_cut: int,
    delete_cut: int,
    video_url: str,
    start_time: str,
    end_time: str,
    cut_file_name: str,
    landed: bool,
    stance: str,
    body_rotation_type: str,
    body_rotation_number: int,
    shov_it_type: str,
    shov_it_number: int,
    flip_type: str,
    flip_number: int,
    current_cut: str,
) -> list[dict]:
    trigg = dash.callback_context.triggered[0]["prop_id"]

    data = utils.get_cuts_data()
    
    trick_info = {
        "landed": landed,
        "stance": stance,
        "body_rotation_type": body_rotation_type,
        "body_rotation_number": body_rotation_number,
        "shov_it_type": shov_it_type,
        "shov_it_number": shov_it_number,
        "flip_type": flip_type,
        "flip_number": flip_number
    }
    if trigg=="btn-create-cut.n_clicks":
        start = float(start_time.split(":")[-1]) 
        end = float(end_time.split(":")[-1])
        data = utils.update_cuts(data.copy(), video_url, cut_file_name, start, end, trick_info)
    elif trigg=="delete-cut.n_clicks":
        if current_cut is not None:
            data = utils.delete_cuts(data.copy(), video_url, current_cut)

    with open("batb11/tricks_cut.json", 'w') as f:
        json.dump(data, f)

    return [{"label": key, "value":key } for key in data[video_url].keys()] if video_url in data else []


@app.callback(
    Output("video-player", "seekTo"),
    [Input("dd-cut", "value"), Input("video-player", "currentTime")],
    [State("video-player", "url")]
)
def play_cut(cut_name: str, current_time: float, video_url: str) -> float:
    if cut_name is None or cut_name=="No results":
        return
    data = utils.get_cuts_data()
    if current_time < data[video_url][cut_name]["interval"][0]:
        return data[video_url][cut_name]["interval"][0]
    if current_time > data[video_url][cut_name]["interval"][-1]:
        return data[video_url][cut_name]["interval"][0]
