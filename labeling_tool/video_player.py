import json

import dash_player
from dash import html, dcc
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app 
import const
import utils

dummy_div = html.Div(id="dummy-div", children="")

videos_play_list = dcc.Dropdown(
    id="playlist_url",
    options=[{"label": title, "value": url} for title, url in const.VIDEO_SOURCES.items()],
    value=const.VIDEO_SOURCES[const.DEFAULT_SOURCE],
    placeholder="Select the Source",
    style={"margin-top": "10px"},
)

video_drop_down = dcc.Dropdown(
    id="dd-my-videos",
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
    options=[{"label": stance.title(), "value":stance} for stance in const.STANCES],
    value=const.STANCES[0],
    placeholder="Select the stance"
)

trick_dropdown = dcc.Dropdown(
    id="cut-trick",
    options=[{"label": name.title(), "value": name} for name in const.TRICK_NAMES.keys()],
    value=list(const.TRICK_NAMES.keys())[0],
    placeholder="Select the Trick"
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

cut_collapse = dbc.Collapse(
    dbc.Card(
        [
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col([dbc.Label("Status"), landed_status_radio]),
                            dbc.Col([dbc.Label("Stance"), stance_dropdown]),
                            dbc.Col([dbc.Label("Trick"), trick_dropdown])
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
)

update_cut_btn = dbc.Button(
    "Update cut",
    color="success",
    id="update-btn",
    size="lg",
    style={"width": "100%"}
)

range_slider = dcc.RangeSlider(
    min=0,
    max=1000,
    step=0.1,
    value=[0, 100],
    marks=None,
    tooltip={"placement": "bottom", "always_visible": True},
    id="cut-range-time"
)

edit_collapse = dbc.Collapse(
    dbc.Card(
        [
            dbc.CardBody(
                [
                    dbc.Row(
                        [ 
                            dbc.Col([dbc.Label("Time Interval"), range_slider]),
                            dbc.Col(update_cut_btn)
                        ]
                    )
                ]
            )
        ],
        color="dark",
        outline=True
    ),
    is_open=True,
    id="collapse-edit",
    style={"margin-top": "25px", "margin-bottom": "25px"}
)

cut_panel_btn = dbc.Button(
    "Cut Panel",
    color="info",
    id="btn-collapse",
    size="lg",
    style={"width": "100%", "display": "inline-block"}
)

edit_cut_btn = dbc.Button(
    "Edit Cut",
    color="warning",
    id="edit-btn",
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
        dbc.Col(
            dbc.Row(
                [
                    dbc.Col(videos_play_list), 
                    dbc.Col(video_drop_down),
                ],
                style={"padding": "15px"}
            )
        ),
        cut_collapse,
        edit_collapse,
        dbc.Row(
            [
                dbc.Col(cut_panel_btn, md=2),
                dbc.Col(edit_cut_btn, md=2),
                dbc.Col(delete_cut_btn, md=2),
                dbc.Col(cut_dropdown, md=6)
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
        ),
        dummy_div
    ]
)


@app.callback(
    Output("dd-my-videos", "options"),
    [Input("playlist_url", "value")]
)
def select_source_videos(playlist_url: str) -> dict:
    if not playlist_url:
        return {}
    source_name = {val: key for key, val in const.VIDEO_SOURCES.items()}[playlist_url]
    video_options = const.VIDEOS_PER_SOURCE[source_name]
    return [{"label": title, "value": url} for title, url in video_options.items()]
    

@app.callback(
    Output("collapse", "is_open"),
    [Input("btn-collapse", "n_clicks"), State("collapse", "is_open")]
)
def toggle_collapse(n_clicks: int, is_open: bool) -> bool:
    if n_clicks is not None:
        return not is_open

@app.callback(
    Output("collapse-edit", "is_open"),
    [Input("edit-btn", "n_clicks"), State("collapse-edit", "is_open")]
)
def toggle_collapse_update(n_clicks: int, is_open: bool) -> bool:
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
        State("cut-landed", "value"),
        State("cut-stance", "value"),
        State("cut-trick", "value"),
        State("playlist_url", "value"),
        State("dd-cut", "value")
    ]
)
def make_cut(
    create_cut: int,
    delete_cut: int,
    video_url: str,
    start_time: str,
    end_time: str,
    landed: bool,
    stance: str,
    trick_name: str,
    video_source: str,
    current_cut: str,
) -> list:
    trigg = dash.callback_context.triggered[0]["prop_id"]

    data = utils.get_cuts_data()
    trick_info = const.TRICK_NAMES[trick_name].copy()
    trick_info["landed"] = landed
    trick_info["stance"] = stance
    trick_info["trick_name"] = trick_name
    source = utils.key_from_value(const.VIDEO_SOURCES, video_source)
    
    if trigg=="btn-create-cut.n_clicks":
        start = float(start_time.split(":")[-1]) 
        end = float(end_time.split(":")[-1])
        data = utils.update_cuts(data.copy(), video_url, start, end, trick_info, source)
    elif trigg=="delete-cut.n_clicks":
        if current_cut is not None:
            data = utils.delete_cuts(data.copy(), video_url, current_cut)

    with open(const.TRICKS_JSON_PATH, 'w') as f:
        json.dump(data, f)

    return [{"label": key, "value":key } for key in data[video_url].keys()] if video_url in data else []

@app.callback(
    Output("dummy-div", "children"),
    Input("update-btn", "n_clicks"),
    State('cut-range-time', 'value'),
    State('dd-cut', 'value'),
    State('video-player', 'url')
)
def update_cut_interval(update_cut: int, interval: list, cut_name: str, video_url: str) -> str:
    if cut_name is None or cut_name=="No results":
        return ""
    data = utils.get_cuts_data()
    data[video_url][cut_name]["interval"] = interval

    with open(const.TRICKS_JSON_PATH, 'w') as f:
        json.dump(data, f)
    
    return ""

@app.callback(
    [
        Output("cut-range-time", "min"), 
        Output("cut-range-time", "max"), 
        Output("cut-range-time", "value")
    ],
    [Input("dd-cut", "value")],
    [State("video-player", "url")]
)
def set_cut_interval(cut_name: int, video_url: str) -> list:
    if cut_name is None or cut_name=="No results":
        return [0, 1000, [0, 100]]
    data = utils.get_cuts_data()
    interval = data[video_url][cut_name]["interval"]
    min_ = interval[0]
    max_ = interval[1]

    return [min_ - 2, max_ + 2, [min_, max_]]

@app.callback(
    Output("video-player", "seekTo"),
    [Input("cut-range-time", "value"), Input("video-player", "currentTime")],
    [Input("dd-cut", "value")]
)
def play_cut(interval: list, current_time: float, cut_name: str) -> float:
    if cut_name is None or cut_name=="No results":
        return
    if current_time < interval[0]:
        return interval[0]
    if current_time > interval[-1]:
        return interval[0]
