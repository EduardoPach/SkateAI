from dash import html, dcc 
import dash_bootstrap_components as dbc

from app import app
from video_player import controls

app.layout = dbc.Container(
    children=[
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Slider(
                            id="slider-playback-rate",
                            min=0.25,
                            max=5,
                            step=None,
                            marks={i: str(i) + "x" for i in [0.25, 1, 2, 5]},
                            value=1
                        ),
                        controls
                    ],
                )
            ]
        )
    ],
    fluid=True,
    style={"padding": "60px 60px"}

)


if __name__ == "__main__":
    app.run_server(debug=True)