import plotly.graph_objects as go
import numpy as np
from shapely import Polygon
import os
from pathlib import Path

from src.functions import *

ROUTE_PATH = Path(".\\Google Earth Files\\dfw_small") if os.name != 'posix' else Path(
    "./Google Earth Files/dfw_small")
airspace: dict = prepareSpace(ROUTE_PATH)

print(airspace.keys())
print(airspace['points'])

airspace_poly = np.asarray(airspace['airspace'].exterior.coords.xy)


fig = go.Figure()

center = np.mean(airspace_poly, axis=1)

fig.add_trace(go.Scattermapbox(
        fill='toself',
        fillcolor='rgba(0,135,0,0.2)',
        opacity= 0,
        lon = airspace_poly[0,:],
        lat=airspace_poly[1,:],
        marker = { 'size': 0, 'color': "green", 'opacity':0 },
        showlegend=False
    ))

for name, nfz in airspace['nfzs'].items():
    nfz_poly = np.asarray(nfz.exterior.coords.xy)
    
    fig.add_trace(go.Scattermapbox(
        fill='toself',
        fillcolor='rgba(135,0,0,0.5)',
        opacity= 0,
        lon = nfz_poly[0,:],
        lat=nfz_poly[1,:],
        marker = { 'size': 0, 'color': "red", 'opacity':0 },
        showlegend=False
    ))
    
for i, vert in enumerate(airspace['points']):
    coords = np.asarray(vert.coords.xy)
    
    fig.add_trace(go.Scattermapbox(
        mode = "markers+text",
        marker=go.scattermapbox.Marker(
        size=14
        ),
        name=f'Vertiport {i}',
            lon = coords[0],
            lat = coords[1],
            text = [str(f'Vertiport {i}')],
            textfont={'color':'royalblue', 'family':'Arial','size':16},
            textposition="bottom right",
            marker_color = 'royalblue',
            # texttemplate='(%{lat}, %{lon})%{text}'
        ))





fig.update_layout(
    hovermode='closest',
    mapbox = {
        'style': "stamen-terrain",
        'center': {'lon': center[0], 'lat': center[1] },
        'zoom': 11},
    showlegend = False)
fig.show()