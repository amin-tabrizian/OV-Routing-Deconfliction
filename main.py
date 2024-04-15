"""This program generates routes for UAM aircraft as well as operational volumes
"""


# Imports
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

from src.functions import *
from src.GoogleEarthOut import EarthKML
from src.OVGeneration import (dryvrConcurr, dryvrMain, spheroidConcurr,
                              spheroidMain, OV, Segment, spheroidNewConcurr, spheroidNewMain)
from src.RouteBuilding import RRTStar, AStar
from src.Airspace import Airspace
from src.Simulation import Simulator

# Dunders
__author__ = "Ellis Thompson"

# Constants
ROUTE_PATH = Path(".\\Google Earth Files\\Dallas_map") if os.name != 'posix' else Path(
    "./Google Earth Files/Dallas_map")
OUT_PATH = Path('.\\out\\') if os.name != 'posix' else Path('./out/')
COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255),
           (255, 255, 0), (0, 255, 255), (255, 0, 255), (180, 40, 200)]


def main(points: List[int] = None, show_plots: bool = True, max_workers: int = 20, ov_mode='dryvr', n_ac: int = 100, method: str = 'AStar', speed_bounds: List[int] = [23, 28]):
    start_time = perf_counter()
    # Import airspace from file
    airspace: dict = prepareSpace(ROUTE_PATH)
    ovs = Airspace(airspace['airspace'])
    # ovs.load_airspace(['Airspace OVs/8-2 20 Operational Volumes 1.npy',
    #                    'Airspace OVs/2-0 80 Operational Volumes 2.npy',
    #                    'Airspace OVs/4-8 120 Operational Volumes 3.npy',
    #                    'Airspace OVs/2-5 60 Operational Volumes 4.npy',
    #                    'Airspace OVs/5-0 100 Operational Volumes 5.npy',
    #                    'Airspace OVs/4-7 100 Operational Volumes 6.npy',
    #                    'Airspace OVs/5-8 210 Operational Volumes 7.npy',
    #                    'Airspace OVs/0-5 230 Operational Volumes 8.npy',
    #                    'Airspace OVs/8-5 160 Operational Volumes 9.npy',
    #                    'Airspace OVs/2-3 120 Operational Volumes 10.npy'
    #                    ], 
    #                   [248, 188, 22, 157, 117,7,109,0,202, 166, 41,83,223,94,92, 52, 79,18,64,125,131,289,163,13,300, 1, 14, 38,97,188, 249])
    # ovs.load_airspace(['Airspace OVs/shperoid Operational Volumes.npy'], [0])

    # ovs_np = np.load('Airspace OVs\Operational Volumes.npy', allow_pickle=True)
    # ovs = [((i*60)-180, Polygon(ov)) for i, ov in enumerate(ovs_np)]

    # Pick some random points if none are provided
    if not points or len(points) < 2:
        points: np.ndarray = np.random.choice(
            len(airspace["points"]), 2, replace=False)

    # Prepare output folders
    date_path: Path = Path.joinpath(
        OUT_PATH, f'{datetime.now().strftime("%d-%m-%Y %H-%M-%S")}-Route ({points[0]}-{points[1]})')
    os.mkdir(date_path)
    path: Path = Path.joinpath(date_path, 'routes.kml')

    s = perf_counter()

    # Generate route
    if method == 'AStar':
        route = genAStar(points[0], points[1], airspace,
                         ovs, speed_bounds, show_plots=show_plots)

        route = np.asarray([[point.x, point.y] for point in route.route])
        optimised = []
        print(route)
    elif method == 'RRT':
        print('USING RRT')
        route: RRTStar = genRRT(
            points[0], points[1], airspace, show_plots=show_plots, M=200)
    else:
        raise ValueError(
            f'{method}, is not a valid route generation technique')
        
    np.save(Path.joinpath(date_path,'Route.npy'), route)
    # return

    # Prepare KML output
    # GEarth: EarthKML = EarthKML(str(path))
    # GEarth.addLine(
    #     routeToLL(airspace, rrt.optimised_route),
    #     f'Path {points[0]}-{points[1]}',
    #     colour=(255, 255, 255),
    # )
    # GEarth.save()

    # Save the route
    # unoptimised, optimised = rrt.saveRoutes(
    #     f'{str(date_path)}/({points[0]}-{points[1]})', airspace)
    # Run the simulation
    traces, operational_volumes = runSim(
        route, airspace=airspace, speed_bounds=speed_bounds, show_plots=True, n_ac=n_ac)
    
    print(perf_counter()-s)
    

    # Save sim traces
    np.save(Path.joinpath(date_path, 'Sim Traces.npy'), traces)

    # Generatr OVs for the supplied route
    # operational_volumes = genOVs(
    #     traces, airspace, optimised, max_workers=max_workers, ov_mode=ov_mode, show_plots=show_plots)

    if not ov_mode == 'spheroid':
        np.save(Path.joinpath(date_path, 'Operational Volumes.npy'),
            operational_volumes)
    else:
        contract = []
        for ov in operational_volumes:
            contract.append(ov.as_numpy())
        np.save(Path.joinpath(date_path, 'shperoid Operational Volumes.npy'),
            contract)

    if show_plots:
        drawArrayOVs(airspace, operational_volumes, optimised, traces)

    # System completion time:
    print('========== COMPLETION TIME ==========')
    print(
        f'System took {round(perf_counter()-start_time, 2)} seconds to complete')
    print('=====================================')


def genRRT(start: int, goal: int, airspace: dict, n_iter=5000, step_size=500, end_dist=1000, M=200, show_plots: bool = False) -> RRTStar:
    """Wrapper code for generating a route through RRT*

    Args:
        start (int): The index of the starting node.
        goal (int): The index of the goal node.
        airspace (dict): A dictionary describing the airspace features.
        n_iter (int, optional): The maximum number of iterations to run the path finding algorithm for. Defaults to 5000.
        step_size (int, optional): The maximum distance between nodes when generating the RRT graph. Defaults to 500.
        end_dist (int, optional): The maximum distance from the end node for a path to be connected directly to it. Defaults to 500.
        M (int, optional): The maximum number of nodes avaliable in the graph. Defaults to 200.
        show_plots (bool, optional): If the plots should be drawn. Defaults to True.

    Returns:
        RRTStar: Returns the RRT object
    """

    # Prepare local airspace
    # print(airspace)
    converted_airspace: dict = airspaceToXY(airspace)

    # Initilise RRT class
    rrt: RRTStar = RRTStar(converted_airspace['points'][start],
                           converted_airspace['points'][goal], converted_airspace, n_iter=n_iter, step_size=step_size, end_dist=end_dist, M=M, orig=airspace)

    # Path finding
    rrt.findPath()

    # Route display
    if show_plots:
        displayRouteAirspace(airspace, routeToLL(airspace, rrt.route), routeToLL(
            airspace, rrt.optimised_route), airspace["points"][start], airspace["points"][goal], None)

    return rrt


def genAStar(start: int, goal: int, airspace: dict, ovs, speed_bounds, n_iter=50000, dist=2000, show_plots=True):
    # Initilise RRT class
    offset = np.random.poisson(lam=5)*20
    # offset = -27
    
    found = False
    
    while offset <= 300 and not found:
        astar: AStar = AStar(
        airspace['airspace'], airspace['nfzs'].values(), offset = offset)
        print('OFFSET = :',astar.offset)
        print(start,goal)

        start_pos = (airspace['points'][start].x, airspace['points'][start].y)
        goal_pos = (airspace['points'][goal].x, airspace['points'][goal].y)

        path = astar.find_path(
        start_pos, goal_pos, ovs, n_iter=n_iter, dist=dist, break_on_found=False)
        
        if path is None:
            offset += 30
        else:
            found = True

    # Route display
    if show_plots:
        displayRouteAirspace(
            airspace, path, [], airspace["points"][start], airspace["points"][goal], ovs)

    return astar


def runSim(route: np.ndarray, airspace: dict, speed_bounds, max_itr: int = 100, n_ac: int = 100, th: int = 60, show_plots: bool = False):
    route[:, [0, 1]] = route[:, [1, 0]]
    state_storer = None
    ovs = []

    # Initilise simulator
    sim: Simulator = Simulator(speed_bounds=speed_bounds, airspace=airspace)

    for i in range(max_itr):
        print(f'====== Running iteration {i} ======')

        sim.reset()

        if i == 0:
            sim.initilise(route, n=n_ac, sphere=None)

        complete: bool = sim.run(th, interval=1)

        starting_waypoints: dict = sim.byActiveWaypoint(t=-1)
        sim.resetSim()
        sim.initiliseFromWpts(starting_waypoints, n=n_ac, offset=-1)

        run_states = np.array(list(sim.states.values()))

        if state_storer is None:
            state_storer = np.array([run_states])
        else:
            try:
                state_storer = np.append(
                    state_storer, [run_states], axis=0)
            except:
                print(
                    np.array([np.array(list(sim.states.values()))]).shape, state_storer.shape)

        run_states[:, :, [1, 2]] = run_states[:, :, [2, 1]]

        print(run_states.shape)

        ov = spheroidNewMain(run_states)

        ov_poly = Polygon(ov.shape)

        # print(airspace['nfzs'].items())

        print(any(ov_poly.intersects(nfz)
              for nfz in airspace['nfzs'].values()))

        ovs.append(ov)

        if complete:
            break

    if show_plots:
        plt.show()

    return state_storer, np.asarray(ovs, dtype=object)


def genOVs(traces: np.ndarray, airspace: dict, route, n_traces: int = 20, max_workers: int = 20, show_plots: bool = False, ov_mode='dryvr'):
    print('*'*50)
    print('{:^50}'.format(f'Attempting to generate {traces.shape[0]} OVs'))
    print('*'*50)

    operational_volumes = np.empty(
        (traces.shape[0]), dtype=object)

    args = ((traces[i], n_traces) for i in range(len(traces)))

    if ov_mode == 'dryvr':
        method = dryvrConcurr
    elif ov_mode == 'spheroid':
        method = spheroidConcurr
    else:
        raise ValueError(f'"{ov_mode}" is not a valid OV generation mode')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, result in enumerate(executor.map(method, args)):
            operational_volumes[i] = result

    if not show_plots:
        return operational_volumes

    if ov_mode == 'dryvr':
        drawDryvrOVs(airspace, operational_volumes, route, traces)
    elif ov_mode == 'spheroid':
        drawArrayOVs(airspace, operational_volumes, route, traces)
    else:
        raise ValueError(f'"{ov_mode}" is not a valid OV generation mode')

    return operational_volumes


if __name__ == "__main__":
    main(list(np.random.choice(list(range(2)),size = 2, replace=False)), show_plots=True,
         max_workers=20, method='AStar', ov_mode='spheroid', n_ac=1, speed_bounds=[17,21])
