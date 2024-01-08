import os
from time import perf_counter, sleep

import bluesky as bs
import geopy
import geopy.distance
import numpy as np
import pyproj
import tqdm
from bluesky.simulation import ScreenIO
from matplotlib.patches import Ellipse
from math import isnan
from shapely import Point, Polygon

geodesic = pyproj.Geod(ellps='WGS84')


class ScreenDummy(ScreenIO):
    """
    Dummy class for the screen. Inherits from ScreenIO to make sure all the
    necessary methods are there. This class is there to reimplement the echo
    method so that console messages are printed.
    """

    def echo(self, text='', flags=0):
        """Just print echo messages"""
        print("BlueSky console:", text)


# initialize bluesky as non-networked simulation node
bs.init(mode='sim', detached=True)

# initialize dummy screen
bs.scr = ScreenDummy()


class Simulator:
    def __init__(self, speed_bounds, airspace):
        self.states = {}
        self.globalsim = []
        self.i = 0
        self.speed_bounds = tuple(speed_bounds)
        self.airspace = airspace

    def initilise(self, route, n=100, sphere=None, start=1):
        self.route = route

        if sphere is None:
            sphere = self.getStartSphere([route[0]], n)
        else:
            n = sphere.shape[0]

        for i in range(n):
            h = geodesic.inv(sphere[i, 1], sphere[i, 0], route[start, 1], route[start, 0])[
                0]+np.random.uniform(-45, 45)
            if isnan(h):
                h = 0
            bs.traf.cre(acid=f'D{self.i}', actype='AMZN',
                        aclat=sphere[i, 0], aclon=sphere[i, 1], acalt=200, achdg=h, acspd=np.random.randint(*self.speed_bounds))

            for j in range(start, len(route)):
                bs.stack.stack(f'ADDWPTMODE D{self.i} FLYOVER')
                bs.stack.stack(f'ADDWPT D{self.i} {route[j,0]},{route[j,1]}')

            # bs.stack.stack(f'ADDWPTMODE D{self.i} FLYOVER')
            bs.stack.stack(f'ADDWPT D{self.i} {route[-1,0]},{route[-1,1]}')
            bs.stack.stack(f'VNAV D{self.i} ON')
            bs.stack.stack(f'LNAV D{self.i} ON')

            self.i += 1

    def initiliseFromWpts(self, wptAircraft: dict, n: int = 100, offset: int = -1):
        c = 0
        distributions = np.zeros(len(wptAircraft))

        for i, wpt in enumerate(wptAircraft):
            distributions[i] = len(wptAircraft[wpt])

        if sum(distributions) == 0:
            return

        # distributions = distributions/np.sum(distributions)

        pbar = tqdm.tqdm(total=len(wptAircraft))
        pbar.set_description(
            f"Generating aircraft from {len(wptAircraft)} start positions")

        max_arg = np.argmax(distributions)

        for i, (wpt, aircraft) in enumerate(wptAircraft.items()):
            to_gen = distributions[i]

            if i == max_arg and np.sum(distributions) < n:
                to_gen += n-np.sum(distributions)
            orig_pos = [self.states[ac][offset][1:3] for ac in aircraft]
            positions = np.array(remove_outliers(
                np.array([self.states[ac][offset][1:3] for ac in aircraft])))
            
            if len(positions) ==0:
                positions = orig_pos
            # try:
            new_starts = self.getStartSphere(
                positions, to_gen)
            self.initilise(self.route, len(new_starts), new_starts, wpt)
            # except Exception:
            #     print('Error generating starting shpere with input positions:')
            #     print(np.asarray(positions).shape)

            c += to_gen
            pbar.update(1)
        pbar.close()

    # def getPointDistribution(self, points, n):

    #     points = np.asarray(points)
    #     min_, max_ = points.min(axis=0), points.max(axis=0)
    #     mu = np.mean(points, axis=0)
    #     std = np.std(points, axis=0)

    #     points = np.clip(np.random.normal(
    #         mu, std, size=(n, points.shape[-1])), min_, max_)

    #     print(points.min(axis=0), points.min(axis=0))

    #     return points

    def getStartSphere(self, points, n, distance=30):

        points = np.asarray(points)

        try:
            start = geopy.Point(np.median(points, axis=0))
        except ValueError:
            print('\n', points)
            print(np.median(points, axis=0))
            raise ValueError()

        d = geopy.distance.distance(meters=distance)

        ls = [d.destination(point=start, bearing=0), d.destination(point=start, bearing=90), d.destination(
            point=start, bearing=180), d.destination(point=start, bearing=270)]
        ls = [[p.latitude, p.longitude] for p in ls]

        sphere = []

        pbar = tqdm.tqdm(total=n)
        pbar.set_description(f"Generating {n} points in sphere")

        while len(sphere) < n:
            new_p = np.random.uniform(
                np.min(ls, axis=0), np.max(ls, axis=0), size=2)

            if any(nfz.contains(Point(new_p[1], new_p[0])) for nfz in self.airspace['nfzs'].values()):
                continue

            if geopy.distance.geodesic(np.median(points, axis=0), new_p).meters <= distance:
                sphere.append(new_p)
                pbar.update(1)

        pbar.close()

        return np.array(sphere)

    def step(self, save=False):
        if save:
            lats = bs.traf.lat[:]
            lons = bs.traf.lon[:]

            for i in range(len(lats)):
                d = geopy.distance.distance(meters=np.random.uniform(0, 5))
                r = d.destination(
                    point=(lats[i], lons[i]), bearing=bs.traf.hdg[i]+np.random.choice([90, -90], 1))
                lats[i] = r.latitude
                lons[i] = r.longitude

            self.globalsim.append([bs.sim.simt, bs.traf.id[:], lats[:],
                                   lons[:], bs.traf.alt[:], bs.traf.hdg[:], bs.traf.tas[:], bs.traf.actwp.lat[:], bs.traf.actwp.lon[:]])
        bs.sim.step()

        self.checkRemove()

    def convertData(self):
        for data in tqdm.tqdm(self.globalsim, desc='Converting Data'):
            sim_t = data[0]
            traf = data[1]
            data = np.array(data[2:])

            for idx, _id in enumerate(traf):
                state = np.array([sim_t, *data[:, idx]])
                if _id in self.states:
                    self.states[_id] = np.append(
                        self.states[_id], [state], axis=0)
                else:
                    self.states[_id] = np.array([state])

    def checkRemove(self, remove=False):
        if not remove:
            for i in np.where(bs.traf.swlnav == False)[0][::-1]:

                pos = -2 if len(self.globalsim) > 1 else 0

                self.globalsim[-1][2][i] = self.globalsim[pos][2][i]
                self.globalsim[-1][3][i] = self.globalsim[pos][3][i]
                self.globalsim[-1][4][i] = self.globalsim[pos][4][i]
                self.globalsim[-1][5][i] = self.globalsim[pos][5][i]
                bs.stack.stack(f'HDG {bs.traf.id[i]} {bs.traf.hdg[i]}')
        else:
            for i in np.where(bs.traf.swlnav == False)[0][::-1]:
                bs.traf.delete(i)

    def run(self, seconds=None, interval=1):
        s = False
        i = 0

        n = float('inf') if seconds is None else seconds / 0.1

        pbar = tqdm.tqdm(total=n)
        pbar.set_description(
            f"Running aircraft simulation to {'completion' if n > 1e50 else n}")

        while (bs.traf.ntraf > 0 or not s) and i < n:
            self.step(bs.sim.simt % interval == 0)
            i += 1

            pbar.update(1)

        pbar.close()
        self.convertData()
        return not any(bs.traf.swlnav)

    def getAtPosition(self, t):
        return np.array([item[t, 1:3] for item in self.states.values()])

    def byActiveWaypoint(self, t=-1):
        data = {}
        datum = self.globalsim[t]

        time = datum[0]
        ids = datum[1]
        datum = np.array(datum[-2:])

        for idx, _id in enumerate(ids):
            if not bs.traf.swlnav[idx]:
                continue

            act_wpt = [datum[0, idx], datum[1, idx]]
            wpt_idx = np.where(
                abs(act_wpt[0]-self.route[:, 0]) <= 1e-10)[0][0]

            if wpt_idx not in data:
                data[wpt_idx] = [_id]
            else:
                data[wpt_idx].append(_id)

        return data

    def reset(self):
        self.states = {}
        self.globalsim = []
        self.i = 0
        bs.core.simtime.setdt(0.1)
        bs.stack.stack('FF')

    def resetSim(self):
        bs.sim.reset()


def remove_outliers(coordinates):
    lat = coordinates[:, 0]
    lon = coordinates[:, 1]

    lat_5th_percentile = np.percentile(lat, 10)
    lon_5th_percentile = np.percentile(lon, 10)

    lat_95th_percentile = np.percentile(lat, 90)
    lon_95th_percentile = np.percentile(lon, 90)

    return [
        coord
        for coord in coordinates
        if lat_5th_percentile <= coord[0] <= lat_95th_percentile
        and lon_5th_percentile <= coord[1] <= lon_95th_percentile
    ]
