from pathlib import Path

from shapely import Point
import simplekml


class EarthKML:
    def __init__(self, path: Path):
        self.path = path

        print(self.path)

        self.kml = simplekml.Kml()

    def addLine(self, path: list, name: str, colour=(0, 0, 0), width=2):
        ls = self.kml.newlinestring(name=name)

        if isinstance(path[0], Point):
            path = [(p.x, p.y) for p in path]

        ls.coords = path
        ls.style.linestyle.width = width
        ls.style.linestyle.color = simplekml.Color.rgb(*colour)

    def save(self, path: Path = None):
        save_path = self.path if path is None else path

        self.kml.save(save_path)
