from __future__ import annotations
import dataclasses
import functools
import time
import numpy as np
import datetime
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import dateutil.parser


def parse_date(date_or_num_days_in_the_future: str | int = 0):
    try:
        # can be an integer representing number of days past or future
        date_value = datetime.datetime.today() + datetime.timedelta(days=int(date_or_num_days_in_the_future))
    except ValueError:
        # or else a string parsable as the date
        date_value = dateutil.parser.parse(date_or_num_days_in_the_future).date()
    return datetime.datetime.combine(date_value, datetime.datetime.min.time())


class CelestialObjectTrace:  # define the objects: the Sun, Earth, Mercury, etc

    def __init__(self, name, positions, velocities):
        self.name = name
        self.positions = np.array(positions, dtype=np.float64)
        self.velocities = np.array(velocities, dtype=np.float64)

    @property
    def current_position(self):
        return self.positions[-1]

    @property
    def current_velocity(self):
        return self.velocities[-1]


class SolarSystem:

    def __init__(self, start_date=0, end_date=1000):
        self.times = np.array([parse_date(start_date)])
        self.planets = []
        print("Querying JPL for planet positions and velocities...", end="")
        for i, nasaid in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):  # The 8 planets
            obj = Horizons(id=nasaid, location="@sun", epochs=Time(self.times[0]).jd).vectors()
            self.planets.append(CelestialObjectTrace(
                nasaid,
                [[np.double(obj[xi]) for xi in ['x', 'y', 'z']]],
                [[np.double(obj[vxi]) for vxi in ['vx', 'vy', 'vz']]],
            ))
        print("Done.")
        print("Calculating planet positions and velocities...", end="")
        self.calculate_up_to(end_date)
        print("Done.")

    def save_to_npy(self, file_name):
        arrays = {"times": self.times, "planet_names": [p.name for p in self.planets]}
        for planet in self.planets:
            arrays[f"{planet.name}_positions"] = planet.positions
            arrays[f"{planet.name}_velocities"] = planet.velocities
        np.savez(file_name, **arrays)

    @classmethod
    def load_from_npy(cls, file_name):
        arrays = np.load(file_name, 'r', allow_pickle=True)
        new_solar_system = object.__new__(cls)
        new_solar_system.times = arrays["times"]
        new_solar_system.planets = [
            CelestialObjectTrace(planet_name, arrays[f"{planet_name}_positions"], arrays[f"{planet_name}_velocities"])
            for planet_name in arrays["planet_names"]
        ]
        return new_solar_system

    def calculate_up_to(self, date_to_calculate_to, dt=1.0):
        new_times = [self.times[-1]]
        new_planet_positions = {
            p: [p.current_position] for p in self.planets
        }
        new_planet_velocities = {
            p: [p.current_velocity] for p in self.planets
        }

        date_to_calculate_to = parse_date(date_to_calculate_to)
        while new_times[-1] < date_to_calculate_to:
            new_times.append(new_times[-1] + datetime.timedelta(days=dt))
            for p in self.planets:
                planet_positions, planet_velocities = new_planet_positions[p], new_planet_velocities[p]
                planet_positions.append(planet_positions[-1] + planet_velocities[-1] * dt)
                acc = -2.959e-4 * planet_positions[-1] / np.sum(planet_positions[-1] ** 2) ** (3. / 2)  # in units of AU/day^2
                planet_velocities.append(planet_velocities[-1] + acc * dt)
        self.times = np.concatenate([self.times, new_times[1:]])
        for p in self.planets:
            p.positions = np.concatenate([p.positions, new_planet_positions[p][1:]])
            p.velocities = np.concatenate([p.velocities, new_planet_velocities[p][1:]])

    # def calculate_up_to_inefficient(self, date_to_calculate_to, dt=1.0):
    #     date_to_calculate_to = parse_date(date_to_calculate_to)
    #     while self.times[-1] < date_to_calculate_to:
    #         self.times = np.concatenate([self.times, [self.times[-1] + datetime.timedelta(days=dt)]])
    #         for p in self.planets:
    #             p.positions = np.concatenate([p.positions, [p.positions[-1] + p.velocities[-1] * dt]])
    #             acc = -2.959e-4 * p.positions[-1] / np.sum(p.positions[-1] ** 2) ** (3. / 2)  # in units of AU/day^2
    #             p.velocities = np.concatenate([p.velocities, [p.velocities[-1] + acc * dt]])

    @functools.lru_cache(64)
    def get_snapshot(self, snapshot_date:  str | float) -> SolarSystemSnapshot:
        if isinstance(snapshot_date, str):
            index = (parse_date(snapshot_date) - self.times[0]) / datetime.timedelta(days=1)
        else:
            index = snapshot_date
        if index == int(index):
            index = int(index)
            return SolarSystemSnapshot(
                [np.array([0, 0, 0])] + [p.positions[index] for p in self.planets],
                [np.array([0, 0, 0])] + [p.velocities[index] for p in self.planets]
            )
        else:
            return self.get_snapshot(int(index)).average(self.get_snapshot(int(index) + 1))

    def __getitem__(self, item):
        return self.get_snapshot(item)


planets = ("sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune")


@dataclasses.dataclass
class SolarSystemSnapshot:

    planet_positions: list
    planet_velocities: list

    def _get_obj_index(self, planet_name):

        if not planet_name.lower() in planets:
            raise ValueError(f"Planet '{planet_name}' not recognized.")

        return planets.index(planet_name.lower())

    def get_position(self, obj):
        return self.planet_positions[self._get_obj_index(obj)]

    def get_velocity(self, obj):
        return self.planet_velocities[self._get_obj_index(obj)]

    def get_speed(self, obj):
        return np.linalg.norm(self.get_velocity(obj))

    def get_distance(self, obj1, obj2):
        return np.linalg.norm(self.get_position(obj1) - self.get_position(obj2))

    def get_angle(self, obj):
        pos = self.get_position(obj)
        return np.math.atan2(pos[1], pos[0])

    def average(self, other: SolarSystemSnapshot):
        return SolarSystemSnapshot(
            [(x + y) / 2 for x, y in zip(self.planet_positions, other.planet_positions)],
            [(x + y) / 2 for x, y in zip(self.planet_velocities, other.planet_velocities)],
        )
