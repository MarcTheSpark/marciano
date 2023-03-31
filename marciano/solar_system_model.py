import numpy as np
import datetime
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from scamp import wait


class Object:  # define the objects: the Sun, Earth, Mercury, etc
    def __init__(self, name, r, v):
        self.name = name
        self.r = np.array(r, dtype=np.float64)
        self.v = np.array(v, dtype=np.float64)


class SolarSystem:

    def __init__(self, thesun, start_time):
        self.thesun = thesun
        self.planets = []
        self.time = start_time

    def add_planet(self, planet):
        self.planets.append(planet)

    def evolve(self, dt=1.0):  # evolve the trajectories
        self.time += dt

        for p in self.planets:
            p.r += p.v * dt
            acc = -2.959e-4 * p.r / np.sum(p.r ** 2) ** (3. / 2)  # in units of AU/day^2
            p.v += acc * dt


ss = None
_update_loop_clock = None


def _update_loop(days_per_beat, update_interval):
    while True:
        ss.evolve(days_per_beat * update_interval)
        wait(update_interval)


def start(scamp_session, days_per_beat, start_date=None, update_interval=0.02):
    """
    Start the solar system, or restart the

    :param scamp_session: SCAMP session on which to fork the update function
    :param days_per_beat: how many days should pass per beat in the session
    :param start_date: When the solar system should start from (defaults to current date)
    :param update_interval: How often (in beats) the solar system should update.
    """
    global ss, _update_loop_clock

    if _update_loop_clock is not None:
        _update_loop_clock.kill()
        _update_loop_clock = None

    if ss is None:
        ss = SolarSystem(Object("Sun", [0, 0, 0], [0, 0, 0]),
                         start_time=Time(datetime.datetime.now()).jd if start_date is None else Time(start_date).jd)
        for i, nasaid in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):  # The 1st, 2nd, 3rd, 4th planet in solar system
            obj = Horizons(id=nasaid, location="@sun", epochs=ss.time).vectors()
            ss.add_planet(Object(nasaid,
                                 [np.double(obj[xi]) for xi in ['x', 'y', 'z']],
                                 [np.double(obj[vxi]) for vxi in ['vx', 'vy', 'vz']]))

    _update_loop_clock = scamp_session.fork(_update_loop, args=(days_per_beat, update_interval))


def pause():
    """
    Pause but don't kill the solar system. Can still read data from it.
    """
    global _update_loop_clock
    if _update_loop_clock is not None:
        raise RuntimeError("Cannot pause solar system that was not started.")
    _update_loop_clock.kill()
    _update_loop_clock = None


def stop():
    """
    Stop the current solar system model.
    """
    global ss, _update_loop_clock
    ss = None
    _update_loop_clock.kill()
    _update_loop_clock = None

planets = ("mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune")


def _get_object(planet_name):
    if planet_name.lower() == "sun":
        return ss.thesun

    if not planet_name.lower() in planets:
        raise ValueError(f"Planet '{planet_name}' not recognized.")
    if ss is None:
        raise RuntimeError("Solar system was not started.")
    return ss.planets[planets.index(planet_name.lower())]


def get_position(obj):
    return _get_object(obj).r


def get_velocity(obj):
    return _get_object(obj).v


def get_speed(obj):
    return np.linalg.norm(_get_object(obj).v)


def get_distance(obj1, obj2):
    return np.linalg.norm(get_position(obj1) - get_position(obj2))


def get_angle(obj):
    pos = _get_object(obj).r
    return np.math.atan2(pos[1], pos[0])