import random
import threading
import time
from enum import Enum
from itertools import count
from typing import Sequence, Tuple, Callable, List
from dataclasses import dataclass, field
from abc import ABC
from expenvelope import Envelope
from scamp.utilities import round_to_multiple
from scamp import wait
from marciano.utility_funcs import clamp


class PairingMode(Enum):
    CHOICE = 0
    RANGE = 1


@dataclass
class TraitInfo:
    name: str
    bounds: Tuple[float, float] = (0, 1)
    pairing_mode: PairingMode = PairingMode.CHOICE
    mutation_width: float = 0.1
    mutation_probability: float = 0.5
    quantization: float = None

    def replicate(self, value):
        if random.random() < self.mutation_probability:
            new_value = clamp(random.uniform(value - self.mutation_width, value + self.mutation_width), *self.bounds)
            return round_to_multiple(new_value, self.quantization) if self.quantization else new_value
        return value

    def random_value(self):
        value = random.uniform(*self.bounds)
        return round_to_multiple(value, self.quantization) if self.quantization else value


class Individual(ABC):

    genotype_info: Tuple[TraitInfo] = NotImplemented
    _id_counter = count()

    def __init__(self, *genotype: float):
        if type(self).genotype_info is NotImplemented:
            raise NotImplementedError(f"Class {type(self)} does not define class variable 'genotype_info'")
        self.check_valid_genotype(genotype)
        self.genotype_array = genotype
        self.id_num = next(Individual._id_counter)

    def __init_subclass__(cls, **kwargs):
        cls.genotype_info_dict = {
            trait_info.name: trait_info for trait_info in cls.genotype_info
        }

    @classmethod
    def new_random(cls):
        return cls(*(trait_info.random_value() for trait_info in cls.genotype_info))

    @classmethod
    def check_valid_genotype(cls, genotype: Sequence[float]):
        if not len(genotype) == len(cls.genotype_info):
            raise ValueError(f"Phenotype {genotype} of incorrect length (should be {len(cls.genotype_info)})")
        for x, trait_info in zip(genotype, cls.genotype_info):
            if not trait_info.bounds[0] <= x <= trait_info.bounds[1]:
                raise ValueError(f"Value {x} out of range for trait {trait_info.name} "
                                 f"(should be within {trait_info.bounds})")

    def mutate(self):
        self.genotype_array = tuple(trait_info.replicate(x) for x, trait_info in zip(self.genotype_array, self.genotype_info))
        return self

    def mutated(self):
        return type(self)(*self.genotype_array).mutate()

    def get_offspring(self, how_many):
        return tuple(self.mutated() for _ in range(how_many))

    def pair(self, other):
        if not type(self) is type(other):
            raise ValueError("Cannot pair different species.")
        return type(self)(*(
            random.choice([self_trait, other_trait]) if trait_info.pairing_mode == PairingMode.CHOICE
            else random.uniform(self_trait, other_trait)
            for self_trait, other_trait, trait_info in zip(self.genotype_array, other.genotype_array, self.genotype_info)
        ))

    @classmethod
    def get_random_fitness_func(cls):
        def fitness_func(individual):
            normalized = tuple(
                (genotype - genotype_info.bounds[0]) / (genotype_info.bounds[1] - genotype_info.bounds[0])
                for genotype, genotype_info in zip(individual.genotype_array, cls.genotype_info)
            )
            return sum(n * c for n, c in zip(normalized, fitness_func.coefficients))

        fitness_func.coefficients = tuple(random.uniform(-1, 1) for _ in range(len(cls.genotype_info)))
        return fitness_func

    def __getattr__(self, item):
        if item in self.genotype_info_dict:
            return self.genotype_array[type(self).genotype_info.index(self.genotype_info_dict[item])]

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(str(x) for x in self.genotype_array)})"


@dataclass
class Population:
    individuals: List[Individual]
    fitness_function: Callable[[Individual], float]
    reproduction_curve: Envelope = field(default_factory=lambda: Envelope.from_levels([2, 5]))
    reproduction_dither: float = 1.0

    def __post_init__(self):
        if not isinstance(self.individuals, list):
            self.individuals = list(self.individuals)
        self.individuals.sort(key=self.fitness_function)
        self._individuals_list_lock = threading.Lock()
        self._new_generation_in_progress = []
        self._new_generation_is_sorted = False

    @classmethod
    def generate(cls, individual_type, size, fitness_function,
                 reproduction_curve: Envelope = Envelope.from_levels([2, 5]),
                 reproduction_dither: float = 1.0):
        return cls([individual_type.new_random() for _ in range(size)], fitness_function=fitness_function,
                   reproduction_curve=reproduction_curve, reproduction_dither=reproduction_dither)

    def _get_fitness_percentile(self, individual):
        return self.individuals.index(individual) / len(self.individuals)

    def _get_num_offspring(self, individual):
        return self.reproduction_curve.value_at(self._get_fitness_percentile(individual))

    def _next_generation_breed_once(self, sex_prob):
        if random.random() < sex_prob:
            pair = random.sample(self.individuals, k=2)
            num_offspring = int(
                (self._get_num_offspring(pair[0]) + self._get_num_offspring(pair[1])) / 2 +
                random.uniform(-self.reproduction_dither, self.reproduction_dither)
            )
            for _ in range(num_offspring):
                self._new_generation_in_progress.append(pair[0].pair(pair[1]))
                if len(self._new_generation_in_progress) >= len(self.individuals):
                    break
        else:
            individual = random.choice(self.individuals)

            num_offspring = int(
                self._get_num_offspring(individual) +
                random.uniform(-self.reproduction_dither, self.reproduction_dither)
            )
            for _ in range(num_offspring):
                self._new_generation_in_progress.append(individual.mutated())
                if len(self._new_generation_in_progress) >= len(self.individuals):
                    break

    def next_generation(self, sex_prob=0):
        with self._individuals_list_lock:
            while len(self._new_generation_in_progress) < len(self.individuals):
                self._next_generation_breed_once(sex_prob)
            if self._new_generation_is_sorted:
                self.individuals = list(self._new_generation_in_progress)
            else:
                self.individuals = sorted(self._new_generation_in_progress, key=self.fitness_function)
            self._new_generation_in_progress.clear()
            self._new_generation_is_sorted = False

    def evolve_continuously(self, evolutions_per_minute, sex_prob=0, new_generation_action=None, clock=None):
        generation_period = 60 / evolutions_per_minute
        if new_generation_action:
            new_generation_action(self)

        def _step_through_generations():
            while True:
                if new_generation_action:
                    new_generation_action(self)
                self.next_generation(sex_prob)
                wait(generation_period)

        def _evolve_continuously():
            while True:
                wait_time = 0.05
                with self._individuals_list_lock:
                    if len(self._new_generation_in_progress) < len(self.individuals):
                        self._next_generation_breed_once(sex_prob)
                        wait_time = 1e-5
                    elif not self._new_generation_is_sorted:
                        self._new_generation_in_progress.sort(key=self.fitness_function)
                        self._new_generation_is_sorted = True
                time.sleep(wait_time)

        if clock:
            clock.fork_unsynchronized(_evolve_continuously)
            clock.fork(_step_through_generations)
        else:
            threading.Thread(target=_evolve_continuously, daemon=True).start()
            threading.Thread(target=_step_through_generations, daemon=True).start()

    def get_individual(self, min_percentile=0, max_percentile=1):
        with self._individuals_list_lock:
            min_index = int(round(min_percentile * len(self.individuals)))
            max_index = int(round(max_percentile * (len(self.individuals) - 1)))
            return random.choice(self.individuals[min_index: max_index])

    def mean_fitness(self):
        with self._individuals_list_lock:
            return sum(self.fitness_function(individual) for individual in self.individuals) / len(self.individuals)

    def __repr__(self):
        return f"Population({self.individuals})"


if __name__ == '__main__':
    class Cat(Individual):
        genotype_info = (
            TraitInfo("whiskeriness"),
            TraitInfo("num_legs", (3, 12))
        )

    pop = Population.generate(Cat, 100)
    for _ in range(20):
        print(pop)
        print("--------------------------")
        pop.next_generation()
