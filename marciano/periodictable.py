import pandas as pd
import numpy as np
import pathlib


pt_dataframe = pd.read_csv(pathlib.Path(__file__).parent / "PeriodicTable.csv")
pt_dataframe.set_index("Element", drop=False, inplace=True)
pt_dataframe["Electronegativity"].fillna(0, inplace=True)

for column in "Metal", "Nonmetal", "Metalloid", "Radioactive":
    pt_dataframe[column].fillna(False, inplace=True)
    pt_dataframe[column].replace("yes", True, inplace=True)
pt_dataframe["Electronegativity"].fillna(0, inplace=True)
pt_dataframe["Electronegativity"].fillna(0, inplace=True)
pt_dataframe["Year"].fillna(0, inplace=True)

def _snake_to_camel(s):
    return ''.join(x.title() for x in s.split("_"))


class Element:

    def __init__(self, name_or_number):
        if isinstance(name_or_number, str):
            self.element_info = pt_dataframe.loc[name_or_number.capitalize()]
        elif isinstance(name_or_number, int):
            self.element_info = pt_dataframe.iloc[name_or_number - 1]
        else:
            raise ValueError("name_or_number must be either a string or int")
    
    def print_info(self):
        print(self.element_info)
        
    def phase_at(self, kelvin_temp):
        if self.AtomicNumber > 99:
            raise ValueError("Phase undefined for elements number 100 and higher.")
        elif kelvin_temp >= self.boiling_point:
            return "gas"
        elif kelvin_temp >= self.melting_point or np.isnan(self.melting_point):
            return "liquid"
        else:
            return "solid"
        
    def discovered_by(self, year):
        return True if np.isnan(self.year) else year >= self.year

    @property
    def metalicity(self):
        return "Metal" if self.metal else "Metalloid" if self.metalloid else "Nonmetal"

    Metalicity = metalicity

    @property
    def period(self):
        return int(self.element_info["Period"])

    @property
    def group(self):
        return f"La{self.atomic_number - 56}" if 57 <= self.atomic_number <= 71 \
            else f"Ac{self.atomic_number - 88}" if 89 <= self.atomic_number <= 103 \
            else int(self.element_info["Group"])

    def __getattr__(self, attr):
        if attr == "name":
            return self.element
        elif attr in self.element_info:
            return self.element_info[attr]
        elif (camel_version := _snake_to_camel(attr)) in self.element_info:
            return self.element_info[camel_version]
        else:
            raise AttributeError(f"Element does not have attribute '{attr}'")
    
    def __repr__(self):
        return f"Element({self.AtomicNumber})"


num_elements = len(pt_dataframe)
all_elements = [Element(i + 1) for i in range(num_elements)]


def get_attribute_sequence(attribute_name):
    return [getattr(element, attribute_name) for element in all_elements]


def get_attribute_range(attribute_name):
    attribute_sequence = get_attribute_sequence(attribute_name)
    if any(isinstance(x, str) for x in attribute_sequence):
        return set(attribute_sequence)
    else:
        return min(attribute_sequence), max(attribute_sequence)
