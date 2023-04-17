import pandas as pd
import numpy as np
import pathlib


pt_dataframe = pd.read_csv(pathlib.Path(__file__).parent / "PeriodicTable.csv")
pt_dataframe.set_index("Element", drop=False, inplace=True)
pt_dataframe["Electronegativity"].fillna(0, inplace=True)


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
        
    def __getattr__(self, attr):
        if attr in self.element_info:        
            return self.element_info[attr]
        elif (camel_version := _snake_to_camel(attr)) in self.element_info:
            return self.element_info[camel_version]
        else:
            raise AttributeError(f"Element does not have attribute '{attr}'")
    
    def __repr__(self):
        return f"Element({self.AtomicNumber})"
