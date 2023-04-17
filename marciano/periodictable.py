import pandas as pd
import numpy as np
import pathlib


pt_dataframe = pd.read_csv(pathlib.Path(__file__).parent / "PeriodicTable.csv")
pt_dataframe.set_index("Element", drop=False, inplace=True)


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
        elif kelvin_temp >= self.BoilingPoint:
            return "gas"
        elif kelvin_temp >= self.MeltingPoint or np.isnan(self.MeltingPoint):
            return "liquid"
        else:
            return "solid"
        
    def __getattr__(self, attr):
        return self.element_info[attr]
    
    def __repr__(self):
        return f"Element({self.AtomicNumber})"
