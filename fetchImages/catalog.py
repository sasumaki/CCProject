import csv
from typing import List, Dict, TypeVar, Hashable
import numpy as np

import os


def load_catalog() -> List[List[str]]:
    """
    Loads the catalog.csv file which contains information about the dataset.
    :return:
    """
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'catalog.csv')) as f:
        reader = csv.reader(f, delimiter=';')
        return list(reader)[1:]


def art_paths(catalog: List[List[str]]) -> List[str]:
    return [row[6] for row in catalog]


def time_frames(catalog: List[List[str]]) -> List[str]:
    return [row[10] for row in catalog]


def schools(catalog: List[List[str]]) -> List[str]:
    return [row[9] for row in catalog]


def types(catalog: List[List[str]]) -> List[str]:
    return [row[8] for row in catalog]


def forms(catalog: List[List[str]]) -> List[str]:
    return [row[7] for row in catalog]


def categorical_to_numerical_rules(categorical_values: List[str]) -> Dict[str, int]:
    """
    Generates a translation key for encoding a list of categorical string values into integers.
    :param categorical_values: A list of categorical string values.
    :return: A dict where string category -> integer category
    """
    unique_values = set(categorical_values)
    return {category: idx for idx, category in enumerate(sorted(list(unique_values)))}


def transform_categorical_to_numerical(categorical_values: List[str]) -> List[int]:
    """
    Encodes a list of categorical string values into integers.
    :param categorical_values: A list of categorical string values.
    :return: A list of encoded categories.
    """
    rules = categorical_to_numerical_rules(categorical_values)
    return np.array([rules[categorical] for categorical in categorical_values])


if __name__ == '__main__':
    print(categorical_to_numerical_rules(types(load_catalog())))
