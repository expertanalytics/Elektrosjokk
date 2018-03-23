"""A  collection of namedtuples used for data type specifications."""

from collections import namedtuple


Data_spec = namedtuple("data_spec", ("data", "label", "title", "ylabel"))

Plot_spec = namedtuple("data_spec", ("line", "name", "title", "ylabel"))
