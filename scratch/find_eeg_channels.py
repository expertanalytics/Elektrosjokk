import operator

from pathlib import Path


# Read channel.pos
with open(Path.home() / "Documents/ECT-data/zhi/channel.pos") as infile:
    lines = list(map(lambda x: x[:2], map(lambda x: x.split(","), infile.readlines())))
    pos_dict = {index: name.strip() for index, name in lines}

# Read the eeg header for names and indices
with open(Path.home() / "Documents/Elektrosjokk/Scratch/test_eegheader.txt") as infile:
    header_dict = {}
    for line in infile.readlines():
        index, rest = line.split("=")
        header_dict[index] = rest.split(",")[0]


pos_names = set(pos_dict.values())
header_names = set(header_dict.values())
print(header_names - pos_names)
print(header_dict["Ch31"])
print(header_dict["Ch32"])
print(header_dict["Ch62"])

# difference = header_names - pos_names
# print(difference)
