# License: BSD 3-clause
# Author: Boris Reuderink

#very little modification from the original script
#the regular expression stim_code = int(re.match(r'S\s*(\d+)', mdesc).group(1))
#now it matches correctly also markers without spaces ex: "S102"

# Adapted for python3 by Jakob Schreiner

import logging
import re
import itertools

import numpy as np

from pathlib import Path
from collections import namedtuple
from configparser import SafeConfigParser

from typing import (
    Iterable,
    List,
    Tuple,
    Optional
)


# TODO:
# - add encoding of commas (\1)
# - verify units for resolution in UTF8


log = logging.getLogger('__main__')


HeaderData = namedtuple(
    "DataSpec", (
        "sample_rate",
        "channel_label_list",
        "channel_resolution_list",
        "eeg_file_name",
        "marker_file_name"
))

MarkerData = namedtuple(
    "MarkerData", (
        "name",
        "type",
        "description",
        "duration",
        "channels"
))


def read_header(file_name: Path) -> HeaderData:
    """Read the data header.

    The sample_rate, channel labels, channel resolution, eeg and marker file_names are
    returned as a namedtuple.

    Arguments:
        file_name: Path to header file.
    """
    with open(file_name) as file_handle:
        # setup config reader
        header = "Brain Vision Data Exchange Header File Version 1.0"
        assert file_handle.readline().strip() == header

        # Break when reachibng [Comment]
        lines = itertools.takewhile( lambda x: '[Comment]' not in x, file_handle.readlines())
        cfg = SafeConfigParser()
        cfg.readfp(lines)

        # Samplling interval is given in micro seconds. Convert to seconds -- Hence 1e6
        sample_rate = 1e6/cfg.getfloat('Common Infos', 'SamplingInterval')
        num_channels = cfg.getint('Common Infos', 'NumberOfChannels')

        log.info(f"Found sample rate of {sample_rate:.2f} Hz, {num_channels:d} channels.")

        # check binary format
        assert cfg.get('Common Infos', 'DataOrientation') == 'MULTIPLEXED'
        assert cfg.get('Common Infos', 'DataFormat') == 'BINARY'
        assert cfg.get('Binary Infos', 'BinaryFormat') == 'INT_16'

        # load channel labels
        channel_label_list = ["UNKNOWN"]*num_channels
        channel_resolution_list = [np.nan]*num_channels
        for chan, props in cfg.items('Channel Infos'):
            n = int(re.match(r'ch(\d+)', chan).group(1))
            name, refname, resolution, unit = props.split(',')[:4]
            del refname

            channel_label_list[n - 1] = name
            channel_resolution_list[n - 1] = float(resolution)

        # locate EEG and marker files
        eeg_file_name = cfg.get('Common Infos', 'DataFile')
        marker_file_name = cfg.get('Common Infos', 'MarkerFile')

        return HeaderData(
            sample_rate=sample_rate,
            channel_label_list=channel_label_list,
            channel_resolution_list=channel_resolution_list,
            eeg_file_name=eeg_file_name,
            marker_file_name=marker_file_name
        )


def read_eeg(file_name: Path, channel_resolution: Iterable[float]) -> np.ndarray:
    """Read the binary data file.

    The eeg file must follow the specifications from the header (.vhdr).

    Arguments:
        file_name: Name of binary data file
        channel_resolution: The resolution of each channel

    Returns:
        The eeg channels scaled by their respective resolution
    """
    _channel_resolution = np.asarray(channel_resolution, dtype="f8")
    num_channels = _channel_resolution.size

    with open(file_name, 'rb') as file_handle:
        raw = file_handle.read()
        size = len(raw)//2      # TODO: Why 2?
        ints = np.ndarray(
            shape=(num_channels, size//num_channels),
            dtype='<i2',
            order='F',
            buffer=raw
        )
        return ints*_channel_resolution[:, None]


def read_markers(file_name: Path) -> List[MarkerData]:
    """Parse the marker header and return the each key-value pair.

    Arguments:
        file_name: Path to marker header (*.vmrk).
    """
    with open(file_name) as file_handle:
        header = "Brain Vision Data Exchange Marker File, Version 1.0"
        assert file_handle.readline().strip() == header

        cfg = SafeConfigParser()
        cfg.readfp(file_handle)

        events = []
        for marker, info in cfg.items("Marker Infos"):
            events.append(MarkerData(*info.split(",")[:5]))
        return events


def read_brainvis_triplet(
        header_file_name: str,
        marker_file_name: Optional[str] = None,
        eeg_file_name: Optional[str] = None
) -> Tuple[HeaderData, List[MarkerData], np.ndarray]:
    """ Read BrainVision Recorder header file, locate and read the marker and EEG file.
    Returns a header dictionary, a matrix with events and the raw EEG.

    This is a convenience wrapper around `read_header`, `read_eeg` and `read_markers`.
    """
    header_path = Path(header_file_name)
    assert header_path.exists(), header_path
    header_spec = read_header(header_path)

    if marker_file_name is None:
        marker_fname = header_path.parent / header_spec.marker_file_name
    marks = read_markers(marker_fname)

    if eeg_file_name is None:
        eeg_file_name = header_path.parent / header_spec.eeg_file_name
    X = read_eeg(eeg_file_name, header_spec.channel_resolution_list)
    return header_spec, marks, X
