"""Dataset paths and protocol constants for the AgriEdge study.

All paths are resolved relative to ``DATASET_ROOT`` so the project can be
relocated (or run on Colab) by overriding a single environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ENV_DATASET_ROOT = "AGRIEDGE_DATASET_ROOT"

_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "Edge-IIoTset dataset"


def dataset_root() -> Path:
    """Return the Edge-IIoTset root directory.

    Raises:
        FileNotFoundError: if the resolved directory does not exist.
    """
    root = Path(os.environ.get(ENV_DATASET_ROOT, _DEFAULT_ROOT))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Edge-IIoTset root not found at {root!s}. "
            f"Set the {ENV_DATASET_ROOT} environment variable to its location."
        )
    return root


def project_root() -> Path:
    """Return the agriedge project directory (parent of ``src``)."""
    return Path(__file__).resolve().parents[2]


def results_dir() -> Path:
    """Return the results directory, creating it if absent."""
    path = project_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- Curated subsets shipped with the dataset ------------------------------

CURATED_SUBDIR = "Selected dataset for ML and DL"
ML_SUBSET_FILE = "ML-EdgeIIoT-dataset.csv"
DNN_SUBSET_FILE = "DNN-EdgeIIoT-dataset.csv"


def curated_subset_path(which: str = "ml") -> Path:
    """Return the path to a curated subset.

    Args:
        which: ``"ml"`` for the 157.8k-row ML subset, ``"dnn"`` for the
            2.22M-row DNN subset.

    Raises:
        ValueError: if ``which`` is not a recognised subset name.
        FileNotFoundError: if the resolved file does not exist.
    """
    files = {"ml": ML_SUBSET_FILE, "dnn": DNN_SUBSET_FILE}
    key = which.strip().lower()
    if key not in files:
        raise ValueError(
            f"Unknown curated subset {which!r}; expected one of {sorted(files)}."
        )
    path = dataset_root() / CURATED_SUBDIR / files[key]
    if not path.is_file():
        raise FileNotFoundError(f"Curated subset not found: {path!s}")
    return path


# --- Labels ----------------------------------------------------------------

BINARY_LABEL = "Attack_label"
MULTICLASS_LABEL = "Attack_type"
LABEL_COLUMNS = (BINARY_LABEL, MULTICLASS_LABEL)
NORMAL_CLASS = "Normal"


# --- The official preprocessing recipe -------------------------------------
# Reproduced verbatim from Readme.txt distributed with Edge-IIoTset
# (Ferrag et al., 2022), Steps 4 and 5. These are the columns the dataset
# authors instruct researchers to drop and to dummy-encode respectively.

README_DROP_COLUMNS = (
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "icmp.transmit_timestamp",
    "http.request.uri.query",
    "tcp.options",
    "tcp.payload",
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
    "mqtt.msg",
)

README_DUMMY_COLUMNS = (
    "http.request.method",
    "http.referer",
    "http.request.version",
    "dns.qry.name.len",
    "mqtt.conack.flags",
    "mqtt.protoname",
    "mqtt.topic",
)


# --- Raw per-device captures ------------------------------------------------

NORMAL_TRAFFIC_SUBDIR = "Normal traffic"
ATTACK_TRAFFIC_SUBDIR = "Attack traffic"


@dataclass(frozen=True)
class DeviceSpec:
    """A physical device in the Edge-IIoTset testbed.

    Attributes:
        name: Directory and file stem under ``Normal traffic``.
        agricultural: Whether the device belongs to the precision-agriculture
            deployment scenario modelled in this study.
        layer: Architectural layer the device occupies.
    """

    name: str
    agricultural: bool
    layer: str


# Agricultural relevance follows the Edge-IIoTset testbed description
# (Ferrag et al., 2022, Table 2). Heart_Rate, IR_Receiver, Sound_Sensor and
# Flame_Sensor are healthcare//building-automation devices and are excluded
# from the agricultural scenario; Distance is retained as a silo/tank level
# proxy only when `include_ambiguous` is set by the caller.
DEVICES: tuple[DeviceSpec, ...] = (
    DeviceSpec("Soil_Moisture", agricultural=True, layer="perception"),
    DeviceSpec("phValue", agricultural=True, layer="perception"),
    DeviceSpec("Water_Level", agricultural=True, layer="perception"),
    DeviceSpec("Temperature_and_Humidity", agricultural=True, layer="perception"),
    DeviceSpec("Modbus", agricultural=True, layer="actuation"),
    DeviceSpec("Distance", agricultural=False, layer="perception"),
    DeviceSpec("Flame_Sensor", agricultural=False, layer="perception"),
    DeviceSpec("Heart_Rate", agricultural=False, layer="perception"),
    DeviceSpec("IR_Receiver", agricultural=False, layer="perception"),
    DeviceSpec("Sound_Sensor", agricultural=False, layer="perception"),
)

AGRICULTURAL_DEVICES: tuple[str, ...] = tuple(
    d.name for d in DEVICES if d.agricultural
)


def device_capture_path(device: str) -> Path:
    """Return the CSV capture path for a normal-traffic device.

    Raises:
        ValueError: if ``device`` is not a known testbed device.
        FileNotFoundError: if the capture file is missing.
    """
    known = {d.name for d in DEVICES}
    if device not in known:
        raise ValueError(
            f"Unknown device {device!r}; expected one of {sorted(known)}."
        )
    path = dataset_root() / NORMAL_TRAFFIC_SUBDIR / device / f"{device}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Device capture not found: {path!s}")
    return path


# Attack CSV stems as shipped, mapped to the canonical Attack_type label used
# in the curated subsets. The shipped filenames are not internally consistent
# (e.g. "DDoS_HTTP_Flood_attack" -> "DDoS_HTTP"), so the mapping is explicit.
ATTACK_FILE_TO_TYPE: Mapping[str, str] = {
    "Backdoor_attack": "Backdoor",
    "DDoS_HTTP_Flood_attack": "DDoS_HTTP",
    "DDoS_ICMP_Flood_attack": "DDoS_ICMP",
    "DDoS_TCP_SYN_Flood_attack": "DDoS_TCP",
    "DDoS_UDP_Flood_attack": "DDoS_UDP",
    "MITM_attack": "MITM",
    "OS_Fingerprinting_attack": "Fingerprinting",
    "Password_attack": "Password",
    "Port_Scanning_attack": "Port_Scanning",
    "Ransomware_attack": "Ransomware",
    "SQL_injection_attack": "SQL_injection",
    "Uploading_attack": "Uploading",
    "Vulnerability_scanner_attack": "Vulnerability_scanner",
    "XSS_attack": "XSS",
}


def attack_capture_path(file_stem: str) -> Path:
    """Return the CSV capture path for an attack class.

    Raises:
        ValueError: if ``file_stem`` is not a known attack capture.
        FileNotFoundError: if the capture file is missing.
    """
    if file_stem not in ATTACK_FILE_TO_TYPE:
        raise ValueError(
            f"Unknown attack capture {file_stem!r}; "
            f"expected one of {sorted(ATTACK_FILE_TO_TYPE)}."
        )
    path = dataset_root() / ATTACK_TRAFFIC_SUBDIR / f"{file_stem}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Attack capture not found: {path!s}")
    return path


# --- Reproducibility --------------------------------------------------------

RANDOM_SEED = 20260814
