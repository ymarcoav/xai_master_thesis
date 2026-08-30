from pathlib import Path

import numpy as np
import pandas as pd

from xai_thesis.datasets.base import BaseDataset

_DUPLICATE_COLUMN = "Fwd Header Length.1"
_SOURCE_COLUMN = "source_file"

# CIC-IDS2017 is modeled as multiclass over 7 categories (Benign + 6 attack families),
# matching the scheme used by Arreche et al. 2024 ("E-XAI: Evaluating Black-Box Explainable
# AI Frameworks for Network Intrusion Detection", IEEE Access) - confirmed against that
# paper's own Figure 21 SHAP-plot legend, which explicitly lists these 7 class names
# (the paper's prose description is internally inconsistent, claiming "six distinct attack
# profiles" while naming seven, so the figure legend - not the prose - is the source of truth
# here). DoS and DDoS are merged into one DoS/DDoS bucket, as are the two brute-force tools
# (FTP-Patator, SSH-Patator) into "Brute Force" and the three Web Attack subtypes into "Web
# Attack"; Heartbleed folds into DoS/DDoS too (it's a DoS-class vulnerability, and native
# CIC-IDS2017 has only a handful of Heartbleed rows total - far too few to model as its own
# class). PortScan, Infiltration, and Bot are already single native labels.
_ATTACK_CATEGORY: dict[str, str] = {
    "DoS Hulk": "DoS/DDoS",
    "DDoS": "DoS/DDoS",
    "DoS GoldenEye": "DoS/DDoS",
    "DoS slowloris": "DoS/DDoS",
    "DoS Slowhttptest": "DoS/DDoS",
    "Heartbleed": "DoS/DDoS",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "PortScan": "PortScan",
    "Infiltration": "Infiltration",
    "Bot": "Bot",
}


def _map_to_category(value: str) -> str:
    """Map a native CIC-IDS2017 label to Benign or one of the 6 attack categories.

    The three native "Web Attack - <subtype>" labels don't get an exact dict entry: the
    separator byte between "Web Attack" and the subtype is a corrupted, non-ASCII character
    baked into the raw files themselves (confirmed via raw byte inspection: literal UTF-8
    encoding of U+FFFD, not a decoding artifact on this project's end), so matching by
    startswith() is robust to however pandas happens to decode that byte, rather than
    depending on an exact string match against a specific corrupted character.
    """
    if value == "BENIGN":
        return "Benign"
    if value.startswith("Web Attack"):
        return "Web Attack"
    return _ATTACK_CATEGORY[value]


class CICIDS2017Dataset(BaseDataset):
    """CIC-IDS-2017 adapter."""

    def load_raw(self) -> pd.DataFrame:
        raw_path = Path(self.config.raw_path)
        frames = []
        for file_path in sorted(raw_path.glob("*.csv")):
            frame = pd.read_csv(file_path)
            frame[_SOURCE_COLUMN] = file_path.stem
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # Column names are inconsistently prefixed with a leading space in the raw files.
        data = data.rename(columns=lambda column: column.strip())

        # "Fwd Header Length" is duplicated verbatim in every raw file; the second copy
        # (auto-suffixed ".1" by pandas) is byte-for-byte identical to the first.
        data = data.drop(columns=[_DUPLICATE_COLUMN], errors="ignore")

        # Configured zero-variance columns (see configs/datasets/cic_ids2017.yaml) - a constant
        # column can't inform any classifier and only adds noise to SHAP/permutation-importance
        # rankings and PDP plots.
        data = data.drop(columns=list(self.config.drop_columns), errors="ignore")

        # Flow Bytes/s and Flow Packets/s divide by flow duration; where duration is 0 the
        # result is +Inf (~1.5k rows in Flow Bytes/s, ~2.9k in Flow Packets/s; -Inf is
        # handled too but does not occur in the raw data). Treat as missing so the shared
        # median imputer fills them - same as the ~1.4k genuine blanks Flow Bytes/s already
        # carries.
        data = data.replace([np.inf, -np.inf], np.nan)

        # Flow Duration and *IAT columns are occasionally negative due to a known
        # flow-exporter timestamp bug; these rows are corrupted, not valid traffic.
        duration_columns = [column for column in data.columns if column == "Flow Duration" or "IAT" in column]
        data = data.loc[~(data[duration_columns] < 0).any(axis=1)]

        # Fwd Header Length, Bwd Header Length, and min_seg_size_forward carry physically
        # impossible negative values in 35 rows (as low as -3.22e10 - a byte-length/
        # segment-size field can never be negative), almost certainly a CICFlowMeter integer-
        # underflow bug rather than valid traffic. Found via pairwise feature correlation:
        # these ~35-row outliers were extreme enough to flip Fwd/Bwd Header Length's measured
        # correlation from ~0 to ~1 depending on which rows landed in a given sample. Left in
        # place, a single such row would anchor MinMaxScaler's global min to a nonsensical
        # value, corrupting these otherwise-informative columns for every other row.
        non_negative_columns = ["Fwd Header Length", "Bwd Header Length", "min_seg_size_forward"]
        data = data.loc[~(data[non_negative_columns] < 0).any(axis=1)]

        # Multiclass classification: native ~15-way label collapsed into 7 categories - see
        # _ATTACK_CATEGORY/_map_to_category above.
        label_column = self.config.label_column
        data[label_column] = data[label_column].map(_map_to_category)

        # Dedupe on flow content only, ignoring the source-file tag, so a flow
        # duplicated across two source files is still caught.
        content_columns = [column for column in data.columns if column != _SOURCE_COLUMN]
        return data.loc[~data.duplicated(subset=content_columns)].reset_index(drop=True)
