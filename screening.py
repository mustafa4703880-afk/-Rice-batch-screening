import csv
import io
import re

import numpy as np
import pandas as pd

from train_model import FEATURES


def read_measurements(content):
    try:
        reader = csv.reader(io.StringIO(content.decode("utf-8-sig")), strict=True)
        headers = next(reader, [])
        keys = [re.sub(r"[^a-z0-9]", "", name.casefold()) for name in headers]
        if len(set(keys)) != len(keys):
            raise ValueError("Two columns have the same name. Keep one copy of each measurement.")
        columns = dict(zip(keys, headers))
        aliases = {"Major_Axis_Length": "majoraxis", "Minor_Axis_Length": "minoraxis"}
        names = {
            feature: columns.get(feature.replace("_", "").lower()) or columns.get(aliases.get(feature))
            for feature in FEATURES
        }
        missing = [feature for feature, header in names.items() if header is None]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}.")
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(headers):
                raise ValueError(f"CSV line {reader.line_num} has the wrong number of columns.")
            rows.append(row)
        if not rows:
            raise ValueError("Add at least one grain row.")
        return pd.DataFrame(rows, columns=headers)[list(names.values())].set_axis(FEATURES, axis=1)
    except (UnicodeError, csv.Error) as error:
        raise ValueError("Upload a UTF-8, comma-separated CSV.") from error


def screen_grains(grains, profile, declared, threshold):
    values = grains.apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(values) | values.le(0)
    invalid["Eccentricity"] = (
        ~np.isfinite(values["Eccentricity"]) | ~values["Eccentricity"].between(0, 1)
    )
    invalid["Extent"] |= values["Extent"].gt(1)
    valid = ~invalid.any(axis=1)
    result = values.mask(invalid)
    result.insert(0, "Grain row", range(1, len(grains) + 1))
    result["Predicted variety"] = ""
    result["Model score (%)"] = np.nan
    result["Screening result"] = "Invalid input"
    result["Reason"] = [
        ", ".join(invalid.columns[row]) for row in invalid.to_numpy()
    ]
    if valid.any():
        measurements = values.loc[valid, FEATURES]
        outside = (measurements.lt(profile["min"]) | measurements.gt(profile["max"])).any(axis=1)
        model = profile["model"]
        scores = model.predict_proba(measurements)
        labels = model.classes_[scores.argmax(axis=1)]
        confidence = scores.max(axis=1)
        uncertain = outside.to_numpy() | (confidence < threshold)
        result.loc[valid, "Predicted variety"] = labels
        result.loc[valid, "Model score (%)"] = (confidence * 100).round(1)
        result.loc[valid, "Screening result"] = np.select(
            [uncertain, labels == declared], ["Uncertain", "Declared match"], default="Different variety"
        )
        result.loc[valid, "Reason"] = np.select(
            [outside.to_numpy(), confidence < threshold, labels != declared],
            ["Outside training measurement range", "Model score below threshold", "Does not match declared variety"],
            default="",
        )
    return result
