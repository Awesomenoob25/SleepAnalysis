import csv
from datetime import datetime


SLEEP_POSITIONS = [
    "Back",
    "Left Side",
    "Right Side",
    "Stomach",
    "Unknown",
    "No Person Detected"
]


def parse_timestamp(value):
    """
    Convert a CSV Timestamp value into a datetime.
    Supports the format currently written by sleepmonitor.py
    and a few common alternatives.
    """

    value = str(value).strip()

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f"
    ]

    for timestamp_format in formats:
        try:
            return datetime.strptime(
                value,
                timestamp_format
            )
        except ValueError:
            continue

    raise ValueError(
        f"Unsupported timestamp format: {value}"
    )


def format_duration(seconds):
    seconds = max(
        0,
        int(round(seconds))
    )

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


def analyse_csv(csv_path):
    """
    Analyse one Sleep Monitor CSV session.

    Duration is calculated from the time between
    consecutive CSV rows.

    The final CSV row written when a session stops
    gives the end time for new recordings.
    """

    records = []

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        if reader.fieldnames is None:
            raise ValueError(
                "The selected CSV file has no header."
            )

        required_columns = {
            "Timestamp",
            "Position"
        }

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "The CSV file is missing required columns: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        for row in reader:

            timestamp_text = (
                row.get("Timestamp", "")
                or ""
            ).strip()

            position = (
                row.get("Position", "")
                or "Unknown"
            ).strip()

            if not timestamp_text:
                continue

            try:
                timestamp = parse_timestamp(
                    timestamp_text
                )
            except ValueError:
                continue

            if not position:
                position = "Unknown"

            records.append({
                "timestamp": timestamp,
                "position": position
            })

    if not records:
        raise ValueError(
            "No valid sleep-position records were found in the CSV file."
        )

    records.sort(
        key=lambda item:
            item["timestamp"]
    )

    position_seconds = {
        position: 0.0
        for position in SLEEP_POSITIONS
    }

    # Preserve any unexpected future classifications.
    for record in records:
        if (
            record["position"]
            not in position_seconds
        ):
            position_seconds[
                record["position"]
            ] = 0.0

    total_seconds = 0.0

    # Every row describes the position that remains active
    # until the next logged timestamp.
    for index in range(
        len(records) - 1
    ):

        current_record = records[index]
        next_record = records[index + 1]

        duration_seconds = (
            next_record["timestamp"]
            - current_record["timestamp"]
        ).total_seconds()

        if duration_seconds < 0:
            continue

        position = current_record[
            "position"
        ]

        position_seconds[
            position
        ] = (
            position_seconds.get(
                position,
                0.0
            )
            + duration_seconds
        )

        total_seconds += (
            duration_seconds
        )

    # Count only genuine changes in classification.
    position_changes = 0

    for index in range(
        1,
        len(records)
    ):

        if (
            records[index]["position"]
            != records[index - 1]["position"]
        ):
            position_changes += 1

    no_person_seconds = (
        position_seconds.get(
            "No Person Detected",
            0.0
        )
    )

    # "Total hours slept" excludes periods where
    # no person was detected.
    sleep_seconds = max(
        0.0,
        total_seconds
        - no_person_seconds
    )

    breakdown = []

    for position, seconds in (
        position_seconds.items()
    ):

        if total_seconds > 0:
            percentage = (
                seconds
                / total_seconds
            ) * 100
        else:
            percentage = 0.0

        breakdown.append({
            "position": position,
            "seconds": round(
                seconds,
                1
            ),
            "hours": round(
                seconds / 3600,
                2
            ),
            "duration": format_duration(
                seconds
            ),
            "percentage": round(
                percentage,
                1
            )
        })

    position_order = {
        position: index
        for index, position
        in enumerate(
            SLEEP_POSITIONS
        )
    }

    breakdown.sort(
        key=lambda item: (
            position_order.get(
                item["position"],
                999
            )
        )
    )

    valid_sleep_positions = [
        item
        for item in breakdown
        if item["position"]
        not in {
            "No Person Detected"
        }
    ]

    dominant_position = None

    if valid_sleep_positions:

        dominant = max(
            valid_sleep_positions,
            key=lambda item:
                item["seconds"]
        )

        if dominant["seconds"] > 0:
            dominant_position = (
                dominant["position"]
            )

    return {
        "start_time":
            records[0][
                "timestamp"
            ].strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "end_time":
            records[-1][
                "timestamp"
            ].strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "total_seconds":
            round(
                total_seconds,
                1
            ),

        "total_hours":
            round(
                total_seconds / 3600,
                2
            ),

        "total_duration":
            format_duration(
                total_seconds
            ),

        "sleep_seconds":
            round(
                sleep_seconds,
                1
            ),

        "sleep_hours":
            round(
                sleep_seconds / 3600,
                2
            ),

        "sleep_duration":
            format_duration(
                sleep_seconds
            ),

        "position_changes":
            position_changes,

        "dominant_position":
            dominant_position
            or "N/A",

        "breakdown":
            breakdown,

        "record_count":
            len(records)
    }
