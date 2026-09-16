import os
import pandas as pd

# Converts a given number of seconds into a human-readable string 
# formatted as hours, minutes, and/or seconds (e.g. '2h 15m 30s')
# Returns '0s' if the value is invalid, zero, or missing.
def format_duration(seconds):
    if pd.isna(seconds) or seconds <= 0: return "0s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}h {m}m {s}s"
    elif m > 0: return f"{m}m {s}s"
    else: return f"{s}s"

# Calculates a custom fortnight label
def get_fortnight(dt):
    week_num = dt.isocalendar().week
    fortnight_num = ((week_num - 1) // 2) + 1
    return f"{dt.year} - Fortnight {fortnight_num}"

# Reads the list of selected session CSV filepaths, cleans and normalizes the data
# Computes multi-night metrics across the timeframes (Daily, Weekly, etc)
# Returns the dictionaries: (period_data, global_data)
def perform_analysis(filepaths):
    dfs = []

    # Loop through each provided file path
    for filepath in filepaths:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, skip_blank_lines=True)
            
            # Clean column headers
            df.columns = df.columns.str.strip()
            
            # Identify the timestamp column, handling the shifted column layout
            time_col = "Timestamp"
            if time_col in df.columns and df[time_col].isnull().all():
                time_idx = df.columns.get_loc(time_col)
                if time_idx + 1 < len(df.columns):
                    time_col = df.columns[time_idx + 1]
            
            # Extract only the active timestamp and position columns
            df = df[[time_col, "Position"]].copy()
            df.columns = ["Timestamp", "Position"]
            dfs.append(df)
    
    # Raise an exception if no readable data was found
    if not dfs:
        raise ValueError("No valid data found in selected files.")

    # Combine all individual session dataframes into one large dataframe
    master_df = pd.concat(dfs, ignore_index=True)
    
    # Cleans 'Timestamp' column into proper datetime objects
    master_df["Timestamp"] = pd.to_datetime(
        master_df["Timestamp"].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.strip(), 
        dayfirst=True, format='mixed'
    )
    master_df["Position"] = master_df["Position"].astype(str).str.strip()
    
    # Drop rows with missing values and filter out duplicate entries based on timestamp and position
    master_df = master_df.dropna(subset=['Timestamp', 'Position'])
    master_df = master_df.drop_duplicates(subset=["Timestamp", "Position"]).sort_values('Timestamp').reset_index(drop=True)

    # If no valid data remains after cleaning, raise an exception
    if master_df.empty:
        raise ValueError("Uploaded files contained no readable data.")

    # --- 2. NIGHT CALCULATION LOGIC ---
    master_df["Logical_Date"] = (master_df["Timestamp"] - pd.Timedelta(hours=12)).dt.date
    
    # Find the earliest timestamp for each night to establish when the session began
    night_Timestamps = master_df.groupby("Logical_Date")["Timestamp"].transform('min')
    master_df["Night_Timestamp"] = night_Timestamps
    master_df["Night_ID"] = master_df["Logical_Date"].astype(str)

    # Sort chronologically by Night ID and Timestamp time to properly sequence tracking blocks
    master_df = master_df.sort_values(["Night_ID", "Timestamp"])

    # Flag row entries where a positional shift occurred compared to the previous row (restricted within the same night)
    master_df["Is_Change"] = (master_df["Position"] != master_df["Position"].shift(1)) & (master_df["Night_ID"] == master_df["Night_ID"].shift(1))

    # Calculate the duration (in seconds) for each position block based on the time until the next entry
    master_df["End"] = master_df.groupby("Night_ID")["Timestamp"].shift(-1)
    master_df["End"] = master_df["End"].fillna(master_df["Timestamp"] + pd.Timedelta(seconds=60))
    master_df["Duration_Sec"] = (master_df["End"] - master_df["Timestamp"]).dt.total_seconds()

    # Assigns each row to various time period labels for aggregation
    master_df["Daily"] = master_df["Night_ID"]
    master_df["Weekly"] = master_df["Night_Timestamp"].dt.to_period('W-MON').dt.to_timestamp().dt.strftime('Week %Y-%m-%d')
    master_df["Fortnightly"] = master_df["Night_Timestamp"].apply(get_fortnight)
    master_df["Monthly"] = master_df["Night_Timestamp"].dt.strftime('%Y-%m (Month)')
    master_df["Quarterly"] = master_df["Night_Timestamp"].dt.to_period('Q').dt.strftime('%Y-Q%q')
    master_df["Yearly"] = master_df["Night_Timestamp"].dt.strftime('%Y (Year)')

    periods = ["Daily", "Weekly", "Fortnightly", "Monthly", "Quarterly", "Yearly"]
    period_data = {}

    # Generates statistical breakdowns for each timeframe category
    for p in periods:
        period_data[p] = {}
        p_totals = master_df.groupby(p)["Duration_Sec"].sum()
        p_nights = master_df.groupby(p)["Night_ID"].nunique()
        p_changes = master_df.groupby(p)["Is_Change"].sum() 
        p_group = master_df.groupby([p, "Position"])["Duration_Sec"].sum().reset_index()

        for period_label, total_sec in p_totals.items():
            nights_count = int(p_nights[period_label])
            
            # Sort positions within this specific period group from highest duration to lowest
            pos_data = p_group[p_group[p] == period_label].sort_values(by="Duration_Sec", ascending=False)
            pos_dict = {}
            for _, row in pos_data.iterrows():
                pos = row["Position"]
                sec = row["Duration_Sec"]
                pos_dict[pos] = {
                    "Sec": float(sec),
                    "Str": format_duration(sec),
                    "Pct": round((sec / total_sec * 100), 1) if total_sec > 0 else 0
                }

            sleep_positions = {k: v for k, v in pos_dict.items() if k != "No Person Detected"}
            dominant_pos = max(sleep_positions, key=lambda k: sleep_positions[k]["Sec"]) if sleep_positions else "None"

            period_data[p][period_label] = {
                "Total_Time_Sec": float(total_sec),
                "Total_Time_Str": format_duration(total_sec),
                "Total_Hours": format_duration(total_sec),
                "Average_Time_Str": format_duration(total_sec / nights_count) if nights_count > 0 else "0s",
                "Nights": nights_count,
                "Position_Changes": int(p_changes.get(period_label, 0)),
                "Dominant_Position": dominant_pos,
                "Positions": pos_dict
            }

    global_total = master_df["Duration_Sec"].sum()
    global_nights = master_df["Night_ID"].nunique()
    global_changes = master_df["Is_Change"].sum()
    
    global_pos = master_df.groupby("Position")["Duration_Sec"].sum().sort_values(ascending=False)
    
    global_pos_dict = {}
    for pos, sec in global_pos.items():
        global_pos_dict[pos] = {
            "Sec": float(sec),
            "Str": format_duration(sec),
            "Pct": round((sec / global_total * 100), 1) if global_total > 0 else 0
        }

    global_data = {
        "Total_Time_Str": format_duration(global_total),
        "Total_Hours": format_duration(global_total),
        "Total_Nights": int(global_nights),
        "Total_Changes": int(global_changes),
        "Average_Night_Str": format_duration(global_total / global_nights) if global_nights > 0 else "0s",
        "Average_Changes": round(float(global_changes) / global_nights, 1) if global_nights > 0 else 0,
        "Positions": global_pos_dict
    }

    return period_data, global_data
    # Computes metrics
    global_total = master_df["Duration_Sec"].sum()
    global_nights = master_df["Night_ID"].nunique()
    global_changes = master_df["Is_Change"].sum()
    
    # Aggregate all time positions and sort them by total duration
    global_pos = master_df.groupby("Position")["Duration_Sec"].sum().sort_values(ascending=False)
    
    global_pos_dict = {}
    for pos, sec in global_pos.items():
        global_pos_dict[pos] = {
            "Sec": float(sec),
            "Str": format_duration(sec),
            "Pct": round((sec / global_total * 100), 1) if global_total > 0 else 0
        }

    # Package statistics for export
    global_data = {
        "Total_Time_Str": format_duration(global_total),
        "Total_Hours": format_duration(global_total),
        "Total_Nights": int(global_nights),
        "Total_Changes": int(global_changes),
        "Average_Night_Str": format_duration(global_total / global_nights) if global_nights > 0 else "0s",
        "Average_Changes": round(float(global_changes) / global_nights, 1) if global_nights > 0 else 0,
        "Positions": global_pos_dict
    }

    return period_data, global_data