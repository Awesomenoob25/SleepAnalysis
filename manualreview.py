import os
import pandas as pd
import re
from datetime import datetime

# Converts a given number of seconds into a human-readable string 
# formatted as hours, minutes, and/or seconds (e.g., '2h 15m 30s').
# Returns '0s' if the value is invalid, zero, or missing.
def format_duration(seconds):
    if pd.isna(seconds) or seconds <= 0: return "0s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}h {m}m {s}s"
    elif m > 0: return f"{m}m {s}s"
    else: return f"{s}s"

# Reads a single session CSV, processes sleep metrics,
# and returns individual_stats, pie_chart_data, and an error message (if any).
def perform_individual_analysis(csv_path):
    individual_stats = {}
    pie_chart_data = {}
    
    try:
        # Load the CSV file, skipping any empty lines for data robustness
        df = pd.read_csv(csv_path, skip_blank_lines=True)
        
        # Clean whitespace from column header names
        df.columns = df.columns.str.strip()
        
        # Handle edge-case CSV layouts where timestamp data is shifted into the adjacent column
        time_col = "Timestamp"
        if time_col in df.columns and df[time_col].isnull().all():
            time_idx = df.columns.get_loc(time_col)
            if time_idx + 1 < len(df.columns):
                time_col = df.columns[time_idx + 1]
        
        df = df[[time_col, "Position"]].copy()
        df.columns = ["Start", "Position"]
            
        # Parse timestamp strings into proper datetime objects, stripping bracketed log artifacts
        df["Start"] = pd.to_datetime(
            df["Start"].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.strip(), 
            dayfirst=True, format='mixed', errors='coerce'
        )
        df["Position"] = df["Position"].astype(str).str.strip()
        df = df.dropna(subset=['Start', 'Position']).sort_values('Start').reset_index(drop=True)
        
        if df.empty:
            raise ValueError("No valid data found in the CSV.")

        # --- 1. CALCULATE DURATIONS ---
        # Determine duration blocks by finding the time difference to the subsequent row entry
        df["End"] = df["Start"].shift(-1)
        df["End"] = df["End"].fillna(df["Start"].iloc[-1] + pd.Timedelta(seconds=60))
        df["Duration_Sec"] = (df["End"] - df["Start"]).dt.total_seconds().fillna(0)
        
        # --- 2. KEY METRICS CALCULATION ---
        total_time_sec = df["Duration_Sec"].sum()
        
        # Filter out tracking failure states to calculate true 'Time in Bed'
        bed_df = df[~df["Position"].isin(["No Person Detected", "Empty Bed", "Unknown"])]
        time_in_bed_sec = bed_df["Duration_Sec"].sum()
        
        # Determine the most frequent sleeping position based on cumulative duration rather than frequency count
        pos_totals = df.groupby("Position")["Duration_Sec"].sum()
        most_frequent = pos_totals.idxmax() if not pos_totals.empty else "N/A"
        
        # Calculate positional shifts (rollovers) by detecting row-to-row label changes
        position_changes = (df["Position"] != df["Position"].shift(1)).sum() - 1
        if position_changes < 0: position_changes = 0
        
        individual_stats = {
            "total_time": format_duration(total_time_sec),
            "time_in_bed": format_duration(time_in_bed_sec),
            "most_frequent": most_frequent,
            "position_changes": int(position_changes)
        }
        
        # --- 3. FORMAT PIE CHART OUTPUT ---
        pie_chart_data = pos_totals.to_dict()
        
        return individual_stats, pie_chart_data, None
        
    except Exception as e:
        return {}, {}, f"Error processing session data: {str(e)}"

# Scans the CSV for uncertain positions and matches them to the absolute closest snapshot image
# by fuzzy-matching the session's designated subfolder, accounting for split-second delays.
def get_review_items(session_name, csv_path, snapshots_folder):
    review_items = []
    try:
        df = pd.read_csv(csv_path, skip_blank_lines=True)
        df.columns = df.columns.str.strip()
        
        time_col = "Timestamp" if "Timestamp" in df.columns else df.columns[0]
        if df[time_col].isnull().all():
            time_idx = df.columns.get_loc(time_col)
            if time_idx + 1 < len(df.columns):
                time_col = df.columns[time_idx + 1]
                
        df_clean = df[[time_col, "Position"]].copy()
        df_clean.columns = ["Start", "Position"]
        df_clean["Start"] = pd.to_datetime(
            df_clean["Start"].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.strip(), 
            dayfirst=True, format='mixed', errors='coerce'
        )
        df_clean["Position"] = df_clean["Position"].astype(str).str.strip()
        
        # Define keywords representing uncertain model classifications requiring human review
        search_terms = ["unknown", "no person detected", "empty bed"]
        uncertain_df = df_clean[df_clean["Position"].str.lower().isin(search_terms)]
        
        print(f"\n--- MANUAL REVIEW DIAGNOSTICS ---")
        print(f"Uncertain CSV rows found: {len(uncertain_df)}")
        
        if uncertain_df.empty:
            return review_items
            
        uncertain_times = uncertain_df["Start"].dropna().tolist()
        available_images = []
        
        # --- 1. FUZZY MATCH THE FOLDER NAME ---
        # Ignore exact seconds to bridge minor discrepancies between session title generation and directory names
        session_snapshots_dir = None
        
        if re.search(r'_\d{6}$', session_name):
            search_prefix = session_name[:-2] # Strip the trailing seconds
        else:
            search_prefix = session_name
            
        if os.path.exists(snapshots_folder):
            for item in os.listdir(snapshots_folder):
                full_path = os.path.join(snapshots_folder, item)
                if os.path.isdir(full_path) and item.startswith(search_prefix):
                    session_snapshots_dir = full_path
                    break
        
        # --- 2. EXTRACT TIMESTAMPED IMAGES ---
        if session_snapshots_dir:
            print(f"Scanning target folder: {session_snapshots_dir}")
            folder_name = os.path.basename(session_snapshots_dir)
            
            for filename in os.listdir(session_snapshots_dir):
                fn_lower = filename.lower()
                if fn_lower.startswith('snapshot_') and fn_lower.endswith(('.png', '.jpg', '.jpeg')):
                    match = re.search(r'snapshot_(\d{4}-\d{2}-\d{2}_\d{6})', fn_lower)
                    if match:
                        try:
                            img_time = datetime.strptime(match.group(1), "%Y-%m-%d_%H%M%S")
                            rel_path = f"{folder_name}/{filename}"
                            available_images.append({"filename": rel_path, "time": img_time})
                        except ValueError:
                            continue
        else:
            print(f"Session folder starting with '{search_prefix}' not found.")
                            
        print(f"Valid timestamped snapshots found: {len(available_images)}")
        
        # --- 3. MATCH CLOSEST AVAILABLE IMAGE ---
        # Iterate over each uncertain timestamp and find the snapshot image closest in time
        for u_time in uncertain_times:
            best_match = None
            smallest_diff = float('inf')
            
            for img in available_images:
                time_diff = abs((img["time"] - u_time).total_seconds())
                if time_diff < smallest_diff:
                    smallest_diff = time_diff
                    best_match = img["filename"]
            
            if best_match:
                print(f"-> Matched CSV [{u_time}] to Image [{best_match.split('/')[-1]}] (Gap: {int(smallest_diff)}s)")
                review_items.append({
                    "filename": best_match,
                    "time": u_time.strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                print(f"-> No images available for timestamp [{u_time}].")
                review_items.append({
                    "filename": None,
                    "time": u_time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
        print(f"---------------------------------\n")
                    
    except Exception as e:
        print(f"Error getting review items: {e}")
        
    review_items.sort(key=lambda x: x["time"])
    return review_items

# Overwrites the old CSV position label by finding the exact timestamp row.
def update_csv_label(csv_path, timestamp_str, new_label):
    try:
        df = pd.read_csv(csv_path, skip_blank_lines=True)
        
        time_col = "Timestamp" if "Timestamp" in df.columns else df.columns[0]
        if df[time_col].isnull().all() and df.columns.get_loc(time_col) + 1 < len(df.columns):
            time_col = df.columns[df.columns.get_loc(time_col) + 1]
            
        temp_times = pd.to_datetime(
            df[time_col].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.strip(), 
            dayfirst=True, format='mixed', errors='coerce'
        )
        
        target_time = pd.to_datetime(timestamp_str)
        time_diffs = abs((temp_times - target_time).dt.total_seconds())
        closest_idx = time_diffs.idxmin()
        
        # Validate that the closest timestamp row falls within an acceptable 5-second window threshold
        if pd.isna(closest_idx) or time_diffs[closest_idx] > 5:
            return False, "Could not find the matching timestamp row in the CSV."
            
        pos_col = "Position"
        df.at[closest_idx, pos_col] = new_label
        df.to_csv(csv_path, index=False)
        
        return True, ""
    except Exception as e:
        return False, str(e)

# Retrieves a chronologically sorted list of all snapshot images for a given session.
# Used to construct a pseudo-video player interface when an actual video recording is missing.
def get_all_snapshots(session_name, snapshots_folder):
    available_images = []
    try:
        session_snapshots_dir = None
        
        # Fuzzy Match the Folder Name by stripping trailing timestamp seconds if present
        if re.search(r'_\d{6}$', session_name):
            search_prefix = session_name[:-2]
        else:
            search_prefix = session_name
            
        if os.path.exists(snapshots_folder):
            for item in os.listdir(snapshots_folder):
                full_path = os.path.join(snapshots_folder, item)
                if os.path.isdir(full_path) and item.startswith(search_prefix):
                    session_snapshots_dir = full_path
                    break
        
        # Extract all valid image files within the discovered directory
        if session_snapshots_dir:
            folder_name = os.path.basename(session_snapshots_dir)
            for filename in os.listdir(session_snapshots_dir):
                fn_lower = filename.lower()
                if fn_lower.startswith('snapshot_') and fn_lower.endswith(('.png', '.jpg', '.jpeg')):
                    rel_path = f"{folder_name}/{filename}"
                    available_images.append(rel_path)
                    
        # Alphabetical sorting aligns the timestamp-based naming convention (YYYY-MM-DD_HHMMSS) chronologically
        available_images.sort()
        return available_images
        
    except Exception as e:
        print(f"Error getting all snapshots: {e}")
        return []