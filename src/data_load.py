"""
ETL functions for loading in historical data from CSVs and cleaning it up
Created by: Charles Tang
For: BJ's Wholesale Robotics Team
"""

from datetime import datetime, timedelta
import random
import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from src.distance_map import load_distance_map


######## GENERIC FUNCTIONS ##########
def get_list_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get list of unique users from df
    """
    return list(set(df["user"]))


def convert_to_datetime(date_string: str) -> datetime:
    """
    Converts a string in format
        'DD-MMM-YY hour minute second.microseconds AM/PM'
    to a datetime object
    """
    # Define the format string with corresponding codes
    am_pm = date_string[-2:]
    date_string = date_string[:-13] + " " + am_pm
    format_string = "%d-%b-%y %I.%M.%S %p"

    # Parse the string and create a datetime object
    datetime_obj = datetime.strptime(date_string, format_string)
    return datetime_obj


def plot_time_taken_histogram(df: pd.DataFrame) -> None:
    """
    Plots histogram of values (distribution) time_taken column of df
    """
    plt.hist(df["time_taken"], bins=75, color="#D31242")
    ax = plt.gca()
    plt.xlabel("Seconds (per pallet delivery)")
    plt.xlim((0, 350))
    ax.get_yaxis().set_visible(False)
    plt.show()


def remove_leading_zeros(text: str) -> str:
    """Removes leading zeros from a string."""
    return text.lstrip("0")


def get_date_tasks(df: pd.DataFrame, date: datetime) -> pd.DataFrame:
    """
    Given df and date (datetime obj), filters data only on that day
    """
    day = date.day
    month = date.month
    year = date.year
    condition = f"(from_time.dt.day == {day}) & (from_time.dt.month == {month}) & (from_time.dt.year == {year})"
    df = df.query(condition)
    df = df.sort_values(by="from_time")
    return df


def load_aggregate_data(
    date: datetime = None,  # type: ignore
    dc: int = 800,
) -> pd.DataFrame:
    """
    Loads in xdock, ptc->shipping, and receiving->ptc CSVs and puts them together
    """
    file_path = f"Data/Cleaned Data/DC{dc}.csv"
    # if already cleaned
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=["from_time", "to_time"])
        if date:
            df = get_date_tasks(df, date)
        # prevent from becoming int
        df[["from_locn", "to_locn"]] = df[["from_locn", "to_locn"]].astype(str)
        return df

    # xdock data
    df = load_historical_xdock_data(f"Data/Dock to Dock/DC{dc} Xdock v1.csv")
    df = clean_xdock_data(df)

    # ptc -> shipping data
    df2 = load_historical_ptc_shipping_data(f"Data/PTC to Shipping/DC{dc} PTC Out.csv")
    df2 = clean_ptc_shipping_data(df2)

    # receiving -> ptc data
    df3 = load_historical_receiving_ptc_data(f"Data/Receiving to PTC/DC{dc} to PTC.csv")
    df3 = clean_receiving_ptc_data(df3, dc)

    # receiving -> ibnp data
    df4 = load_historical_receiving_ibnp_data(
        f"Data/Receiving to IBNP/DC{dc} Putaway IBNP.csv", dc
    )
    df4 = clean_receiving_ibnp_data(df4)

    # combine all data
    df_combined = pd.concat([df, df2, df3, df4])

    if date:
        df_combined = get_date_tasks(df_combined, date)

    # anonymize users
    id_map = {}

    def anonymize_id(user_id):
        if user_id in id_map:
            return id_map[user_id]
        id_map[user_id] = len(id_map.keys()) + 1
        return id_map[user_id]

    df_combined["user"] = df_combined["user"].apply(anonymize_id)

    # sort by time
    df_combined = df_combined.sort_values(by="from_time")

    df_combined[["from_locn", "to_locn"]] = df_combined[
        ["from_locn", "to_locn"]
    ].astype(str)

    return df_combined


def get_user_tasks(df: pd.DataFrame, user: str) -> pd.DataFrame:
    """
    Given df and user, filters data only on that user
    """
    df = df[df["user"] == user]
    return df


def get_shipping_doors_volume(tasks: pd.DataFrame, dc: int = -1) -> List[List]:
    """
    Returns list of list corresponding shipping door with its volume of LPNs

    Inputs: tasks (df)
            dc (int)
    Outputs: shipping_doors_counts ([[Door #, Volume], [140, 2000], ...])
    """

    df = tasks

    # Set shipping doors
    if dc == 800:
        dock_doors = set(
            list(range(47, 168, 2)) + list(range(14, 61, 2)) + list(range(138, 161, 2))
        )
        dock_doors_to_remove = set([46, 87, 135])
        dock_doors = sorted(list(dock_doors - dock_doors_to_remove))
    elif dc == 820:
        dock_doors = set(
            list(range(23, 47))
            + list(range(48, 89, 2))
            + list(range(106, 139, 2))
            + list(range(141, 176))
        )
        dock_doors_to_remove = set([60])
        dock_doors = sorted(list(dock_doors - dock_doors_to_remove))
    elif dc == 840:
        dock_doors = set(
            list(range(155, 174, 2))
            + list(range(106, 175, 2))
            + list(range(55, 88, 2))
            + list(range(56, 89, 2))
        )
        dock_doors_to_remove = set([142])
        dock_doors = sorted(list(dock_doors - dock_doors_to_remove))
    else:
        dock_doors = list(
            set(list(set(tasks["from_locn"])) + list(set(tasks["to_locn"])))
        )

    # Count volume for each shipping door
    shipping_doors_counts = []
    for dock_door in dock_doors:
        dock_door = str(dock_door)
        df_filtered_to = df[df["to_locn"] == dock_door]
        df_filtered_from = df[df["from_locn"] == dock_door]
        df_filtered = pd.concat([df_filtered_from, df_filtered_to])
        shipping_doors_counts.append([int(dock_door), df_filtered.shape[0]])

    # sort by volume ascending
    shipping_doors_counts.sort(key=lambda x: x[1], reverse=False)

    return shipping_doors_counts


######## LOAD XDOCK DATA ############


def load_historical_xdock_data(filename: str) -> pd.DataFrame:
    """
    Loads in Xdock CSV and cleans it up as df
    """

    df = pd.read_csv(filename)
    try:
        # fix time stamps
        cols = [
            "CROSS_DOCK_CREATE_DATE",
            "CROSS_DOCK_LAST_TOUCHED",
            "ANCHOR_LAST_TOUCHED",
            "DISPO_CREATE_DATE",
        ]
        for col in cols:
            try:
                df[col] = df[col].apply(convert_to_datetime)
            except TypeError:
                pass
        # sort by starting times
        df = df.sort_values(by=["CROSS_DOCK_LAST_TOUCHED"])
        # create new col to find time taken
        df["time_taken"] = df["ANCHOR_LAST_TOUCHED"] - df["CROSS_DOCK_LAST_TOUCHED"]
        df["time_taken"] = df["time_taken"].apply(lambda x: x.total_seconds())
    except TypeError:
        pass
    return df


def clean_xdock_data(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """
    Cleans out df
    """
    # clean out outlier times
    df = df.sort_values(by="time_taken")
    threshold_index = int(len(df) * threshold)
    df = df.iloc[:threshold_index]
    df = df[df["time_taken"] > 0]
    # clean out diff users
    df = df[df["DISPO_USER"] == df["ANCHOR_USER"]]
    # remove when ILPN != OLPN
    df = df[df["ILPN"] == df["OLPN"]]
    # remove PROBLEM
    df = df[df["ORIGINAL_LOCN"] != df["ANCHOR_LOCN"]]
    df = df[~df["ORIGINAL_LOCN"].str.contains("PROBLEM", case=False)]
    df = df[~df["ANCHOR_LOCN"].str.contains("PROBLEM", case=False)]
    # remove everything from ORIGINAL_LOCN that doesn't end with R or start w C
    df = df[
        df["ORIGINAL_LOCN"].str.endswith("R") | df["ORIGINAL_LOCN"].str.startswith("C")
    ]
    # replace C*** with just C
    df["ORIGINAL_LOCN"] = df["ORIGINAL_LOCN"].str.replace("^C", "C", regex=True)

    # remove endings with R
    def remove_last_r(locn):
        """Removes 'R' from the last character if it exists."""
        return locn[:-1] if locn.endswith("R") else locn

    df["ORIGINAL_LOCN"] = df["ORIGINAL_LOCN"].apply(remove_last_r)

    # keep everything that ends with S or A for ANCHOR_LOCN
    df = df[
        df["ANCHOR_LOCN"].str.endswith("S")
        | df["ANCHOR_LOCN"].str.endswith("A")
        | df["ANCHOR_LOCN"].str.endswith("B")
    ]
    df["ANCHOR_LOCN"] = df["ANCHOR_LOCN"].apply(lambda x: x[:-1])

    # for ANCHOR and ORIGINAL, if they are numbers, remove zeros infront of them
    df["ORIGINAL_LOCN"] = df["ORIGINAL_LOCN"].apply(remove_leading_zeros)
    df["ANCHOR_LOCN"] = df["ANCHOR_LOCN"].apply(remove_leading_zeros)

    # remove unecessary cols
    cols = [
        "Unnamed: 0",
        "WHSE",
        "LOCN_MAYBE",
        "CROSS_DOCK_CREATE_DATE",
        "STAT_CODE",
        "DISPO_USER",
        "OLPN",
        "DISPO_CREATE_DATE",
    ]
    for col in cols:
        try:
            df.drop(columns=[col], inplace=True)
        except KeyError:
            continue
    renames = {
        "ANCHOR_USER": "user",
        "ILPN": "id",
        "CROSS_DOCK_LAST_TOUCHED": "from_time",
        "ORIGINAL_LOCN": "from_locn",
        "ANCHOR_LAST_TOUCHED": "to_time",
        "ANCHOR_LOCN": "to_locn",
    }
    df.rename(columns=renames, inplace=True)
    return df


######## LOAD PTC -> Shipping DATA ############
def load_historical_ptc_shipping_data(filename: str) -> pd.DataFrame:
    """
    Loads in PTC->Shipping CSV and cleans it up as df
    """
    df = pd.read_csv(filename)
    # convert columns to standard datetime format
    df["BEGIN_DATE"] = pd.to_datetime(df["BEGIN_DATE"])
    df["END_DATE"] = pd.to_datetime(df["END_DATE"])
    # add time_taken column and convert to seconds
    df["time_taken"] = df["END_DATE"] - df["BEGIN_DATE"]
    df["time_taken"] = df["time_taken"].apply(lambda x: x.total_seconds())
    return df


def clean_ptc_shipping_data(df: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    """
    Cleans out df
    """
    # drop uneceesary columns
    columns_to_drop = [
        df.columns[0],
        "RECORD_CREATED",
        "MODULE_NAME",
        "MENU_OPTN_NAME",
        "NBR_UNITS",
    ]
    for col in columns_to_drop:
        try:
            df.drop(columns=[col], inplace=True)
        except KeyError:
            continue

    # format locn strings
    df["FROM_LOCN"] = df["FROM_LOCN"].astype(str)
    df["TO_LOCN"] = df["TO_LOCN"].astype(str)
    df["FROM_LOCN"] = df["FROM_LOCN"].str[:3]

    # clear out outlier values
    df = df.sort_values(by="time_taken")
    threshold_index = int(len(df) * threshold)
    df = df.iloc[:threshold_index]
    df = df[df["time_taken"] > 0]

    # sort by start time
    df = df.sort_values(by="BEGIN_DATE")

    # rename columns
    renames = {
        "CNTR_NBR": "id",
        "FROM_LOCN": "from_locn",
        "TO_LOCN": "to_locn",
        "USER_ID": "user",
        "BEGIN_DATE": "from_time",
        "END_DATE": "to_time",
    }
    df.rename(columns=renames, inplace=True)
    # keep everything that is: 612, 622, 632, 642, 652, 662, 672, 682, 692
    possible_from_locations = [
        "612",
        "622",
        "632",
        "642",
        "652",
        "662",
        "672",
        "682",
        "692",
    ]

    df = df[df["from_locn"].isin(possible_from_locations)]
    # keep everything that ends with S or A for TO_LOCN
    df = df[
        df["to_locn"].str.endswith("S")
        | df["to_locn"].str.endswith("A")
        | df["to_locn"].str.endswith("CDY1")
        | df["to_locn"].str.endswith("CDY2")
    ]
    # replace CDY with S
    df["to_locn"] = df["to_locn"].str.replace(r"(CDY1|CDY2)$", "S", regex=True)

    # remove S, A from to_locn
    df["to_locn"] = df["to_locn"].apply(lambda x: x[:-1])

    # for to_locn, if they are numbers, remove zeros infront of them
    df["to_locn"] = df["to_locn"].apply(remove_leading_zeros)

    # print(
    #     f"Cleaned out {(original_num_rows - df.shape[0]) / original_num_rows * 100} % of entries"
    # )
    return df


######## LOAD RECEIVING -> IBNP DATA ############
def load_historical_receiving_ibnp_data(filename: str, dc: int) -> pd.DataFrame:
    """
    Loads in CSV for Receiving -> IBNP
    """
    df = pd.read_csv(filename)
    # apply time formatting
    cols = ["CREATE_DATE_TIME", "MOD_DATE_TIME"]
    for col in cols:
        try:
            df[col] = df[col].apply(convert_to_datetime)
        except TypeError:
            pass
    # sort by starting times
    df = df.sort_values(by=["CREATE_DATE_TIME"])
    # Fix ending times so they are extrapolated from starting time
    dist_map = load_distance_map(dc)

    def calculate_mod_date_time(row):
        """Calculates MOD_DATE_TIME based on CREATE_DATE_TIME, LOCN_BRCD, and dist_map."""
        locn_brcd = row["LOCN_BRCD"]
        create_date_time = row["CREATE_DATE_TIME"]
        dist_in_seconds = dist_map.get((locn_brcd, "IBNP"), random.randint(50, 100))
        return create_date_time + timedelta(seconds=dist_in_seconds)

    df["MOD_DATE_TIME"] = df.apply(calculate_mod_date_time, axis=1)
    # create new col to find time taken
    df["time_taken"] = df["MOD_DATE_TIME"] - df["CREATE_DATE_TIME"]
    df["time_taken"] = df["time_taken"].apply(lambda x: x.total_seconds())
    return df


def clean_receiving_ibnp_data(
    df: pd.DataFrame, threshold: float = 0.95
) -> pd.DataFrame:
    """
    Cleans out time_taken column based on bottom % threshold and removes negative times
    """
    # drop uneceesary columns
    columns_to_drop = [df.columns[0], "WHSE", "TASK_ID", "STAT_CODE"]
    for col in columns_to_drop:
        try:
            df.drop(columns=[col], inplace=True)
        except KeyError:
            continue
    # filter out only values that say Drop IBNP
    df = df[df["TEST"].str.contains("IBNP")]

    # drop TEST col
    df = df.drop(columns=["TEST"])

    # threshold top 95% of values by time_taken
    df = df.sort_values(by="time_taken")
    threshold_index = int(len(df) * threshold)
    df = df.iloc[:threshold_index]
    df = df[df["time_taken"] > 0]

    # format locn strings
    df = df[df["LOCN_BRCD"].str.endswith("R")]
    # remove R
    df["LOCN_BRCD"] = df["LOCN_BRCD"].apply(lambda x: x[:-1])
    # for LOCN_BRCD, if they are numbers, remove zeros infront of them
    df["LOCN_BRCD"] = df["LOCN_BRCD"].apply(remove_leading_zeros)
    df["LOCN_BRCD_1"] = "IBNP"

    # sort by start time
    df = df.sort_values(by="LOCN_BRCD")

    # rename columns
    renames = {
        "CNTR_NBR": "id",
        "LOCN_BRCD": "from_locn",
        "LOCN_BRCD_1": "to_locn",
        "USER_ID": "user",
        "CREATE_DATE_TIME": "from_time",
        "MOD_DATE_TIME": "to_time",
    }
    df.rename(columns=renames, inplace=True)
    return df


######## LOAD RECEIVING -> PTC DATA ############
def load_historical_receiving_ptc_data(filename: str) -> pd.DataFrame:
    """
    Loads in CSV for Receiving -> PTC
    """
    df = pd.read_csv(filename)
    # drop unecessary columns
    cols = [df.columns[0], "WHSE", "NBR_OF_CASES", "NBR_UNITS", "MENU_OPTN_NAME"]
    for col in cols:
        try:
            df.drop(columns=[col], inplace=True)
        except KeyError:
            continue
    # rename columns
    renames = {
        "CNTR_NBR": "id",
        "TO_LOCN": "to_locn",
        "FROM_LOCN": "from_locn",
        "CREATE_DATE": "from_time",
        "USER_ID": "user",
    }
    try:
        df.rename(columns=renames, inplace=True)
    except KeyError:
        print("Rename not successful")
    # convert time values
    df["from_time"] = pd.to_datetime(df["from_time"])
    # ensure it is object type
    df["to_locn"] = df["to_locn"].astype("str")
    return df


def clean_receiving_ptc_data(df: pd.DataFrame, dc: int) -> pd.DataFrame:
    """
    Cleans out data and extrapolates time_takens
    """
    # Remove NaN locn values
    df.dropna(subset=["from_locn"], inplace=True)
    df.dropna(subset=["to_locn"], inplace=True)

    # Remove same to_locn and from_locn
    df = df[df["from_locn"] != df["to_locn"]]

    # Filter from_locn
    df = df[
        (df["from_locn"].str.endswith("R"))  # receiving dock
        | (df["from_locn"].str.startswith("C"))  # cart
        | (df["from_locn"].str.endswith("S"))  # shipping dock
        | (df["from_locn"].isin(["BRK", "INBP", "MNOB", "BRK1"]))  # special case
        | (df["from_locn"].astype(str).str.contains("692|682"))  # special case
        | (df["from_locn"].str.startswith("6"))
    ]

    # Remove trailing Rs and make 682/692*** onWly 682 and 692 and make BRK1 -> BRK
    def custom_clean(x):
        if x[-1] == "R":
            return x[:-1]
        if x[-1] == "S":
            return x[:-1]
        if x[0] == "C":
            return "C"
        if len(x) == 6 and x[0] == "6":
            return x[:3]
        if x == "BRK1":
            return "BRK"
        return x

    df["from_locn"] = df["from_locn"].apply(custom_clean)
    # Remove trailing 0s in from_locn
    df["from_locn"] = df["from_locn"].apply(remove_leading_zeros)
    # Remove QC for DC840 data
    df = df[df["from_locn"] != "QC"]
    df = df[df["to_locn"] != "QC"]
    # Filter to_locn
    df = df[
        df["to_locn"].isin(
            ["611", "621", "631", "641", "651", "661", "671", "681", "691"]
        )
    ]

    # Remove PTC -> PTC data
    ptc = [
        "611",
        "621",
        "631",
        "641",
        "651",
        "661",
        "671",
        "681",
        "691",
        "612",
        "622",
        "632",
        "642",
        "652",
        "662",
        "672",
        "682",
        "692",
        "IBNP",
        "MNOB",
    ]
    df = df[(~df["to_locn"].isin(ptc)) | (~df["from_locn"].isin(ptc))]

    # Extrapolate to_times
    dist_map = load_distance_map(dc)

    def calculate_to_date_time(row):
        """Calculates to_time based on from_time, from_locn, and dist_map."""
        from_locn = row["from_locn"]
        to_locn = row["to_locn"]
        from_time = row["from_time"]
        dist_in_seconds = dist_map.get((from_locn, to_locn), random.randint(50, 150))
        return from_time + timedelta(seconds=dist_in_seconds)

    df["to_time"] = df.apply(calculate_to_date_time, axis=1)

    # Calculate time_taken
    df["time_taken"] = df["to_time"] - df["from_time"]
    df["time_taken"] = df["time_taken"].apply(lambda x: x.total_seconds())

    # Sort by start time
    df = df.sort_values(by="from_time")
    return df
