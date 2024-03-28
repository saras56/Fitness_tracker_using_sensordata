import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
data_path = "../../data/raw/MetaMotion\\"
f = files[2]

participant = f.split("-")[0].replace(data_path, " ")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("2")

df = pd.read_csv(f)

df["participant"] = participant
df["label"] = label
df["category"] = category
df

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    participant = f.split("-")[0].replace(data_path, " ")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("23").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)

    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])
    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])

# set is a count for each acc(94) and gyr(93) file.
# total of 187 files thus 187 sets
# each set has 150-200 entries

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
acc_df.info()
pd.to_datetime(df["epoch (ms)"], unit="ms")
pd.to_datetime(df["time (01:00)"])
# converting to datetime allows us to use datetime
# variables like dt.week
# Use unix timestamp as its standardised representation

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

acc_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)
gyr_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")


def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, " ")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("23").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    acc_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)
    gyr_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
df_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
# But as the timestanp of acc and gyr are different,there are many rows
# with Nan values. So we need to adjust the timestamp to avoid rows with Nan values

df_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------
# Accelerometer:    12.500HZ (1/12.5 = .04) # measured every .04s
# Gyroscope:        25.000Hz (1/25 = .08)# measured every .08s
# resamplimg works only on pandas datetime format
# here there are a lot of rows with Nan values as acc and gyr 
# measure in different time intervals. So we resample so as to get
# just enough data by skipping the nan and setting an appropriate
# interval (here 200ms)

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

df_merged[:1000].resample(rule="200ms").apply(sampling)
# applyin this resampling function on the whole dataframe 
# would create unnecessary rows in 200 ms interval and could 
# potentially blow up the dataframe. So we create a
# df for each day and then apply the resampling function

days = [g for n, g in df_merged.groupby(pd.Grouper(freq="D"))]

days[-1]
len(days)  
# we have 10 different dfs as o/p of days
# We have to concatenate it while resampling

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)
# thus we reduced 70k rows to 9k rows

data_resampled.info()

data_resampled["set"] = data_resampled["set"].astype("int")


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
# pickle format is good when dataset contains timeformat
# dont have to worry about conversion when reading the file later.
# also very fast and small in size files