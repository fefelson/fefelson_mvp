import os
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split

from ..src.agents.file_agent import JSONAgent, PickleAgent

mlbPath = os.path.join(os.environ["HOME"], "FEFelson/Sports/MLB")
trainFilePath = os.path.join(os.environ["HOME"], "FEFelson/fefelson_mvp/data/train_pitches.pkl")
testFilePath = os.path.join(os.environ["HOME"], "FEFelson/fefelson_mvp/data/test_pitches.pkl")



def zscore(index, df):
    """
    Cleans and z-score normalizes a column. Uses provided mean/std if given.
    """
    mean = df[index].mean()
    std = df[index].std()
    if std == 0 or pd.isna(std):
        print(f"Warning: {index} has no variance or all NaN, setting to 0")
        df[index] = 0
    else:
        df[index] = (df[index] - mean) / std
    return df



def process_records(files):
    """
    Process atbats and normalize hit_distance, keeping all columns.
    """
    
    allPitches = []
    for filePath in files:

        info = JSONAgent.read(filePath)

        try:
            atbats = pd.DataFrame(info["gameData"]["play_by_play"].values())
            hitValues = atbats[["play_num", "hit_distance", "hit_style", "hit_angle", "ball_hit"]].copy()
            hitValues['play_num'] = pd.to_numeric(hitValues["play_num"], errors='coerce').astype(int) -1
            pitches = pd.DataFrame(info["gameData"]["pitches"].values())
            pitches['play_num'] = pd.to_numeric(pitches["play_num"], errors='coerce').astype(int)
            pitches = pd.merge(pitches, hitValues, "left", on="play_num")
            pitches = pitches.drop(['pitch_num', 'play_type'], axis=1)

            for col in ['velocity', 'vertical', 'horizontal']:
                pitches[col] = pd.to_numeric(pitches[col], errors='coerce')
            pitches = pitches.dropna(subset=['velocity', 'vertical', 'horizontal'])
            # Convert to int after cleaning
            pitches[['velocity', 'vertical', 'horizontal']] = pitches[['velocity', 'vertical', 'horizontal']].astype(int)

            pitches['result'] = pitches['result'].apply(lambda x: 6 if int(x) == 10 else x)
            pitches['is_swing'] = pitches['result'].apply(lambda x: 0 if int(x) in [0, 1] else 1)
            pitches = pitches.rename(columns={'ball_hit': 'is_hit'})
            pitches['is_hit'] = pitches['is_hit'].fillna(0)

            allPitches.append(pitches)
        except KeyError:
            print(f"Skipping {filePath}: No pitch or play by play data")
            continue
    
    return pd.concat(allPitches, ignore_index=True)




def main():
    # Collect JSON files
    jsonFiles = [
        os.path.join(dirName, fileName)
        for dirName, _, files in os.walk(mlbPath)
        for fileName in files if fileName.endswith('.json')
    ]

    records = process_records(jsonFiles)
    for index in ("velocity", "horizontal", "vertical"):
        records = zscore(index, records)

    # Split the DataFrame into training and testing sets (e.g., 80% train, 20% test)
    train_df, test_df = train_test_split(records, test_size=0.2, random_state=42)

    PickleAgent.write(trainFilePath, train_df)
    PickleAgent.write(testFilePath, test_df)

    


if __name__ == "__main__":
    main()