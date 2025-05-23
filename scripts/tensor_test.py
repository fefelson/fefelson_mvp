import pandas as pd
import torch

from ..src.tensors.baseball.baseball_atomics import (PitchTypeSelect, IsHit, NumBasesIfHit)
from ..src.tensors.baseball.datasets import (PitchTypeDataset, IsHitDataset, NumBasesIfHitDataset)
from ..src.tensors.trainer import BinaryTrainer, ClassifyTrainer

from ..src.database.models.database import get_db_session

PITCH_TYPE_LABELS = ["changeup", "curve", "cutter", "two-seam fb", "fastball", "forkball", "four-seam fb",
                                         "sweeper", "splitter", "screwball", "sinker", "slider", "slow curve", "slurve", 
                                         "knuckle curve", "knuckleball", "eephus pitch"]


BATTER_QUERY = """SELECT DISTINCT player_id, first_name, last_name, bats
                FROM players AS p
                INNER JOIN at_bats AS ab on p.player_id = ab.batter_id        
                """


PITCHER_QUERY = """SELECT DISTINCT player_id, first_name, last_name, throws
                FROM players 
                INNER JOIN pitches AS p ON players.player_id = p.pitcher_id
                GROUP BY p.pitcher_id
                HAVING COUNT(pitch_id) > 999
                """


STADIUM_QUERY = """SELECT DISTINCT s.stadium_id, s.name
                FROM at_bats AS ab
                INNER JOIN games AS g ON ab.game_id = g.game_id 
                INNER JOIN stadiums AS s on g.stadium_id = s.stadium_id               
                """


def query_db(query):
    
    # print(query)
    with get_db_session() as session:
        df = pd.read_sql(query, session.bind)
    return df


def num_bases():
    
    atomicTrainer = ClassifyTrainer(class_labels=["single", "double", "triple", "hr"])
    atomicTrainer.dojo(NumBasesIfHit(), NumBasesIfHitDataset(), patience=5)
    

    for _, row in query_db(STADIUM_QUERY).iterrows():

        stadiumId = row["stadium_id"]
        print(f"\n\n{row['name']}\n")

        atomicTrainer = ClassifyTrainer(class_labels=["single", "double", "triple", "hr"])
        try:
            atomicTrainer.dojo(NumBasesIfHit(stadiumId=stadiumId), NumBasesIfHitDataset(entityId=stadiumId))
        except ValueError:
            pass


def is_hit():
    
    atomicTrainer = BinaryTrainer(class_labels=["out", "hit"])
    atomicTrainer.dojo(IsHit(), IsHitDataset(), patience=5)
    

    for _, row in query_db(STADIUM_QUERY).iterrows():

        stadiumId = row["stadium_id"]
        print(f"\n\n{row['name']}\n")

        atomicTrainer = BinaryTrainer(class_labels=["out", "hit"])
        try:
            atomicTrainer.dojo(IsHit(stadiumId=stadiumId), IsHitDataset(entityId=stadiumId))
        except ValueError:
            pass


# def is_hr():
    
    
#     for _, row in query_db(STADIUM_QUERY).iterrows():

#         stadiumId = row["stadium_id"]
#         print(f"\n\n{row['name']}\n")

#         atomicTrainer = BinaryTrainer(class_labels=["weak", "HR"])
#         try:
#             atomicTrainer.dojo(IsHR(stadiumId=stadiumId), IsHRDataset(entityId=stadiumId))
#         except ValueError:
#             pass


# def is_swing():
    
#     atomicTrainer = BinaryTrainer(class_labels=["NO", "YES"])
#     for bats in ("R", "L", "S"):
#         atomicTrainer.dojo(IsSwing(entityId=bats), IsSwingDataset(condition=f"batter.bats = '{bats}'"))


# def swing_result():
    
#     atomicTrainer = ClassifyTrainer(class_labels=["swinging strike", "foul ball", "in play"])
#     for bats in ("R", "L", "S"):
#         atomicTrainer.dojo(SwingResult(entityId=bats), SwingResultDataset(condition=f"batter.bats = '{bats}'")) 


def pitch_type_select():

    
    atomicTrainer = ClassifyTrainer(class_labels=PITCH_TYPE_LABELS)
    for throws in ("R", "L"):
        atomicTrainer.dojo(PitchTypeSelect(entityId=throws), PitchTypeDataset(condition=f"pitcher.throws = '{throws}'"), patience=-1) 

    for _, row in query_db(PITCHER_QUERY).iterrows():

        atomicTrainer = ClassifyTrainer(class_labels=PITCH_TYPE_LABELS)
        try:
            atomicTrainer.dojo(PitchTypeSelect(entityId=row['player_id']), PitchTypeDataset(entityId=row['player_id'])) 
        except ValueError:
            pass


if __name__ == "__main__":

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    # num_bases()
    pitch_type_select()
        
    
