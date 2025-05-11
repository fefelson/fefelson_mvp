import pandas as pd 
from pprint import pprint 

from ..src.database.models.database import get_db_session

pitch_query = """
                SELECT COUNT(at_bat_id) FROM at_bats
                """

ab_query = """
            WITH hr_counts AS (
    SELECT 
        SUM(CASE WHEN contact_type_id = 7 THEN 1 ELSE 0 END) AS hr_count,
        SUM(CASE WHEN contact_type_id != 7 THEN 1 ELSE 0 END) AS non_hr_count,
        COUNT(*) AS total_count
    FROM at_bats AS ab
    INNER JOIN contact_types AS ct 
        ON ab.at_bat_type_id = ct.at_bat_type_id
)
SELECT 
    hr_count,
    non_hr_count,
    total_count,
    (non_hr_count * 1.0 / hr_count) AS pos_weight,  -- For BCE pos_weight
    (total_count * 1.0 / hr_count) AS hr_class_weight,  -- For balanced class weighting
    (total_count * 1.0 / non_hr_count) AS non_hr_class_weight  -- For balanced class weighting
FROM hr_counts;
            """

def run_query(query):

    with get_db_session() as session:
        return  pd.read_sql(query, session.bind)   

        
        


if __name__ == "__main__":
    pprint(run_query(pitch_query))
    
