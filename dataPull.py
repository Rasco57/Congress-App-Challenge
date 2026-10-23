import pandas as pd
import statsapi
from pybaseball import statcast_batter
from datetime import datetime, timedelta

def getIDparquet(season, sportsID, fileName):
    params = {"season": season, "sportId": sportsID}
    data = statsapi.get("sports_players", params)
    players = data.get("people", [])
    playerID = pd.DataFrame([p.get("id") for p in players], columns=["mlbam_id"])
    playerID.to_parquet(fileName, engine="pyarrow", index=False)

def getID(season, sportsID):
    params = {"season": season, "sportId": sportsID}
    data = statsapi.get("sports_players", params)
    players = data.get("people", [])
    playerID = pd.DataFrame([p.get("id") for p in players], columns=["mlbam_id"])
    return playerID

def getStats(id):
    playerStats = statsapi.player_stat_data(id, group="hitting", type="career")
    return playerStats

def preStat(start, end, MBLAMID):
    a = datetime.now().date() - timedelta(days=end)
    pre = a.strftime("%Y-%m-%d")
    data = statcast_batter(start, pre, player_id=MBLAMID)
    return data

def getPbyP(start, MLBAMID):
    todayDate = datetime.today().date().isoformat()
    data = statcast_batter(start, todayDate, player_id=MLBAMID)
    return data

def countEvents(date, df):
    allEvents = df['events'].unique()
    filteredDf = df.loc[df['game_date'] == date]
    dayCounts = filteredDf['events'].value_counts()
    fullCounts = pd.DataFrame(dayCounts.reindex(allEvents, fill_value=0)).T.rename(index={'count':date})
    return fullCounts