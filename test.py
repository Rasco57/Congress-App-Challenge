import pandas as pd
import dataPull
from pandasDebug import display
import ai
import pybaseball

print("Please enter the first name of the player you would like to analyze:")
firstName=input()
print("Please enter the last name of the player you would like to analyze:")
lastName=input()
playerID=pybaseball.playerid_lookup(lastName, firstName)
playerID=playerID.iat[0, 2]
data=pd.DataFrame(dataPull.getPbyP('2026-03-25', playerID))
data=data[data['events'] != 'NaN']
data=data.loc[data['events'].notna(), ['game_date', 'events']].copy()
data=data.reset_index(drop=True)

uniqueDates = data['game_date'].unique()
uniqueDatesDF = pd.DataFrame(data['game_date'].unique())
dataList = []
tempDF=pd.DataFrame()
for date in uniqueDates:
    tempDF=dataPull.countEvents(date, data)
    tempDF['date']=date
    dataList.append(tempDF)
dataList = pd.DataFrame(pd.concat(dataList, ignore_index=True))

rates = ai.calculate_rates(dataList)
fig = ai.plot_ops(rates)
fig.show()