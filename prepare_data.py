import pandas as pd
import os
import json

df = pd.read_csv(os.path.join('./data', 'bpi12_complete.csv'))
df.rename({
    'case': 'CaseID',
    'event': 'ActivityName',
    'startTime': 'StartTime',
    'completeTime': 'CompleteTime',
    'AMOUNT_REQ': 'AmountReq',
    'org:resource': 'ResourceID'
}, inplace=True, axis='columns')
df = df.sort_values(by=['StartTime', 'CompleteTime'])
df['StartTime'] = pd.to_datetime(df['StartTime'])
df['CompleteTime'] = pd.to_datetime(df['CompleteTime'])
df['ActivityID'] = ''

prefixes = ['A', 'W', 'O']
activity_map = {}
for pref in prefixes:
    mask = df.ActivityName.str.startswith(pref)
    df[mask] = df[mask].assign(
        ActivityID=(pref + (df[mask]['ActivityName']).astype('category').cat.codes.astype(str)))
    ids = df[mask]['ActivityID'].unique()
    for aid in ids:
        name = df[mask][df[mask]['ActivityID'] == aid].iloc[0]['ActivityName']
        activity_map[aid] = name
with open('./data/activity_map.json', 'w') as f:
    json.dump(activity_map, f)
df['Duration'] = (df['CompleteTime'] - df['StartTime']).view(int) / 1e9  # ns -> s

df.to_csv('./data/bpi12_all.csv', index=False)
prefixes = ['A', 'W', 'O']
for pref in prefixes:
    mask = df.ActivityName.str.startswith(pref)
    df[mask].to_csv(f'./data/bpi12_{pref.lower()}.csv', index=False)
