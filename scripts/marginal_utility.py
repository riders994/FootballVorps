import nfl_data_py as nfl
import pandas as pd
import json

YEARS = list(range(2012, 2022))
ROMAN_SCORING = {
    'Rec': 1,
    'PassingYds': .04,
    'PassingTD': 6,
    'RushingYds': .1,
    'RushingTD': 6,
    'ReceivingYds': .1,
    'ReceivingTD': 6,
    'Int': -2,
    'Fumbles': 0,
    'FumblesLost': -2
}

COLUMN_FIX = {
    'interceptions': 'Int',
    'passing_tds': 'PassingTD',
    'rushing_tds': 'RushingTD',
    'receiving_tds': 'ReceivingTD',
    'fumbles': 'Fumbles',
    'fumbles_lost': 'FumblesLost',
    'passing_yards': 'PassingYds',
    'rushing_yards': 'RushingYds',
    'receiving_yards': 'ReceivingYds',
    'receptions': 'Rec',
    'games': 'G',
    'player_name': 'Player',
    'team': 'Tm',
    'position': 'Pos',
    'age': 'Age'
}

POSITIONS = {'WR', 'TE', 'RB', 'QB'}
NORM_COLS = [
    'iNorm',
    'pNorm',
    'dNorm',
    'RpNorm',
    'RdNorm',
]
FINAL_COLS = [
    'Player',
    'Tm',
    'PosRating',
    'Pos',
    'Age',
    'G',
    'season',
    'TotalPoints',
    'AvgPoints',
] + NORM_COLS

FULL_COLS = [col for col in FINAL_COLS if col != 'PosRating']


def get_season(year: int) -> pd.DataFrame:
    player_info = nfl.import_seasonal_rosters(year)[[
        'player_id', 'last_name', 'football_name', 'season', 'player_name', 'team', 'position', 'age'
    ]]
    player_stats = nfl.import_seasonal_data(year, 'REG')

    return pd.merge(player_info, player_stats, on=['player_id', 'season'])


def fix_fumbles(season_df: pd.DataFrame) -> None:
    season_df['fumbles'] = season_df['sack_fumbles'] + season_df['receiving_fumbles'] + season_df['rushing_fumbles']
    season_df['fumbles_lost'] = season_df['sack_fumbles_lost'] + season_df['receiving_fumbles_lost'] + season_df['rushing_fumbles_lost']


def load_scores(stat_df: pd.DataFrame, scoring=ROMAN_SCORING) -> pd.DataFrame:
    if isinstance(scoring, str):
        scoring = json.load(scoring)
    elif not isinstance(scoring, dict):
        raise ValueError
    stat_df['TotalPoints'] = sum([stat_df[k] * v for k, v in scoring.items()])
    stat_df['AvgPoints'] = stat_df['TotalPoints']/stat_df['G']
    return stat_df


def normalize(frame, over='TotalPoints'):
    metric = frame[over]
    mu = metric.mean()
    sig = metric.std()
    normed = (metric - mu)/sig
    return normed


def make_norms(frame, over='TotalPoints'):
    frame['iNorm'] = normalize(frame, over)
    res = list()
    for pos in POSITIONS:
        pos_frame = frame[frame['Pos'] == pos].copy().reset_index(drop=True)
        pos_frame['pNorm'] = normalize(pos_frame, over)
        pos_frame['dNorm'] = normalize(pos_frame, 'iNorm')
        res.append(pos_frame)
    res_frame = pd.concat(res, ignore_index=True).reset_index(drop=True)
    res_frame['RpNorm'] = normalize(res_frame, 'pNorm')
    res_frame['RdNorm'] = normalize(res_frame, 'dNorm')
    return res_frame[FULL_COLS]


def make_margins(frame, over='TotalPoints'):
    main_frame = frame.sort_values(by=over)
    main_frame['Marginal_{}'.format(over)] = main_frame[over] - main_frame[over].shift()
    res = list()
    for pos in POSITIONS:
        pos_frame = frame[frame['Pos'] == pos].copy().reset_index(drop=True).sort_values(by=over)
        pos_frame['PosRating'] = [pos + '{}'.format(n) for n in range(pos_frame[over].shape[0], 0, -1)]
        pos_frame = pos_frame[FINAL_COLS]
        pos_frame['PosMarginal_{}'.format(over)] = pos_frame[over] - pos_frame[over].shift()
        for norm in NORM_COLS:
            pos_frame['PosMarginal_{0}_{1}'.format(over, norm)] = pos_frame[norm] - pos_frame[norm].shift()
        res.append(pos_frame)
    res_frame = pd.concat(res).sort_values(by=over)
    res_frame['Marginal_{}'.format(over)] = main_frame['Marginal_{}'.format(over)]
    return res_frame


def _run(ys: int = 2013, ye: int = 2023):
    tot = dict()
    avg = dict()
    for y in range(ys, ye):
        season_frame = get_season([y])
        fix_fumbles(season_frame)
        scores = load_scores(season_frame.rename(columns=COLUMN_FIX))
        normed_totals = make_norms(scores)
        normed_averages = make_norms(scores, 'AvgPoints')
        tot.update({str(y): make_margins(normed_totals)})
        avg.update({str(y): make_margins(normed_averages, 'AvgPoints')})
        print('test')
    return tot, avg

if __name__ == '__main__':
    _run()
