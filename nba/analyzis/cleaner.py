import copy

import numpy as np
import pandas as pd
from scipy import stats  # for sampling
import os


class Cleaner:
    def __init__(self):
        self.ranking_data = self.__ranking_data_cleaning(
            os.path.abspath("nba/data") + "/ranking.csv")
        self.game_data = self.__game_data_cleaning(
            os.path.abspath("nba/data") + "/games.csv")
        
        
    def get_formated_data(self):
        _games = self.game_data[self.game_data['SEASON'] >= 2005]
        games_formated = self.prepare_data(_games)
        games_formated = games_formated.merge(self.game_data[['GAME_ID','GAME_DATE_EST','SEASON','HOME_TEAM_WINS']], on='GAME_ID', how='left')
        games_formated = games_formated.loc[games_formated["SEASON"] >= 2005].reset_index(drop=True)
        games_formated.to_csv(os.path.abspath("nba/data") + "/games_formated.csv", index=False)
        return games_formated
        

    def prepare_data(self, games):
        # Get ranking stats before game
        rank_stats = self.get_team_ranking_before_game(games)
        
        team_info = games[['GAME_ID','HOME_TEAM_ID', 'VISITOR_TEAM_ID','FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
                        'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away',]]
        # Get stats before game 2 previous games
        game_stats_2g = self.get_games_stats_before_game(games, n=2)
        # Get stats before game 4 previous games
        game_stats_4g = self.get_games_stats_before_game(games, n=4)
        # Get stats before game 6 previous games
        game_stats_6g = self.get_games_stats_before_game(games, n=6)
        # Get stats before game 2 previous games
        game_stats_8g = self.get_games_stats_before_game(games, n=8)
        # Get stats before game 10 previous games
        game_stats_10g = self.get_games_stats_before_game(games, n=10)
        # Get stats before game 2 previous games
        game_stats_15g = self.get_games_stats_before_game(games, n=15)
        formated_games = rank_stats.merge(game_stats_2g, on='GAME_ID')
        formated_games = formated_games.merge(game_stats_4g, on='GAME_ID')
        formated_games = formated_games.merge(game_stats_6g, on='GAME_ID')
        formated_games = formated_games.merge(game_stats_8g, on='GAME_ID')
        formated_games = formated_games.merge(game_stats_10g, on='GAME_ID')
        formated_games = formated_games.merge(game_stats_15g, on='GAME_ID')
        formated_games = formated_games.merge(team_info, on='GAME_ID')  
        return formated_games
    
    def get_team_ranking_before_game(self, games):
        _games = games.copy()
        
        def _get_ranking(game):
            date = game['GAME_DATE_EST'].values[0]
            home_team = game['TEAM_ID_home'].values[0]
            away_team = game['TEAM_ID_away'].values[0]
            
            h_rank = self.__get_team_ranking_before_date(home_team, date)
            a_rank = self.__get_team_ranking_before_date(away_team, date)
            
            h_rank.columns += '_home'
            a_rank.columns += '_away'
            
            return pd.concat([h_rank, a_rank], axis=1)
        
        _games = _games.groupby('GAME_ID').apply(_get_ranking)
        _games = _games.reset_index().drop(columns='level_1')
        
        return _games.reset_index(drop=True)

    def get_games_stats_before_game(self, games, n, stats_cols=['PTS','FG_PCT','FT_PCT','FG3_PCT','AST','REB']):
        _games = games.copy()
        
        def _get_stats(game):
            date = game['GAME_DATE_EST'].values[0]
            home_team = game['TEAM_ID_home'].values[0]
            away_team = game['TEAM_ID_away'].values[0]
            
            h_stats = self.__get_games_stats_before_date(home_team, date, n, stats_cols, game_type='all')
            h_stats.columns += '_home_%ig'%n
            h_stats = h_stats.mean().to_frame().T
            
            a_stats = self.__get_games_stats_before_date(away_team, date, n, stats_cols, game_type='all')
            a_stats.columns += '_away_%ig'%n
            a_stats = a_stats.mean().to_frame().T
            
            return pd.concat([h_stats, a_stats], axis=1)
            
            
        _games = _games.groupby('GAME_ID').apply(_get_stats)
        _games = _games.reset_index().drop(columns='level_1')
    
        return _games.reset_index(drop=True)

    def __game_data_cleaning(self, game_link):
        df = pd.read_csv(game_link)
        # Sort games dataframe by date
        df = df.sort_values(by='GAME_DATE_EST').reset_index(drop=True)
        # Drop empty entries, games data before 2003 contains NaN
        df = df.loc[df['GAME_DATE_EST'] >= "2004-10-28"].reset_index(drop=True)
        return df

    def __ranking_data_cleaning(self, ranking_link):
        rf = pd.read_csv(ranking_link)
        rf = self.__format_rankings(rf)
        rf = rf.drop(columns="RETURNTOPLAY")
        rf = rf.sort_values(by='STANDINGSDATE').reset_index(drop=True)
        rf = rf.loc[rf['STANDINGSDATE'] >= "2005-10-28"].reset_index(drop=True)
        return rf

    def __format_record(self, record):
        """
            Split win and loss of home and road record
        """
        w = int(record[0])
        l = int(record[1])
        n = w+l

        if n == 0:
            return np.NaN

        return w / n

    def __format_rankings(self, ranking):
        """
            Format ranking dataset
        """
        home_record = ranking.loc[:, 'HOME_RECORD'].str.split(
            '-').apply(self.__format_record)
        road_record = ranking.loc[:, 'ROAD_RECORD'].str.split(
            '-').apply(self.__format_record)

        ranking.loc[:, 'HOME_RECORD'] = home_record
        ranking.loc[:, 'ROAD_RECORD'] = road_record

        ranking.loc[:, 'SEASON_ID'] = ranking.loc[:,
                                                  'SEASON_ID'].astype(str).str[1:]

        return ranking
    
    def __get_team_ranking_before_date(self, team_id, date, min_games=10):
        """Returned a dataframe with the team id, 
        Number of games played, win percentage, home and road record for
        current and previous season.
        
        Current and previous season are based on the date    
        """
        _ranking = self.ranking_data.loc[self.ranking_data['STANDINGSDATE'] < date]
        _ranking = _ranking.loc[_ranking['TEAM_ID'] == team_id]

        if _ranking.tail(1)['G'].values < min_games:
            _ranking = _ranking.loc[_ranking['SEASON_ID']
                                    < _ranking['SEASON_ID'].max()]

        _prev_season = _ranking.loc[_ranking['SEASON_ID']
                                    < _ranking['SEASON_ID'].max()]
        _prev_season = _prev_season.loc[_prev_season['STANDINGSDATE']
                                        == _prev_season['STANDINGSDATE'].max()]

        _current_season = _ranking[_ranking['STANDINGSDATE']
                                == _ranking['STANDINGSDATE'].max()]

        _current_season = _current_season[['TEAM_ID', 'G', 'W_PCT']]
        _prev_season = _prev_season[['TEAM_ID', 'G', 'W_PCT']]

        return _current_season.merge(_prev_season, on='TEAM_ID', suffixes=('', '_prev')).drop(columns='TEAM_ID')
    
    def __get_games_stats_before_date(self, team_id, date, n, stats_cols, game_type='all'):
        """
        """
        
        if game_type not in ['all','home','away']:
            raise ValueError('game_type must be all, home or away')
        
        _games = self.game_data.loc[self.game_data['GAME_DATE_EST'] < date]
        _games = _games.loc[(_games['TEAM_ID_home'] == team_id) | (_games['TEAM_ID_away'] == team_id)]
        
        _games.loc[:,'is_home'] = _games['TEAM_ID_home'] == team_id
        
        if game_type == 'home':
            _games = _games.loc[_games['is_home']]
            
        elif game_type == 'away':
            _games = _games.loc[~_games['is_home']]
            
        _games.loc[:,'WIN_PRCT'] = _games['is_home'] == _games['HOME_TEAM_WINS']
        
        for col in stats_cols:
            _games.loc[:,col] = np.where(_games['is_home'], _games['%s_home'%col], _games['%s_away'%col])
        
        cols = ['WIN_PRCT'] + stats_cols
        
        if len(_games) < n:
            return _games[cols]
        
        return _games.tail(n)[cols]
