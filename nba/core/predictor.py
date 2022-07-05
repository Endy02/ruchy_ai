import copy
from copyreg import pickle
from dis import dis  # for copying nested dictionaries
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from collections import Counter
from fitter import Fitter  # for fitting the best distribution
from nba.analyzis.nba_pipeline import NbaPipeline
from nba.analyzis.nbaencorder import NBAEncoder
from sklearn import metrics # for evaluation
# for Generator
from scipy import stats  # for sampling


class Predictor:
    """
        Predictor class
    """
    
    
    """
        Prediction pipeline
    """
    maker = NbaPipeline()
    """
        Probabilities generator distribution
    """
    selected_distributions = [
        'norm','t', 'f', 'chi', 'cosine', 'alpha', 
        'beta', 'gamma', 'dgamma', 'dweibull',
        'maxwell', 'pareto', 'fisk', 'levy', 'moyal',
        'rayleigh']
    GEN = {
            'alpha': stats.alpha.rvs,
            'beta': stats.beta.rvs,
            'chi': stats.chi.rvs,
            'cosine': stats.cosine.rvs,
            'dgamma': stats.dgamma.rvs,
            'dweibull':stats.dweibull.rvs,
            'f':stats.f.rvs,
            'fisk':stats.fisk.rvs,
            'gamma': stats.gamma.rvs,
            'maxwell':stats.maxwell.rvs,
            'norm':stats.norm.rvs,
            'pareto':stats.pareto.rvs,
            't':stats.t.rvs,
            'levy': stats.levy.rvs,
            'moyal': stats.moyal.rvs,
            'rayleigh': stats.rayleigh.rvs
        }
    
    """
        Predictions features
    """
    features = [
            'G_home', 'G_away',
            'W_PCT_home', 'W_PCT_prev_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
            'W_PCT_away', 'W_PCT_prev_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away',
            'WIN_PRCT_home_2g', 'FG_PCT_home_2g', 'FT_PCT_home_2g', 'FG3_PCT_home_2g', 'AST_home_2g', 'REB_home_2g',
            'WIN_PRCT_away_2g', 'FG_PCT_away_2g', 'FT_PCT_away_2g', 'FG3_PCT_away_2g', 'AST_away_2g', 'REB_away_2g',
            'WIN_PRCT_home_4g', 'FG_PCT_home_4g', 'FT_PCT_home_4g', 'FG3_PCT_home_4g', 'AST_home_4g', 'REB_home_4g',
            'WIN_PRCT_away_4g', 'FG_PCT_away_4g', 'FT_PCT_away_4g', 'FG3_PCT_away_4g', 'AST_away_4g', 'REB_away_4g',
            'WIN_PRCT_home_6g', 'FG_PCT_home_6g', 'FT_PCT_home_6g', 'FG3_PCT_home_6g', 'AST_home_6g', 'REB_home_6g',
            'WIN_PRCT_away_6g', 'FG_PCT_away_6g', 'FT_PCT_away_6g', 'FG3_PCT_away_6g', 'AST_away_6g', 'REB_away_6g',
            'WIN_PRCT_home_8g', 'FG_PCT_home_8g', 'FT_PCT_home_8g', 'FG3_PCT_home_8g', 'AST_home_8g', 'REB_home_8g',
            'WIN_PRCT_away_8g', 'FG_PCT_away_8g', 'FT_PCT_away_8g', 'FG3_PCT_away_8g', 'AST_away_8g', 'REB_away_8g',
            'WIN_PRCT_home_10g', 'FG_PCT_home_10g', 'FT_PCT_home_10g', 'FG3_PCT_home_10g', 'AST_home_10g', 'REB_home_10g',
            'WIN_PRCT_away_10g', 'FG_PCT_away_10g', 'FT_PCT_away_10g', 'FG3_PCT_away_10g', 'AST_away_10g', 'REB_away_10g',
            'WIN_PRCT_home_15g', 'FG_PCT_home_15g', 'FT_PCT_home_15g', 'FG3_PCT_home_15g', 'AST_home_15g', 'REB_home_15g',
            'WIN_PRCT_away_15g', 'FG_PCT_away_15g', 'FT_PCT_away_15g', 'FG3_PCT_away_15g', 'AST_away_15g', 'REB_away_15g'
        ]
        
    def __init__(self, random_state = None):
        self.random_state = random_state # keep this to None for making simulations
        self.teams = pd.read_csv(os.path.abspath("nba/data") + "/teams.csv")
        self.data = self.__format_dataset(os.path.abspath("nba/data") + "/games_formated.csv")
        self.pipeline = self.maker.make_nba_pipeline()
        self.all_team_sim_data = self.__format_all_team_sim_data(self.data)
        # self.create_mega_data(self.all_team_sim_data)
        self.big_data = self.get_mega_data()
    
    def make_prediction(self, home, away):
        """
            Make predictions purcentage based on N games 
        """
        home_team_name = self.teams.loc[self.teams['TEAM_ID'] == int(home)]['TEAM_NAME'].values
        away_team_name = self.teams.loc[self.teams['TEAM_ID'] == int(away)]['TEAM_NAME'].values
        
        home_win = 0
        away_win = 0
        games = 100 # !important : simulation number
        
        for i in range(0, games):
            win_loss = self.__predict(home, away)
            if list(win_loss)[0] == 1:
                home_win += 1
            else:
                away_win += 1
        pct_home = round(home_win / games * 100, 2)
        pct_away = round(away_win / games * 100, 2)
        print(f"Home Team | {home_team_name} : {pct_home} %")       
        print(f"Away Team | {away_team_name} : {pct_away} %")       
        
    def __predict(self, homeTeam, awayteam, num_games = 1):
        """
            Predict win or loss of n game(s) played by two tems
        """
        home_team_feature_data = self.big_data[homeTeam]
        away_team_feature_data = self.big_data[awayteam]
        new_features = []
                    
        for feature_paras_1 in home_team_feature_data:
            sample_1 = self.sampling(feature_paras_1, size= num_games)
            new_features.append(sample_1)
            
        for feature_paras_2 in away_team_feature_data:
            sample_2 = self.sampling(feature_paras_2, size= num_games)
            new_features.append(sample_2)
        
        new_features = np.array(new_features).T
        win_loss = self.pipeline.predict(new_features)

        return list(win_loss) # a list of win/loss from num_games
    
    
    def sampling(self, dic, size, random_state = None):
        """
            generate feature values used for making win/loss prediction
        """         
        dis_name = list(dic.keys())[0] # get the type
        paras = list(dic.values())[0] # get the paras
        
        # Generate sample
        sample = self.GEN[dis_name](**paras, size=size, random_state = random_state)
        return sample 
    
    
    def get_mega_data(self):
        f = open(os.path.abspath("nba/data") + "/clean_format.json")
        data = json.load(f)
        return data
    
    def create_mega_data(self, all_team_sim_data):
        """
            Format mega data for predictions
        """
        # TODO : Improve probabilities generator
        unique_teams = self.data['HOME_TEAM_ID'].unique()
        megadata = {} # store the data that our Generator will rely on
        for team_name in unique_teams:
            
            feature_dis_paras = []
            data = all_team_sim_data[team_name]
            
            # All features for each team
            for i in range(np.size(data, 1)): 
                f = Fitter(data[:, i]) # initalize a Fitter instance
                f.distributions = self.selected_distributions # use only the selected distributions
                f.fit() # do the fitting 
                best_paras = f.get_best(method='sumsquare_error') # get the best fitted params
                feature_dis_paras.append(best_paras)
                
            megadata[str(team_name)] = feature_dis_paras
        
        with open(os.path.abspath("nba/data") + "/mega_data.json", "w+") as f:
            json.dump(megadata, f, indent=4)
        
    
    def __format_all_team_sim_data(self, data):
        unique_teams = data['HOME_TEAM_ID'].unique()
        sim_data = {}
        for team_name in unique_teams:
    
            # find games where the team is either at home or away
            df_team = data.loc[(data['HOME_TEAM_ID'] == team_name) | (data['VISITOR_TEAM_ID'] == team_name)]
            # it is home team, select all features
            df_1 = df_team.loc[df_team['HOME_TEAM_ID'] == team_name][self.features]
            # it is away team, select all features
            df_0 = df_team.loc[df_team['VISITOR_TEAM_ID'] == team_name][self.features]

            # combine them
            df_0.columns = df_1.columns # before concating, match the column names
            df_s = pd.concat([df_1, df_0], axis = 0)
                      
            # convert the pandas.DataFrame to numpy array
            sim_data[team_name] = df_s.to_numpy()

        return sim_data
    
    
    def __format_dataset(self, link):
        """
            Format dataset to take NBA games since 2007
            return: Pandas dataset
        """
        df = pd.read_csv(link)
        df = df.sort_values(by='GAME_DATE_EST').reset_index(drop=True)
        date = datetime.today()
        if date < datetime.strptime(str(date.year)+"-10-28","%Y-%m-%d"):
            current_year = date.year - 1
        else:
            current_year = date.year

        df = df.loc[df["GAME_DATE_EST"] >= str(current_year) + "-10-28"].reset_index(drop=True)
        
        return df
