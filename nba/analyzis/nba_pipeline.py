from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline # for prediction
from sklearn.linear_model import LogisticRegression # For data analizis
from sklearn.preprocessing import StandardScaler # for feature scaling
from sklearn.model_selection import train_test_split # for train-test split
import os


class NbaPipeline:
    def __init__(self):
        self.data = self.__format_dataset(os.path.abspath("nba/data") + "/games_formated.csv")
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.__get_features(self.data)

    def make_nba_pipeline(self):
        scaler = self.__st_scaler()
        ml_model = self.__ml_model()
        return make_pipeline(scaler, ml_model)

    def __ml_model(self):
        """
            Initialize Machine learning algorythm
            return LogisticRegression object
        """
        model = LogisticRegression(max_iter = 7000, C = 10, penalty = 'l2', solver = 'lbfgs')
        model.fit(self.X_train, self.Y_train)
        
        return model

    def __st_scaler(self):
        """
            Initial Standard scaler to fit with the machine learning algorythm
        """
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        return scaler

    def __get_features(self, data):
        """
            Create all features needed for the algorythm
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
        X = data[features]
        Y = data['HOME_TEAM_WINS']
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X.values, Y, train_size=0.7, random_state=0, stratify=self.data['HOME_TEAM_WINS'])

        return X_train, X_test, Y_train, Y_test
    
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