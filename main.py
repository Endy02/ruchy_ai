from nba.analyzis.nba_pipeline import NbaPipeline
from nba.analyzis.cleaner import Cleaner
from nba.core.predictor import Predictor
import time


# [1610612759 1610612747 1610612744 1610612739 1610612743 1610612740
#  1610612763 1610612754 1610612753 1610612761 1610612751 1610612760
#  1610612748 1610612762 1610612741 1610612737 1610612746 1610612750
#  1610612756 1610612766 1610612738 1610612755 1610612764 1610612749
#  1610612745 1610612742 1610612752 1610612765 1610612758 1610612757]

if __name__ == '__main__':
    
    s_time = time.time()
    # print("---------------- Pipeline execution ----------------")
    # pip_s_time = time.time()
    # maker = NbaPipeline()
    # print(maker.make_nba_pipeline())
    # pip_e_time = time.time()
    # print(f'Execution time : {round(pip_e_time - pip_s_time, 2)} s')
    # print("---------------- END Pipeline  ----------------")
    print("\n")
    # print("---------------- Cleaner execution ----------------")
    # clean_s_time = time.time()
    # cleaner = Cleaner()
    # print(cleaner.get_formated_data())
    # clean_e_time = time.time()
    # print(f'Execution time : {round(clean_e_time - clean_s_time, 2)} s')
    # print("---------------- END Cleaner  ----------------")
    # print("\n")
    print("---------------- Predictor execution ----------------")
    pred_s_time = time.time()
    predictor = Predictor()
    winner = predictor.make_prediction("1610612761", "1610612748")
    pred_e_time = time.time()
    print(f'Execution time : {round(pred_e_time - pred_s_time, 2)} s')
    print("---------------- END Predictor  ----------------")
    e_time = time.time()
    print("\n")
    print("---------------- Total execution ----------------")
    print(f'Execution time : {round(e_time - s_time, 2)} s')