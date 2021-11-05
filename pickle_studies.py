import pickle
from datetime import datetime

import optuna

if __name__ == '__main__':
    db = 'postgresql://postgres:i5gMr!Pfcdm$dn8YqhTf#$hL?jkb@localhost:5432/postgres'

    trials = {
        s.study_name: s.best_trial
        for s in optuna.get_all_study_summaries(db)
    }
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    pickle.dump(trials, open(f'trials-{timestamp}.pickle', 'wb'), protocol=0)
