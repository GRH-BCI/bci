import pickle
from pathlib import Path

import optuna

if __name__ == '__main__':
    out_path = Path('.')
    out_path.mkdir(parents=True, exist_ok=True)

    db = 'postgresql://postgres:i5gMr!Pfcdm$dn8YqhTf#$hL?jkb@localhost:5432/postgres'
    study_summaries = optuna.get_all_study_summaries(db)

    for study_summary in study_summaries:
        pickle.dump(study_summary.best_trial, open(out_path / f'{study_summary.study_name}.pickle', 'wb'))
