import pandas as pd
from typing import Optional
from pathlib import Path


ROOT: Path = Path(__file__).resolve().parent.parent.parent

def load_baseline(path: Optional[str] = None) -> pd.DataFrame:
    if not path:
        path = ROOT.joinpath("data/baseline.csv")
    else:
        path = Path(path)

    if not path.exists():
        create_baseline(path)
        print(f"baseline data saved to {path}.")
        
    return pd.read_csv(path)

def create_baseline(path: str) -> None:

    baseline_dict = {
        'model': {
            0: 'xgboost_classification',
            1: 'support_vector_classification',
            2: 'random_forest_classification',
            3: 'neural_network_classification',
            4: 'logistic_regression'
        },
        'model_type': {
            0: 'baseline',
            1: 'baseline',
            2: 'baseline',
            3: 'baseline',
            4: 'baseline'
        },
        'accuracy_min': {
            0: 94.406, 1: 90.21, 2: 95.105, 3: 87.413, 4: 92.308
        },
        'accuracy_max': {
            0: 99.301, 1: 96.902, 2: 100.0, 3: 96.503, 4: 98.601
        },
        'accuracy_avg': {
            0: 97.203, 1: 94.406, 2: 97.902, 3: 92.308, 4: 95.804
        },
        'precision_min': {
            0: 93.673, 1: 90.814, 2: 95.102, 3: 86.51, 4: 91.555
        },
        'precision_max': {
            0: 99.468, 1: 98.182, 2: 100.0, 3: 95.973, 4: 98.576
        },
        'precision_avg': {
            0: 97.002, 1: 94.768, 2: 97.94, 3: 91.635, 4: 95.503
        }
    }

    pd.DataFrame.from_dict(baseline_dict).to_csv(path, index=False)
