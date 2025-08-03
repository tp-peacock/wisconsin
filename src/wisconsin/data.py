import urllib.request
import zipfile
from pathlib import Path
from typing import Optional,List
import pandas as pd

ROOT: Path = Path(__file__).resolve().parent.parent.parent

COLUMN_NAMES: List[str] = [
    "ID", "Diagnosis", "radius1", "texture1", "perimeter1", "area1", "smoothness1",
    "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2", "smoothness2", "compactness2", "concavity2",
    "concave_points2", "symmetry2", "fractal_dimension2", "radius3", "texture3", "perimeter3",
    "area3", "smoothness3", "compactness3", "concavity3", "concave_points3", "symmetry3",
    "fractal_dimension3"
]

def download(unzip: bool = True):
    url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
    data_dir = ROOT.joinpath("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir.joinpath("breast_cancer_wisconsin_diagnostic.zip")
    urllib.request.urlretrieve(url, zip_path)

    if unzip:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Extracted contents to {data_dir}")



def load(path: Optional[str] = None) -> pd.DataFrame:
    if not path:
        path = ROOT.joinpath("data/wdbc.data")
    else:
        path = Path(path)

    if not path.exists():
        download()
        
    return pd.read_csv(path, header=None, names=COLUMN_NAMES)


if __name__ == "__main__":
    download(unzip=True)
    df = load()
