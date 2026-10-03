import numpy as np, yaml, os
from sklearn.model_selection import train_test_split
p = yaml.safe_load(open("params.yaml"))["preprocess"]
d = np.load("data/raw/fashion.npz")
x = d["xtr"].astype("float32")/255.0
xte = d["xte"].astype("float32")/255.0
xtr, xv, ytr, yv = train_test_split(x, d["ytr"], test_size=p["test_size"], random_state=p["seed"])
os.makedirs("data/processed", exist_ok=True)
np.savez("data/processed/data.npz", xtr=xtr, ytr=ytr, xv=xv, yv=yv, xte=xte, yte=d["yte"])
# work in progress
