import numpy as np, os
from tensorflow import keras
(xtr, ytr), (xte, yte) = keras.datasets.fashion_mnist.load_data()
os.makedirs("data/raw", exist_ok=True)
np.savez("data/raw/fashion.npz", xtr=xtr, ytr=ytr, xte=xte, yte=yte)
