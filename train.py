import numpy as np, yaml, os, pandas as pd
from tensorflow import keras
p = yaml.safe_load(open("params.yaml"))["train"]
d = np.load("data/processed/data.npz")
m = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(p["dense_units"], activation="relu"),
    keras.layers.Dropout(p["dropout_rate"]),
    keras.layers.Dense(10, activation="softmax")])
m.compile(optimizer=keras.optimizers.Adam(p["learning_rate"]),
          loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h = m.fit(d["xtr"], d["ytr"], validation_data=(d["xv"], d["yv"]),
          epochs=p["epochs"], batch_size=p["batch_size"])
os.makedirs("models", exist_ok=True)
m.save("models/model.h5")
pd.DataFrame(h.history).to_csv("models/history.csv", index=False)
