import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
d = np.load("data/processed/data.npz")
m = keras.models.load_model("models/model.h5")
loss, acc = m.evaluate(d["xte"], d["yte"], verbose=0)
pred = m.predict(d["xte"], verbose=0).argmax(1)
ConfusionMatrixDisplay(confusion_matrix(d["yte"], pred)).plot()
plt.savefig("confusion_matrix.png")
json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, open("metrics.json", "w"))
