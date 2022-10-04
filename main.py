import matplotlib as plt
from sklearn.datasets import make_blobs
from tensorflow import keras
from keras import layers 
from keras import models 
from keras import optimizers 

n = 1000
classes = 3
dimensions = 2

train_size = int(n * 0.8)

X,y = make_blobs(
    n_samples=n,
    centers=classes,
    n_features=dimensions,
    cluster_std = 2,
    random_state =42
)

X_train,X_test = X[:train_size,:], X[train_size:, :]
y_train, y_test = y[:train_size], y[train_size:]

def fit_model(batch_size):
    model = models.Sequential([
        layers.Dense(32,input_dim=dimensions, activation='relu'),
        layers.Dense(classes, activation='softmax'),
    ])

    model.compile(
        optimizers = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=batch_size
    )

    return model, history


def evaluate(model, history):
    _, train_accuracy = model.evaluate(X_train, y_train)
    _, test_accuracy = model.evaluate(X_test, y_test)

    print(f"Training accuracy: {train_accuracy:2f}")
    print(f"Test accuracy: {test_accuracy:2f}")

    plt.figure(figsize=(6, 4), dpi = 160)

    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Test loss")
    plt.legend()
    plt.show()

# Experiment 1 - Stochastic Gradient Descent (batch_size=1) i.e one sample at a time
%%timeit -n 1 -r 1
model, history = fit_model(batch_size=1)
evaluate(model, history)


# Experiment 2 - Batcg Gradient Descent (batch_size=n) i.e using all data at once
%%timeit -n 1 -r 1
model, history = fit_model(batch_size=train_size)
evaluate(model, history)

# Experiment 3 - Mini-Batch Gradient Descent (batch_size=32) i.e some data at once
%%timeit -n 1 -r 1
model, history = fit_model(batch_size=32)
evaluate(model, history)

