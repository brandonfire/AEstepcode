from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential([
    Dense(10, activation = "relu"),
    Dense(2, activation = "softmax")])
    

model.compile(
    optimizer = "rmsprop",
    loss = "sparse_categorical_crossentropy")

#model.summary()
print(model.trainable_variables)
print(model.predict([[1, 2], [1, 3], [1, 1]]))