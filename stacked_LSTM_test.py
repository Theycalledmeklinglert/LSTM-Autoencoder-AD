from keras import Sequential
from keras.src.layers import LSTM, Dense
from numpy import array

# Define the stacked LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))  #todo: try ", activation='sigmoid'" -->should be sigmoid by default i believe
model.add(LSTM(50, return_sequences=True))  # Second LSTM layer
model.add(LSTM(50))  # Third LSTM layer, does not return sequences
model.add(Dense(1, activation='sigmoid'))  # Output layer for regression (use appropriate activation for classification tasks)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')  # Use 'binary_crossentropy' for binary classification #standard: mse

# Print model summary to verify the structure
model.summary()

# Example input data
data = array([[0.1] * 100]).reshape((1, 100, 1))  # Single sequence of length 100 with 1 feature

# Example label (for regression)
label = array([0.5])  # Adjust according to the task

# Train the model (in practice, use more data and more epochs)
model.fit(data, label, epochs=10)

# Make and show prediction
print(model.predict(data))
