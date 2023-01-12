# Goals
The goal of the scripts here is to create something that can help optimally prepare athletes for a competition.
## Method
One of the methods I have found is to use a CNN or a RNN to rank the difficulty of a segment in order to determine which segments are critical and which segemnts are not. If I end up using a CNN I will have to make it a sliding window application.
## Models
I can use the maximum lengths that i can set in the segmentation, and then either just zero-pad, or artificially augment the segemnts so that they become of the same size, at which point i can use a CNN
here is some code for this:
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model

# Input layers
coordinates_input = Input(shape=(max_coordinates_length, 2))
heartrate_input = Input(shape=(max_heartrate_length,))
velocity_input = Input(shape=(max_velocity_length,))
cadence_input = Input(shape=(max_cadence_length,))

# Hidden layers
x = LSTM(units=32)(coordinates_input)
y = LSTM(units=32)(heartrate_input)
z = LSTM(units=32)(velocity_input)
w = LSTM(units=32)(cadence_input)

# Concatenate the output of all inputs
concat = concatenate([x, y, z, w])

# Add fully connected layers
concat = Dense(64, activation='relu')(concat)
concat = Dense(32, activation='relu')(concat)

# Output layer
output = Dense(1, activation='sigmoid')(concat)

# Create the model
model = Model(inputs=[coordinates_input, heartrate_input, velocity_input, cadence_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#### another solution
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# Input layer
inputs = Input(shape=(max_data_length,))

# Embedding layer
x = Embedding(vocab_size, embedding_dim, input_length=max_data_length)(inputs)

# RNN layers
x = LSTM(units=32, return_sequences=True)(x)
x = LSTM(units=32)(x)

# Fully connected layers
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)

# Output layer
output = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
## RNN-model with variyng input length
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# Input layer
inputs = Input(shape=(None,))  # variable length input

# Embedding layer
x = Embedding(vocab_size, embedding_dim)(inputs)

# RNN layer
x = LSTM(units=32, return_sequences=True)(x)

# Fully connected output layer
output = Dense(1)(x)

# Create the model
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
