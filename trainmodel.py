import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Define your variables
actions = ['A', 'B', 'C']  # Actions from A to Z
no_sequences = 15  # Number of sequences per action
sequence_length = 14  # Length of each sequence
DATA_PATH = 'MP_Datas'  # Path to your data directory (adjusted for the extraction structure)

# Create label map
label_map = {label: num for num, label in enumerate(actions)}

# Initialize lists for sequences and labels
sequences = []
labels = []

# Iterate over each action (A to Z)
for action in actions:
    # Iterate over each sequence number (0 to 14)
    for sequence in range(no_sequences):
        # Initialize empty list for the current sequence
        window = []
        # Iterate over each frame number within the sequence length
        for frame_num in range(sequence_length):
            # Construct the path to load the numpy array
            file_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            # Load the numpy array for the current frame
            res = np.load(file_path)
            window.append(res)  # Append the loaded array to the current sequence window
        sequences.append(window)  # Append the sequence window to the list of sequences
        labels.append(label_map[action])  # Append the label index for the current action

# Convert sequences and labels to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Set up TensorBoard callback
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define your LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))  # Output layer with softmax activation

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Print model summary
model.summary()

# Save the model architecture as JSON
model_json = model.to_json()
with open("models.json", "w") as json_file:
    json_file.write(model_json)

# Save the trained weights
model.save('models.h5')
