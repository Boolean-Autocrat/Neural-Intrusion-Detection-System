# import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#loading the dataset and preprocessing the data
df = pd.read_csv('intrusion_detection_data.csv')
df = df.dropna()
X = df.drop('label', axis=1)
y = df['label']
encoder = LabelEncoder()
y = encoder.fit_transform(y)

#spliting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#defining the model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#training the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

#evaluating the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
