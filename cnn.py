from misc import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score


# citirea tuturor datelor: imagini si etichete de antrenare, validare, testare
train_images, train_labels, validation_images, validation_labels, test_images = read_all(for_CNN=True)

# preprocesare: one-hot encoding pentru label-uri si aducerea valorilor pixelilor in intervalul [0, 1]
train_labels = to_categorical(train_labels, NUM_CLASSES)
validation_labels = to_categorical(validation_labels, NUM_CLASSES)
train_images /= 255
validation_images /= 255
test_images /= 255

# crearea unui model CNN
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

model.fit(
    train_images,
    train_labels,
    batch_size=256,
    epochs=20,
    verbose=2,
    validation_data=(validation_images, validation_labels)
)

# Calculul predictiilor pe datele de validare
# (Folosim urmatoarea linie in loc de model.predict_classes(test_images), care, conform unui avertisment la rulare, este deprecated:)
validation_predictions = np.argmax(model.predict(validation_images), axis=-1)

# readucem label-urile din one-hot encoding in forma numerica, pentru a calcula matricea de confuzie:
validation_labels = [np.where(vector == 1)[0][0] for vector in validation_labels]

accuracy_validation = accuracy_score(validation_labels, validation_predictions)
print(f'Accuracy on validation images: {accuracy_validation}')

validation_confusion_matrix = confusion_matrix(validation_labels, validation_predictions)
print(validation_confusion_matrix)

# calculul predictiilor pe imaginile de test si scrierea lor in fisier
predictions = np.argmax(model.predict(test_images), axis=-1)
write_predictions(
    './submissions/CNN_try10.disasterprevention.txt',
    predictions
)
