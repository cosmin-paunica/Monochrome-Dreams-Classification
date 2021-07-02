from misc import *
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import confusion_matrix, accuracy_score


# citirea tuturor datelor: imagini si etichete de antrenare, validare, testare
train_images, train_labels, validation_images, validation_labels, test_images = read_all()

# preprocesarea imaginilor
transformer = StandardScaler()
transformer.fit(train_images)
train_images = transformer.transform(train_images)
validation_images = transformer.transform(validation_images)
test_images = transformer.transform(test_images)

# definirea modelului: clasificatorul liniar SVC
model = SVC(C=100)

# antrenarea modelului si calculul predictiilor pe datele de validare
model.fit(train_images, train_labels)
validation_predictions = model.predict(validation_images)
accuracy_validation = accuracy_score(validation_labels, validation_predictions)
print(f'Accuracy on validation images: {accuracy_validation}')
validation_confusion_matrix = confusion_matrix(validation_labels, validation_predictions)
print(validation_confusion_matrix)

# calculul predictiilor pe imaginile de test si scrierea lor in fisier
predictions = model.predict(test_images)
write_predictions(
    './submissions/SVC_C=100_StandardScaler.txt',
    predictions
)
