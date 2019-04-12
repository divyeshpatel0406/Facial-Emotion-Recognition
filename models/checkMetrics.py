import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import model_from_json

x = np.load("fdataX.npy")
y = np.load("flabels.npy")

# x -= np.mean(x, axis=0)
# x /= np.std(x, axis=0)
# splitting into training, validation and testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=41)


np.save('modXtest', X_test)
np.save('modytest', y_test)

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#loading the model
json_file = open('fer_70.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fer_70.hdf5")
print("Loaded model from disk")


y_pred = loaded_model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(classification_report(y_test, y_pred, target_names=labels))

# # predict crisp classes for test set
# yhat_classes = loaded_model.predict_classes(X_test, verbose=0)
#
#
# # reduce to 1d array
# yhat_classes = yhat_classes.ravel()
# print(yhat_classes)
#
# # accuracy: (tp + tn) / (p + n)
# accuracy = metrics.accuracy_score(y_test, yhat_classes)
# print('Accuracy: %f' % accuracy)
# # precision tp / (tp + fp)
# precision = metrics.precision_score(y_test, yhat_classes)
# print('Precision: %f' % precision)
# # recall: tp / (tp + fn)
# recall = metrics.recall_score(y_test, yhat_classes)
# print('Recall: %f' % recall)
# # f1: 2 tp / (2 tp + fp + fn)
# f1 = metrics.f1_score(y_test, yhat_classes)
# print('F1 score: %f' % f1)
