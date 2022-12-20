import cv2
import numpy as np
import os
import time
startTime = time.time()

training = ['imagedb_btsd/imagedb']
testing = ['imagedb_btsd/imagedb_test']
sift = cv2.xfeatures2d_SIFT.create()


# sift
def descriptor(path):
    img = cv2.imread(path)
    _, d = sift.detectAndCompute(img, None)
    return d


# encoding
def encoding(desc, vocabulary):
    bow_desc = np.zeros((1, vocabulary.shape[0]))
    for x in range(desc.shape[0]):
        distances = np.sum((desc[x, :] - vocabulary) ** 2, axis=1)
        mini = np.argmin(distances)
        bow_desc[0, mini] += 1
    return bow_desc / np.sum(bow_desc)


# descriptors for training
def train(training_path):
    traindesc = np.zeros((0, 128))
    for folder in training_path:
        for i in os.scandir(folder):
            for fl in os.listdir(i):
                path = os.path.join(i, fl)
                train_img_desc = descriptor(path)
                if train_img_desc is None:
                    continue
                traindesc = np.concatenate((traindesc, train_img_desc), axis=0)
    select = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
    trainer = cv2.BOWKMeansTrainer(60, select, 1, cv2.KMEANS_PP_CENTERS)
    vocab = trainer.cluster(traindesc.astype(np.float32))
    return vocab


# bag of visual words
def bovw(diction, train_path):
    img_directories = []
    bowdescs = np.zeros((0, diction.shape[0]))
    for fldr in train_path:
        for x in os.scandir(fldr):
            for file in os.listdir(x):
                paths = os.path.join(x, file)
                desc = descriptor(paths)
                if desc is None:
                    continue
                bow_desc1 = encoding(desc, diction)
                img_directories.append(paths)
                bowdescs = np.concatenate((bowdescs, bow_desc1), axis=0)
    return bowdescs, img_directories


def classes_num(train_path):
    class_count = []
    count = 0
    class_number = []
    for folder in train_path:
        for subfolder in os.listdir(folder):
            count = count + 1
            path = os.path.join(folder, subfolder)
            class_count.insert(count, path)
            class_number.insert(count, subfolder)
    class_count = np.array(class_count)
    return class_count, class_number


svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_INTER)
svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))


def svmtrain(bowdesc, classnum, paths):
    if os.path.isdir('SVMs') == True:
        for b in range(len(classnum)):
            labels = np.array([classnum[b] in a for a in paths], np.int32)
            svm.trainAuto(bowdesc.astype(np.float32), cv2.ml.ROW_SAMPLE, labels)
            svm.save('SVMs/svm' + str(classnum[b]))
    else:
        os.mkdir('SVMs')
        for b in range(len(classnum)):
            labels = np.array([classnum[b] in a for a in paths], np.int32)
            svm.trainAuto(bowdesc.astype(np.float32), cv2.ml.ROW_SAMPLE, labels)
            svm.save('SVMs/svm' + str(classnum[b]))
    return


dictionary = train(training)
# np.save('dictionary.npy', dictionary)
# dictionary = np.load('dictionary.npy')

bow_descs, img_paths = bovw(dictionary, training)
test_bovw, test_paths = bovw(dictionary, testing)
# np.save('index.npy', bow_descs)
# bow_descs = np.load('index.npy')

classes, classnumber = classes_num(training)
test_classes, testnums = classes_num(testing)

svmtrain(bow_descs, classnumber, img_paths)
svms = []
for svm_file in os.listdir('SVMs'):
    svms.append(svm.load('SVMs/' + str(svm_file)))

correct = 0
class_success = np.zeros((1, len(classes)))
for j in range(test_bovw.shape[0]):
    query = test_bovw[j]
    query = np.expand_dims(query, axis=1)
    query = np.transpose(query)
    predictions = []
    for class_svm in svms:
        response = class_svm.predict(query.astype(np.float32), flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
        guess = response[1]
        predictions.append(guess)
    min_idx = np.argmin(predictions)
    for num in classnumber:
        if (num in classes[min_idx]) & (num in test_paths[j]):
            idx = classnumber.index(num)
            correct += 1
            class_success[:, idx] += 1
success_rate = correct / len(test_paths)
success_rate_byclass = np.array(class_success)


print('Using SVM,the successful predictions are', correct, 'images out of', len(test_paths),
      'images,while the success rate is', (success_rate * 100).__round__(2), '%')
for z in range(len(classes)):
    success_rate_byclass[:, z] = class_success[:, z] / len(os.listdir(test_classes[z]))
    print('Successful Prediction Rate,using SVM classification for class', classnumber[z], 'is ',
          np.round((success_rate_byclass[:, z] * 100), 2), '%')
