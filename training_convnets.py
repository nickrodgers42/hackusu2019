import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, max_pool_2d

import numpy as np
import sys
import cv2
import os
import math

expression_dict = {
    0: 'neutral',
    1: 'anger',
    2: 'contempt',
    3: 'disgust',
    4: 'fear',
    5: 'happy',
    6: 'sadness',
    7: 'surprise'
}

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255

def format_image(image):
    image = to_grayscale(image)
    image = cv2.resize(image, (64, 49))
    return image

def readImages(dirName, images):
    imgArr = []
    for img in images:
        image = format_image(cv2.imread(os.path.join(dirName, img)))
        imgArr.append(image)
        imgArr.append(cv2.flip(image, 1))
    return imgArr

def load_image_data():
    image_dir = 'cohn-kanade-images'
    emotion_dir = 'Emotion'
    emotion_images = []
    emotion_labels = []
    for (root, _, files) in os.walk(emotion_dir):
        for fileName in files:
            with open(os.path.join(root, fileName)) as f:
                emotion = int(float(f.readline()))
            image_path = image_dir + root[len(emotion_dir):]
            image_name = sorted(os.listdir(image_path))[-1]
            images = sorted(os.listdir(image_path))
            images = images[math.floor(len(images) / 2):]
            images = readImages(image_path, images)
            emotion_images.extend(images)
            emotion_array = [0 for _ in range(8)]
            emotion_array[emotion] += 1
            emotion_labels.extend([emotion_array for _ in range(len(images))])
    for dirName in os.listdir('./additionalImages'):
        # for fileName in os.listdir(os.path.join('additionalImages', dirName)):
        emotion_array = [0 for _ in range(8)]
        if (dirName == 'happy'):
            emotion_array[5] += 1
        elif(dirName == 'sad'):
            emotion_array[6] += 1
        elif(dirName == 'angry'):
            emotion_array[1] += 1
        else:
            emotion_array[4] += 1
        images = readImages(os.path.join('./additionalImages', dirName), os.listdir(os.path.join('additionalImages', dirName)))
        emotion_images.extend(images)
        emotion_labels.extend([emotion_array for _ in range(len(images))])                 

    return (emotion_images, emotion_labels)

def shuffleAndSplitData(images, labels):
    images, labels = shuffle(images, labels)
    numImages = len(images)
    numTrain = math.floor(numImages * 0.7)
    numTest = math.floor(numImages * 0.2)
    trainImages = images[:numTrain]
    trainLabels = labels[:numTrain]
    testImages = images[numTrain:numTrain + numTest]
    testLabels = labels[numTrain:numTrain + numTest]
    validImages = images[numTrain + numTest:]
    validLabels = labels[numTrain + numTest:]

    test = cv2.imread('./test.jpg')
    test = format_image(test)
    print(test)
    trainImages = np.concatenate((trainImages, [test]))
    expression_arr = [0 for _ in range(8)]
    expression_arr[6] += 1
    trainLabels = np.concatenate((trainLabels, [expression_arr]))
    
    test2 = cv2.imread('./test2.jpg')
    test2 = format_image(test2)
    trainImages = np.concatenate((trainImages, [test2]))
    expression_arr = [0 for _ in range(8)]
    expression_arr[5] += 1
    trainLabels = np.concatenate((trainLabels, [expression_arr]))

    test3 = cv2.imread('./test3.jpg')
    test3 = format_image(test3)
    trainimages = np.concatenate((trainImages, [test3]))
    expression_arr = [0 for _ in range(8)]
    expression_arr[1] += 1
    trainLabels = np.concatenate((trainLabels, [expression_arr]))

    trainImages = np.array(trainImages)
    testImages = np.array(testImages)
    validImages = np.array(validImages)

    return (trainImages, trainLabels, testImages, testLabels, validImages, validLabels)

def train_emotion_convnet():
    images, labels = load_image_data()
    trainImages, trainLabels, testImages, testLabels, validImages, validLabels = shuffleAndSplitData(images, labels)
    trainImages = trainImages.reshape([-1, 64, 49, 1])
    testImages = testImages.reshape([-1, 64, 49, 1])
    validImages = validImages.reshape([-1, 64, 49, 1])
    
    NUM_EPOCHS = 50
    BATCH_SIZE = 10
    MODEL = build_emotion_convent()
    MODEL.fit(
        trainImages, 
        trainLabels,
        n_epoch=NUM_EPOCHS,
        shuffle=True,
        validation_set=(testImages, testLabels),
        show_metric=True,
        batch_size=BATCH_SIZE,
        run_id='emotion_convnet'
    )
    acc = evaluate_net(MODEL, validImages, validLabels)
    print('Validation accuracy = ' + str(acc))
    save_path = './convnets/EmotionConvnet2.tfl'
    MODEL.save(save_path)

def build_emotion_convent():
    input_layer = input_data(shape=[None, 64, 49, 1])
    conv_layer_1 = conv_2d(input_layer, nb_filter=25, filter_size=3, activation='relu', name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1, nb_filter=30, filter_size=3, activation='relu', name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2, nb_filter=45, filter_size=3, activation='relu', name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    conv_layer_4 = conv_2d(pool_layer_3, nb_filter=50, filter_size=5, activation='relu', name='conv_layer_4')
    pool_layer_4 = max_pool_2d(conv_layer_4, 2, name='pool_layer_4')
    fc_layer_1 = fully_connected(pool_layer_4, 100, activation='relu', name='fc_layer_1')
    dropout_layer = dropout(fc_layer_1, 0.5, name='dropout_layer')
    fc_layer_2 = fully_connected(dropout_layer, 8, activation='softmax', name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.01)
    model = tflearn.DNN(network)
    return model

def evaluate_net(model, data, labels):
    results = []
    for i in range(len(data)):
        prediction = model.predict(data[i].reshape([-1, 64, 49, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == np.argmax(labels[i]))
    return sum((np.array(results) == True)) / len(results)

def load_emotion_convnet(path):
    input_layer = input_data(shape=[None, 64, 49, 1])
    conv_layer_1 = conv_2d(input_layer, nb_filter=25, filter_size=3, activation='relu', name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1, nb_filter=30, filter_size=3, activation='relu', name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2, nb_filter=45, filter_size=3, activation='relu', name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    conv_layer_4 = conv_2d(pool_layer_3, nb_filter=50, filter_size=5, activation='relu', name='conv_layer_4')
    pool_layer_4 = max_pool_2d(conv_layer_4, 2, name='pool_layer_4')
    fc_layer_1 = fully_connected(pool_layer_4, 100, activation='relu', name='fc_layer_1')
    dropout_layer = dropout(fc_layer_1, 0.5, name='dropout_layer')
    fc_layer_2 = fully_connected(dropout_layer, 8, activation='softmax', name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(path)
    return model

def predict_image(model, path):
    img = cv2.imread(path)
    img = to_grayscale(img)
    img = cv2.resize(img, (64, 49))
    prediction = model.predict(np.array(img).reshape([-1, 64, 49, 1]))
    # print(prediction)
    x = [i for i in range(8)]
    # print(x)
    x = [x for _,x in sorted(zip(prediction[0], x), reverse=True)]
    # print(x)
    # for i in x:
    #     print(expression_dict[i])
    prediction = np.argmax(prediction)
    return expression_dict[prediction]

if __name__ == '__main__':
    # train_emotion_convnet()

    tf.reset_default_graph()
    model = load_emotion_convnet('./convnets/EmotionConvnet2.tfl')
    sys.stdout.flush()
    if (len(sys.argv) > 1):
        filePath = sys.argv[1]
        print(predict_image(model, filePath))
        sys.stdout.flush()
    else:
        print(predict_image(model, './test.jpg'))
        print(predict_image(model, './test2.jpg'))
        print(predict_image(model, './test3.jpg'))
        print(predict_image(model, './test4.jpg'))