import Hyperspectral_input
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as ls
import sys
import time
from sklearn import metrics
import Datasets as D

PCA_BAND = 12


def Attention(x, ch):
    
    GAP = ls.GlobalAveragePooling3D()(x)
    fc_1 = ls.Dense(int(ch / 2), 'relu')(GAP)
    fc_2 = ls.Dense(ch)(fc_1)
    attention = ls.Activation('hard_sigmoid')(fc_2)
    

    Fuse = ls.multiply([x, attention])
    return Fuse


def MSDNSA():
    kind = np.unique(D.gt_data).shape[0] - 1

    inputs = ls.Input(shape=(PCA_BAND, 12, 12, 1))

    conv1 = ls.Conv3D(32, 3, padding='valid')(inputs)
    conv1_bn = ls.BatchNormalization()(conv1)
    conv1_bnac = ls.ReLU()(conv1_bn)
    att_1 = Attention(conv1_bnac, 32)

    conv2 = ls.Conv3D(32, 3, padding='same')(ls.add([conv1_bnac, att_1]))
    conv2_bn = ls.BatchNormalization()(conv2)
    conv2_bnac = ls.ReLU()(conv2_bn)
    att_2 = Attention(conv2_bnac, 32)

    conv3_input = ls.add([att_2, att_1, conv2_bnac, conv1_bnac])
    conv3 = ls.Conv3D(32, 3, padding='same')(conv3_input)
    conv3_bn = ls.BatchNormalization()(conv3)
    conv3_bnac = ls.ReLU()(conv3_bn)
    att_3 = Attention(conv3_bnac, 32)

    conv4_input = ls.add([conv3_bnac, conv2_bnac, conv1_bnac, att_1, att_2, att_3])
    conv4 = ls.Conv3D(128, 3, padding='valid')(conv4_input)
    conv4_bn = ls.BatchNormalization()(conv4)
    conv4_bnac = ls.ReLU()(conv4_bn)

    fc_input = ls.Flatten()(conv4_bnac)

    fc1 = ls.Dense(512, activation='sigmoid')(fc_input)
    fc1_drop = ls.Dropout(0.5)(fc1)
    fc2 = ls.Dense(256, activation='sigmoid')(fc1_drop)
    fc2_drop = ls.Dropout(0.5)(fc2)
    fc3 = ls.Dense(kind, activation='softmax')(fc2_drop)

    model = tf.keras.Model(inputs=inputs, outputs=fc3)

    return model


def accuracy_eval(label_tr, label_pred):
    overall_accuracy = metrics.accuracy_score(label_tr, label_pred)
    avarage_accuracy = np.mean(metrics.precision_score(label_tr, label_pred, average=None))
    kappa = metrics.cohen_kappa_score(label_tr, label_pred)
    cm = metrics.confusion_matrix(label_tr, label_pred)
    return overall_accuracy, avarage_accuracy, kappa, cm


def train(train_batch, train_labels, test_batch, test_labels, all_batch):
    # Train
    model = MSDNSA()

    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    model.fit(x=train_batch, y=train_labels, batch_size=128, epochs=45,validation_data=(test_batch, test_labels), verbose=2)

    # Test
    test_num = test_batch.shape[0]
    predictions = model.predict(test_batch)
    label = np.argmax(test_labels, 1)
    target = np.argmax(predictions, 1)

    OA, AA, Kappa, cm = accuracy_eval(label, target)
    print(cm)
    print("\nall %d samples has the total_accuracy (MSDNSA) :\n OA: %g, AA:%g , Kappa:%g " % (test_num, OA, AA, Kappa))

    # ALL
    pred_all = model.predict(all_batch)
    target_all = np.argmax(pred_all, 1)

    all_num = all_batch.shape[0]
    all_pixels = np.where(D.gt_data != 0)
    restore_pic = np.zeros_like(D.gt_data)

    for k in range(all_num):
        row = all_pixels[0][k]
        col = all_pixels[1][k]
        restore_pic[row, col] = target_all[k] + 1
        
    Hyperspectral_input.dis_groundtruth(restore_pic, ' ')
#    Hyperspectral_input.dis_groundtruth(D.gt_data,' ')   


start = time.perf_counter()

Z, _, _ = D.hyperdate.pca_reduction(PCA_BAND)

(train_pixels, test_pixels) = D.hyperdate.batch_select()

train_batch, train_labels = Hyperspectral_input.get_batchs(Z, train_pixels)
test_batch, test_labels = Hyperspectral_input.get_batchs(Z, test_pixels)
all_batch, all_labels = Hyperspectral_input.get_batchs(Z, D.gt_data)
train(train_batch, train_labels, test_batch, test_labels, all_batch)

end = time.perf_counter()
print("running time is %g min" % ((end - start) / 60))
sys.exit()
