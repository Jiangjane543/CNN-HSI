import Hyperspectral_input
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as ls
from matplotlib import pyplot as plt
from sklearn import metrics
import time
import sys
import Datasets as D

PCA_BAND = 12 
    
    
def HSI_2DCNN():

    kind = np.unique(D.gt_data).shape[0]-1
    
    inputs =ls.Input(shape=(10, 10, 1))
    
    conv1 = ls.Conv2D(16, 1, padding='valid')(inputs)
    conv1_bn = ls.BatchNormalization()(conv1)
    conv1_bnac = ls.Activation('relu')(conv1_bn)
    
    conv2 = ls.Conv2D(32, 3, padding='valid')(conv1_bnac)
    conv2_bn = ls.BatchNormalization()(conv2)
    conv2_bnac = ls.Activation('relu')(conv2_bn)

    conv3 = ls.Conv2D(64, 3, padding='valid')(conv2_bnac)
    conv3_bn = ls.BatchNormalization()(conv3)
    conv3_bnac = ls.Activation('relu')(conv3_bn)
    
    conv4 = ls.Conv2D(128, 3, padding='valid')(conv3_bnac)
    conv4_bn = ls.BatchNormalization()(conv4)
    conv4_bnac = ls.Activation('relu')(conv4_bn)
    
    conv5 = ls.Conv2D(256,3, padding='valid')(conv4_bnac)
    conv5_bn = ls.BatchNormalization()(conv5)
    conv5_bnac = ls.Activation('relu')(conv5_bn)

    
    
    
    fc_input = ls.Flatten()(conv5_bnac)
    
    fc1 = ls.Dense(512, activation='sigmoid')(fc_input)
    fc1_drop = ls.Dropout(0.5)(fc1)
    fc2 = ls.Dense(256, activation='sigmoid')(fc1_drop)
    fc2_drop = ls.Dropout(0.5)(fc2)
    fc3 = ls.Dense(kind, activation='softmax')(fc2_drop)
    
    model = tf.keras.Model(inputs=inputs, outputs=fc3)
    
    return model
    
    

def accuracy_eval(label_tr, label_pred):
    overall_accuracy = metrics.accuracy_score(label_tr, label_pred)
    avarage_accuracy = np.mean(metrics.precision_score(label_tr, label_pred, average = None))
    kappa = metrics.cohen_kappa_score(label_tr, label_pred)
    cm = metrics.confusion_matrix(label_tr, label_pred)
    return overall_accuracy, avarage_accuracy, kappa, cm




def train(train_batch, train_labels, test_batch, test_labels, all_batch,  all_labels):
    
    #Train
    model=HSI_2DCNN()
    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    
    model.fit(x=train_batch, y=train_labels, batch_size=128, epochs=125,validation_data=(test_batch, test_labels),verbose=2)
    
    
    #Test
    test_num = test_batch.shape[0]    
    predictions=model.predict(test_batch)
    label = np.argmax(test_labels,1)
    target = np.argmax(predictions,1)

    OA, AA, Kappa,cm = accuracy_eval(label, target)
    print(cm)
    
    #混淆矩阵
    cm1=np.diag(cm)/cm.sum(axis=0)#OA 
    cm1=np.round(cm1,4)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    
    plt.imshow(cm, interpolation='nearest')    
    plt.colorbar()
    plt.axis('off')
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    
    for first_index in range(len(cm)):    
       for second_index in range(len(cm)):  
           if first_index==second_index:  
              plt.text(first_index, second_index,cm1[first_index])
    plt.savefig('cm.png')
    plt.show()
    
    print ("\nall %d samples has the total_accuracy (2DCNN) :\n OA: %g, AA:%g , Kappa:%g " % (test_num, OA, AA, Kappa))
    
    
    #ALL    
    pred_all=model.predict(all_batch)
    target_all = np.argmax(pred_all,1)
    
    all_num = all_batch.shape[0]
    all_pixels = np.where(D.gt_data != 0)   
    restore_pic = np.zeros_like(D.gt_data)
    
    
    for k in range(all_num):
        row = all_pixels[0][k]
        col = all_pixels[1][k]
        restore_pic[row, col] = target_all[k]+1
        
    Hyperspectral_input.dis_groundtruth(restore_pic,' ')    
#    Hyperspectral_input.dis_groundtruth(D.gt_data,' ') 
    
    
         

    
start = time.perf_counter()


norm_data = D.hyperdate.spectrum_normlized()
(train_pixels,test_pixels) = D.hyperdate.batch_select()
(train_batch, train_labels) = Hyperspectral_input.get_2dimbatches(norm_data, train_pixels)
(test_batch, test_labels) = Hyperspectral_input.get_2dimbatches(norm_data, test_pixels)
(all_batch, all_labels) = Hyperspectral_input.get_2dimbatches(norm_data, D.gt_data)

train(train_batch, train_labels, test_batch, test_labels, all_batch, all_labels)

end = time.perf_counter()
print("running time is %g min \n" % ((end-start)/60))
sys.exit()
 

    

















