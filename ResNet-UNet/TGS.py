import tensorflow as tf
from UNet1 import Forward_UNet2
from lovasz_losses_tf import lovasz_hinge
from Data_Processing import training_data_augmentation_, get_data_folds_original
from MIOU import MIOU, Kaggle_MIOU
import numpy as np
from CheckPointMaker import make_checkpoint
import warnings
from ResNet34 import ResNet34, Decoder
from FPA import Feature_Pyramid_Attention

warnings.filterwarnings("ignore")

img_path = r'D:\Information Technology\Deep Learning\Projects\TGS Project\TGS Salt Identification Challenge\Data\Train\images'
mask_path = r'D:\Information Technology\Deep Learning\Projects\TGS Project\TGS Salt Identification Challenge\Data\Train\masks'
depths_path = r'D:/Information Technology/Deep Learning/Projects/TGS Project/TGS Salt Identification Challenge/Data/depths.csv'
check_point_path = r'C:/Users/MSabry/Desktop/TGS Project/ResNet-UNet/Check Point/'
meta_graph_path = r'C:/Users/MSabry/Desktop/TGS Project/ResNet-UNet/Check Point/.meta'
every_ckpt_path = r'C:/Users/MSabry/Desktop/TGS Project/ResNet-UNet/Every Check Point/'
last_ckpt_path = r'C:/Users/MSabry/Desktop/TGS Project/ResNet-UNet/Last Check Point/'
x_train, y_train, x_test, y_test = training_data_augmentation_(img_path, mask_path, depths_path, img_size = 192, padding = 16, n_folds = 6, fold = 1)

x_train = x_train.reshape(-1, 224, 224, 1).astype(np.float32)/255.0
y_train = y_train.astype(np.float32)
x_test = x_test.reshape(-1, 224, 224, 1).astype(np.float32)/255.0
y_test = y_test.astype(np.float32)


epochs = 500
batch_size = 32
train_accuraciess = []
test_accuraciess = []
thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
losses = []
tf.reset_default_graph()

x = tf.placeholder(tf.float32,[batch_size, 224, 224, 1], name =  'x')
y = tf.placeholder(tf.float32,[batch_size, 101, 101], name = 'y')

is_training = tf.placeholder(tf.bool, name = 'is_training')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
decay = tf.placeholder(tf.float32, name = 'decay')

global_step = tf.Variable(0, trainable = False)
LR = tf.Variable(0.0003, trainable = False, name = 'LR')
lr = tf.train.exponential_decay(LR, global_step, 25*len(x_train)//batch_size, 0.95)

#predictions = Forward_UNet2(x, is_training, batch_size, decay, initial_filters = 5, keep_prob = keep_prob, batch_normalize = True)
encoded = ResNet34(x, 5, is_training, decay, True, False)
#mid_layer = Feature_Pyramid_Attention(encoded).FPA()
predictions = Decoder(encoded, is_training , decay , True , False)
predictions = tf.reshape(predictions, [batch_size, 101, 101])
sigmoided = tf.nn.sigmoid(predictions, name = 'Prediction')

loss = tf.losses.hinge_loss(logits = predictions, labels = y)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss, name = 'optimization')
train_op = tf.group([optimizer, update_ops])

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.333
config.gpu_options.allow_growth = True  

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep = 3)
    early_stop = make_checkpoint(check_point_path, every_ckpt_path, last_ckpt_path, saver, sess, await_epochs = 100)

    for epoch in range(epochs):
        
        train_accuracies = []
        ktrain_accuracies = []
        test_accuracies = []
        ktest_accuracies = []
        
        print('epoch ', epoch+1, 'has started out of ', epochs, 'epochs')
        epoch_loss = 0
        start = 0
        end = batch_size
        
        for batch in range(len(x_train)//batch_size):
            
            print('Batch number', batch+1, 'has Started out of', len(x_train)//batch_size, 'Batches', 'of epoch', epoch+1)
            
            ex = x_train[start:end].astype(np.float16)
            ey = y_train[start:end].astype(np.float16)
            start += batch_size
            end += batch_size
            
            _, c= sess.run([train_op, loss], feed_dict = {x: ex, y: ey, is_training:True, keep_prob : 1.0, decay: 0.95})
            
            epoch_loss += c
            
            p = sigmoided.eval(feed_dict = {x: ex, y: ey, is_training:False, keep_prob : 1.0, decay: 0.95}).astype(np.float32)
            ye = y.eval(feed_dict = {x: ex, y: ey, is_training:False, keep_prob : 1.0, decay: 0.95}).astype(np.float32)
#            print('Predictions: ', p)
            accuracy_train = MIOU(p, ye, 0.5, batch_size)
            kaccuracy_train = Kaggle_MIOU(p, ye, thresholds, batch_size)
            train_accuracies.append(accuracy_train)
            ktrain_accuracies.append(kaccuracy_train)
            
        start = 0
        end = batch_size
        for i in range(len(x_test)//batch_size):      
            
            xxx= x_test[start: end]
            yyy= y_test[start: end]
            start += batch_size
            end += batch_size
            
            p = sigmoided.eval(feed_dict = {x: xxx, y: yyy, is_training:False, keep_prob : 1.0, decay: 0.95}).astype(np.float32)
            ye = y.eval(feed_dict = {x: xxx, y: yyy, is_training:False, keep_prob : 1.0, decay: 0.95}).astype(np.float32)
            
            kaccuracy_test = Kaggle_MIOU(p, ye, thresholds, batch_size)
            accuracy_test = MIOU(p, ye, 0.5, batch_size)
            test_accuracies.append(accuracy_test)
            ktest_accuracies.append(kaccuracy_test)
            
        test_acc = np.array(test_accuracies)
        ktest_acc = np.array(ktest_accuracies)
        train_acc = np.array(train_accuracies)
        ktrain_acc = np.array(ktrain_accuracies)
        
        test_acc = np.mean(test_acc.ravel())
        ktest_acc = np.mean(ktest_acc.ravel())
        train_acc = np.mean(train_acc.ravel())
        ktrain_acc = np.mean(ktrain_acc.ravel())
        
        train_accuraciess.append((epoch+1, train_acc, ktrain_acc))
        test_accuraciess.append((epoch+1, test_acc, ktest_acc))
        losses.append(epoch_loss)
        early_stop.add(epoch+1, test_acc)
        print('epoch number ', epoch+1, 'has finished. train accuracy is: ', train_acc, 'kaggle acc is: ',ktrain_acc , 'test accuracy is: ', test_acc,'kaggle test is: ', ktest_acc, 'loss is: ', epoch_loss)
        if early_stop.end_epochs == True:
            break
        
#        print('train accuracy is: ', train_acc, 'kaggle acc is: ',test_acc ,'loss is: ', epoch_loss)#, 'r is: ', k)

    
xs = list(range(len(losses)))
ys = losses
yys = [i[1] for i in train_accuraciess]
yyys = [i[1] for i in test_accuraciess]
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
plt.plot(xs, ys)
#plt.plot(xs, yys, c = 'r')
#plt.plot(xs, yyys, c = 'k')
plt.show()
    
    

#    