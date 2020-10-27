import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data

def feature_normalize(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

#hello 
def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += int(size / 2)

def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    print(len(data['timestamp']))
    for (start, end) in windows(data['timestamp'], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if (len(data['timestamp'][start:end]) == window_size):
            print((start, end))
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
    return segments, labels

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')

def apply_depthwise_conv(x, filter_height, filter_width, in_channels, out_channels):
    weights = weight_variable([filter_height, filter_width, in_channels, out_channels])
    biases = bias_variable([in_channels*out_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))

def apply_max_pool(x, filter_height, filter_width, stride_height, stride_width):
    return tf.nn.max_pool2d(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_height, stride_width, 1], padding='VALID')


dataset = read_data('WISDM_ar_v1.1/WISDM_ar_v1.1_900.txt')
dataset['x-axis'] = feature_normalize(dataset['x-axis'])
dataset['y-axis'] = feature_normalize(dataset['y-axis'])
dataset['z-axis'] = feature_normalize(dataset['z-axis'])

segments, labels = segment_signal(dataset)
#打印标签和对应关系
#print(labels)
#print(pd.get_dummies(labels))
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
reshaped_segments = segments.reshape(len(segments), 1, 90, 3)
train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
train_x = reshaped_segments
train_y = labels
#train_x = reshaped_segments[train_test_split]
#train_y = labels[train_test_split]
#test_x = reshaped_segments[~train_test_split]
#test_y = labels[~train_test_split]


realdata = read_data('WISDM_ar_v1.1/WISDM_ar_v1.1_real.txt')
realdata['x-axis'] = feature_normalize(realdata['x-axis'])
realdata['y-axis'] = feature_normalize(realdata['y-axis'])
realdata['z-axis'] = feature_normalize(realdata['z-axis'])

realdata, reallabel = segment_signal(realdata)
reshaped_realdata = realdata.reshape(len(realdata), 1, 90, 3)

batch_size = 10

# 24403 // 10
total_batchs = reshaped_segments.shape[0] // batch_size
# X: (?, 1, 90, 3)
X = tf.placeholder(tf.float32, shape=[None, 1, 90, 3])
# Y: (?, 6)
Y = tf.placeholder(tf.float32, shape=[None, 6])
# conv1: (?, 1, 31, 180)
conv1 = apply_depthwise_conv(X, 1, 60, 3, 60)
# pool1: (?, 1, 6, 180)
pool1 = apply_max_pool(conv1, 1, 20, 1, 2)
# conv2: (?, 1, 1, 1080)
conv2 = apply_depthwise_conv(pool1, 1, 6, 180, 6)

# shape: [None, 1, 1, 1080]
shape = conv2.get_shape().as_list()
# conv2_flat: (?, 1080)
conv2_flat = tf.reshape(conv2, [-1, 1080])
# full_weights: (1080, 1000)
full_weights = weight_variable([1080, 1000])
# full_biases: (1000,)
full_biases = bias_variable([1000])
# full: (?, 1000)
full = tf.nn.tanh(tf.add(tf.matmul(conv2_flat, full_weights),full_biases))
# out_weights: (1000, 6)
out_weights = weight_variable([1000, 6])
# out_biases: (6,)
out_biases = bias_variable([6])
# prediction: (?, 6)
prediction = tf.nn.softmax(tf.matmul(full, out_weights) + out_biases)
# loss: ()
loss = -tf.reduce_sum(Y * tf.math.log(prediction))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.0001).minimize(loss)
# correction_prediction: (?,bool)  tf.argmax(prediction, 1): [[5],[2],...,[0]], tf.argmax(Y, 1): [[0],[2],...,[5]], 
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
# accuracy: ()
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# cost_history: (1, )
cost_history = np.empty(shape=[1], dtype=float)

with tf.compat.v1.Session() as session:
    tf.compat.v1.global_variables_initializer().run()
    for epoch in range(8):
        for b in range(total_batchs):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
        print("Epoch {}: Training Loss = {}, Training Accuracy = {}".format(epoch, c, session.run(accuracy, feed_dict={X: train_x, Y: train_y})))

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, './checkpoint_dir/MyModel')


