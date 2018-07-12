import numpy as np
import pandas as pd
import tensorflow as tf

import sys
from os import environ
from sklearn import preprocessing
import progressbar
from scipy.io import mmread as load

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_gpus', 1, "How many GPUs to use.")

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

k = 10

epochs = 1000

learning_rate = 0.3

train_data = sys.argv[1]
matrix_dir = sys.argv[2]


# Reading dataset

df = pd.read_csv(train_data, sep='\t', names=['user', 'item', 'rating', 'timestamp'], header=None)
df = df.drop('timestamp', axis=1)

users = list(df.user.unique())

# begin = int(sys.argv[3])
# end = int(sys.argv[4])
# users = list(range(begin, end, 1))

num_users = len(users)

# Normalize in [0, 1]

r = df['rating'].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(r.reshape(-1, 1))
df_normalized = pd.DataFrame(x_scaled)
df['rating'] = df_normalized

#config = tf.ConfigProto(allow_soft_placement=True)
config = tf.ConfigProto(device_count={'GPU': 0})

with progressbar.ProgressBar(max_value=num_users) as bar:
    processed_user = 0
    for u in range(0, num_users, FLAGS.num_gpus):
        for gpu in range(FLAGS.num_gpus):

            user = u + gpu
            with tf.device('/cpu:0'):

                matrix = load(matrix_dir + '/{}/matrix.mtx'.format(users[user])).todense().reshape([1, -1]).astype(np.float32)
                mask = load(matrix_dir + '/{}/mask.mtx'.format(users[user])).todense().astype(np.float32)

                with open(matrix_dir + '/{}/features'.format(users[user])) as f:
                    features = list(f.read().splitlines())

                # Model
                num_input = matrix.shape[1]  # num of items
                num_features = mask.shape[1]

                M = tf.placeholder(tf.float32, [None, num_features])

                X = tf.placeholder(tf.float32, [None, num_input])

                W1 = tf.get_variable("w1", shape=[num_input, num_features], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.01))

                hidden = tf.nn.sigmoid(tf.matmul(X, tf.multiply(W1, tf.cast(mask, dtype=tf.float32))))

                W2 = tf.get_variable("w2", shape=[num_features, num_input], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.01))

                output = tf.nn.sigmoid(tf.matmul(hidden, tf.multiply(W2, tf.cast(mask.T, dtype=tf.float32))))

                # Prediction

                y_pred = output

                # Targets are the input data.

                y_true = X

                loss = tf.losses.mean_squared_error(y_true, y_pred)

                grad_W1 = tf.gradients(loss, W1)[0]
                grad_W2 = tf.gradients(loss, W2)[0]

                new_W1 = W1.assign(tf.multiply(W1, M) - learning_rate * grad_W1)
                new_W2 = W2.assign(tf.multiply(W2, tf.transpose(M)) - learning_rate * grad_W2)

                extract_op = tf.matmul(X, W1)

                # Initialize the variables (i.e. assign their default value)
                init = tf.global_variables_initializer()

                with tf.Session(config=config) as session:
                    session.run(init)

                    l = 0.0
                    for e in range(epochs):
                        _, _, l = session.run([new_W1, new_W2, loss], feed_dict={X: matrix, M: mask})
                    print("User: {} - Loss: {}".format(users[user], l))

                    print("User: {} - Extracting features".format(users[user]))

                    values = session.run(extract_op, feed_dict={X: matrix, M: mask})
                    min_max_scaler = preprocessing.MinMaxScaler()
                    values = values.reshape([-1, 1])
                    values = values.ravel()
                    x_scaled = min_max_scaler.fit_transform(values)

                    values = x_scaled.squeeze()

                    up = {}
                    for i, v in enumerate(values):
                        up[features[i]] = v

                    s = [(k, up[k]) for k in sorted(up, key=up.get, reverse=True)]

                    with open("UP/{}.tsv".format(users[user]), "w") as file:
                        for k, v in s:
                            file.write("{}\t{:.16f}\n".format(k, v))

                    processed_user += 1
                    bar.update(processed_user)

                tf.reset_default_graph()
