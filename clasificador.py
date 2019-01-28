import PIL
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def index_permutation(n,images_array):
	index_p=np.random.permutation(n)
	new_array= np.zeros((n,28,28))
	for i in range(n):
		new_array[i]=images_array[index_p[i]]
	index_p= index_p/(n/10)
	return new_array,index_p.astype(int)


def one_hot(label_array):
	size=len(label_array)
	new_label_array= np.zeros((size,10))
	for i in range (size):
		new_label_array[i][label_array[i]]=1
	return new_label_array


train_size= 60000 #6000
testing_size= 10000 #1000

train_images = tf.get_variable("train_images", shape=[train_size,28,28])
test_images = tf.get_variable("test_images", shape=[testing_size,28,28])

saver = tf.train.Saver()

sess= tf.Session()
saver.restore(sess, "fashion/Preprocesado2.ckpt")

print("Permutando imagenes de entrenamiento ...")
train_images, train_labels= index_permutation(train_size, train_images.eval(session=sess))
print("Permutando imagenes de prueba ...")
test_images, test_labels = index_permutation(testing_size, test_images.eval(session=sess))


print("Conviertiendo etiquetas test")
test_labels= one_hot(test_labels)
print("Conviertiendo etiquetas train")
train_labels = one_hot(train_labels)

class_names = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapato deportivo', 'Bolso', 'Botines']

"""
plt.figure(figsize=(5,5))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i+150], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(train_labels[i+150])])
plt.show()
"""

train_images=train_images.reshape(train_size,784)
test_images=test_images.reshape(testing_size,784)

"""
x = tf.placeholder("float", [None, 784])
W = tf.Variable(name="W",initial_value=tf.zeros([784,10]))
b = tf.Variable(name="b",initial_value=tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs= train_images[i*25:25*(i+1)]
    batch_ys= train_labels[i*25:25*(i+1)]
    #print(batch_ys.shape)
    #print(batch_xs.shape)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print (sess.run(accuracy, feed_dict={x: train_images, y_: train_labels}))


saver = tf.train.Saver()
save_path = saver.save(sess, "fashion/red.ckpt")
"""


tf.reset_default_graph()

# Create some variables.
W = tf.get_variable("W", shape=[784,10])
b = tf.get_variable("b", shape=[10])

x = tf.placeholder("float", [None, 784])


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "fashion/red6.ckpt")
    print("Model restored.")
    #print("v1 : %s" % W.eval())

    y = tf.nn.softmax(tf.matmul(x,W) + b)
    y_ = tf.placeholder("float", [None,10])

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    
    for i in range(2400):
        batch_xs= train_images[i*25:25*(i+1)]
        batch_ys= train_labels[i*25:25*(i+1)]
        #print(batch_ys.shape)
        #print(batch_xs.shape)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print (sess.run(accuracy, feed_dict={x: test_images, y_: test_labels}))
    saver = tf.train.Saver()
    save_path = saver.save(sess, "fashion/red7.ckpt")
    """
    batch_xs= train_images[0:25]
    batch_ys= train_labels[0:25]
    k=sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})
    u=tf.argmax(k, 1).eval()
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(batch_xs[i].reshape(28,28), cmap=plt.cm.binary)
        plt.xlabel(class_names[u[i]])
    plt.show()
    """

