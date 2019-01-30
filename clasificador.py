import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

train_size= 60000
testing_size= 10000

train_images = tf.get_variable("train_images", shape=[train_size,28,28])
test_images = tf.get_variable("test_images", shape=[testing_size,28,28])

saver = tf.train.Saver()

sess= tf.Session()
saver.restore(sess, "fashion/Preprocesado2.ckpt")

def index_permutation(images_array):
    n=len(images_array)
    index_p=np.random.permutation(n)
    new_array= np.zeros((n,28,28))
    for i in range(n):
        new_array[i]=images_array[index_p[i]]
    index_p= index_p/(n/10)
    return new_array,index_p.astype(int)

print("Permutando imagenes de entrenamiento ...")
train_images, train_labels= index_permutation(train_images.eval(session=sess))
print("Permutando imagenes de prueba ...")
test_images, test_labels = index_permutation(test_images.eval(session=sess))

def one_hot(label_array):
    size=len(label_array)
    new_label_array= np.zeros((size,10))
    for i in range (size):
        new_label_array[i][label_array[i]]=1
    return new_label_array

print("Conviertiendo etiquetas test")
test_labels= one_hot(test_labels)
print("Conviertiendo etiquetas train")
train_labels = one_hot(train_labels)


class_names = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapato deportivo', 'Bolso', 'Botines']

plt.figure(figsize=(5,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(train_labels[i])])
plt.show()

train_images=train_images.reshape(train_size,784)
test_images=test_images.reshape(testing_size,784)

###########----------------Entrenar modelo----------------#################

"""
x = tf.placeholder("float", [None, 784])
W = tf.Variable(name="W",initial_value=tf.zeros([784,10]))
b = tf.Variable(name="b",initial_value=tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2400):
    batch_xs= train_images[i*25:25*(i+1)]
    batch_ys= train_labels[i*25:25*(i+1)]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print (sess.run(accuracy, feed_dict={x: train_images, y_: train_labels}))


saver = tf.train.Saver()
save_path = saver.save(sess, "fashion/red.ckpt")
"""

##############----------Modelo ya entrenado-----------###############


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  true_label = np.argmax(true_label)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({} {:2.0f}%) ".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label], 100*np.max(predictions_array[true_label])),
                                color=color)
  #plt.show()

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  true_label= np.argmax(true_label)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  _ = plt.xticks(range(10), class_names, rotation=90)
  #plt.show()


tf.reset_default_graph()

# Create some variables.
W = tf.get_variable("W", shape=[784,10])
b = tf.get_variable("b", shape=[10])

x = tf.placeholder("float", [None, 784])
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "fashion/red6.ckpt")
    print("Model restored.")

    n=10  #interaciones
    m=6   #tamaño del lote
    for j in range(n):
        batch_xs= test_images[m*j:m*(j+1)]
        batch_ys= test_labels[m*j:m*(j+1)]
        u=sess.run(y, feed_dict={x: batch_xs})
        num_rows = 2
        num_cols = 3
        num_images = num_rows*num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for i in range(num_images):
          plt.subplot(num_rows, 2*num_cols, 2*i+1)
          plot_image(i, u, batch_ys, batch_xs.reshape(m,28,28))
          plt.subplot(num_rows, 2*num_cols, 2*i+2)
          plot_value_array(i, u, batch_ys)
        plt.show()
