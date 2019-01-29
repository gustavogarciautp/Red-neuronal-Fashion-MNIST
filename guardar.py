import PIL
import numpy as np
import tensorflow as tf

train_size= 60000
testing_size= 10000

def png_to_np(folder,n):
	l=[]
	for i in range(10):
		for j in range(n):
			I = np.asarray(PIL.Image.open('mfashion/'+folder+'/'+str(i)+'/'+str(i)+' ('+str(j+1)+').png'))
			l.append(tf.constant(I))
	return tf.convert_to_tensor(l)


print("Leyendo imagenes de entrenamiento ...")
train_images=png_to_np('training',train_size)
print("Leyendo imagenes de prueba ...")
test_images = png_to_np('testing',testing_size)

import matplotlib.pyplot as plt

sess= tf.Session()
plt.figure()
plt.imshow(train_images[0].eval(session=sess))
plt.colorbar()
plt.grid(False)
plt.show()

train_images= train_images/255
test_images = test_images/255

plt.figure()
plt.imshow(train_images[0].eval(session=sess),cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# Create some variables.
v1 = tf.Variable(name="train_images",initial_value=train_images)
v2 = tf.Variable(name="test_images",initial_value=test_images)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  save_path = saver.save(sess, "fashion/Preprocesado.ckpt")
  print("Variables guardadas en: %s" % save_path)