import tensorflow as tf
import keras
import os

# Set parameters
mnist = keras.datasets.mnist
batch_size = 128
num_class = 10
epochs = 10

# Load dataset
(train_data, train_label), (test_data, test_label) = mnist.load_data()

# Reshaping input data
train_data = train_data.reshape(train_data.shape[0], 
                                (train_data.shape[1]*train_data.shape[2]))
test_data = test_data.reshape(test_data.shape[0],
                              (test_data.shape[1]*test_data.shape[2]))

# Data Normalization
train_data = train_data / 255.0
test_data = test_data / 255.0

def prepare():
    if not os.path.isdir('keras'):
        os.mkdir('keras')
        print('keras folder model created!')
    else:
        print('keras folder model already exist!')

    if not os.path.isdir('tf'):
        os.mkdir('tf')
        print('tf folder model created!')
    else:
        print('tf folder model already exist!')
    
    return True


# Build Model Architecture
# Use tf.keras.models.Sequential instead of keras.Sequential
# When saving model tf.keras.models.Sequential produce 3 files
# - checkpoint
# - model_name.ckpt.data-0000-of-00001
# - model_name.ckpt.index
####
# When saving model with keras.Sequential, it will produce 1 file only
# - model_name.ckpt
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(num_class, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.train.AdamOptimizer(),
                metrics=['accuracy'])

    return model

prepare()
model = create_model()
model.summary()

# Create checkpoint callback
# Model with keras based
checkpoint_path = "keras/model.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(train_data, train_label,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(test_data, test_label),
                    callbacks=[cp_callback])

# Model with TF Based
sess = tf.keras.backend.get_session()
saver = tf.train.Saver()
save_path = saver.save(sess, "tf/model.ckpt")

# Evaluation
score = model.evaluate(test_data, test_label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])