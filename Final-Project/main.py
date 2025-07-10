import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

train_images_gray = tf.image.rgb_to_grayscale(train_images)
test_images_gray = tf.image.rgb_to_grayscale(test_images)

model = models.Sequential([
    
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    
    # لایه اول (Conv1)
    layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.SpatialDropout2D(0.1),  
    layers.MaxPooling2D((2, 2), strides=(1, 1)),
    
    # لایه دوم (Conv2)
    layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.SpatialDropout2D(0.15), 
    layers.MaxPooling2D((2, 2), strides=(1, 1)),
    
    # لایه سوم (Conv3)
    layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.SpatialDropout2D(0.2),  
    layers.GlobalAveragePooling2D(),
    
    # طبقه‌بندی
    layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(10)
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(train_images_gray, train_labels, 
                    epochs=100, 
                    batch_size=8,
                    validation_data=(test_images_gray, test_labels),
                    verbose=1)


plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch') 
plt.ylabel('Loss')
plt.legend()
plt.show()


test_loss, test_acc = model.evaluate(test_images_gray, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc*100:.2f}%')


model.save('cifar10_model.h5')