import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# بارگذاری داده‌های CIFAR10
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# نرمال‌سازی تصاویر به محدوده [0,1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# تبدیل به تصاویر خاکستری (اگر لازم است)
train_images_gray = tf.image.rgb_to_grayscale(train_images)
test_images_gray = tf.image.rgb_to_grayscale(test_images)

model = models.Sequential([
    # لایه اول (Conv1)
    layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', input_shape=(32, 32, 1)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2), strides=(1, 1)),
    
    # لایه دوم (Conv2)
    layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2), strides=(1, 1)),
    
    # لایه سوم (Conv3)
    layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.AveragePooling2D((2, 2), strides=(1, 1)),
    
    # لایه‌های تمام متصل
    layers.Flatten(),
    layers.Dense(64, activation='relu'),  # FC1
    layers.Dense(10)  # FC2 (10 کلاس خروجی)
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(train_images_gray, train_labels, 
                    epochs=100, 
                    batch_size=32,  # مقدار $ در جدول را با 32 جایگزین کنید
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