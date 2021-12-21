import tensorflow as tf
from tensorflow.keras.models import load_model


#initialize denseNet model with None = random weights
model = tf.keras.applications.DenseNet121(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
)

#Gewichte ausgeben lassen
print(model.get_weights())

#model im TF = Tensorflow -Format (= .pb Format abspeichern, führt zur Fehlermeldung)
#model.save('mymodel' ,save_format='tf')

#sollte model per Default eigentlich auch im TF Format abspeichern, führt aber zu hbf Ordner?)
model.save('densenet121_model')

#Gewichte abspeichern
model.save_weights('densenet121_random.h5')

#Gewichte wieder in Model zu laden scheint aber zu funktionieren
model.load_weights('denseNet121_random.h5')
print(model.get_weights())






