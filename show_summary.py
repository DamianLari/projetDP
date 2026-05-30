import tensorflow as tf
m = tf.keras.models.load_model('tune_results/model_trial_07.keras')
m.summary()
