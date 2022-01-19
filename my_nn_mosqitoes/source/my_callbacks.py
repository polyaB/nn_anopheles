import tensorflow as tf

class weightCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
   print(self.model.trainable_variables)

class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        lr = self.model.optimizer.lr
        print(f'learning rate {lr}')
        tf.summary.scalar('learning rate', data=lr, step=epoch)
