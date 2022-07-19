import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from loss_functions import dice_coef_loss, tversky_loss
from segmentation_models.dataset.load_dataset import load_dataset_for_training
from segmentation_models.models.fcn_model import fcn_model
from segmentation_models.models.unet_model import unet_model


def train_net(num_epochs=10, batch_size=64, val_subsplits=5, net_model='unet', net_name='unet_dice', loss_type='dice'):
    train_dataset, test_dataset = load_dataset_for_training()
    train_batches = train_dataset.batch(batch_size).repeat()
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    steps_per_epoch = len(train_dataset) // batch_size

    test_batches = test_dataset.batch(batch_size)
    validation_steps = len(test_dataset) // batch_size // val_subsplits
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint('/content/drive/MyDrive/dataset/models/unet_softmax_zero_one.h5', verbose=1,
                        save_weights_only=True)
    ]
    with tf.device('gpu'):
        if net_model == 'unet':
            net_model = unet_model()
        else:
            net_model = fcn_model((256, 256))
        if loss_type=='dice':
            loss_function = [dice_coef_loss]
        else:
            loss_function = [tversky_loss]
        net_model.compile(optimizer=Adam(learning_rate=0.0001), loss=loss_function, metrics=['accuracy'])

        net_model.summary()
        model = net_model.fit(train_batches,
                              epochs=num_epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              validation_data=test_batches,
                              callbacks=callbacks)

        model.save('trained_model/' + net_name + '.h5')