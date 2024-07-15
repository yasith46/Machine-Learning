import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Reading data and processing
ds_path = '/kaggle/input/digit-recognizer/train.csv'
ds = pd.read_csv(ds_path)

y_ds = ds.pop('label')
y_ds = y_ds.values.reshape(-1)
x_ds = ds


# Splitting test-validation set
x_ds_train, x_ds_val, y_ds_train, y_ds_val = train_test_split(x_ds, y_ds, test_size=0.25, random_state=1)

# Model
model2 = keras.Sequential([
    layers.Dense(784, activation = 'sigmoid', input_shape=[784]),
    layers.Dropout(rate = 0.3),
    
    layers.Dense(256, activation = 'sigmoid'),
    layers.Dropout(rate = 0.2),
    
    layers.Dense(64, activation = 'sigmoid'),
    layers.Dropout(rate=0.2),
    
    layers.Dense(10, activation = 'softmax'),
])

model2.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
)

early_stopping = EarlyStopping(
    min_delta = 0.001,
    patience  = 15,
    verbose   = 1,
    restore_best_weights = True,
)

history = model2.fit(
    x_ds_train, y_ds_train,
    validation_data = (x_ds_val, y_ds_val),
    batch_size = 256,
    epochs = 200,
    callbacks = [early_stopping],
)


# Predicting for the test data
labels = [0,1,2,3,4,5,6,7,8,9]
test_path = '/kaggle/input/digit-recognizer/test.csv'
test_ds = pd.read_csv(test_path)

submission_pr = model2.predict(test_ds)
submission_df = pd.DataFrame(submission_pr, columns=labels)
submission_df = submission_df.idxmax(axis=1)

# Processing to submission format
submission_final = pd.DataFrame({
    'ImageId': submission_df.index+1,
    'Label'  : submission_df.values
})

# Submission
submission_final.to_csv('submission.csv', index=False)
print('Submitted')
