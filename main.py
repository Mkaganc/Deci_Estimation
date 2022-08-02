import pandas as pd


def csv_to_pandas_dataframe(csv_path):
    print('----------------------------------------------------------------')
    print('OPENING FILES FROM SYSTEM DATA...')
    file = open(csv_path, 'r')
    return pd.read_csv(file)


def train_network(pandas_dataframe):
    print('IMPORTING ALL THE NECESSARY LIBRARIES...')
    from tensorflow import keras
    from keras.models import Sequential, model_from_json
    from keras.layers import Dense
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from tensorflow.keras.optimizers import RMSprop

    df = pandas_dataframe

    print('----------------------------------------------------------------')
    print('CREATING NEW DATA FRAMES FOR TRAINING AND TESTING THE MODEL...')
    raw_train_data = pd.DataFrame()  # Creating a new dataframe

    test_data = pd.DataFrame()
    train_data = pd.DataFrame()
    print('----------------------------------------------------------------')
    print('PROCESSING NECESSARY MODIFICATIONS ON DATA...')
    raw_train_data = df.loc[df['total_volume_cm3'] != 0.00]  # Select the data with total volume input

    raw_train_data = raw_train_data.loc[df['yurtici_deci'] != 0.0]  # Data with yurtici desi

    raw_train_data = raw_train_data.drop(2)
    raw_train_data = raw_train_data.drop('index', axis='columns')
    ## Finally define train and test data

    test_data = raw_train_data['yurtici_deci'].copy()
    train_data = raw_train_data[['calculated_deci', 'total_weight_gr', 'total_volume_cm3']].copy()

    # train_data.fillna(0)

    def reset_dataframe_index(dataframe):
        dataframe = dataframe.reset_index()


    reset_dataframe_index(train_data)
    reset_dataframe_index(test_data)

    from pandas.core.frame import DataFrame
    print('----------------------------------------------------------------')
    print('CONVERTING DATAFRAMES TO NUMPY ARRAYS...')

    ## Converting my pandas dataframes to numpy arrays.

    def turn_numpy_arrays(dataframe):
        try:
            dataframe = dataframe.to_numpy()
            return DataFrame
        except AttributeError:
            pass

    turn_numpy_arrays(train_data)
    turn_numpy_arrays(test_data)

    print('----------------------------------------------------------------')
    print('CREATING TENSORS FOR FINAL VERSION...')

    def create_tensor(array):
        array = tf.convert_to_tensor(array, dtype=tf.float32)
        return array
    train_data = create_tensor(train_data)
    test_data = create_tensor(test_data)

    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')
    print('CREATING BASELINE NEURAL NETWORK MODEL...')

    # Create model and add layers
    model = Sequential()
    model.add(Dense(3, input_shape=(3,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, input_shape=(3,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    opt = RMSprop(learning_rate=0.0002)
    model.compile(optimizer=opt, loss='mean_squared_error')

    print('----------------------------------------------------------------')
    print('----------------------------------------------------------------')
    print('EVALUATING MODEL...')

    history = model.fit(train_data, test_data, epochs=140, batch_size=3, verbose=1)

    scores = model.evaluate(train_data, test_data, verbose=0)

    try:
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    except IndexError:
        pass

    # After the training you should look is your model trained correctly

    def plot_history(history, key):
        plt.plot(history.history['loss'])

        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'test'], loc='upper left')
        plt.show()

    plot_history(history, 'mean_absolute_percentage_error')
    return model






# Save trained neural network model as JSON file
def save_model_as_json(model_input):
    """Saves model named object in the script to default location. You can change that if you want."""
    model_json = model_input.to_json()   # You can specify path and name
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

# Save last given weights(input tensors) as h5 file
def save_weights_as_h5(model_input):
    """Saves network's weights(inputs) to default location as .h5 file. """
    model_input.save_weights("model.h5")  # You can specify path and name
    print("Saved model to disk.")




df = csv_to_pandas_dataframe('nn_weights.csv')
model = train_network(df)
save_model_as_json(model)
save_weights_as_h5(model)


# Details for modification given in the ReadMe.txt file

