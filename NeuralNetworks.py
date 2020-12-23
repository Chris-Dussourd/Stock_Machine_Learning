"""
Neural Network Machine Learning functions. Takes in input and output pandas DataFrames and returns predictions and params.

CD 12/2020  - Create neuralnet_category_sigmoid function
"""

def neuralnet_category_sigmoid(input_data_df,output_data_df,hidden_layers,hidden_units):
    """
    Uses the sigmoid function as the activiation function in a neural network machine learning algorithm.
    Takes in input and output DataFrames and returns the neural network parameters and the predicted output.

    input_data_df - input data, each column is a new feature.
    output_data_df - output data, should contain only one column of data.
    hidden_layers - number of the hidden layers to include in the neural network
    hidden_units - number of units to include in each hidden layer

    Returns (theta,predicted_output)
        theta - parameters to the neural network machine learning algorithm
        predicted_output - the prediction of the neural network algorithm
    """
    import tensorflow as tf
    from os import path, getcwd, chdir

    class trainNNCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if (logs.get('accuracy'))>0.99:
                print("\nWe reached an accuracy above 99% so end training.")
                self.model.stop_training=True

    #Callback used to check if a certain accuracy is reached before ending the training
    callback=trainNNCallback()

    #Number of features
    num_features = input_data_df.shape[1]

    #Number of output categories
    num_cat = output_data_df.max()+1
    
    #Create a neural network model in keras
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=hidden_units,activation=tf.nn.relu,input_shape=[num_features]),
        #tf.keras.layers.Dense(units=hidden_units,activation=tf.nn.relu,input_shape=[num_features]),
        tf.keras.layers.Dense(units=num_cat,activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) 
    
    # model fitting
    history = model.fit(input_data_df,output_data_df,epochs=20,callbacks=[callback])
    # model fitting
    return model