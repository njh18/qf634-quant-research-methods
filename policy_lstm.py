import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras import Model
import numpy as np

class PolicyLSTM(object):
    '''
    Using this class we will build policy for the LSTM network.
    '''

    def __init__(self, ohlc_feature_num, ticker_num, num_trading_periods, trading_cost, cash_bias_init, interest_rate, 
                 equiweight_vector, adjusted_rewards_alpha, num_filter_layer, lstm_neurons = 20, layers = 2):

        super(PolicyLSTM, self).__init__()
        # parameters
        self.ohlc_feature_num = ohlc_feature_num
        self.ticker_num = ticker_num
        self.num_trading_periods =  num_trading_periods
        self.trading_cost = trading_cost
        self.cash_bias_init = cash_bias_init
        self.interest_rate = interest_rate
        self.equiweight_vector = equiweight_vector
        self.adjusted_rewards_alpha = adjusted_rewards_alpha    
        #self.optimizer = optimizer
        #self.sess = sess
        self.num_filter_layer = num_filter_layer
        #layers=2
        self.lstm_neurons = lstm_neurons
        self.layers = layers
        #define placeholders
        self.X_t = Input(shape=(ohlc_feature_num, ticker_num, num_trading_periods))  # Input shape
        self.weights_previous_t = Input(shape=(ticker_num + 1,))
        self.pf_previous_t = Input(shape=(1,))
        self.daily_returns_t = Input(shape=(ticker_num,))

        # LSTM network
        self.lstm_layers = [LSTM(lstm_neurons, return_sequences=True, dropout=0.3) for _ in range(layers)]
        
        # Convolution layer for policy output
        self.conv_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(num_filter_layer + 1, 1),
                                                 activation='relu', padding='same')
        # Output layer to compute the portfolio allocation
        self.output_layer = tf.keras.layers.Softmax(axis=1)

        #self.X_t = tf.placeholder(tf.float32, [None, self.ohlc_feature_num, self.ticker_num, self.num_trading_periods])
        #self.weights_previous_t = tf.placeholder(tf.float32, [None, self.ticker_num + 1])
        #self.pf_previous_t = tf.placeholder(tf.float32, [None, 1])
        #self.daily_returns_t = tf.placeholder(tf.float32, [None, self.ticker_num]) 

        '''
        cash_bias = tf.get_variable('cash_bias', shape=[1, 1, 1, 1], initializer = tf.constant_initializer(self.cash_bias_init))
        shape_X_t = tf.shape(self.X_t)[0]
        self.cash_bias = tf.tile(cash_bias, tf.stack([shape_X_t, 1, 1, 1]))
        '''
    
    def call_lstm(self, X_t):
        
        # Prepare input for LSTM (reshape for batch processing)
        #network = tf.transpose(X_t, [0, 1, 3, 2])  # Adjust the shape for LSTM processing
        network = tf.transpose(X_t, [0, 2, 3, 1])
        batch_size, timesteps, features = network.shape[0], network.shape[2], network.shape[3] * network.shape[1]
        network = tf.reshape(network, (batch_size, timesteps, features))
        # Normalize data
        #network = network / network[:, :, -1, 0, None, None]
        # Pass through LSTM layers
        for lstm_layer in self.lstm_layers:
            network = lstm_layer(network)
        # Expand dimensions to 4D for Conv2D
        network = tf.expand_dims(network, axis=-1)
        # Apply convolution to the LSTM output
        conv_output = self.conv_layer(network)

        # Squeeze and concatenate the cash bias
        cash_bias = tf.constant(self.cash_bias_init, shape=[1, 1, 1, 1])
        replication_factors = tf.concat([tf.shape(conv_output)[:3], [1]], axis=0)  # Make sure it's a 4D vector
        cash_bias = tf.tile(cash_bias, replication_factors)
        #cash_bias = tf.tile(cash_bias, tf.shape(conv_output)[:3])  # Tile the cash bias over the batch
        concatenated_tensor = tf.concat([cash_bias, conv_output], axis=2)

        # Check the shape before squeezing
        # Ensure squeezing only dimensions == 1
        tensor_squeeze = tf.squeeze(concatenated_tensor, axis=[-1]) 
        # Concatenate the cash bias and convolution output
        #tensor_squeeze = tf.squeeze(tf.concat([cash_bias, conv_output], axis=2), [1, 3])

        # Apply Softmax to compute action (portfolio weights)
        action_chosen = self.output_layer(tensor_squeeze)

        return action_chosen
        '''
        def lstm(X_t):
            network = tf.transpose(X_t, [0, 1, 3, 2])
            network = network / network[:, :, -1, 0, None, None]

            for layer_number in range(layers):
                resultlist = []
                reuse = False

                for i in range(self.ticker_num):
                    if i > 0:
                        reuse = True

                    result = tflearn.layers.lstm(X_t[:,:,:, i],
                                                    lstm_neurons,
                                                    dropout=0.3,
                                                    scope="lstm"+str(layer_number),
                                                    reuse=reuse)
            

                    resultlist.append(result)
                network = tf.stack(resultlist)
                network = tf.transpose(network, [1, 0, 2])
                network = tf.reshape(network, [-1, 1, self.ticker_num, lstm_neurons])
                # print('dhsegfhebgfhewf', network.shape)
            return network

        def policy_output(network, cash_bias):
            with tf.variable_scope("Convolution_Layer"):
                self.conv = tf.layers.conv2d(
                    inputs = network,
                    activation = tf.nn.relu,
                    filters = 1,
                    strides = (num_filter_layer + 1, 1),
                    kernel_size = (1, 1),
                    padding = 'same')

            with tf.variable_scope("Policy-Output"):
                tensor_squeeze = tf.squeeze(tf.concat([cash_bias, self.conv], axis=2), [1,3])
                self.action = tf.nn.softmax(tensor_squeeze)
            return self.action
        '''

    def reward(self, shape_X_t, action_chosen, interest_rate, weights_previous_t, pf_previous_t, daily_returns_t, trading_cost):
        #Calculating reward for current Portfolio
        #with tf.variable_scope("Reward"):
        cash_return = tf.tile(tf.constant(1 + interest_rate, shape=[1, 1]), tf.stack([shape_X_t, 1]))
        y_t = tf.concat([cash_return, daily_returns_t], axis=1)

        # Expand dimensions of pf_previous_t to match the shape of action_chosen
        pf_previous_t_expanded = tf.expand_dims(pf_previous_t, axis=-1)  # Shape: [batch_size, ticker_num, 1]

        # Expand dimensions of weights_previous_t to match the shape of action_chosen
        weights_previous_t_expanded = tf.expand_dims(weights_previous_t, axis=-1)  # Shape: [batch_size, ticker_num, 1]

        # Ensure action_chosen has the same shape as the others
        pf_vector_t = action_chosen * pf_previous_t_expanded  # Shape: [batch_size, ticker_num, 1]

        # For pf_vector_previous, ensure it matches the same shape as pf_vector_t
        pf_vector_previous = weights_previous_t_expanded * pf_previous_t_expanded 

        
        #pf_vector_t = action_chosen * tf.expand_dims(pf_previous_t, axis=-1)
        #pf_vector_previous = weights_previous_t * pf_previous_t
        #pf_vector_previous = tf.expand_dims(weights_previous_t, axis=-1) * tf.expand_dims(pf_previous_t, axis=-1)
        total_trading_cost = trading_cost * tf.norm(pf_vector_t - pf_vector_previous, ord=1, axis=[1, 2])
        #total_trading_cost = trading_cost * tf.norm(pf_vector_t - pf_vector_previous, ord=1, axis=1) * tf.constant(1.0, shape=[1])
        total_trading_cost = tf.expand_dims(total_trading_cost, axis = -1)
        zero_vector = tf.zeros_like(total_trading_cost) 
        #zero_vector = tf.tile(tf.constant(np.array([0.0] * ticker_num).reshape(1, ticker_num), shape=[1, ticker_num], dtype=tf.float32), tf.stack([shape_X_t, 1]))
        #cost_vector = tf.concat([total_trading_cost, zero_vector], axis=1)
        cost_vector = total_trading_cost + zero_vector
        # Broadcasting cost_vector to match the shape of pf_vector_t
        cost_vector_broadcasted = tf.tile(cost_vector, [1, ticker_num, 1])  # This will give [batch_size, ticker_num, 1]

        # Perform the subtraction
        pf_vector_second_t = pf_vector_t - cost_vector_broadcasted

        #pf_vector_second_t = pf_vector_t - cost_vector
        final_pf_vector_t = tf.multiply(pf_vector_second_t, y_t)
        portfolio_value = tf.norm(final_pf_vector_t, ord=1)
        instantaneous_reward = (portfolio_value - pf_previous_t) / pf_previous_t
            
        #Calculating Reward for Equiweight portfolio
        #with tf.variable_scope("Reward-Equiweighted"):
        cash_return = tf.tile(tf.constant(1 + interest_rate, shape=[1, 1]), tf.stack([shape_X_t, 1]))
        y_t = tf.concat([cash_return, daily_returns_t], axis=1)

        pf_vector_eq = self.equiweight_vector * pf_previous_t

        portfolio_value_eq = tf.norm(tf.multiply(pf_vector_eq, y_t), ord=1)
        instantaneous_reward_eq = (portfolio_value_eq - pf_previous_t) / pf_previous_t

        #Calculating Adjusted Rewards
        #with tf.variable_scope("Reward-adjusted"):
        self.adjusted_reward = instantaneous_reward - instantaneous_reward_eq - self.adjusted_rewards_alpha * tf.reduce_max(action_chosen)
            
        return self.adjusted_reward

    '''
    self.lstm_layer = lstm(self.X_t)
    self.action_chosen = policy_output(self.lstm_layer, self.cash_bias)
    self.adjusted_reward = reward(shape_X_t, self.action_chosen, self.interest_rate, self.weights_previous_t, self.pf_previous_t, self.daily_returns_t, self.trading_cost)
    self.train_op = optimizer.minimize(-self.adjusted_reward)
    '''
    @tf.function
    def compute_weights(self, X_t_, weights_previous_t_):
        # tf.print(self.action_chosen)
        return self(X_t_, weights_previous_t_, self.pf_previous_t, self.daily_returns_t)
        #return self(tf.squeeze(self.action_chosen), feed_dict={self.X_t: X_t_, self.weights_previous_t: weights_previous_t_})

    def train_lstm(self, X_t_, weights_previous_t_, pf_previous_t_, daily_returns_t_):
        """
        training the neural network
        """
        with tf.GradientTape() as tape:
            action_chosen = self.call_lstm(X_t_)
            adjusted_reward = self.reward(X_t_.shape[0], action_chosen, self.interest_rate, weights_previous_t_, pf_previous_t_, daily_returns_t_, self.trading_cost)
            loss = -tf.reduce_mean(adjusted_reward)  # Minimize negative reward
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        '''
        self.sess.run(self.train_op, feed_dict={self.X_t: X_t_,
                                                self.weights_previous_t: weights_previous_t_,
                                                self.pf_previous_t: pf_previous_t_,
                                                self.daily_returns_t: daily_returns_t_})
        '''


# Instantiate the model
policy_lstm_model = PolicyLSTM(ohlc_feature_num=10, ticker_num=5, num_trading_periods=100, 
                               trading_cost=0.001, cash_bias_init=0.0, interest_rate=0.01, 
                               equiweight_vector=np.ones(5)/5, adjusted_rewards_alpha=0.1, 
                               num_filter_layer=3)

# Sample data generation (e.g., for 5 time steps, 3 tickers, and 4 features)
num_samples = 100  # 100 data samples (for demonstration)
ohlc_feature_num = 4  # Open, High, Low, Close
ticker_num = 3
num_trading_periods = 5  # Number of trading periods

# Generating random sample data for OHLC, previous weights, and daily returns
X_t_sample = np.random.random((num_samples, ohlc_feature_num, ticker_num, num_trading_periods)).astype(np.float32)
weights_previous_t_sample = np.random.random((num_samples, ticker_num + 1)).astype(np.float32)  # +1 for cash bias
pf_previous_t_sample = np.random.random((num_samples, 1)).astype(np.float32)
daily_returns_t_sample = np.random.random((num_samples, ticker_num)).astype(np.float32)

pf_previous_t_sample_expanded = np.tile(pf_previous_t_sample, (1, ticker_num))  # [num_samples, ticker_num]
weights_previous_t_sample = weights_previous_t_sample[:, :-1]

# Parameters
trading_cost = 0.01
cash_bias_init = 0.1
interest_rate = 0.05
equiweight_vector = np.ones(ticker_num) / ticker_num  # Equally weighted portfolio
adjusted_rewards_alpha = 0.1
num_filter_layer = 1
lstm_neurons = 20
layers = 2

# Initialize the model
model = PolicyLSTM(
    ohlc_feature_num=ohlc_feature_num,
    ticker_num=ticker_num,
    num_trading_periods=num_trading_periods,
    trading_cost=trading_cost,
    cash_bias_init=cash_bias_init,
    interest_rate=interest_rate,
    equiweight_vector=equiweight_vector,
    adjusted_rewards_alpha=adjusted_rewards_alpha,
    num_filter_layer=num_filter_layer,
    lstm_neurons=lstm_neurons,
    layers=layers
)

# Optimizer
optimizer = tf.optimizers.Adam()

# Set optimizer to the model
model.optimizer = optimizer

# Training loop (for demonstration, we'll run for 10 epochs)
for epoch in range(10):
    print(f"Epoch {epoch + 1}/{10}")
    model.train_lstm(X_t_sample, weights_previous_t_sample, pf_previous_t_sample, daily_returns_t_sample)




