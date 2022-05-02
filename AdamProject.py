import numpy as np
import matplotlib.pyplot as plt

class Optimizers():

    def __init__(self, all_weights):
        #all_weights is a vector of all of a neural networks weights concatenated into a one-dimensional vector
        
        self.all_weights = all_weights

        # The following initializations are only used by adam.
        # Only initializing mt, vt, beta1t and beta2t here allows multiple calls to adam to handle training
        # with multiple subsets (batches) of training data.
        self.mt = np.zeros_like(all_weights)
        self.vt = np.zeros_like(all_weights)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.beta1t = 1
        self.beta2t = 1 

        
    def sgd(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=0.001, error_convert_f=None):
    #error_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.
    #gradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error with respect to each weight.
    #error_convert_f: function that converts the standardized error from error_f to original T units.


        error_trace = []
        epochs_per_print = n_epochs // 10

        for epoch in range(n_epochs):

            error = error_f(*fargs)
            grad = gradient_f(*fargs)

            # Update all weights using -= to modify their values in-place.
            self.all_weights -= learning_rate * grad

            if error_convert_f:
                error = error_convert_f(error)
            error_trace.append(error)

            if (epoch + 1) % max(1, epochs_per_print) == 0:
                print(f'sgd: Epoch {epoch+1:d} Error={error:.5f}')

        return error_trace

    def adam(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=0.001, error_convert_f=None):
    #error_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.
    #gradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error with respect to each weight.
    #error_convert_f: function that converts the standardized error from error_f to original T units.

        alpha = learning_rate  # learning rate called alpha in original paper on adam
        epsilon = 1e-8
        error_trace = []
        epochs_per_print = n_epochs // 10

        for epoch in range(n_epochs):

            error = error_f(*fargs)
            grad = gradient_f(*fargs)
            
            self.mt = self.beta1 * self.mt + (1- self.beta1) * grad
            self.vt = self.beta2 * self.vt + (1- self.beta2) * np.square(grad)

            self.beta1t *= self.beta1
            self.beta2t *= self.beta2
            mhat = self.mt / (1-self.beta1t)
            vhat = self.vt / (1-self.beta2t)
            self.all_weights-= alpha * mhat / (np.sqrt(vhat) + epsilon)

            if error_convert_f:
                error = error_convert_f(error)
            error_trace.append(error)

            if (epoch + 1) % max(1, epochs_per_print) == 0:
                print(f'Adam: Epoch {epoch+1:d} Error={error:.5f}')

        return error_trace

def test_optimizers():

    def parabola(wmin):
        return ((w - wmin) ** 2)[0]

    def parabola_gradient(wmin):
        return 2 * (w - wmin)

    w = np.array([0.0])
    optimizer = Optimizers(w)

    wmin = 5
    optimizer.sgd(parabola, parabola_gradient, [wmin], n_epochs=100, learning_rate=0.1)
    print(f'sgd: Minimum of parabola is at {wmin}. Value found is {w}')

    w = np.array([0.0])
    optimizer = Optimizers(w)
    optimizer.adam(parabola, parabola_gradient, [wmin], n_epochs=100, learning_rate=0.1)
    print(f'adam: Minimum of parabola is at {wmin}. Value found is {w}')

test_optimizers()

class NeuralNetwork():


    def __init__(self, n_inputs, n_hiddens_per_layer, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Set self.n_hiddens_per_layer to [] if argument is 0, [], or [0]
        if n_hiddens_per_layer == 0 or n_hiddens_per_layer == [] or n_hiddens_per_layer == [0]:
            self.n_hiddens_per_layer = []
        else:
            self.n_hiddens_per_layer = n_hiddens_per_layer

        # Initialize weights, by first building list of all weight matrix shapes.
        n_in = n_inputs
        shapes = []
        for nh in self.n_hiddens_per_layer:
            shapes.append((n_in + 1, nh))
            n_in = nh
        shapes.append((n_in + 1, n_outputs))

        # self.all_weights:  vector of all weights
        # self.Ws: list of weight matrices by layer
        self.all_weights, self.Ws = self.make_weights_and_views(shapes)

        # Define arrays to hold gradient values.
        # One array for each W array with same shape.
        self.all_gradients, self.dE_dWs = self.make_weights_and_views(shapes)

        self.trained = False
        self.total_epochs = 0
        self.error_trace = []
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None


    def make_weights_and_views(self, shapes):
        # vector of all weights built by horizontally stacking flattened matrices
        # for each layer initialized with uniformly-distributed values.
        all_weights = np.hstack([np.random.uniform(size=shape).flat / np.sqrt(shape[0])
                                 for shape in shapes])
        # Build list of views by reshaping corresponding elements from vector of all weights
        # into correct shape for each layer.
        views = []
        start = 0
        for shape in shapes:
            size =shape[0] * shape[1]
            views.append(all_weights[start:start + size].reshape(shape))
            start += size
        return all_weights, views


    # Return string that shows how the constructor was called
    def __repr__(self):
        return f'NeuralNetwork({self.n_inputs}, {self.n_hiddens_per_layer}, {self.n_outputs})'


    # Return string that is more informative to the user about the state of this neural network.
    def __str__(self):
        if self.trained:
            return self.__repr__() + f' trained for {self.total_epochs} epochs, final training error {self.error_trace[-1]}'


    def train(self, X, T, n_epochs, learning_rate, method='sgd'):

    #  train: 
    #    X: n_samples x n_inputs matrix of input samples, one per row
    #    T: n_samples x n_outputs matrix of target output values, one sample per row
    #    n_epochs: number of passes to take through all samples updating weights each pass
    #    learning_rate: factor controlling the step size of each update
    #    method: is either 'sgd' or 'adam'
        

        # Setup standardization parameters
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xstds[self.Xstds == 0] = 1  # So we don't divide by zero when standardizing
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            
        # Standardize X and T
        X = (X - self.Xmeans) / self.Xstds
        T = (T - self.Tmeans) / self.Tstds

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = Optimizers(self.all_weights)

        # Define function to convert value from error_f into error in original T units.
        error_convert_f = lambda err: (np.sqrt(err) * self.Tstds)[0] # to scalar

        if method == 'sgd':

            error_trace = optimizer.sgd(self.error_f, self.gradient_f,
                                        fargs=[X, T], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        error_convert_f=error_convert_f)

        elif method == 'adam':

            error_trace = optimizer.adam(self.error_f, self.gradient_f,
                                         fargs=[X, T], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         error_convert_f=error_convert_f)

        else:
            raise Exception("method must be 'sgd' or 'adam'")
        
        self.error_trace = error_trace

        return self
   
    def forward_pass(self, X):
        self.Ys = [X]
        for W in self.Ws[:-1]:
            self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
        last_W = self.Ws[-1]
        self.Ys.append(self.Ys[-1] @ last_W[1:, :] + last_W[0:1, :])
        return self.Ys

    # Function to be minimized by optimizer method, mean squared error
    def error_f(self, X, T):
        Ys = self.forward_pass(X)
        mean_sq_error = np.mean((T - Ys[-1]) ** 2)
        return mean_sq_error

    # Gradient of function to be minimized for use by optimizer method
    def gradient_f(self, X, T):
        #Assumes forward_pass just called with layer outputs in self.Ys.
        error = T - self.Ys[-1]
        n_samples = X.shape[0]
        n_outputs = T.shape[1]
        delta = - error / (n_samples * n_outputs)
        n_layers = len(self.n_hiddens_per_layer) + 1
        # Step backwards through the layers to back-propagate the error (delta)
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            self.dE_dWs[layeri][1:, :] = self.Ys[layeri].T @ delta
            # gradient of just the bias weights
            self.dE_dWs[layeri][0:1, :] = np.sum(delta, 0)
            # Back-propagate this layer's delta to previous layer
            delta = delta @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri] ** 2)
        return self.all_gradients

    def use(self, X):
        #X assumed to not be standardized. Return the unstandardized prediction

        X = X.copy()
        X= (X - self.Xmeans) / self.Xstds
       
        y=self.forward_pass(X)
        #print(y, y.shape)
        y=y[-1]
        #=np.insert(y, 0, 1, axis=1)
        #y=[b for x in y for b in x]
        #print(y)
        #print(self.Tstds)
        #print(self.Tmeans)
        y = y * self.Tstds + self.Tmeans
        return y
        




np.random.seed(42)
np.random.uniform(-0.1, 0.1, size=(2, 2))

np.random.uniform(-0.1, 0.1, size=(2, 2))

np.random.seed(42)
np.random.uniform(-0.1, 0.1, size=(2, 2))




def test_neuralnetwork():
    
    np.random.seed(42)
    
    X = np.arange(100).reshape((-1, 1))
    T = np.sin(X * 0.04)

    n_hiddens = [10, 10]
    n_epochs = 2000
    learning_rate = 0.01
    
    nnetsgd = NeuralNetwork(1, n_hiddens, 1)
    nnetsgd.train(X, T, n_epochs, learning_rate, method='sgd')

    print()  # skip a line
    
    nnetadam = NeuralNetwork(1, n_hiddens, 1)
    nnetadam.train(X, T, n_epochs, learning_rate, method='adam')

    Ysgd = nnetsgd.use(X)
    Yadam = nnetadam.use(X)

    plt.figure(figsize=(15,10))
    plt.subplot(1, 3, 1)
    plt.plot(nnetsgd.error_trace, label='SGD')
    plt.plot(nnetadam.error_trace, label='Adam')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(T, Ysgd, 'o', label='SGD')
    plt.plot(T, Yadam, 'o', label='Adam')
    a = min(np.min(T), np.min(Ysgd))
    b = max(np.max(T), np.max(Ysgd))
    plt.plot([a, b], [a, b], 'k-', lw=3, alpha=0.5, label='45 degree')
    plt.xlabel('Target')
    plt.ylabel('Predicted')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(Ysgd, 'o-', label='SGD')
    plt.plot(Yadam, 'o-', label='Adam')
    plt.plot(T, label='Target')
    plt.xlabel('Sample')
    plt.ylabel('Target or Predicted')
    plt.legend()

    plt.tight_layout()


test_neuralnetwork()


class NeuralNetwork():


    def __init__(self, n_inputs, n_hiddens_per_layer, n_outputs,activation_function='tanh'):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Set self.n_hiddens_per_layer to [] if argument is 0, [], or [0]
        if n_hiddens_per_layer == 0 or n_hiddens_per_layer == [] or n_hiddens_per_layer == [0]:
            self.n_hiddens_per_layer = []
        else:
            self.n_hiddens_per_layer = n_hiddens_per_layer

        # Initialize weights, by first building list of all weight matrix shapes.
        n_in = n_inputs
        shapes = []
        for nh in self.n_hiddens_per_layer:
            shapes.append((n_in + 1, nh))
            n_in = nh
        shapes.append((n_in + 1, n_outputs))

        # self.all_weights:  vector of all weights
        # self.Ws: list of weight matrices by layer
        self.all_weights, self.Ws = self.make_weights_and_views(shapes)

        # Define arrays to hold gradient values.
        # One array for each W array with same shape.
        self.all_gradients, self.dE_dWs = self.make_weights_and_views(shapes)

        self.trained = False
        self.total_epochs = 0
        self.error_trace = []
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None
        self.activationf=activation_function

    def make_weights_and_views(self, shapes):
        # vector of all weights built by horizontally stacking flatenned matrices
        # for each layer initialized with uniformly-distributed values.
        all_weights = np.hstack([np.random.uniform(size=shape).flat / np.sqrt(shape[0])
                                 for shape in shapes])
        # Build list of views by reshaping corresponding elements from vector of all weights
        # into correct shape for each layer.
        views = []
        start = 0
        for shape in shapes:
            size =shape[0] * shape[1]
            views.append(all_weights[start:start + size].reshape(shape))
            start += size
        return all_weights, views


    # Return string that shows how the constructor was called
    def __repr__(self):
        return f'NeuralNetwork({self.n_inputs}, {self.n_hiddens_per_layer}, {self.n_outputs})'


    # Return string that is more informative to the user about the state of this neural network.
    def __str__(self):
        if self.trained:
            return self.__repr__() + f' trained for {self.total_epochs} epochs, final training error {self.error_trace[-1]}'


    def train(self, X, T, n_epochs, learning_rate, method='sgd'):
        '''
train: 
  X: n_samples x n_inputs matrix of input samples, one per row
  T: n_samples x n_outputs matrix of target output values, one sample per row
  n_epochs: number of passes to take through all samples updating weights each pass
  learning_rate: factor controlling the step size of each update
  method: is either 'sgd' or 'adam'
        '''

        # Setup standardization parameters
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
            self.Xstds[self.Xstds == 0] = 1  # So we don't divide by zero when standardizing
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
            
        # Standardize X and T
        X = (X - self.Xmeans) / self.Xstds
        T = (T - self.Tmeans) / self.Tstds

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = Optimizers(self.all_weights)

        # Define function to convert value from error_f into error in original T units.
        error_convert_f = lambda err: (np.sqrt(err) * self.Tstds)[0] # to scalar

        if method == 'sgd':

            error_trace = optimizer.sgd(self.error_f, self.gradient_f,
                                        fargs=[X, T], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        error_convert_f=error_convert_f)

        elif method == 'adam':

            error_trace = optimizer.adam(self.error_f, self.gradient_f,
                                         fargs=[X, T], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         error_convert_f=error_convert_f)

        else:
            raise Exception("method must be 'sgd' or 'adam'")
        
        self.error_trace = error_trace

        return self

   
    def forward_pass(self, X):
        '''X assumed already standardized. Output returned as standardized.'''
        self.Ys = [X]
        for W in self.Ws[:-1]:
            #print("activationf is:",str(self.activationf))
            if self.activationf == 'tanh':
             #   print("in tanh")
                self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
            else:
              #  print("in relu")
                self.Ys.append(self.relu(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
        last_W = self.Ws[-1]
        #print("ys:",self.Ys[-1],"last_w:", last_W[1:, :])
        self.Ys.append(self.Ys[-1] @ last_W[1:, :] + last_W[0:1, :])
        return self.Ys

    # Function to be minimized by optimizer method, mean squared error
    def error_f(self, X, T):
        Ys = self.forward_pass(X)
        mean_sq_error = np.mean((T - Ys[-1]) ** 2)
        return mean_sq_error

    # Gradient of function to be minimized for use by optimizer method
    def gradient_f(self, X, T):
        '''Assumes forward_pass just called with layer outputs in self.Ys.'''
        error = T - self.Ys[-1]
        n_samples = X.shape[0]
        n_outputs = T.shape[1]
        delta = - error / (n_samples * n_outputs)
        n_layers = len(self.n_hiddens_per_layer) + 1
        # Step backwards through the layers to back-propagate the error (delta)
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            
            self.dE_dWs[layeri][1:, :] = self.Ys[layeri].T @ delta
            # gradient of just the bias weights
            self.dE_dWs[layeri][0:1, :] = np.sum(delta, 0)
            # Back-propagate this layer's delta to previous layer
            if(self.activationf=='tanh'):
               # print("in tanh")
                delta = delta @ self.Ws[layeri][1:, :].T * (1- self.Ys[layeri] ** 2)#1-
            else:
                #print("for relu: ",self.Ys[layeri] ** 2)
                #print("in relu")
                delta = delta @ self.Ws[layeri][1:, :].T * (1-self.grad_relu((self.Ys[layeri]) ** 2))#1-
        return self.all_gradients
    def relu(self, s):
        Y=s
        #Y=s.copy()
        Y[Y<0]=0
        return Y
    def grad_relu(self, s):
        Y=s
        #Y=s.copy()
        Y[Y<0]=0
        Y[Y>0]=1
        return Y
    def use(self, X):
        '''X assumed to not be standardized. Return the unstandardized prediction'''

        X = X.copy()
        X= (X - self.Xmeans) / self.Xstds
       
        y=self.forward_pass(X)
        #print(y, y.shape)
        y=y[-1]
        #=np.insert(y, 0, 1, axis=1)
        #y=[b for x in y for b in x]
        #print(y)
        #print(self.Tstds)
        #print(self.Tmeans)
        y = y * self.Tstds + self.Tmeans
        return y
        #OLD RELU VERSION
        



def partition(X, T, n_folds, random_shuffle=True):
    np.random.seed(42)
    rows = np.arange(X.shape[0])
    if(random_shuffle): 
        np.random.shuffle(rows)  # shuffle the row indices in-place (rows are changed)
    X = X[rows, :]
    T = T[rows, :]
    n_samples = X.shape[0]
    n_per_fold = n_samples // n_folds
    n_last_fold = n_samples - n_per_fold * (n_folds - 1)  # handles case when n_samples not evenly divided by n_folds

    folds = []
    start = 0
    for foldi in range(n_folds-1):
        folds.append( (X[start:start + n_per_fold, :], T[start:start + n_per_fold, :]) )
        start += n_per_fold
    folds.append( (X[start:, :], T[start:, :]) )
    
    Xvalidate, Tvalidate = folds[0]
    Xtest, Ttest = folds[1]
    Xtrain, Ttrain = np.vstack([X for (X, _) in folds[2:]]), np.vstack([T for (_, T) in folds[2:]])
    return Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest


import pandas
import matplotlib.pyplot as plt

data = pandas.read_csv('auto-mpg.data', header=0, delim_whitespace=True, na_values="?")
data=data.dropna()
data = data.iloc[:,:-1]

print(data)




def run_experiment(X, T, n_folds,n_epochs_choices,n_hidden_units_per_layer_choices,activation_function_choices):
    Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = partition(X,T,n_folds)
    results=[]
    learning_rate = 0.01
    for epochnum in n_epochs_choices:
        for nhidden in n_hidden_units_per_layer_choices:
            for activation in activation_function_choices:
                #print("new attempt with: ",n_hidden_units_per_layer_choices[j], 1,activation_function_choices[k],n_epochs_choices[i])
                attempt = NeuralNetwork(Xtrain.shape[1],nhidden, Ttrain.shape[1],activation)
                #          def __init__(n_inputs, n_hiddens_per_layer, n_outputs,activation_function='tanh'):
                #(self, n_inputs, n_hiddens_per_layer, n_outputs,activation_function='tanh'):
                attempt.train(Xtrain, Ttrain, epochnum, learning_rate, method='adam')
                #np.sqrt(np.mean((A-B)**2))
                #train =(Ttrain,attempt.use(Xtrain))
                train=np.sqrt(np.mean((Ttrain-attempt.use(Xtrain))**2))
                #val = rmse(Tvalidate,attempt.use(Xvalidate))
                val=np.sqrt(np.mean((Tvalidate-attempt.use(Xvalidate))**2))
                #test=rmse(Ttest,attempt.use(Xtest))
                test=np.sqrt(np.mean((Ttest-attempt.use(Xtest))**2))
                #print("result: ",[epochnum,n_hidden_units_per_layer_choices[j],
                #               learning_rate,activation_function_choices[k],
                #             train, val,test])
                results.append([epochnum,nhidden,
                               learning_rate,activation,
                              train, val,test])
                
    #results=None
    return pandas.DataFrame(results, columns=('epochs', 'nh', 'lr', 'act func','RMSE Train', 'RMSE Val','RMSE Test'))

df = run_experiment(X, T, n_folds=5, 
                           n_epochs_choices=[1000, 5000,10000],
                           n_hidden_units_per_layer_choices=[[0], [20], [20, 20, 20, 20]],activation_function_choices=['tanh', 'relu'])

print(df)

plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
dftanh = df[(df['act func'] == 'tanh')]
xs = range(dftanh.shape[0])
plt.plot(xs, dftanh[['RMSE Train', 'RMSE Val', 'RMSE Test']], 'o-')
xticks = dftanh[['epochs', 'nh']].apply(lambda x: f'{x[0]}, {x[1]}', axis=1) # converting to strings
plt.xticks(range(len(xticks)), xticks, rotation=25, ha='right')
plt.xlabel('Epochs, Architecture')
plt.ylabel('RMSE')
plt.legend(('Train tanh', 'Val tanh', 'Test tanh'))
plt.grid('on')

plt.subplot(1, 2, 2)
dfrelu = df[(df['act func'] == 'relu')]
xs = range(dfrelu.shape[0])
#RMSE Train  RMSE Val  RMSE Test
plt.plot(xs, dfrelu[['RMSE Train', 'RMSE Val', 'RMSE Test']], 'o-')
xticks = dfrelu[['epochs', 'nh']].apply(lambda x: f'{x[0]}, {x[1]}', axis=1) # converting to strings
plt.xticks(range(len(xticks)), xticks, rotation=25, ha='right')
plt.xlabel('Epochs, Architecture')
plt.ylabel('RMSE')
plt.legend(('Train relu', 'Val relu', 'Test relu'))
plt.grid('on')

