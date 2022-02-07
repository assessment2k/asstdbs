from base import BaseMLP 
class NeuralNetworkClassifier(BaseMLP):
	# (self, hidden_layer_sizes, batch_size, learning_rate, max_iter, random_state, momentum)
    

    def fit(self, x_train, y_train):
        
        samples = len(x_train)

        # training 
        for i in range(self.epochs):
            err = 0
            for j in range(samples):
                # forward prop
                output = x_train[j]
                for layer in self.hidden_layer_sizes:
                    output = layer.forward_propagation(output)

                
                err += self.loss(y_train[j], output)

                # back prop
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.hidden_layer_sizes):
                    error = layer.backward_propagation(error, self.learning_rate)

            # error calculation
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, self.epochs, err))    	
        

    def predict(self, input_data):
        
        samples = len(input_data)
        result = []

        
        for i in range(samples):
            # forward prop
            output = input_data[i]
            for layer in self.hidden_layer_sizes:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def add(self, layer):
        self.hidden_layer_sizes.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
