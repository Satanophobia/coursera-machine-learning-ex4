function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
y_matrix = [];
for i = 1:max(y)
	y_matrix = [y_matrix y==(size(y_matrix,2)+1)];
end

a1 = [ones(size(X,1),1) X]';
z2 = Theta1 * a1;
a2 = [ones(1, size(z2,2)); sigmoid(z2)];  % add bias unit for hidden layer

z3 = Theta2 * a2;
h3 = sigmoid(z3);

J_unreg =  (1/m) * sum(sum( (-y_matrix' .* log(h3)) - ((1-y_matrix') .* log(1-h3))));

Theta1_reg = Theta1;
Theta1_reg(:,1) = 0;

Theta2_reg = Theta2;
Theta2_reg(:,1) = 0;

J = J_unreg + (lambda/(2*m)) * ( sum(sum(Theta1_reg .* Theta1_reg)) + sum(sum(Theta2_reg .* Theta2_reg)) );


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients


delta3 = zeros(size(y_matrix,2), 1);
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for t = 1:m
	a1 = [1; X(t,:)'];
	z2 = Theta1 * a1;
	a2 = [ones(1, size(z2,2)); sigmoid(z2)]; 
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	
	for k = 1:size(y_matrix,2)
		delta3(k,:) = a3(k) - y_matrix(t,k);
	end

	delta2 = Theta2(:,2:end)' * delta3 .* sigmoidGradient(z2);
	%delta2 = [0; delta2(2:end)];

	Theta1_grad = Theta1_grad + delta2 * a1';
	Theta2_grad = Theta2_grad + delta3 * a2';
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

%Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * Theta1_reg;
%Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * Theta2_reg;

% Unroll gradients
grad=[Theta1_grad(:) ; Theta2_grad(:)];

















% -------------------------------------------------------------

% =========================================================================

end
