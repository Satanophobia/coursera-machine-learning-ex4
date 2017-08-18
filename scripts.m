clear;

load('ex4data1.mat');
m = size(X, 1);
num_labels = 10;
t = 1;

load('ex4weights.mat');

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

y2 = zeros(m, num_labels);
y2(sub2ind(size(y2), (1:length(y))', y)) = 1;

a_1 = X(t, :)';
z_2 = Theta1 * [1; a_1];
a_2 = sigmoid(z_2);
z_3 = Theta2 * [1; a_2];
a_3 = sigmoid(z_3);
delta_3 = a_3 - y2(t, :)';
Theta2_t = Theta2';
delta_2 = Theta2_t(2:end, :) * delta_3 .* sigmoidGradient(z_2);

Theta2_grad = Theta2_grad + delta_3 * [1; a_2]';
Theta1_grad = Theta1_grad + delta_2 * [1; a_1]';