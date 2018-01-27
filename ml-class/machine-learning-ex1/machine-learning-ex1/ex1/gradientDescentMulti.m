function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


    sum = zeros(length(theta), 1);
    for i = 1:m
        sum(1) = sum(1) + (theta' * X(i,:)' - y(i));
        for j = 2:length(theta)
            sum(j) = sum(j) + (theta' * X(i,:)' - y(i)) * X(i,j);
        end
    end
    for j = 1:length(theta)
        sum(j) = sum(j) / m;
    theta = theta - alpha * sum;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
