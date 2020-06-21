function [grad] = sigmoidGradient(z)
  
  %returns the gradient of the sigmoid activation function
  grad = sigmoid(z).*(1-sigmoid(z));
endfunction
