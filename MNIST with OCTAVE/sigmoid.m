function [val] = sigmoid(z)
  %returns the value of the sigmoid function for matrices we pass from propagations.m
  val = 1./(1+exp(-z));
endfunction
