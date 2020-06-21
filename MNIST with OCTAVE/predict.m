function [result] = predict(X,theta,m,n,h1,h2)
  
  %predicts the results of the model after optimisation
  theta1 = reshape(theta(1 : (h1 * (n(2) + 1))), h1, (n(2) + 1));
  theta2 = reshape(theta((1 + (h1 * (n(2) + 1))):(h1 * (n(2) + 1))+h2*(h1+1)),h2, (h1 + 1));
  theta3 = reshape(theta((1+(h1 * (n(2) + 1))+h2*(h1+1)):end) , 10 , (h2+1));
  
  a1 = [ones(m(1),1) X];
  z2 = a1*theta1';
  a2 = [ones(m(1),1) sigmoid(z2)];
  z3 = a2*theta2';
  a3 = [ones(m(1),1) sigmoid(z3)];
  z4 = a3*theta3';
  
  %a3 contains the results in the form of a [1x10] array for each example
  a4 = sigmoid(z4);
  result = zeros(m(1),1);
  
  %result takes the index of '1' in the a3 array for each example
  [max_values,result] =max(a4,[],2);
  result = result-1;
endfunction
