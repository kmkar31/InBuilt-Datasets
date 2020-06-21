function [J,grad] = propagations(X,y,m,n,h1,h2,theta,lambda)
  
  %unrolling theta into matrices to apply forward and back propagations 
  theta1 = reshape(theta(1 : (h1 * (n(2) + 1))), h1, (n(2) + 1));
  theta2 = reshape(theta((1 + (h1 * (n(2) + 1))):(h1 * (n(2) + 1))+h2*(h1+1)),h2, (h1 + 1));
  theta3 = reshape(theta((1+(h1 * (n(2) + 1))+h2*(h1+1)):end) , 10 , (h2+1));
  
  
  
  %Adding the bias unit to yhe input layer and all subsequent layers
  % Performing forward propagation
  a1 = [ones(m(1),1) X];
  z2 = a1*theta1';
  a2 = [ones(m,1) sigmoid(z2)];
  z3 = a2*theta2';
  a3 = [ones(m,1) sigmoid(z3)];
  z4 = a3*theta3';
  a4 = sigmoid(z4);
  
  %Cost Function with regualrisation
  penalty = (lambda/(2*m(1)))*(sum(sum(theta1(:, 2:end).^2, 2)) + sum(sum(theta2(:,2:end).^2, 2)) + sum(sum(theta3(:, 2:end).^2, 2)));
  J = (1/m(1))*sum(sum((-y).*log(a4) - (1-y).*log(1-a4)));
  J = J + penalty;
  
  %Backpropagation  
  d4 = a4 - y;
  d3 = (d4*theta3 .* sigmoidGradient([ones(size(z3, 1), 1) z3]))(:, 2:end);
  d2 = (d3*theta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2]))(:, 2:end);
  
  D1 = d2'*a1;
  D2 = d3'*a2;
  D3 = d4'*a3;
  
  %Gradients
  Theta1_grad = D1./m(1) + (lambda/m(1))*[zeros(size(theta1,1), 1) theta1(:, 2:end)];
  Theta2_grad = D2./m(1) + (lambda/m(1))*[zeros(size(theta2,1), 1) theta2(:, 2:end)];
  Theta3_grad = D3./m(1) + (lambda/m(1))*[zeros(size(theta3,1), 1) theta3(:, 2:end)];
  grad = [Theta1_grad(:);Theta2_grad(:);Theta3_grad(:)];
  
endfunction
