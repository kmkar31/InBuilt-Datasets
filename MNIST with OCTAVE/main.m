%Digit Recognition using MNIST dataset
%Author KARTHIK KARUMANCHI

%Loading Data and obtaining training , test seta ad their corresponding labels
I = load('mnist.mat');
Xtrain = double(I.trainX);
ytr = double(I.trainY);
Xtest = double(I.testX);
yt = double(I.testY);

% mtrain/mtest are the number of examples in the training and test sets
mtrain = size(Xtrain(:,1));
n = size(Xtrain(1,:));
mtest = size(Xtest(:,1));

%converting labels from integers to an array of 10 integers
% These arrays have 0's everywhere except at te integer index where it has 1
t = zeros(1,10);
ytrain = zeros(mtrain,10);
ytest = zeros(mtest,10);
for i=1:10
  t(1,i) = i-1;
endfor  

for i=1:mtrain(1)
  ytrain(i,:) = t==ytr(i);
  
endfor

for i=1:mtest(1)
  ytest(i,:) = t==yt(i);
endfor

%Starting train
% h = hidden layer size = 50
h1 = 200;
h2 = 200;
lambda = 15;

%randomly initializing theta to be between -epsilon and +epsilon
epsilon = 0.2;
theta1_init = rand(h1,(n(2)+1))*(2*epsilon) - epsilon;
theta2_init = rand(h2,(h1+1))*(2*epsilon) - epsilon;
theta3_init = rand(10,(h2+1))*(2*epsilon) - epsilon;
%unrolling theta1 and theta2 into a single vector theta_init
%in order to run advanced optimisations like fmincg
theta_init = [theta1_init(:);theta2_init(:);theta3_init(:)];



%Running fmincg
options = optimset('MaxIter', 50);
Propagations = @(p) propagations(Xtrain, ytrain,mtrain,n,h1,h2,p,lambda);
[theta_optimised , cost] = fmincg(Propagations, theta_init , options);



%Running the model on the test set
%Obtaining errors on training and test sets .
%these errors are later used to determine the number of hidden layer units 
%and the value of lambda
result = predict(Xtest,theta_optimised,mtest,n,h1,h2);
errortest = sum(result!=yt');
result2 = predict(Xtrain,theta_optimised,mtrain,n,h1,h2);
errortrain = sum(result2!=ytr');
erortestpercent = (errortest/mtest(1))*100

plot(result , 'r');
hold on;
plot(yt , 'b');
hold off;

%Testing with my own input
R2 = own();
Answer = predict(R2,theta_optimised,1,n,h1,h2)





