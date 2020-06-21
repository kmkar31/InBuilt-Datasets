function [] = optimise(Xtrain,ytrain,ytr,Xtest,yt,mtrain,mtest,n,h1,theta_init)
  
  error = zeros(8);
  h2 = [10,20,30,40,50,50,70,80];
  lambda = [0.002,0.0033,0.02,0.033,0.2,0.33,2,3.33];
  for i=1:8
    for j=1:8
      options = optimset('MaxIter', 50);
      Propagations = @(p) propagations(Xtrain, ytrain,mtrain,n,h1,h2(i),p,lambda(j));
      [theta_optimised , cost] = fmincg(Propagations, theta_init , options);
      
      Rtest = predict(Xtest,theta_optimised,mtest,n,h1,h2(i));
      error(i,j) = sum(Rtest!=yt');
    endfor
 endfor 

  meshc(error,h2,lambda); 
endfunction
