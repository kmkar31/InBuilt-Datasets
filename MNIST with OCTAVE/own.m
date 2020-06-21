function [R2] = own()  
  
  T = imread('input2(resize).png');
  
  image(T);
  R2 = reshape(T,1,784);
endfunction