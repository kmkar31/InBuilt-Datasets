This program uses a Recurrent Neural Network (RNN) to train the data . A Recurrent Neural Network contains information about its 'history' 
i.e., its current computation is affected by its previous computations .
This is mainly used for data that contains some notion of a sequence like sentences , videos , stock predictions etc .
For this model , the 28x28 imge is taken and considered as a set of 28 "timesteps"(read columns), each a 28x1 vector of pixel values .
In this approach , the model learns how a number would be written if it was created one column at a time .
Using this knowledge , it predicts an unknown image . 
