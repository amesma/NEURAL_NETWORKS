%--------Set parameters---------



%--------Randomly Generate the input---------
%Create 100x100 random numbers from 0.00 to 0.02
StimulusInputStore = rand(100)/50;
StimulusTargetStore = zeros(100,100);

NoiseInputStore = rand(100)/50;
NoiseTargetStore = zeros(100,100);

%Insert a random [0.00 to 1.00] into one of each vector (column), and
%insert a 1 into the corresponding location in the vector
for i = 1:100
    %Generate the value of the stimulus
    randStimulus = rand;
    %Generate the locu of the stimulus within the vector
    randLocus = randi([1,100]);
    %Insert the stimulus into the input vector
    StimulusInputStore(randLocus,i) = randStimulus;
    %Insert the stimulus into the target vector
    StimulusTargetStore(randLocus,i) = 1;
end


%Shuffle and mix the stimulus and noise into one large 100 x 200 input matrix
%Preallocate memory
BothInputStore = zeros(100,200);
BothTargetStore = zeros(100,200);

%Fill in Inputs from signal and noise
BothInputStore(:,1:100) = StimulusInputStore;
BothInputStore(:,101:200) = NoiseInputStore;

%Fill in Targets from signal and noise
BothTargetStore(:,1:100) = StimulusTargetStore;
BothTargetStore(:,101:200) = NoiseTargetStore;

%Create new stores for random inputs and targets
RandBothInputStore = zeros(100,200);
RandBothTargetStore = zeros(100,200);

%Create a random permutation from 1 to 200
thePermutation = randperm(200);

%Fill in the random input and target stores
for i = 1:200
   RandBothInputStore(:,i) = BothInputStore(:,thePermutation(i));
   RandBothTargetStore(:,i) = BothTargetStore(:,thePermutation(i));
end


%--------Randomly Generate the weights---------
w_fg = (rand([60 100])*2)-1;
w_gh = (rand([100 60])*2)-1;

%--------Run the columns through the associator--------

%Pre-Training of Data
for i = 1:200
    
    %Run the vector through the assocation matrix
    inputVector = RandBothInputStore(:,i);
    input_to_hidden = w_fg * inputVector;
    hidden_activation = activation_fn(input_to_hidden);
    input_to_output = w_gh * hidden_activation;
    output_activation = activation_fn(input_to_output);
    
    %Calculate the output error
    output_error = RandBothTargetStore(:,1) - output_activation;
    
    % Create the required change in weights by backpropagation
    dw_fg = changeW_FG(learningRate, w_fg, inputPattern, w_gh, output_error, hidden_activation);
    dw_gh = changeW_GH(learningRate,w_gh,hidden_activation,output_error);    
    
    % Add the changes to the current weights to fix them
    w_fg = w_fg + dw_fg;
    w_gh = w_gh + dw_gh;
    
end