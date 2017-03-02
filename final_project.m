%--------Set parameters---------
learningRate = 1;
maxEpochs = 150;
correctAssess = 0;

%--------Randomly Generate the input---------
%Create 100x100 random numbers from 0.00 to 0.02
StimulusInputStore = rand(100)/50;
StimulusTargetStore = zeros(100,100);

NoiseInputStore = rand(100)/50; %range from 0 to 0.02
NoiseTargetStore = zeros(100,100);

%Insert a random [0.00 to 1.00] into one of each vector (column), and
%insert a 1 into the corresponding location in the vector
%each f vector is a column
for i = 1:100
    %Generate the value of the stimulus
    randStimulus = rand;
    %Generate the locus of the stimulus within the vector
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

%Create a vector to keep track of which patterns have the stimulus and
%which are only noise
correctTrials = zeros(200,1);

%Fill in the random input and target stores
% each element column can be either a noise or a stimuli
for i = 1:200
   RandBothInputStore(:,i) = BothInputStore(:,thePermutation(i));
   RandBothTargetStore(:,i) = BothTargetStore(:,thePermutation(i));
   %Fill in the value with 1 if there is stimulus, leave at 0 otherwise
   if(any(RandBothTargetStore(:,i)>0))
       correctTrials(i,1) = 1;
   end
end


%--------Randomly Generate the weights---------
w_fg = (rand([60 100])*2)-1;
w_gh = (rand([100 60])*2)-1;

%--------Run the whole matrix through the epochs--------
epochs = 0;

% Create store for sse
sseStore = zeros(maxEpochs,1);

while (epochs < maxEpochs)
    %Pre-Training of Data
    for i = 1:200

        %Run the vector through the assocation matrix
        inputPattern = RandBothInputStore(:,i);
        input_to_hidden = w_fg * inputPattern;
        hidden_activation = activation_fn(input_to_hidden);
        input_to_output = w_gh * hidden_activation;
        output_activation = activation_fn(input_to_output);

        %Calculate the output error
        output_error = RandBothTargetStore(:,i) - output_activation;

        % Create the required change in weights by backpropagation
        dw_fg = changeW_FG(learningRate, w_fg, inputPattern, w_gh, output_error, hidden_activation);
        dw_gh = changeW_GH(learningRate,w_gh,hidden_activation,output_error);    

        % Add the changes to the current weights to fix them
        w_fg = w_fg + dw_fg;
        w_gh = w_gh + dw_gh;
        

    %End of the for loop that runs through all the columns in the matrix
    end
    
    % ---------Error Calculation---------
    
    % Run the entire pattern through the associator to obtain the errors all
    % at once, without changing the weights. This is to see the progress of
    % the associator at this epoch.
    inputPattern = RandBothInputStore;
    input_to_hidden = w_fg*inputPattern;
    hidden_activation = activation_fn(input_to_hidden);
    input_to_output = w_gh*hidden_activation;
    output_activation = activation_fn(input_to_output);
    output_error = RandBothTargetStore - output_activation;
    
    % Calculate the sum of squares
    sse = trace(output_error' * output_error);
    %[Uncomment below to show sse per epoch]
    %disp(string('sse is ')); disp(sse);
    
    % Increment the epochs to keep track of the current epoch and to see if
    %the maximum epoch has been reached
    epochs = epochs + 1;
    
    % Store the current sum of squares in the sum of squares store vector
    sseStore(epochs,1) = sse;
    
    % Every 10 epochs, show what the sum of squares is
    if mod(epochs,10) == 0
       disp(string('epoch = ') + epochs + string(', sse value = ') + sse);
    end
        
        
% End of the while loop for the epochs        
end

%Plot the sse
 figure(1);
       plot(sseStore(1:epochs,1));
       title('ssError Plot');
       xlabel('epoch');
       ylabel('sse');

%--------Decision of FON--------
for i = 1:200

        %Run the vector through the assocation matrix
        inputPattern = RandBothInputStore(:,i);
        input_to_hidden = w_fg * inputPattern;
        hidden_activation = activation_fn(input_to_hidden);
        input_to_output = w_gh * hidden_activation;
        output_activation = activation_fn(input_to_output);
        
        
        
        %Set Boolean switch to noise by default until it encounters a value
        %that is above threshold
        stimulusPresent = false;
        
        for j = 1:100
            if (output_activation(j,1) >= 0.5)
               stimulusPresent = true; 
            end
        end
        
        
        % Display the results of the decision
        if(stimulusPresent == true)
            disp(string('In pattern ') + i + string(', the stimulus is present.')); 
        elseif (stimulusPresent == false)
            disp(string('In pattern ') + i + string(', the stimulus is absent.'));
        else
            disp(string('There is no way this is supposed to be displaying.'));
        end
        if((correctTrials(i) == 1 && stimulusPresent == true) || (correctTrials(i) == 0 && stimulusPresent == false))
            correctAssess = correctAssess + 1;
            disp(string('   ++ This assessment is correct.'));
        else
            disp(string('   -- This assessment is incorrect.'));
        end
end

disp('Correct assessment');
disp(correctAssess);