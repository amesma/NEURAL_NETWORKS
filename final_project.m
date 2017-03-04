%--------Set parameters---------
learningRate = 0.9;
secondLearningRate = 0.1;
maxEpochs = 150;
correctAssess = 0;
%simple counter to store all errors
secondNetUnit = 0;
wtaCorrected = zeros(100,200);

%----------------Winner Take All Paramters (first type of network)
dim = 100;
half_dim = 50;
upper_limit = 1;
lower_limit = 0;
epsilon = 2;
length_constant = .5;
wta_itr = 2;



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

%Create a store for the Target Vectors of SoN
targetVectorStore = zeros(2,200);

%Fill in the random input and target stores
% each element column can be either a noise or a stimuli
for i = 1:200
   RandBothInputStore(:,i) = BothInputStore(:,thePermutation(i));
   RandBothTargetStore(:,i) = BothTargetStore(:,thePermutation(i));
   
   %Fill in the value with 1 if there is stimulus, leave at 0 otherwise
   if(any(RandBothTargetStore(:,i)>0))
       correctTrials(i,1) = 1;
   end
   
   %Generate target vectors for SoN
   if (correctTrials(i,1) == 1)
       targetVectorStore(:,i) = [1; 0];
   elseif (correctTrials(i,1) == 0)
        targetVectorStore(:,i) = [0; 1];   
   else
       disp(string('Error with generating target vector.'));
   end   
   
end


%--------Randomly Generate the weights for FoN---------
w_fg = (rand([60 100])*2)-1;
w_gh = (rand([100 60])*2)-1;

%--------Randomly Generate the weights for SoN---------
w_co = rand(2,100) * 0.1;

%--------Run the whole matrix through the epochs--------
epochs = 0;

% Create store for sse and sse_2
sseStore = zeros(maxEpochs,1);
sseStore_2 = zeros(maxEpochs,1);

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
     
        
        %first type of WTA network
        inhibit_weights = makeInhibitoryWeights(dim,half_dim,epsilon,length_constant);
        output_activation = compute_inhibited_vect(inhibit_weights,output_activation,output_activation, dim, wta_itr, epsilon);  

          % Create the required change in weights by backpropagation
        dw_fg = changeW_FG(learningRate, w_fg, inputPattern, w_gh, output_error, hidden_activation);
        dw_gh = changeW_GH(learningRate,w_gh,hidden_activation,output_error);    

        % Add the changes to the current weights to fix them
        w_fg = w_fg + dw_fg;
        w_gh = w_gh + dw_gh;
        
        
        %----------------SoN---------------
        % Generate Comparison vector
        comparisonMatrix = inputPattern - output_activation;
        
        % Run it through the SoN
        inputPattern_2 = comparisonMatrix;
        input_to_output_2 = w_co * inputPattern_2;
        output_activation_2 = activation_fn(input_to_output_2);
        
        
        % Generate the error vector for SoN
        output_error_2 = targetVectorStore(:,i) - output_activation_2;
        
        % Calculate the desired change in weights for w_co
        dw_co = changeW_GH(learningRate,w_co,inputPattern_2,output_error_2);
        
        % Change the weights
        w_co = w_co + dw_co;
        
        %second example of WTA network
        %w_wta = rand(size(output_activation_2,1)) * 0.1; 
        %wta(1, output_activation_2, w_wta, 150);

    %End of the for loop that runs through all the columns in the matrix
    end
    
    % ---------Error Calculation---------
    
    % Run the entire pattern through the associator to obtain the errors all
    % at once, without changing the weights. This is to see the progress of
    % the associator at this epoch.
    inputPatternOuter = RandBothInputStore;
    input_to_hidden_outer = w_fg*inputPatternOuter;
    hidden_activation_outer = activation_fn(input_to_hidden_outer);
    input_to_output_outer = w_gh*hidden_activation_outer;
    output_activation_outer = activation_fn(input_to_output_outer);
    
    %Calculate the error for the entire matrix for FoN
    output_error_outer = RandBothTargetStore - output_activation_outer;
    
    % Calculate the sum of squares for FoN
    sse = trace(output_error_outer' * output_error_outer);
    %[Uncomment below to show sse per epoch]
    %disp(string('sse is ')); disp(sse);
    
    % Increment the epochs to keep track of the current epoch and to see if
    %the maximum epoch has been reached
    epochs = epochs + 1;
    
    % Store the current sum of squares for FoN in the sum of squares store vector
    sseStore(epochs,1) = sse;
    
    % Every 10 epochs, show what the sum of squares is
    if mod(epochs,10) == 0
       disp(string('epoch = ') + epochs + string(', sse value of FoN = ') + sse);
    end
    
    
    %---------------------------------------SON
    compMatrix = RandBothInputStore - output_activation_outer;
    % f weight is 1, h weight is -1
    
    %[Notes]
    %compMatrix is 100 x 200
    %inputPattern_2 is 100 x 1
    
    % Run the entire thing through the SoN to test, like FoN above
    inputPatternOuter_2 = compMatrix;
    input_to_hidden_outer_2 = w_co * inputPatternOuter_2;
    output_activation_outer_2 = activation_fn(input_to_hidden_outer_2);
    
    % Calculate the error for the entire matrix for SoN
    output_error_outer_2 = targetVectorStore - output_activation_outer_2;
    
    % Calculate the sum of squares for SoN
    sse_2 = trace(output_error_outer_2' * output_error_outer_2);
    
    % Store the current sum of squares for SoN in the sum of squares store vector
    sseStore_2(epochs,1) = sse_2;
    
    
    % End of the while loop for the epochs        
end

%Plot the sse
 figure(1);
       plot(sseStore(1:epochs,1));
       title('FoN ssError Plot');
       xlabel('epoch');
       ylabel('sse');
       
 figure(2);
       plot(sseStore_2(1:epochs,1));
       title('SoN ssError Plot');
       xlabel('epoch');
       ylabel('sse_2');


%--------Decision of FON--------
for i = 1:200

        %Run the vector through the assocation matrix
        inputPattern = RandBothInputStore(:,i);%new_fon(:,i);
   
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
              % disp(output_activation(j,1));
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
