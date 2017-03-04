%--------Set parameters---------
learningRate = 0.9;
secondLearningRate = 0.9;
maxEpochs = 150;

%simple counter to store all errors
%[Not sure if we are using this secondNetUnit variable]
secondNetUnit = 0;

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
   
   %[Wrong here]
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

% Create a store for the output_activations of FoN
output_activation_store = zeros(100,200);

while (epochs < maxEpochs)
    %Pre-Training of Data
    for i = 1:200
        
        %----------------FoN---------------

        %Run the vector through the assocation matrix
        inputPattern = RandBothInputStore(:,i);
        input_to_hidden = w_fg * inputPattern;
        hidden_activation = activation_fn(input_to_hidden);
        input_to_output = w_gh * hidden_activation;
        output_activation = activation_fn(input_to_output);
        
        %Store the output_activation into output_activation_store
        output_activation_store(:,i) = output_activation;

        %Calculate the output error
        output_error = RandBothTargetStore(:,i) - output_activation;

        % Create the required change in weights by backpropagation
        dw_fg = changeW_FG(learningRate, w_fg, inputPattern, w_gh, output_error, hidden_activation);
        dw_gh = changeW_GH(learningRate,w_gh,hidden_activation,output_error);    

        % Add the changes to the current weights to fix them
        w_fg = w_fg + dw_fg;
        w_gh = w_gh + dw_gh;
        
        %Determine judgment at this point to train SoN
        stimulusPresent = false;
        for j = 1:100
            if (output_activation(j,1) >= 0.5)
               stimulusPresent = true; 
            end
        end
        
        % Determine if the FoN is making a correct judgment
        if ((stimulusPresent == true && correctTrials(i,1) == 1) || (stimulusPresent == false && correctTrials(i,1) == 0))
           FoNIsCorrect = true; 
        elseif  ((stimulusPresent == true && correctTrials(i,1) == 0) || (stimulusPresent == false && correctTrials(i,1) == 1))
           FoNIsCorrect = false;
        else
            disp(string('Error in determining if FONIsCorrect.'));
        end
        
        %----------------SoN---------------
        
        % Generate Comparison vector
        comparisonMatrix = inputPattern - output_activation;
        
        % Run it through the SoN
        inputPattern_2 = comparisonMatrix;
        input_to_output_2 = w_co * inputPattern_2;
        output_activation_2 = activation_fn(input_to_output_2);
        
        % Set up the target vector according to the FoN's accuracy
        if (FoNIsCorrect == true) 
            targetVector = [1;0];
        elseif (FoNIsCorrect == false) 
            targetVector = [0;1]; 
        else
            disp(string('Error in generating targetVector in top 1:200 loop.'));
        end
        
        % Generate the error vector for SoN
        output_error_2 = targetVector - output_activation_2;
        
        % Calculate the desired change in weights for w_co
        dw_co = changeW_GH(secondLearningRate,w_co,inputPattern_2,output_error_2);
        
        % Change the weights
        w_co = w_co + dw_co;
        

    %End of the for loop that runs through all the columns in the matrix
    end
    
    % ---------Outer FoN---------
    
    % Run the entire pattern through the associator to obtain the errors all
    % at once, without changing the weights. This is to see the progress of
    % the associator at this epoch.
    inputPatternOuter = RandBothInputStore;
    input_to_hidden_outer = w_fg*inputPatternOuter;
    hidden_activation_outer = activation_fn(input_to_hidden_outer);
    input_to_output_outer = w_gh*hidden_activation_outer;
    output_activation_outer = activation_fn(input_to_output_outer);
    
    %[Wrong here]
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
    
    
    %----------------- Outer SoN -----------------
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

%w_wta = rand(size(output_activation_2,1)) * 0.1; 
%wta(1, output_activation_2, w_wta, 150);

%-----------Plot the sse--------------
% Plot sse for FoN
 figure(1);
       plot(sseStore(1:epochs,1));
       title('FoN ssError Plot');
       xlabel('epoch');
       ylabel('sse');
       
 % Plot sse for SoN      
 figure(2);
       plot(sseStore_2(1:epochs,1));
       title('SoN ssError Plot');
       xlabel('epoch');
       ylabel('sse_2');

       
%--------Create Counters for the testing loop--------
       
%Create counter for correct assessments
correctAssess = 0;
correctAssess_2 = 0;


% Create counters for high and low wagers
highWagerCount = 0;
lowWagerCount = 0;

% Create counters for SDT
hitsCount = 0;
faCount = 0;
missCount = 0;
crCount = 0;

%--------Loop for Testing--------

for i = 1:200
    
        %--------Decision of FON--------

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
        
        % Determine the correct answer and set the boolean value to reflect
        % accuracy
        
        if((correctTrials(i,1) == 1 && stimulusPresent == true) || (correctTrials(i,1) == 0 && stimulusPresent == false))
            correctAssess = correctAssess + 1;
            disp(string('   ++ This assessment is correct.'));
            FoNIsCorrect = true;
        else
            disp(string('   -- This assessment is incorrect.'));
            FoNIsCorrect = false;
        end
        
        
        %--------Decision of SON--------
        
        compMatrix = RandBothInputStore(:,i) - output_activation_store(:,i);
            
        % Run the vector through the association matrix
        inputPattern_2 = compMatrix;
        input_to_hidden_2 = w_co * inputPattern_2;
        output_activation_2 = activation_fn(input_to_hidden_2);
        
        %Comparison to make a decision
        if (output_activation_2(1,1) > output_activation_2(2,1))
            disp(string('      The SoN decision is High wager.'));
            highWager = true;
            highWagerCount = highWagerCount + 1;
        elseif (output_activation_2(1,1) < output_activation_2(2,1))
            disp(string('      The SoN decision is Low wager.'));
            highWager = false;
            lowWagerCount = lowWagerCount + 1;
        else
             disp(string('Error in making a SoN Decision.'));
        end
        
        
        % Set up the target vector according to the FoN's accuracy
        if (FoNIsCorrect == true) 
            targetVector = [1;0];
        elseif (FoNIsCorrect == false) 
            targetVector = [0;1]; 
        else
            disp(string('Error in generating targetVector in bottom 1:200 loop.'));
        end
        
        
        if (highWager == true &&  FoNIsCorrect == true)
            disp(string('         The SoN assessment is correct!!! :D :D '));
            correctAssess_2 = correctAssess_2 + 1;   
            hitsCount = hitsCount + 1;
        elseif (highWager == false &&  FoNIsCorrect == false)
            disp(string('         The SoN assessment is correct!!! :D :D '));
            correctAssess_2 = correctAssess_2 + 1;   
            crCount = crCount + 1;
        elseif (highWager == true &&  FoNIsCorrect == false)
            disp(string('         The SoN assessment is incorrect.... '));
            faCount = faCount + 1;
        elseif (highWager == false &&  FoNIsCorrect == true)
            disp(string('         The SoN assessment is incorrect.... '));
            missCount = missCount + 1;
        end
        
        
        
        
end
%Display total correct assessments
disp('Correct assessment for FoN');
disp(correctAssess/2);
disp('Correct assessment for SoN');
disp(correctAssess_2/2);

%Display wager count
disp(string('High Wager Count: ') + highWagerCount/2);
disp(string('Low Wager Count: ') + lowWagerCount/2);
disp(' ');

%Display SDT numbers
disp(string('Hits Count: ') + hitsCount/2);
disp(string('False Alarms Count: ') + faCount/2);
disp(string('Correct Rejection Count: ') + crCount/2);
disp(string('Miss Count: ') + missCount/2);

