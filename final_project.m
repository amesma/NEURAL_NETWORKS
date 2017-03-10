%----Shuffle the random number generator----
rng('shuffle');

%--------Set parameters---------
learningRate = 0.9;
secondLearningRate = 0.1;
maxEpochs = 150;


%*****SWITCHES*****
subthresholdTest  = 0;
alternativeSigmoidFoN = 0;
alternativeSigmoidSoN = 0;
sigmoidForComparatorMatrix = 0;
manyRuns = 1;

%simple counter to store all errors
%[Not sure if we are using this secondNetUnit variable]
%secondNetUnit = 0;
wtaCorrected = zeros(100,200);

%----------Winner Take All Parameters (first type of network)-------
%not changing
dim = 100;
half_dim = 50;

upper_limit = 1;
lower_limit = 0;
%experimental
epsilon = 2.3; %keep
length_constant = 0.85;
wta_itr = 10;
max_strength = 5;


%-----Massive Loop for 15 runs-----
runs = 0;

if(manyRuns == 1)
    maxRuns = 15;
elseif (manyRuns == 0)
    maxRuns = 1;
end

%Huge matrix to store all the FoN and SoN accuracy rates across the runs
runsPerformanceStoreFoN = zeros(maxEpochs,maxRuns);
runsPerformanceStoreSoN = zeros(maxEpochs,maxRuns);

while (runs < maxRuns)
runs = runs + 1;

%--------Randomly Generate the input for Pre-Training---------

son_Output = nan(2,200);
fon_Correct = nan(2,200);

%Create 100x100 random numbers from 0.00 to 0.02
StimulusInputStore = rand(100)/50;
StimulusTargetStore = zeros(100,100);

NoiseInputStore = rand(100)/50; %range from 0 to 0.02
NoiseTargetStore = zeros(100,100);

%Create a vector to store the loci of the stimulus
stimulusLoci = zeros(100,1);

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

%Create a store for performance every epoch
epochPerformanceStoreFoN = zeros(maxEpochs,1);
epochPerformanceStoreSoN = zeros(maxEpochs,1);

%Create a store for performance for every vector
tempPerformanceStoreFoN = zeros(200,1);
tempPerformanceStoreSoN = zeros(200,1);

%--------------------------START OF ALL TRAINING-----------------------------

while (epochs < maxEpochs)
    %Pre-Training of Data
    for i = 1:200
        
        %----------------------------------------------------
        %----------------FoN within 1:200 loop---------------
        %----------------------------------------------------

        %Run the vector through the assocation matrix one at a time
        if (alternativeSigmoidFoN == 0)
            inputPattern = RandBothInputStore(:,i);
            input_to_hidden = w_fg * inputPattern; %60
            hidden_activation = activation_fn(input_to_hidden);
            input_to_output = w_gh * hidden_activation; %100
            output_activation = activation_fn(input_to_output);
        elseif (alternativeSigmoidFoN == 1)
            inputPattern = RandBothInputStore(:,i);
            input_to_hidden = w_fg * inputPattern; %60
            hidden_activation = activation_fn_2(input_to_hidden);
            input_to_output = w_gh * hidden_activation; %100
            output_activation = activation_fn_2(input_to_output);
        end
        %Store the output_activation into output_activation_store
        output_activation_store(:,i) = output_activation;

        %Calculate the output error
        output_error = RandBothTargetStore(:,i) - output_activation;
     
        % Create the required change in weights by backpropagation
        if (alternativeSigmoidFoN == 0)
            dw_fg = changeW_FG(learningRate, w_fg, inputPattern, w_gh, output_error, hidden_activation);
            dw_gh = changeW_GH(learningRate,w_gh,hidden_activation,output_error);    
        elseif (alternativeSigmoidFoN == 1)
            dw_fg = changeW_FG2(learningRate, w_fg, inputPattern, w_gh, output_error, hidden_activation);
            dw_gh = changeW_GH2(learningRate,w_gh,hidden_activation,output_error); 
        end
        % Add the changes to the current weights to fix them
        w_fg = w_fg + dw_fg;
        w_gh = w_gh + dw_gh;
        
        %Determine judgment at this point to train SoN
        stimulusPresent = false;
        for j = 1:100
            if (output_activation(j,1) > 0.5)
               stimulusPresent = true; 
            end
        end
        
        % Determine if the FoN is making a correct judgment
        if ((stimulusPresent == true && correctTrials(i,1) == 1) || (stimulusPresent == false && correctTrials(i,1) == 0))
           FoNIsCorrect = true;
           tempPerformanceStoreFoN(i,1) = 1;
        elseif  ((stimulusPresent == true && correctTrials(i,1) == 0) || (stimulusPresent == false && correctTrials(i,1) == 1))
           FoNIsCorrect = false;
           tempPerformanceStoreFoN(i,1) = 0;
        else
            disp(string('Error in determining if FONIsCorrect.'));
        end
        
        %------------Accuracy of FoN-------------
         % Set up the target vector according to the FoN's accuracy
        if (FoNIsCorrect == true) 
            targetVector = [1;0];
            fon_Correct(:,i) = targetVector;
        elseif (FoNIsCorrect == false) 
            targetVector = [0;1]; 
            fon_Correct(:,i) = targetVector;
        else
            disp(string('Error in generating targetVector in top 1:200 loop.'));
        end
        
        %----------------------------------------------------
        %----------------SoN within 1:200 loop---------------
        %----------------------------------------------------
        
        % Generate Comparison vector
        %comparator matrix is initial matrix * 1 weight + result matrix * -1
        comparisonMatrix = inputPattern - output_activation;
        sum_1 = mean(abs(comparisonMatrix));
        
        % Run it through the SoN
        inputPattern_2 = comparisonMatrix;
        if (sigmoidForComparatorMatrix == 1)
            if (alternativeSigmoidSoN == 0)
                inputPattern_2 = activation_fn(inputPattern_2);
            elseif (alternativeSigmoidSoN == 1)
                inputPattern_2 = activation_fn_2(inputPattern_2);
            end
        end
        
        input_to_output_2 = w_co * inputPattern_2;
        if (alternativeSigmoidSoN == 0)
            output_activation_2 = activation_fn(input_to_output_2);
        elseif (alternativeSigmoidSoN == 1)
            output_activation_2 = activation_fn_2(input_to_output_2);
        end
        
        %Determine SoN decision
        if (output_activation_2(1,1) < output_activation_2(2,1))
            highWager = false;
        elseif (output_activation_2(1,1) > output_activation_2(2,1))
            highWager = true;
        end
        
        %Determine is SoN is correct
        if((FoNIsCorrect == true && highWager == true)||(FoNIsCorrect == false && highWager == false))
            tempPerformanceStoreSoN(i,1) = 1;
        elseif ((FoNIsCorrect == true && highWager == false)||(FoNIsCorrect == false && highWager == true))
            tempPerformanceStoreSoN(i,1) = 0;
        end
        
        
        %Generate the error vector for SoN
        %backpropagate through one set of hidden units
        %output_error_2 = sum_1 - output_activation_2;
        output_error_2 = targetVector - output_activation_2;
        
        % Calculate the desired change in weights for w_co
        
        if (alternativeSigmoidSoN == 0)
            dw_co = changeW_GH(secondLearningRate,w_co,inputPattern_2,output_error_2);
        elseif (alternativeSigmoidSoN == 1)
            dw_co = changeW_GH2(secondLearningRate,w_co,inputPattern_2,output_error_2);
        end
        
        % Change the weights
       w_co = w_co + dw_co;
       son_Output(:,i) = output_activation_2;
     
    %End of the for loop that runs through all the columns in the matrix
    end
    %*******************************************************************
    %********************START OF OUTER LOOP****************************
    %*******************************************************************
    
    % Increment the epochs to keep track of the current epoch and to see if
    %the maximum epoch has been reached
    epochs = epochs + 1;
    
    %----------------------------------------
    % ---------------Outer FoN---------------
    %----------------------------------------
    
    %Store the FoN proportion correct in this epoch into the performance
    %store
    epochPerformanceStoreFoN(epochs,1) = mean(tempPerformanceStoreFoN);
    
    %Store into the performance store for Runs
    runsPerformanceStoreFoN(:,runs) = epochPerformanceStoreFoN;
    
    % Run the entire pattern through the associator to obtain the errors all
    % at once, without changing the weights. This is to see the progress of
    % the associator at this epoch.
    if (alternativeSigmoidFoN == 0)
        inputPatternOuter = RandBothInputStore;
        input_to_hidden_outer = w_fg*inputPatternOuter;
        hidden_activation_outer = activation_fn(input_to_hidden_outer);
        input_to_output_outer = w_gh*hidden_activation_outer;
        output_activation_outer = activation_fn(input_to_output_outer);
    elseif (alternativeSigmoidFoN == 1)
        inputPatternOuter = RandBothInputStore;
        input_to_hidden_outer = w_fg*inputPatternOuter;
        hidden_activation_outer = activation_fn_2(input_to_hidden_outer);
        input_to_output_outer = w_gh*hidden_activation_outer;
        output_activation_outer = activation_fn_2(input_to_output_outer);
    end
    
      
    
    %Calculate the error for the entire matrix for FoN
    output_error_outer = RandBothTargetStore - output_activation_outer;
    
    % Calculate the sum of squares for FoN
    sse = trace(output_error_outer' * output_error_outer);
    

    
    % Store the current sum of squares for FoN in the sum of squares store vector
    sseStore(epochs,1) = sse;
    
    % Every 10 epochs, show what the sum of squares is
    if mod(epochs,10) == 0
       %disp(string('epoch = ') + epochs + string(', sse value of FoN = ') + sse);
       %ames commented this out because it was making output too long
    end
    
    %----------------------------------------
    % ---------------Outer SoN---------------
    %----------------------------------------  
    
    %Store the FoN proportion correct in this epoch into the performance
    %store
    epochPerformanceStoreSoN(epochs,1) = mean(tempPerformanceStoreSoN);
    
    %Store into the performance for Runs
    runsPerformanceStoreSoN(:,runs) = epochPerformanceStoreSoN;
    
    compMatrix = RandBothInputStore - output_activation_outer;
    % f weight is 1, h weight is -1
    
    %[Notes]
    %compMatrix is 100 x 200
    %inputPattern_2 is 100 x 1
    
    % Run the entire thing through the SoN to test, like FoN above
    inputPatternOuter_2 = compMatrix;
    if (sigmoidForComparatorMatrix == 1)
        if (alternativeSigmoidSoN == 0)
            inputPatternOuter_2 = activation_fn(inputPatternOuter_2);
        elseif (alternativeSigmoidSoN == 1)
            inputPatternOuter_2 = activation_fn_2(inputPatternOuter_2);
        end
    end
    input_to_hidden_outer_2 = w_co * inputPatternOuter_2;
    if (alternativeSigmoidSoN == 0)
        output_activation_outer_2 = activation_fn(input_to_hidden_outer_2);
    elseif (alternativeSigmoidSoN == 1)
        output_activation_outer_2 = activation_fn_2(input_to_hidden_outer_2);
    end
    
    % Calculate the error for the entire matrix for SoN
    output_error_outer_2 = targetVectorStore - output_activation_outer_2;
    
    % Calculate the sum of squares for SoN
    sse_2 = trace(output_error_outer_2' * output_error_outer_2);
    
    % Store the current sum of squares for SoN in the sum of squares store vector
    sseStore_2(epochs,1) = sse_2;
    
    % End of the while loop for the epochs        
end

%--------------------------END OF ALL TRAINING-----------------------------


%-----------Plot the sse--------------
% Plot sse for FoN

%Plot the sse
%{
 figure(1);
       plot(sseStore(1:epochs,1));
       title('FoN ssError Plot');
       xlabel('epoch');
       ylabel('sse');
   %}    
 % Plot sse for SoN      
 %{
figure(2);
       plot(sseStore_2(1:epochs,1));
       title('SoN ssError Plot');
       xlabel('epoch');
       ylabel('sse_2');
 %}



 
 
 
       
%--------Create Counters for the testing loop--------
       
%Create counter for correct assessments
correctAssess = 0;
correctAssess_2 = 0;


% Create counters for high and low wagers
highWagerCount = 0;
lowWagerCount = 0;

% Create counters for SDT
hitsCountFoN= 0;
faCountFoN = 0;
missCountFoN = 0;
crCountFoN = 0;


hitsCountSoN= 0;
faCountSoN = 0;
missCountSoN = 0;
crCountSoN = 0;

%------Set up Subthreshold stimuli for testing--------

%[Comment chunk below for suprathreshold stimuli, uncomment for
%subthreshold stimuli]
if (subthresholdTest == true)
    disp(string('Testing Subthreshold Stimuli'));
    randStimulusLoci = zeros(1,200);
    for i = 1:200
        for j = 1:100
           if(RandBothTargetStore(j,i) == 1)
              randStimulusLoci(1,i) = j; 
           end
        end
    end

    %Add noise (+0.0012) to every input of the FoN
    RandBothInputStore = RandBothInputStore + 0.0012;

    %Subtract the noise from the stimulus inputs
    for i = 1:200
       if(randStimulusLoci(1,i) > 0)
           RandBothInputStore(randStimulusLoci(1,i),i) = RandBothInputStore(randStimulusLoci(1,i),i) - 0.0012;
       elseif (randStimulusLoci(1,i) == 0)
           %Intentionally blank to test for error below
       else
           disp(string('Error in subtracting noise from stimulus.'));
       end
    end
end
%[Comment/Uncomment for supra/subthreshold stimuli ends]

%--------Loop for Testing--------

%Changeinput to test network=======================================

%RandBothInputStore(100,200) = -1;



for i = 1:200
    
        %--------Decision of FON--------

        %Run the vector through the assocation matrix
        inputPattern = RandBothInputStore(:,i);%new_fon(:,i);
        
        if (alternativeSigmoidFoN == 0)
            input_to_hidden = w_fg * inputPattern;
            hidden_activation = activation_fn(input_to_hidden);
            input_to_output = w_gh * hidden_activation;
            output_activation = activation_fn(input_to_output);
        elseif (alternativeSigmoidFoN == 1)
            input_to_hidden = w_fg * inputPattern;
            hidden_activation = activation_fn_2(input_to_hidden);
            input_to_output = w_gh * hidden_activation;
            output_activation = activation_fn_2(input_to_output);
        end
        %Set Boolean switch to noise by default until it encounters a value
        %that is above threshold
        stimulusPresent = false;
        
        for j = 1:100
            if (output_activation_store(j,i) > 0.5)
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
        
        % Determine the correct answer and set the boolean value to reflect
        % accuracy
        
        if(correctTrials(i,1) == 1 && stimulusPresent == true)
            correctAssess = correctAssess + 1;
            disp(string('   ++ This assessment is correct.'));
            FoNIsCorrect = true;
            hitsCountFoN = hitsCountFoN + 1;
        elseif (correctTrials(i,1) == 0 && stimulusPresent == false)
            correctAssess = correctAssess + 1;
            disp(string('   ++ This assessment is correct.'));
            FoNIsCorrect = true;
            crCountFoN = crCountFoN + 1;
        elseif (correctTrials(i,1) == 1 && stimulusPresent == false)
            disp(string('   -- This assessment is incorrect.'));
            FoNIsCorrect = false;
            missCountFoN = missCountFoN + 1;
        elseif (correctTrials(i,1) == 0 && stimulusPresent == true)
            disp(string('   -- This assessment is incorrect.'));
            FoNIsCorrect = false;
            faCountFoN = faCountFoN + 1;
        end
        
        %--------Decision of SON--------
        %ames jump here
        

       output_activation_store(100,200) = 1;

        compMatrix = RandBothInputStore(:,i) - output_activation_store(:,i);
            
        %Run the vector through the association matrix
        %{
        inputPattern_2 = compMatrix;
        if (sigmoidForComparatorMatrix == 1)
            if (alternativeSigmoidSoN == 0)
                inputPattern_2 = activation_fn(inputPattern_2);
            elseif (alternativeSigmoidSoN == 1)
                inputPattern_2 = activation_fn_2(inputPattern_2);
            end
        end
        input_to_hidden_2 = w_co * inputPattern_2;
        
        if (alternativeSigmoidSoN == 0)
            output_activation_2 = activation_fn(input_to_hidden_2);
        elseif (alternativeSigmoidSoN == 1)
            output_activation_2 = activation_fn_2(input_to_hidden_2);
        end
        
        %test_1 = output_activation_2;
        %}
         %----------------WTA Implementation for SoN-----------------
        %first type of WTA network
        %inhibit_weights_son = makeInhibitoryWeights(2,1,epsilon,max_strength);
        %output_activation_2_wta = compute_inhibited_vect(inhibit_weights_son,output_activation_2,output_activation_2, 2, wta_itr, epsilon);  

        %output_activation_2_wta = son_Output(:,i);
        output_activation_2_wta = output_activation_2;


        output_activation_2_wta = son_Output(:,i);
        %output_activation_2_wta = output_activation_2;

        %Comparison to make a decision
        if (output_activation_2_wta(1,1) < output_activation_2_wta(2,1))
            %disp(string('      The SoN decision is Low wager.'));
            highWager = false;
            lowWagerCount = lowWagerCount + 1;
        elseif (output_activation_2_wta(1,1) > output_activation_2_wta(2,1))
            %disp(string('      The SoN decision is High wager.'));
            highWager = true;
            highWagerCount = highWagerCount + 1;
        else
             disp(string('Error in making a SoN Decision.'));
        end
        disp(output_activation_2_wta);
        
        % Set up the target vector according to the FoN's accuracy
        if (FoNIsCorrect == true) 
            targetVector = [1;0];
        elseif (FoNIsCorrect == false) 
            targetVector = [0;1]; 
        else
            disp(string('Error in generating targetVector in bottom 1:200 loop.'));
        end
        
        
        if (highWager == true &&  FoNIsCorrect == true)
            disp(string('         The SoN assessment is correct!!! '));
            correctAssess_2 = correctAssess_2 + 1;   
            hitsCountSoN = hitsCountSoN + 1;
        elseif (highWager == false &&  FoNIsCorrect == false)
            disp(string('         The SoN assessment is correct!!! '));
            correctAssess_2 = correctAssess_2 + 1;   
            crCountSoN = crCountSoN + 1;
        elseif (highWager == true &&  FoNIsCorrect == false)
            disp(string('         The SoN assessment is incorrect.... '));
            faCountSoN = faCountSoN + 1;
        elseif (highWager == false &&  FoNIsCorrect == true)
            disp(string('         The SoN assessment is incorrect.... '));
            missCountSoN = missCountSoN + 1;
        end
             
end
%----------------------END OF TESTING------------------------------------


%{
%Plot the performance per epoch
plot(epochPerformanceStoreFoN);
hold;
plot(epochPerformanceStoreSoN);
ylim([0 1]);
title('Performance in Recognition and Wagering');
       xlabel('epoch');
       ylabel('Performance');
       xticks([0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150]);
       legend('First Order Network', 'Second Order Network');

 %}
 
%Display if subthreshold test or suprathreshold test
disp(' ');
if subthresholdTest == true
    disp(string('Subthreshold Test:'))
elseif subthresholdTest == false
    disp(string('Suprathreshold Test:'))
else
    disp(string('Error in determining the kind of test.'))
end
%Display if alternative sigmoid function
if (alternativeSigmoidFoN == 0)
    disp(string('FoN: Usual sigmoid function.'));
elseif (alternativeSigmoidFoN == 1)
    disp(string('FoN: Alternative sigmoid function.')); 
end
if (alternativeSigmoidSoN == 0)
    disp(string('SoN: Usual sigmoid function.'));
elseif (alternativeSigmoidSoN == 1)
    disp(string('SoN: Alternative sigmoid function.')); 
end
disp(' ');

%Display SDT numbers for FoN
disp(string('FoN:'));
disp(string('Hits Count: ') + hitsCountFoN/2);
disp(string('False Alarms Count: ') + faCountFoN/2);
disp(string('Correct Rejection Count: ') + crCountFoN/2);
disp(string('Miss Count: ') + missCountFoN/2);
disp(' ');

%Display total correct assessments
disp('Correct assessment for FoN');
disp(correctAssess/2);
disp('Correct assessment for SoN');
disp(correctAssess_2/2);


%Display wager count
disp(string('High Wager Count: ') + highWagerCount/2);
disp(string('Low Wager Count: ') + lowWagerCount/2);
disp(' ');

%Display SDT numbers for SoN
disp(string('SoN:'));
disp(string('Hits Count: ') + hitsCountSoN/2);
disp(string('False Alarms Count: ') + faCountSoN/2);
disp(string('Correct Rejection Count: ') + crCountSoN/2);
disp(string('Miss Count: ') + missCountSoN/2);

end
%^^^ This end here belongs to the massive while loop all the way at the top
%(for number of runs, line 45 or so.)

%average out the performances for each epoch
averagePerformanceFoN = mean(runsPerformanceStoreFoN,2);
averagePerformanceSoN = mean(runsPerformanceStoreSoN,2);

%Plot the performance per epoch
plot(averagePerformanceFoN);
hold;
plot(averagePerformanceSoN);
ylim([0 1]);
title('Performance in Recognition and Wagering');
       xlabel('epoch');
       ylabel('Performance');
       xticks([0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150]);
       legend('First Order Network', 'Second Order Network');