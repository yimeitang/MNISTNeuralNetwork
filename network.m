function network(inputs,targets,nodelayers,numEpochs,batchSize,eta)


%%%Step 1
%%%stack inputs and targets vertically for future random shuffle
training_data = [inputs;targets];
num_examples = size(inputs,2);
num_features = size(inputs,1);

%%%Step 2
%%%Initialize weights and bias for each layer
%%%We don't need weights and bias for the first layer(input layer)
%%%However, for future reference's convenience, we will leave the first
%  layer's weights and bias as empty instead of creating from the second
%  layer
num_layers = size(nodelayers,2);
weights = cell(1, num_layers);
delta_nabla_weights= cell(1, num_layers);
delta_nabla_bias= cell(1, num_layers);
bias=cell(1,num_layers);

for i = 2:num_layers
    weights{i} = randn(nodelayers(i),nodelayers(i-1));
    bias{i}=randn(nodelayers(i),1);
end

%%%Step 3
%%%Choose an activation function
%%%We will be using sigmoid function here 
%%%Improvements: we could have a seperate function designed for 
%  activation funtions.Therefore, we could test different activation
%  functions easily

%%%Step 4 
%%%Randomly shuffle the training data(input data + target data) then 
%  divide the into batches of input_data and batches of target data.
num_of_batches = num_examples/batchSize;
mini_batches = cell(1,num_of_batches);
for i=1:numEpochs
    training_data_shuffled= training_data(:,randperm(size(training_data,2)));
    for j=1:num_of_batches
        mini_batches{j} = training_data_shuffled(:,((j-1)*batchSize+1):j*batchSize);
    end
    
%%%Step 5     
%%%For each mini_batch, we calculate the zvalues and activations values
%  and use back propogation to calculate gradient descent in vectorized
%  function instead of using many for loops
    for j=1:num_of_batches
        zvalues=cell(1,num_layers);
        activations=cell(1,num_layers);
        activations{1}=mini_batches{j}(1:num_features,:);
        
        
%%%It is easy to make mistakes here because we need to keep in mind that
%  the each mini_batch has the input features and target outputs
        mini_batch_targets = mini_batches{j}(num_features+1:end,:);
       
%%%Feed forward
        for k=2:num_layers
            zvalues{k}=weights{k}*activations{k-1}+bias{k};
            activations{k}= 1.0 ./ (1.0 + exp(-zvalues{k}));
        end
        
%%%First, calculate the delta,delta nabla of weights and delta nabla of 
%  bias of output layer
        activation_output=1.0 ./ (1.0 + exp(-zvalues{end}));
        delta_output_layer = (activations{end}-mini_batch_targets)...
                             .*((activation_output).*(1-activation_output));
        delta_nabla_weights{end}=eta*(batchSize .\((delta_output_layer*(activations{end-1})'))); 
        delta_nabla_bias{end}= eta*mean(delta_output_layer,2);
    
        
%%%Then, we do the same things for other layers(except for input layer)        
        delta_previous_layer = delta_output_layer;
        for k=(num_layers-1):-1:2
            activation_output=1.0 ./ (1.0 + exp(-zvalues{k}));
            delta_this_layer = (((weights{k+1})')*delta_previous_layer)...
                .*((activation_output).*(1-activation_output));
            delta_nabla_weights{k}=eta*(batchSize .\((delta_this_layer*(activations{k-1})'))); 
            delta_nabla_bias{k}= eta*mean(delta_this_layer,2);
            delta_previous_layer = delta_this_layer;
        end
        
%%%Last, we could update weights and bias after each mini_batch by the
%  calculations we done in the previous steps
        for k=2:num_layers
            weights{k}=weights{k}-delta_nabla_weights{k};
            bias{k}=bias{k}-delta_nabla_bias{k};
        end
    end

%%%Step 6
%%%After each epoch, calculate MSE and Accuracy by using the final updated
%  version of weights and bias. Here is another round of feed forward,the
%  only difference is that we are using the whole training data in this
%  process instead of using 1 mini batch
    zvalues=cell(1,num_layers);
    activations=cell(1,num_layers);
    activations{1}=training_data_shuffled(1:num_features,:);
    training_data_targets = training_data_shuffled(num_features+1:end,:);
    for k=2:num_layers
        zvalues{k}=weights{k}*activations{k-1}+bias{k};
        activations{k}=1.0 ./ (1.0 + exp(-zvalues{k}));
    end

%%%Calculate the MSE
    D = abs(activations{end}-training_data_targets).^2;
    MSE = sum(D(:))/numel(activations{end});

%%%Calculate the Accuracy score
%%%Improvement: I could have thought of better and simpler solutions for
%  this function

%%%Scenario 1: the output layer has more than 1 node
    if size(training_data_targets,1)>1
        max_index_of_output = (activations{end}==max(activations{end}));
        max_index_of_targets = (training_data_targets==max(training_data_targets));
        correct_items=sum(mean(max_index_of_output==max_index_of_targets)==1);
    end
    
%%%Scenario 2: the output layer has only 1 node
    if size(training_data_targets,1)==1
        one_index = find(activations{end}>0.5);
        activations{end}(one_index)=1;
        correct_items=sum(activations{end}==training_data_targets);
    end
    
    Accuracy=  correct_items/num_examples;
    fprintf('Epoch %3d, MSE:%6.2f, Correct: %6d /%6d, Acc: %6.3f\n',i,MSE,...
             correct_items,num_examples,Accuracy);
    
         
%%% Set target Accuracy 
%%% Here, I set it as 0.95
%%% Improvements: I could have a parameter in the function arguments to
%   specify this target accuracy
    if Accuracy > 0.95
        break
    end

%%%Step 7
%%%Repeat step 4 to step 6 for each epoch(if there is more than 1)
end

end

