function [weights,bias]=network(inputs,targets,split,nodelayers,...
    numEpochs,batchSize,eta,costFunOpetion,actFunOption,momentum,lambda)
%%
%%%Step 1 Prep work
%%%(1) Stack inputs and outputs vertically and shuffle randomly
%%%(2) Split the randomly shuffled dataset into training set, test set and
%      validation set
%%%(3) Preallocate space to store training set and test sets'cost scores
%      and accuracy scores so to plot them after all epochs
%%%(4) Print the format


%%
%%%(1)Stack inputs and outputs vertically and shuffle randomly
%%%If the activation is tanh, then we need to normalize the input data
all_data = [inputs;targets];
all_data_shuffled= all_data(:,randperm(size(all_data,2)));
if actFunOption == 1
    all_data_shuffled=zscore(all_data_shuffled);
end

%%
%%%(2)split the randomly shuffled dataset into training set, test set and
%    validation set
[train_set,test_set,val_set] = dividerand(all_data_shuffled,...
                                split(1)/sum(split),split(2)/sum(split),...
                                split(3)/sum(split));
num_train_examples = size(train_set,2); 
num_test_examples = size(test_set,2);
num_val_examples = size(val_set,2);
num_features = size(inputs,1);


%%
%%%(3)Preallocate space to store training set and test sets'cost scores
%      and accuracy scores so to plot them after all epochs
train_cost_all=zeros(1,numEpochs);
train_accu_all=zeros(1,numEpochs);
test_cost_all=zeros(1,numEpochs);
test_accu_all=zeros(1,numEpochs);


%%
%%% (4)Print the format
fprintf('   |           TRAIN                ||              TEST              ||     VALIDATION\n')
fprintf('------------------------------------------------------------------------------------------------------\n')
fprintf('Ep | Cost  |      Corr       |  Acc ||  Cost |      Corr      | Acc   || Cost|       Corr       | Acc\n')
fprintf('------------------------------------------------------------------------------------------------------\n')


%%
%%%Step 2 Initialize Weights and Bias
%%%(1)Preallocate space each layer's parameters
%%%(2)Initialize weights , bias and delta nablas for each layer

%%
%%%(1)Preallocate space each layer's parameters
num_layers=length(nodelayers);
weights = cell(1, num_layers);                                              
bias=cell(1,num_layers);
delta_nabla_weights= cell(1, num_layers);
delta_nabla_bias= cell(1, num_layers);

%%
%%%(2)Initialize weights , bias and delta nablas for each layer
for i = 2:num_layers
    weights{i} = randn(nodelayers(i),nodelayers(i-1))*(1/sqrt(nodelayers(i-1)));
    bias{i}=randn(nodelayers(i),1);
end

%%
%%%Step 3 Handle Activation Functions
%%%(1)Create function handler for activation functions and according primes


%%
%%%(1)Create function handler for activation functions and according primes
%   if actFunOption == 0 -> sigmoid
%   if actFunOption == 1 -> tanh
%   if actFunOption == 2 -> softmax
%   if actFunOption == 3 -> ReLu
        
        if actFunOption == 0 
            actFun=@Sigmoid_act;
            actFunPrime=@SigmoidPrime;
            outputFun=@Sigmoid_act;
            outputFunPrime=@SigmoidPrime;
        elseif actFunOption ==1
            actFun=@tanh;
            actFunPrime=@TanhPrime;
            outputFun=@tanh;
            outputFunPrime=@TanhPrime;
        elseif actFunOption == 2
            actFun=@Sigmoid_act;
            actFunPrime=@SigmoidPrime;
            outputFun=@softmax;
            outputFunPrime=@SoftmaxPrime;
        else  %actFunOption == 3
            actFun=@Sigmoid_act;
            actFunPrime=@SigmoidPrime;
            outputFun=@poslin;
            outputFunPrime=@ReluPrime;
        end


%%
%%%Step 4 Work with each mini batch
%%%(1)Create for loop to iterate through epochs
%%%(2)In each epoch, create mini batches
%%%(3)Feed forward in each mini batch by using updated weights and bias
%%%(4)Back propagation in each mini batch
%%%(5)Update the weights and bias by the end of each mini batch



%%%(1)Randomly shuffle training set starting in each epoch
num_of_batches = num_train_examples/batchSize;
for i=1:numEpochs
    training_data_shuffled= train_set(:,randperm(size(train_set,2)));



%%%(2)In each epoch, create mini batches
    for j=1:num_of_batches
        current_mini_batch = training_data_shuffled(:,((j-1)*batchSize+1):j*batchSize);


%%%(3)Feed forward in each mini batch by using updated weights and bias
        zvalues=cell(1,num_layers);
        activations=cell(1,num_layers);
        activations{1}=current_mini_batch(1:num_features,:);
        mini_batch_targets = current_mini_batch(num_features+1:end,:);
        
        for k=2:num_layers-1
            zvalues{k}=weights{k}*activations{k-1}+bias{k};
            activations{k}= actFun(zvalues{k});
        end
        zvalues{end}=weights{end}*activations{end-1}+bias{end};
        activations{end}= outputFun(zvalues{end});
        
        
        

%%%(4)Back propagation in each mini batch
%%%Initialize Momentum as 0 for each mini batch
        momentum_weights=cell(1,num_layers);
        momentum_bias=cell(1,num_layers);
        for k=2:num_layers
            momentum_weights{k}=zeros(size(weights{k}));
            momentum_bias{k}=zeros(size(bias{k}));
        end


%%%First, calculate the delta,delta nabla of weights and delta nabla of 
%  bias of output layer
%%%If costFunOption ==0 -> Quadratic
%%%If costFunOption ==1 -> Cross-entropy
%%%If costFunOption ==2 -> log-likelihood(only with softmax)            
        if costFunOpetion ==0
            delta_output_layer = (activations{end}-mini_batch_targets)...
                                .*outputFunPrime(zvalues{end});
        else
            delta_output_layer = activations{end}-mini_batch_targets;
        end
        
        delta_nabla_weights{end}=batchSize .\((delta_output_layer*(activations{end-1})')); 
        delta_nabla_bias{end}= mean(delta_output_layer,2);
        momentum_weights{end}=momentum*momentum_weights{end}+(1-momentum)* delta_nabla_weights{end};
        momentum_bias{end}=momentum*momentum_bias{end}+(1-momentum)* delta_nabla_bias{end};
        
        
%%%Then, we do the same things for other layers(except for input layer)
        delta_previous_layer = delta_output_layer;
        
        for k=(num_layers-1):-1:2
            delta_this_layer = (((weights{k+1})')*delta_previous_layer)...
                .*(actFunPrime(zvalues{k}));
            delta_nabla_weights{k}=batchSize .\((delta_this_layer*(activations{k-1})')); 
            delta_nabla_bias{k}= mean(delta_this_layer,2);
            momentum_weights{k}=momentum*momentum_weights{k}+(1-momentum)* delta_nabla_weights{k};
            momentum_bias{k}=momentum*momentum_bias{k}+(1-momentum)* delta_nabla_bias{k};
            delta_previous_layer = delta_this_layer;
        end
%%        
%%%(5)Update the weights and bias by the end of each mini batch

        for k=2:num_layers
            weights{k}=(1-eta*lambda/batchSize)*weights{k}-eta*momentum_weights{k};
            bias{k}=bias{k}-eta*momentum_bias{k};
        end
        
        
    end
    

%%%Step 6 Work with each epoch
%%%(1)Prep work for training set, test set and validation set
%%%(2)Feed forward seperately with the whole three sets with updated
%     weights and bias
%%%(3)Calculate evluation metrics

%%
%%%(1)Prep work for training set, test set and validation set
    train_zvalues=cell(1,num_layers);
    train_activations=cell(1,num_layers);
    train_activations{1}=train_set(1:num_features,:);
    train_target=train_set(num_features+1:end,:);

%%%for test set
    test_zvalues=cell(1,num_layers);
    test_activations=cell(1,num_layers);
    test_activations{1}=test_set(1:num_features,:);
    test_target=test_set(num_features+1:end,:);
   
%%%for validation set
    val_zvalues=cell(1,num_layers);
    val_activations=cell(1,num_layers);
    val_activations{1}=val_set(1:num_features,:);
    val_target=val_set(num_features+1:end,:);
   
%%%(2)Feed forward seperately with the whole three sets with updated
%     weights and bias
    for k=2:num_layers-1
        train_zvalues{k}=weights{k}*train_activations{k-1}+bias{k};
        train_activations{k}=actFun(train_zvalues{k}); 
        test_zvalues{k}=weights{k}*test_activations{k-1}+bias{k};
        test_activations{k}=actFun(test_zvalues{k});
        val_zvalues{k}=weights{k}*val_activations{k-1}+bias{k};
        val_activations{k}=actFun(val_zvalues{k});
    end
    train_zvalues{end}=weights{end}*train_activations{end-1}+bias{end};
    test_zvalues{end}=weights{end}*test_activations{end-1}+bias{end};
    val_zvalues{end}=weights{end}*val_activations{end-1}+bias{end};
    if actFunOption == 2
            train_activations{end}=softmax(train_zvalues{end}); 
            test_activations{end}=softmax(test_zvalues{end});
            val_activations{end}=softmax(val_zvalues{end});
    else
            train_activations{end}=actFun(train_zvalues{end}); 
            test_activations{end}=actFun(test_zvalues{end});
            val_activations{end}=actFun(val_zvalues{end});
    end 
    
    
%%%(3)Calculate evluation metrics
%%% Cost Score
    train_MSE = immse(train_activations{end},train_target);
     if num_test_examples>0
        test_MSE = immse(test_activations{end},test_target);
   else
        test_MSE =0;
   end
   if num_val_examples>0
        val_MSE = immse(val_activations{end},val_target);
   else
        val_MSE =0; 
   end


%%%Calculate the Accuracy score
%%%Scenario 1: the output layer has more than 1 node
    if size(train_target,1)>1
       [M1,I1] = max(train_activations{end});     
       [M2,I2] = max(train_target);  
       train_correct_items = sum(I1==I2);
       [M1,T1] = max(test_activations{end});     
       [M2,T2] = max(test_target);  
       test_correct_items = sum(T1==T2); 
       [M1,V1] = max(val_activations{end});     
       [M2,V2] = max(val_target);  
       val_correct_items = sum(V1==V2);  
        
    end
    
%%%Scenario 2: the output layer has only 1 node
%%%Improvement needed here
    if size(train_target,1)==1
        one_index = find(train_activations{end}>0.5);
        train_activations{end}(one_index)=1;
        train_correct_items=sum(train_activations{end}==train_target);
        t_one_index = find(test_activations{end}>0.5);
        test_activations{end}(t_one_index)=1;
        test_correct_items=sum(test_activations{end}==test_target);
        v_one_index = find(val_activations{end}>0.5);
        val_activations{end}(v_one_index)=1;
        val_correct_items=sum(val_activations{end}==val_target);
    end
    
    train_Accuracy=  train_correct_items/num_train_examples;
     if num_test_examples>0
        test_Accuracy=  test_correct_items/num_test_examples;
    else
        test_Accuracy=0;
    end
    if num_val_examples>0
        val_Accuracy=  val_correct_items/num_val_examples;
    else 
        val_Accuracy=0;
    end
    train_cost_all(i)=train_MSE;
    train_accu_all(i)=train_Accuracy;
    test_cost_all(i)=test_MSE;
    test_accu_all(i)=test_Accuracy;
    
    fprintf('%-3d| %.3f |  %6d/%6d  | %.3f|| %.3f | %6d/%6d  | %.3f ||%.3f|  %6d/%6d  | %.3f\n',i,train_MSE,...
    train_correct_items,num_train_examples,train_Accuracy,test_MSE,...
    test_correct_items,num_test_examples,test_Accuracy,val_MSE,...
    val_correct_items,num_val_examples,val_Accuracy)

   

%%%Step 7: Set Early Stopping 
    if train_Accuracy-val_Accuracy >= 0.15
        break
    end
end

%%%Step 8: plot
x = 1:i;
y1 = train_cost_all(1:i);
y2 = test_cost_all(1:i);
y3 = train_accu_all(1:i);
y4= test_accu_all(1:i);
figure
subplot(2,1,1)       
plot(x,y1,x,y2)
title('Cost Score Plot')
xlabel('Numbers of Epochs')
ylabel('Cost Score')
legend('Training set','Test set')
subplot(2,1,2)      
plot(x,y3,x,y4)      
title('Accuracy Score')
xlabel('Numbers of Epochs')
ylabel('Accuracy Score')
legend('Training set','Test set')


end