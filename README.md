# Partial Transfer Learning with MAML with CNN Architecture
MAML meta learning basis te idea on that the model adapts fast to a task-specific data, then the learning can transfered to the other data. 
This will make the learning faster because the model can leanrn on subsets of data and the fine tuning can happen once (or very few times) on the rest of the data. 
However, with more inspection of the CNN model, the learning can be accelerated by freezing some layers. 
## CNN architecture inspection and the idea of layer freezing
Upon the inspection of the CNN architecture, we can quickly see that the few beginning layers of the CNN capture the fnie details of the picture. 
Where as the layer layers capture more meta-features. 
![image](https://github.com/user-attachments/assets/0ac778f3-6b91-4fcf-b018-d9d86294743c)

With this inspection, we can tell that while the beginning layers still need to learn to the all dataset to be accurate, the meta-features should be very transferrable. 
Thus, during the all data training (the outer loop of the MAML algorithm given below)
![image](https://github.com/user-attachments/assets/b4cc67a8-d49d-438c-a729-c933c0ef46ee)
We can freeze the upper layers and only learn the lower layesr to learn the detail, because the meta-features are already learned in the inner loop. 
This freezing of the upper layers, of course makes the learning much faster. Especailly because with CNN, the upper layer has more channels. 

## Results
Because the algorithm does not reduce the number of epochs, it does not reduce the total epochs. 
![image](https://github.com/user-attachments/assets/700a4089-4714-4120-9409-6c535bcd82ac)

However, since the computation amount is reduced at the outer loop by not doing the backpropagation of the upper layer, the computation amount (the actual time)
shows a big difference. 
![image](https://github.com/user-attachments/assets/a2426350-61db-45e7-8235-0c1c7e5759d2)

## code explanation
With the partial transfer learning code, you can change from what layers you want to freeze your layer. 
By chnage the transfer_ratio. 
def learning_transfer(meta_model, model, transfer_ratio):
    # first see how many parameters there are in the model 
    i = 0
    for name, param in meta_model.named_parameters():
        i += 1
    transfer_layer_idx = int(round(i * transfer_ratio))

    i = 0
    
    # the parameter after the transfer_ratio are set to False
    for param_first, param_second in zip(meta_model.parameters(), model.parameters()):
        # transfer the weights 
        if i > transfer_layer_idx:
            # then transfer the parameter
            param_second.copy_ = param_first
            # then freeze the parameter
            param_second.requires_grad = False
        else:
            pass
        i += 1
    return model
