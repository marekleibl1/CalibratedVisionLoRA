
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import os 
from os.path import join
import math
import sys
import numpy as np 
from collections import defaultdict

from lora_vit import create_lora_vit_model
from utils import print_trainable_parameters
from dataloaders import create_food_data_loaders


def train_single_model(config): 
    print('New Model Training ...')
    print(config) 
    
    lora_model, transform = create_lora_vit_model(**config)
    print_trainable_parameters(lora_model, detailed=True)

    trainloader, testloader = create_food_data_loaders(transform, **config)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device', device)
    lora_model.to(device)

    model_dir = '../models'
    model_name = config.get('model_name', f'model_lorarank{config["lora_rank"]}')
    export_path = join(model_dir, f'{model_name}.pth') 
    
    os.makedirs(model_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=config['learning_rate'])
    
    best_valid_loss = 9999
    
    for epoch in range(config['epochs']): 
        
        # --- Train the model

        lora_model.train()
        train_losses = []
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
    
            outputs = lora_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            train_losses.append(loss.item())
     
        train_loss = np.mean(train_losses)
        print(f'[{epoch + 1}, {i + 1}] loss: {train_loss:.3f}')


        # --- Compute test loss  & save model
        
        if epoch % 2 == 0: 
            lora_model.eval()
            test_losses = []
            correct, total = 0, 0
    
            with torch.no_grad():
                for i, (images, labels) in enumerate(testloader):
                    if i == 200:
                        break
                    images, labels = images.to(device), labels.to(device)
                    logits = lora_model(images) 
                    
                    test_loss = criterion(logits, labels).item()
                    test_losses.append(test_loss)
     
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
            test_loss = np.mean(test_losses)
            test_accuracy = correct / total
            print(f'Epoch {epoch + 1}, Test loss: {test_loss:.3f}, Test accuracy: {test_accuracy:.3f}')
    
            if test_loss < best_valid_loss:
                best_valid_loss = test_loss
                if config.get('save_model', True): 
                    torch.save(lora_model.state_dict(), export_path)
                    print(f'Model exported to {export_path}')
        
    
    print('Finished Training')
    return best_valid_loss
    

def train_lora_models():
    """
    Train multiple models with different LoRA ranks to determine the optimal rank.
    
    Note that: for more accurate comparison we might need longer training. 
    """

    experiment_name = 'lora_ranks'
    
    config = dict(
        n_classes = 10,
        batch_size =  32,
        lora_rank = 2,                
        lora_layers = [10, 11],
        random_projections=False,
        random_projections_dim=128,
        learning_rate =  0.001,
        epochs = 21, #  11,  
        save_model = False
    )

    n_repeat = 3
    lora_ranks =  [1, 2, 4, 8]
    losses = defaultdict(list)

    # n_repeat = 2 
    # lora_ranks = [1, 2]
    # config['epochs'] = 1
    
    for lora_rank in lora_ranks:
        for _ in range(n_repeat): 
            config['lora_rank'] = lora_rank
            best_valid_loss = train_single_model(config)
            print('lora_rank', lora_rank, 'best_valid_loss', best_valid_loss)
            losses[lora_rank].append(best_valid_loss)

    
    export_dir = '../results'
    os.makedirs(export_dir, exist_ok=True)
    export_path  = os.path.join(export_dir, f'{experiment_name}.json')
    
    with open(export_path, 'w') as f: 
        json.dump(losses, f)
    
    print('Exported to', export_path)
    print(losses)


def default_training():
    """
    Train a model that will be calibrated. 
    """
    
    config = dict(
        model_name = 'lora_model',
        n_classes = 10,
        batch_size =  32,
        lora_rank = 2,                
        lora_layers = [10, 11],
        random_projections=False,
        random_projections_dim=128,
        learning_rate =  0.001,
        epochs = 31, #  11,  
        save_model = True
    )

    best_valid_loss = train_single_model(config)
    print('best_valid_loss', best_valid_loss)


def train_lora_5classes():
    """
    Train a model that will be calibrated.  
    """
    
    config = dict(
        model_name = 'lora_model_5classes',
        n_classes = 5,
        batch_size =  32,
        lora_rank = 1,                
        lora_layers = [10, 11],
        random_projections=False,
        random_projections_dim=128,
        learning_rate =  0.001,
        epochs = 31, #  11,  
        save_model = True
    )

    best_valid_loss = train_single_model(config)
    print('best_valid_loss', best_valid_loss)


def train_lora_5classes_with_random_projections():
    """
    Train a model that will be calibrated.  
    """
    
    config = dict(
        model_name = 'lora_model_5classes_random_proj_dim32',
        n_classes = 5,
        batch_size =  32,
        lora_rank = 1,                
        lora_layers = [10, 11],
        random_projections=True,
        trainable_projections=False,
        random_projections_dim=32,
        learning_rate =  0.001,
        epochs = 31, #  11,  
        save_model = True
    )

    best_valid_loss = train_single_model(config)
    print('best_valid_loss', best_valid_loss)


def train_lora_5classes_with_trainable_projections():
    """
    Train a model that will be calibrated.  
    """
    
    config = dict(
        model_name = 'lora_model_5classes_trainable_proj_dim32',
        n_classes = 5,
        batch_size =  32,
        lora_rank = 1,                
        lora_layers = [10, 11],
        random_projections=True,
        trainable_projections=True,
        random_projections_dim=32,
        learning_rate =  0.001,
        epochs = 31, #  11,  
        save_model = True
    )

    best_valid_loss = train_single_model(config)
    print('best_valid_loss', best_valid_loss)
        


if __name__ == '__main__':
    # train_lora_models()
    # default_training()
    # train_lora_5classes()
    train_lora_5classes_with_random_projections()
    