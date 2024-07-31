

def print_trainable_parameters(model, detailed=False): 
    """
    Calculate total number of parameters and trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Total number of parameters: {total_params}')
    print(f'Number of trainable parameters: {trainable_params}')

    
    if detailed:
        print(f"Trainable parameters per layer:")
        for name, module in model.named_modules():
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if n >0: 
                print(name, n)