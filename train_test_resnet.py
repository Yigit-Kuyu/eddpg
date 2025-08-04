
import torch
import numpy as np
import torch.nn as nn
import random


from utils.load_utils import create_data_loader
from utils.torch_metrics import EpochMetricTracker
from utils.helpers import get_training_logger, load_yaml
from data_modules.class_mod import ResNet50Module  as ResNet50Model



def main():
    args = load_yaml("config/config.yaml")

    if args.seed != 0: 
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)
    
  
    logger = get_training_logger(args.class_model_checkpoint)
    val_tracker = EpochMetricTracker()
    test_tracker = EpochMetricTracker()
    
    model = ResNet50Model(
            num_classes=args.num_classes,
            dropout_prob=args.dropout_prob
        )
    model = model.to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    

    # Training loop
    if args.do_train:  
        train_loader = create_data_loader(args, "train", shuffle=True)
        val_loader   = create_data_loader(args, "val",   shuffle=False)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )

        for epoch in range(args.num_epochs):
            model.train()
            for step, batch in enumerate(train_loader):

                images = batch.image.to(args.device)
                labels = batch.label.to(args.device)
                optimizer.zero_grad()
                outputs,_ = model(images)
                ce_loss = loss_fn(outputs, labels)
                ce_loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation loop
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch.image.to(args.device)
                    labels = batch.label.to(args.device)
                    outputs,_ = model(images)
                    val_tracker.update(outputs, labels)      

    
    # Test loop
    else:
        test_loader   = create_data_loader(args, "test",   shuffle=False)
        loaded_model = torch.load(args.class_model_checkpoint/ 'best_model.pth', map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(loaded_model['state_dict'])

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                images = batch.image.to(args.device)
                labels = batch.label.to(args.device)
                outputs,_ = model(images)
                test_tracker.update(outputs, labels)
            

        stats = test_tracker.compute()
        test_accuracy    = stats["accuracy"]      
        test_recall      = stats["recall"]        
        test_specificity = stats["specificity"]          
        test_auc         = stats["auc"] 
                 
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test Specificity: {test_specificity:.4f}")
        logger.info(f"Test AUC: {test_auc:.4f}")
 

if __name__ == '__main__':
    main()

