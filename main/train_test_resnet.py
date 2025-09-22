
import torch
import numpy as np
import torch.nn as nn
import random


from utils.load_utils import create_data_loader, load_resnet_model_for_inference
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
    
    import os
    checkpoint_dir = os.path.dirname(args.class_model_checkpoint)
    logger = get_training_logger(checkpoint_dir)
    val_tracker = EpochMetricTracker()
    test_tracker = EpochMetricTracker()
    
    model = ResNet50Model(
            num_classes=args.num_classes,
            dropout_prob=args.drop_prob
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
        
        best_val_auc = 0.0

        for epoch in range(args.num_epochs):
            print(f"--- Starting Epoch {epoch+1}/{args.num_epochs} ---")
            model.train()
            for step, batch in enumerate(train_loader):

                images = batch.image.to(args.device)
                labels = batch.label.to(args.device)
                optimizer.zero_grad()
                outputs, logits = model(images)
                ce_loss = loss_fn(logits, labels)
                ce_loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation loop
            model.eval()
            val_tracker.reset()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch.image.to(args.device)
                    labels = batch.label.to(args.device)
                    outputs,_ = model(images)
                    val_tracker.update(outputs, labels)      
            
            val_stats = val_tracker.compute()
            val_auc = val_stats["auc"]
            val_acc = val_stats["accuracy"]
            
            logger.info(f"Epoch {epoch+1}: Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'auc': val_auc,
                }, os.path.join(args.class_model_checkpoint, 'best_model.pth'))
                logger.info(f"New best model saved with AUC: {best_val_auc:.4f}")

    
    # Test loop
    else:
        test_loader = create_data_loader(args, "test", shuffle=False)
        
        
        state_dict = load_resnet_model_for_inference(args)
        model.load_state_dict(state_dict)

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

