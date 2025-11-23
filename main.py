import torch
import numpy as np
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

# Local imports
from dataset import PDScalogram
from dataloader.dataloader import FewshotDataset
from function.function import ContrastiveLoss, seed_func, plot_confusion_matrix, plot_tsne

# Models
from net.cosine import CosineNet
from net.protonet import ProtoNet
from net.covamnet import CovaMNet

def get_args():
    parser = argparse.ArgumentParser(description='PD Scalogram Fewshot Training & Testing')
    
    # Dataset / Paths
    parser.add_argument('--dataset_path', type=str, default='./scalogram_images/', help='Path to dataset')
    parser.add_argument('--path_weights', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--weights', type=str, default=None, help='Path to specific weight file for testing')
    
    # Model
    parser.add_argument('--model', type=str, default='covamnet', choices=['cosine', 'protonet', 'covamnet'], help='Model architecture')
    
    # Fewshot settings
    parser.add_argument('--way_num', type=int, default=3, help='Number of classes per episode')
    parser.add_argument('--shot_num', type=int, default=1, help='Number of support samples per class')
    parser.add_argument('--query_num', type=int, default=1, help='Number of query samples per class')
    
    # Training settings
    parser.add_argument('--episode_num_train', type=int, default=100, help='Number of episodes per epoch (train)')
    parser.add_argument('--episode_num_test', type=int, default=75, help='Number of episodes per epoch (test)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Episodes per batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=10, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='Scheduler gamma')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Execution mode')
    
    return parser.parse_args()

def get_model(model_name, device):
    if model_name == 'cosine':
        model = CosineNet(use_gpu=(device=='cuda'))
    elif model_name == 'protonet':
        model = ProtoNet(use_gpu=(device=='cuda'))
    elif model_name == 'covamnet':
        model = CovaMNet(use_gpu=(device=='cuda'))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)

def train_loop(net, train_loader, val_loader, args):
    device = args.device
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = ContrastiveLoss().to(device)
    
    best_acc = 0.0
    history = {'loss': [], 'acc': []}
    
    print(f"Starting training for {args.num_epochs} epochs...")
    
    for epoch in range(1, args.num_epochs + 1):
        net.train()
        running_loss = 0.0
        total_batches = 0
        
        # Training
        with tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}', unit='batch') as t:
            for query_imgs, query_targets, support_imgs, support_targets in t:
                # Prepare Data
                # Loader returns:
                # query_imgs: (B, Way*Query, C, H, W)
                # support_imgs: (B, Way*Shot, C, H, W)
                # targets: (B, Way*Query) - wait, let's check targets structure
                
                B, NQ_total, C, H, W = query_imgs.shape
                _, NS_total, _, _, _ = support_imgs.shape
                
                # Reshape Support: (B, Way, Shot, C, H, W)
                # Assuming support_imgs is ordered by way then shot
                s = support_imgs.view(B, args.way_num, args.shot_num, C, H, W).to(device)
                
                # Reshape Query: (B, NQ_total, C, H, W) - keep as is, just move to device
                q = query_imgs.to(device)
                
                # Targets
                targets = query_targets.view(-1).to(device) # Flatten to (B*NQ_total)
                
                optimizer.zero_grad()
                
                # Forward
                # Model returns (B*NQ_total, Way) scores
                scores = net(q, s)
                
                # Loss
                loss = loss_fn(scores, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                total_batches += 1
                
                t.set_postfix(loss=running_loss/total_batches)
        
        scheduler.step()
        avg_loss = running_loss / total_batches
        history['loss'].append(avg_loss)
        
        # Validation
        acc = evaluate(net, val_loader, args)
        history['acc'].append(acc)
        
        print(f"Validation Accuracy: {acc:.4f}")
        
        # Save Best
        if acc > best_acc:
            best_acc = acc
            save_name = f"{args.model}_{args.shot_num}shot_best.pth"
            save_path = os.path.join(args.path_weights, save_name)
            torch.save(net.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
            
    return history

def evaluate(net, loader, args):
    net.eval()
    # Using the helper functions from function.py for consistency with legacy metrics
    # But we need to make sure they work with our models. 
    # The helper functions cal_accuracy_fewshot_* iterate the loader themselves.
    # And they assume specific input handling inside.
    
    # CAUTION: function.py's cal_accuracy_fewshot_1shot iterates:
    # for q, qt, s, st in loader:
    #    q = q.permute(1, 0, 2, 3, 4)
    #    for i in range(len(q)): ...
    
    # My models now expect (B, NQ, ...) and (B, Way, Shot, ...).
    # The legacy functions might break if I don't adapt them OR the models.
    # My models forward() method is flexible enough?
    # Legacy function:
    # q input: (Batch, Way, Query, C, H, W) -> permute(1,0,...) -> (Way, Batch, Query, ...)
    # q[i] -> (Batch, Query, ...) -> perfect for my model's (B, NQ, ...)
    # s input: (Batch, Way, Shot, ...) -> permute(1,0,...) -> (Way, Batch, Shot, ...)
    
    # Wait, in `dataloader.py`, I saw:
    # return query_images (B, Way*Query, ...), support_images (B, Way*Shot, ...)
    
    # If `cal_accuracy_fewshot_1shot` in `function.py` expects the OLD format, it might be incompatible with `dataloader.py` OUTPUT.
    # Let's re-read `function.py` carefully.
    
    # function.py:
    # for query_images, query_targets, support_images, support_targets in loader:
    #    q = query_images.permute(1, 0, 2, 3, 4)
    
    # This implies `query_images` has 5 dimensions.
    # But `dataloader.py` returns 5 dims?
    # query_images list of (Query, C, H, W).
    # torch.cat(dim=0) -> (Way*Query, C, H, W).
    # DataLoader adds Batch -> (Batch, Way*Query, C, H, W).
    # This is 5 dimensions.
    
    # So `permute(1, 0, 2, 3, 4)` switches dim 0 (Batch) and dim 1 (Way*Query).
    # Result: (Way*Query, Batch, C, H, W).
    
    # Then `for i in range(len(q))` iterates over `Way*Query`.
    # `q[i]` is `(Batch, C, H, W)`.
    # `net(q[i], s)`
    
    # My models expect:
    # query: (B, NQ, C, H, W). If I pass `q[i]`, it is (B, C, H, W).
    # `q[i].unsqueeze(1)` would make it (B, 1, C, H, W).
    
    # AND support `s` in `function.py`:
    # s = support_images.permute(1, 0, 2, 3, 4) -> (Way*Shot, Batch, C, H, W).
    
    # My models expect support: (B, Way, Shot, C, H, W).
    # The legacy function passes (Way*Shot, Batch, C, H, W) or similar.
    
    # CONCLUSION: I cannot use the legacy `cal_accuracy_fewshot` functions directly with my new `forward` signatures 
    # UNLESS I modify the `forward` signatures to detect and reshape, OR I write a new evaluation loop.
    
    # I will write a NEW evaluation loop here to be clean and professional.
    
    total_correct = 0
    total_samples = 0
    
    device = args.device
    
    with torch.no_grad():
        for query_imgs, query_targets, support_imgs, support_targets in loader:
            B, NQ_total, C, H, W = query_imgs.shape
            
            # Reshape Data
            s = support_imgs.view(B, args.way_num, args.shot_num, C, H, W).to(device)
            q = query_imgs.to(device)
            targets = query_targets.view(-1).to(device)
            
            # Forward
            scores = net(q, s) # (B*NQ_total, Way)
            
            # Predictions
            preds = torch.argmax(scores, dim=1)
            
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            
    return total_correct / total_samples

def test_full(net, loader, args):
    print(f"Testing model: {args.model}, {args.shot_num}-shot")
    acc = evaluate(net, loader, args)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Confusion Matrix & t-SNE
    # We need to gather all preds and targets
    all_preds = []
    all_targets = []
    all_features = []
    
    device = args.device
    net.eval()
    
    with torch.no_grad():
        for query_imgs, query_targets, support_imgs, support_targets in tqdm(loader, desc="Testing"):
            B, NQ_total, C, H, W = query_imgs.shape
            s = support_imgs.view(B, args.way_num, args.shot_num, C, H, W).to(device)
            q = query_imgs.to(device)
            targets = query_targets.view(-1).to(device)
            
            scores = net(q, s)
            preds = torch.argmax(scores, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Features for t-SNE (extract from query)
            # Depending on model, we might want features before classifier
            # All my models have 'encoder' or 'features'
            
            # Flatten q for encoding
            q_flat = q.view(-1, C, H, W)
            if hasattr(net, 'encoder'):
                feat = net.encoder(q_flat)
            elif hasattr(net, 'features'):
                feat = net.features(q_flat)
            else:
                feat = None
            
            if feat is not None:
                 # Global pool if needed
                 if feat.dim() > 2:
                     feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                     feat = feat.view(feat.size(0), -1)
                 all_features.append(feat.cpu().numpy())
                 
    # Plotting
    save_path_cm = f"confusion_matrix_{args.model}_{args.shot_num}shot.png"
    plot_confusion_matrix(all_targets, all_preds, num_classes=args.way_num, save_path=save_path_cm)
    
    if all_features:
        all_features = np.vstack(all_features)
        all_targets_arr = np.array(all_targets)
        save_path_tsne = f"tsne_{args.model}_{args.shot_num}shot.png"
        try:
            plot_tsne(all_features, all_targets_arr, num_classes=args.way_num, save_path=save_path_tsne)
        except Exception as e:
            print(f"Skipping t-SNE: {e}")

def main():
    args = get_args()
    print(f"Configuration: {args}")
    
    # Setup
    seed_func()
    if not os.path.exists(args.path_weights):
        os.makedirs(args.path_weights)
        
    # Data
    print("Loading Dataset...")
    dataset = PDScalogram(args.dataset_path)
    
    # Helper to prep data
    def prep_data(X, y):
        # Convert to tensor and permute to (N, C, H, W)
        X = torch.from_numpy(X.astype(np.float32))
        if X.dim() == 4 and X.shape[3] == 3: # (N, H, W, C)
             X = X.permute(0, 3, 1, 2)
        y = torch.from_numpy(y)
        return X, y
        
    train_X, train_y = prep_data(dataset.X_train, dataset.y_train)
    test_X, test_y = prep_data(dataset.X_test, dataset.y_test)
    
    train_ds = FewshotDataset(train_X, train_y, 
                              episode_num=args.episode_num_train,
                              way_num=args.way_num,
                              shot_num=args.shot_num,
                              query_num=args.query_num)
                              
    test_ds = FewshotDataset(test_X, test_y,
                             episode_num=args.episode_num_test,
                             way_num=args.way_num,
                             shot_num=args.shot_num,
                             query_num=args.query_num)
                             
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # Model
    net = get_model(args.model, args.device)
    
    if args.mode == 'train':
        train_loop(net, train_loader, test_loader, args)
        # Test best model
        best_path = os.path.join(args.path_weights, f"{args.model}_{args.shot_num}shot_best.pth")
        if os.path.exists(best_path):
            print(f"Loading best model for final test: {best_path}")
            net.load_state_dict(torch.load(best_path))
            test_full(net, test_loader, args)
            
    elif args.mode == 'test':
        if args.weights is None:
            # Try to find best model default
            best_path = os.path.join(args.path_weights, f"{args.model}_{args.shot_num}shot_best.pth")
            if os.path.exists(best_path):
                args.weights = best_path
            else:
                print("Warning: No weights provided and no default best model found. Testing with random init.")
        
        if args.weights:
            print(f"Loading weights: {args.weights}")
            net.load_state_dict(torch.load(args.weights))
            
        test_full(net, test_loader, args)

if __name__ == '__main__':
    main()
