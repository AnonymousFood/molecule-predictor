import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(losses, trained_model, test_data, target_feature):
    # Plot training and test loss
    epochs = [loss['epoch'] for loss in losses]
    train_losses = [loss['train_loss'] for loss in losses]
    test_losses = [loss['test_loss'] for loss in losses] 

    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title(f'Loss Over Epochs for {target_feature}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Get final test metrics
    test_metrics = trained_model.evaluate(test_data)

    # Visualize predictions vs actual
    print("\nModel Predictions Analysis:")
    trained_model.eval()
    with torch.no_grad():
        output = trained_model(test_data)
        target = test_data.y.reshape(-1, 1).expand(output.shape[0], -1)
        
        predictions = output.cpu().numpy().flatten()
        actuals = target.cpu().numpy().flatten()
        
        # Calculate average prediction and error
        avg_prediction = np.mean(predictions)
        avg_actual = np.mean(actuals)
        avg_error = np.mean(np.abs(predictions - actuals))
        std_predictions = np.std(predictions)
        
        print(f"Average prediction: {avg_prediction:.6f}")
        print(f"Average actual value: {avg_actual:.6f}")
        print(f"Average absolute error: {avg_error:.6f}")
        print(f"Standard deviation of predictions: {std_predictions:.6f}")
        
        # Plot predictions
        plt.figure(figsize=(10, 6))
        plt.hist(predictions, bins=20, alpha=0.5, color='blue')
        plt.axvline(actuals[0], color='red', linestyle='dashed', linewidth=2, label='Actual Value')
        plt.axvline(avg_prediction, color='green', linestyle='dashed', linewidth=2, label='Mean Prediction')
        plt.xlabel('Predicted Values')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Predictions for {target_feature}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()