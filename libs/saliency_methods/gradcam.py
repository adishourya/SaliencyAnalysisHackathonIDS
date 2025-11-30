import torch
import torch.nn as nn
import numpy as np
import cv2 # For resizing the final heatmap
from load_model import get_conv_model

# --- Global storage for Activations and Gradients ---
# We use hooks to extract feature maps and gradients without modifying the model's structure.
activations = None
gradients = None

# save the activations from each layer during fwd pass
def save_activation_hook(module, input, output):
    """Saves the output of the target layer (activations)."""
    global activations
    # We detach the tensor and store it globally
    activations = output.detach()


# save gradients dlogits/dW during the bwd pass
def save_gradient_hook(module, grad_input, grad_output):
    """Saves the gradients flowing backward through the target layer."""
    global gradients
    # grad_output is a tuple; we care about the first element (gradient w.r.t. output)
    gradients = grad_output[0].detach()


def compute_grad_cam(model: nn.Module, input_tensor: torch.Tensor, target_layer: nn.Module, target_category: int) -> np.ndarray:
    """
    Performs the 5 core steps of the Grad-CAM calculation.
    """
    global activations, gradients
    
    # 2. Register Hooks
    # Attach the hooks to the target layer (e.g., the last conv layer)
    target_layer.register_forward_hook(save_activation_hook)
    target_layer.register_backward_hook(save_gradient_hook)
    print("--- 2. Registered Forward/Backward Hooks ---")

    # 3. Forward Pass to get Model Outputs and Activations
    output = model(input_tensor)
    
    # Reset gradients after the forward pass
    model.zero_grad() 
    print("--- 3. Performed Forward Pass ---")

    # 4. Target Score Selection and Backward Pass
    # We select the score of the specific target category for backpropagation.
    # If target_category is None, use the predicted class (argmax)
    if target_category is None:
        target_category = output.argmax(dim=1).item()
        print(f"No target specified. Using predicted class index: {target_category}")

    # Create a dummy loss function: scalar value for the target class
    # bwd from logits
    target_score = output[:, target_category]
    
    # Calculate gradients w.r.t. the target score
    target_score.backward(retain_graph=True)
    print(f"--- 4. Performed Backward Pass (Target: {target_category}) ---")
    
    # After backward pass, 'gradients' and 'activations' are populated by the hooks

    # --- Grad-CAM CORE CALCULATION ---
    
    # 5. Calculate Neuron Importance Weights (Alpha_k)
    # Global Average Pooling (GAP) of the gradients: Mean across spatial dimensions (H and W)
    # Gradients shape: (B, C, H, W)
    # Weights shape: (B, C)
    
    # Ensure activations and gradients are available
    if activations is None or gradients is None:
        raise RuntimeError("Activations or Gradients were not captured by hooks.")

    weights = torch.mean(gradients, dim=[2, 3], keepdim=True) # shape: (B, C, 1, 1)
    
    print("--- 5. Calculated Weights (GAP of Gradients) ---")

    # 6. Compute Weighted Activations (Linear combination)
    # Element-wise multiplication of weights (1x1) and activations (H x W)
    weighted_activations = weights * activations # shape: (B, C, H, W)

    # Sum across the channel dimension (C) to get the raw heatmap
    cam = torch.sum(weighted_activations, dim=1) # shape: (B, H, W)
    print("--- 6. Combined Weights and Activations ---")

    # 7. Apply ReLU and Normalization/Resizing
    # Apply ReLU to only keep positive contributions
    cam = torch.relu(cam)
    
    # Resize the CAM to the original image size (assuming batch size 1)
    original_size = input_tensor.shape[2:]
    
    # Convert to numpy and resize
    cam_np = cam.squeeze().cpu().numpy()
    
    # The output heatmap needs to be scaled up for visualization
    cam_resized = cv2.resize(cam_np, original_size[::-1], interpolation=cv2.INTER_LINEAR)
    
    # Normalize the heatmap to 0-1 range
    cam_min, cam_max = cam_resized.min(), cam_resized.max()
    if cam_max > cam_min:
        cam_resized = (cam_resized - cam_min) / (cam_max - cam_min)
    
    print("--- 7. Applied ReLU, Resized, and Normalized Heatmap ---")

    return cam_resized


# ====================================================================
# --- TUTORIAL EXECUTION EXAMPLE ---
# ====================================================================

if __name__ == "__main__":
    
    # --- Configuration ---
    # Path should be defined relative to where you execute the script (or use absolute path)
    LOGIT_IDX = 3 # The class index you want to visualize saliency for (e.g., Cat)
    
    # Load Model
    model = get_conv_model()
    model.eval() # double sure!
    
    # Define Target Layer
    try:
        target_layer = model.conv3 # Example: assuming the last conv layer is named 'conv3'
    except AttributeError:
        target_layer = model.layer3 # Assuming SmallCNN uses a Sequential block named layer3/layer4
    
    print(f"\nTarget Layer Selected: {target_layer.__class__.__name__}")
    
    # Create Dummy Input (B, C, H, W)
    # Simulate a single 224x224 RGB image
    dummy_input = torch.randn(1, 3, 224, 224) 
    print(f"Dummy Input Tensor shape: {dummy_input.shape}")

    # Compute Grad-CAM
    try:
        heatmap_np = compute_grad_cam(model, dummy_input, target_layer, LOGIT_IDX)
        print("\nSUCCESS: Grad-CAM Heatmap computed.")
        print(f"Final Heatmap Shape (normalized 0-1): {heatmap_np.shape}")
        
        # In a real application, you would now overlay 'heatmap_np' onto the original image.
        
    except Exception as e:
        print(f"\nFAILURE during Grad-CAM computation: {e}")
