import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk

def test_single_volume_oilspill(image, label, net, classes, patch_size=[256, 256], 
                               test_save_path=None, case=None, z_spacing=1):
    """
    Test function specifically for oil spill detection
    """
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
    # Handle different image dimensions
    if len(image.shape) == 3:
        image = image[0]  # Take first channel if multi-channel
    
    x, y = image.shape[0], image.shape[1]
    if x != patch_size[0] or y != patch_size[1]:
        image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        label = zoom(label, (patch_size[0] / x, patch_size[1] / y), order=0)
    
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()
    
    with torch.no_grad():
        outputs = net(input_tensor)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
        
        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    
    if test_save_path is not None:
        # Create colored visualization for oil spills
        vis_pred = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
        
        # Color mapping for oil spill visualization
        # Background (water) - dark blue
        vis_pred[prediction == 0] = [0, 0, 139]
        # Oil spill - red
        vis_pred[prediction == 1] = [255, 0, 0]
        # Other class if exists - green
        if classes > 2:
            vis_pred[prediction == 2] = [0, 255, 0]
        
        # Save prediction visualization
        from PIL import Image
        pred_image = Image.fromarray(vis_pred)
        pred_image.save(os.path.join(test_save_path, case + '_prediction.png'))
        
        # Also save original image for comparison
        if len(image.shape) == 2:
            orig_image = (image * 255).astype(np.uint8)
            orig_pil = Image.fromarray(orig_image, mode='L')
            orig_pil.save(os.path.join(test_save_path, case + '_original.png'))
    
    return metric_list

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
