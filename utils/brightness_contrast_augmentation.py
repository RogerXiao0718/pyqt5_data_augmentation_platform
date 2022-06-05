import os
import cv2
import json
import numpy as np
from torchvision import transforms

def dataAugmentation_contrast_brightness(image_paths, image_savePath, coloredAugData, logLabel=None):
    colorJitterTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=coloredAugData["brightness"],
                               contrast=coloredAugData["contrast"],
                               saturation=coloredAugData["saturation"],
                               hue=coloredAugData["hue"])
    ])

    for image_path in image_paths:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        # 色調擴增
        transformed_image = colorJitterTransform(image).permute(1, 2, 0)
        transformed_image = (transformed_image.cpu().detach().numpy() * 255).astype(int)
        image_file_basename = os.path.basename(image_path)
        # 反白擴增
        if coloredAugData["highlight"]:
            # transformed_image = (np.vectorize(lambda p: 255 - p))(transformed_image)
            transformed_image = 255 - transformed_image
        cv2.imencode(os.path.splitext(image_file_basename)[1], transformed_image)[1].tofile(
            os.path.join(image_savePath, f"./{image_file_basename}"))

        if logLabel:
            logLabel.setText(f"saved image into {os.path.join(image_savePath, f'{image_file_basename}')}\n")
        else:
            print(f"saved image into {os.path.join(image_savePath, f'{image_file_basename}')}\n")



