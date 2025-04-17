# Object Detection with SSDlite320 MobileNetV3

This project demonstrates how to use a pre-trained `ssdlite320_mobilenet_v3_large` model from PyTorch’s `torchvision` library to perform object detection on the PASCAL VOC 2012 dataset.

## 📌 Project Highlights

- ✅ **Model**: SSDlite320 MobileNetV3 (pretrained on COCO)
- ✅ **Dataset**: PASCAL VOC 2012
- ✅ **Metrics**: Evaluated using `torchmetrics` Mean Average Precision (mAP)
- ✅ **Tools Used**: PyTorch, Torchvision, Torchmetrics, Matplotlib
- ✅ **Platform**: Google Colab (GPU-enabled)

## 📊 Results

| Metric         | Value     |
|----------------|-----------|
| mAP@0.5        | 12.52%    |
| Overall mAP    | 6.94%     |
| Visualization  | Top-3 predictions drawn using bounding boxes |

## 🧠 Key Features

- Custom `prepare_target()` function to parse VOC annotations into PyTorch format.
- Visualize model predictions using `torchvision.utils.draw_bounding_boxes`.
- Evaluation pipeline for computing mAP across 20 samples.
- Efficient use of GPU for inference.
