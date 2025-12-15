# Horse Detection using Single Shot Detector (SSD)

##  Overview
This project implements an **object detection pipeline** using the **Single Shot Detector (SSD)** architecture to detect **horses in video streams**.  
The objective is to explore **deep learning–based object detection** using a single-stage detector and apply it to real-time or near–real-time video inference.

The project focuses on understanding the **SSD detection pipeline**, including feature maps, anchor boxes, and bounding box regression.

---

##  Key Concepts Covered
- Single-stage object detection
- Anchor (default) boxes
- Bounding box regression
- Multi-scale feature maps
- Real-time video inference

---

##  Tech Stack
- **Python**
- **PyTorch**
- **OpenCV**
- **NumPy**

---

##  Project Structure
ssd-object-detection-horses/
├── ssd_model.py
├── inference.py
├── utils/
├── video_input/
├── output_video/
└── README.md

---

##  Detection Pipeline
1. Load a pretrained SSD model
2. Preprocess video frames
3. Generate predictions for bounding boxes and class scores
4. Apply confidence thresholding
5. Scale bounding boxes back to original frame size
6. Visualize detections on video frames

---

## Model Intuition (SSD)
SSD performs **object localization and classification in a single forward pass**.

For each default (anchor) box, the model predicts:
- Class probabilities
- Bounding box offsets

The total loss is a combination of:
- **Localization loss** (Smooth L1 loss)
- **Confidence loss** (Softmax cross-entropy)

\[
L = \frac{1}{N} (L_{conf} + \alpha L_{loc})
\]

where:
- \(L_{conf}\) measures classification error  
- \(L_{loc}\) measures bounding box regression error  

This design allows SSD to achieve **fast inference** compared to two-stage detectors.

---

## Results
- Successfully detects horses in video sequences
- Demonstrates multi-object detection capability
- Achieves reasonable performance using pretrained weights

---

##  Notes
- This project uses **pretrained SSD weights**
- No fine-tuning is performed
- Focus is on **inference and pipeline understanding**
- Intended for learning and experimentation

---

## Future Improvements
- Fine-tuning SSD on a custom dataset
- Evaluating performance using IoU and mAP
- Comparing SSD with YOLO and Faster R-CNN
- Optimizing inference speed on GPU