## 📂 Task-05: Neural Style Transfer

### Description
An implementation of **Neural Style Transfer (NST)** that blends the content of one image with the artistic style of another. By utilizing feature maps from pre-trained convolutional neural networks (like VGG-19), the tool recreates the content image in the "brushstrokes" of the style image.

### Features
 * **Style Intensity Control:** An interactive slider to adjust how heavily the style is applied to the content.
 * **Dual-Image Input:** Easy upload functionality for both the Content image and the Style image.
 * **Real-time Preview:** Displays the original content, the style source, and the final synthesized result.
   
### Tech Stack
 * Python
 * TensorFlow Hub / PyTorch
 * VGG-19 Pre-trained Model
 * Custom GUI Framework
