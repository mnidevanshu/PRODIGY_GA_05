## Task 05: Neural Style Transfer Tool

### 📌 Project Overview
This project implements a **Neural Style Transfer (NST)** application that allows users to apply the artistic style of one image to the content of another. Unlike traditional filters, NST uses deep neural networks to disentangle and recombine the content of the base image with the textures and color patterns of the style image.
The tool features a custom **Graphical User Interface (GUI)** for seamless interaction, allowing users to upload images, adjust style intensity, and view results in real-time.

### 🛠️ Technical Stack
 * **Deep Learning:** TensorFlow, TensorFlow Hub
 * **Model:** Magenta Arbitrary Image Stylization (V1-256)
 * **GUI Framework:** CustomTkinter (Modernized Tkinter wrapper)
 * **Image Processing:** Pillow (PIL), NumPy
   
### 🧬 How It Works
 1. **Model Loading:** The application utilizes a pre-trained **Arbitrary Image Stylization** model from TensorFlow Hub. This model is capable of performing fast stylization in a single forward pass, supporting any arbitrary style image.
 2. **Preprocessing:** Content and style images are converted into float tensors. The content image is resized to 512 \times 512 for optimal detail, while the style image is resized to 256 \times 256 as per model requirements.
 3. **Intensity Blending:** An integrated slider allows for linear interpolation between the original content and the stylized output, giving the user control over the "strength" of the transformation.
<img width="1920" height="1080" alt="output-preview" src="https://github.com/user-attachments/assets/feaf3bcf-943b-40da-9542-727c6dd60089" />


### 🚀 How to Run
 1. **Install dependencies:**
   ```bash
   pip install tensorflow tensorflow-hub customtkinter pillow numpy
   
   ```
 2. **Execute the application:**
   ```bash
   python "Task 5.py"
   
   ```
 3. **Usage:**
   * Click **Upload Content** to pick your base photo.
   * Click **Upload Style** to pick an artistic painting or texture.
   * Adjust the **Style Intensity** slider if desired.
   * Click **Transfer Style** to generate the final artwork.
