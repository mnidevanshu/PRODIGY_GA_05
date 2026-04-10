import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class NSTApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Neural Style Transfer Tool")
        self.geometry("1150x800")

        # Load Model
        self.status_var = tk.StringVar(value="Loading Model...")
        try:
            # This is the fast arbitrary stylization model
            self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
            self.status_var.set("Model Loaded. Ready.")
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load NST model: {e}")
            self.model = None

        # State variables
        self.content_path = None
        self.style_path = None
        self.output_img_ref = None # To prevent garbage collection
        self.content_img_ref = None
        self.style_img_ref = None

        self._setup_ui()
        self._create_placeholders()

    def _setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure((0, 1, 2), weight=1)

        # Header
        ctk.CTkLabel(self.main_frame, text="Neural Style Transfer", font=("Arial", 28, "bold")).grid(row=0, column=0, columnspan=3, pady=20)

        # Buttons
        self.btn_content = ctk.CTkButton(self.main_frame, text="Upload Content", command=self.load_content, fg_color="#2ecc71", hover_color="#27ae60")
        self.btn_content.grid(row=1, column=0, padx=10, pady=10)

        self.btn_style = ctk.CTkButton(self.main_frame, text="Upload Style", command=self.load_style, fg_color="#3498db", hover_color="#2980b9")
        self.btn_style.grid(row=1, column=1, padx=10, pady=10)

        self.btn_run = ctk.CTkButton(self.main_frame, text="Transfer Style", command=self.run_transfer, fg_color="#e67e22", hover_color="#d35400")
        self.btn_run.grid(row=1, column=2, padx=10, pady=10)

        # Image Displays
        self.display_content = ctk.CTkLabel(self.main_frame, text="No Content Loaded", width=320, height=320, fg_color="#2b2b2b", corner_radius=8)
        self.display_content.grid(row=2, column=0, padx=10, pady=20)

        self.display_style = ctk.CTkLabel(self.main_frame, text="No Style Loaded", width=320, height=320, fg_color="#2b2b2b", corner_radius=8)
        self.display_style.grid(row=2, column=1, padx=10, pady=20)

        self.display_output = ctk.CTkLabel(self.main_frame, text="Output Appears Here", width=320, height=320, fg_color="#2b2b2b", corner_radius=8)
        self.display_output.grid(row=2, column=2, padx=10, pady=20)

        # Intensity Slider
        self.slider = ctk.CTkSlider(self.main_frame, from_=0.1, to=1.0, number_of_steps=10)
        self.slider.set(1.0)
        self.slider.grid(row=3, column=0, columnspan=3, pady=(10, 0), padx=50)
        
        ctk.CTkLabel(self.main_frame, text="Style Intensity").grid(row=4, column=0, columnspan=3)

        # Status Bar
        self.status_label = ctk.CTkLabel(self.main_frame, textvariable=self.status_var, font=("Arial", 12, "italic"))
        self.status_label.grid(row=5, column=0, columnspan=3, pady=10)

    def _create_placeholders(self):
        """Creates dummy files if they don't exist for the 'Default' feature."""
        if not os.path.exists("placeholder_c.jpg"):
            Image.new('RGB', (300, 300), color="#e74c3c").save("placeholder_c.jpg")
        if not os.path.exists("placeholder_s.jpg"):
            Image.new('RGB', (300, 300), color="#34495e").save("placeholder_s.jpg")

    def load_img_as_tensor(self, path):
        """Reads image and converts to float tensor [1, H, W, 3]"""
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Content image should be max 512 for speed; style is usually 256
        img = tf.image.resize(img, (512, 512)) 
        return img[tf.newaxis, :]

    def load_content(self):
        path = filedialog.askopenfilename()
        if path:
            self.content_path = path
            img = Image.open(path)
            self.content_img_ref = ctk.CTkImage(img, size=(300, 300))
            self.display_content.configure(image=self.content_img_ref, text="")
            self.status_var.set(f"Loaded Content: {os.path.basename(path)}")

    def load_style(self):
        path = filedialog.askopenfilename()
        if path:
            self.style_path = path
            img = Image.open(path)
            self.style_img_ref = ctk.CTkImage(img, size=(300, 300))
            self.display_style.configure(image=self.style_img_ref, text="")
            self.status_var.set(f"Loaded Style: {os.path.basename(path)}")

    def run_transfer(self):
        if not self.content_path or not self.style_path:
            messagebox.showwarning("Wait!", "Please upload both images first.")
            return
        
        self.status_var.set("Processing style transfer...")
        self.btn_run.configure(state="disabled")
        self.update()

        try:
            # Load Tensors
            content_tensor = self.load_img_as_tensor(self.content_path)
            style_tensor = self.load_img_as_tensor(self.style_path)
            # Style tensor for this model works best at 256x256
            style_tensor = tf.image.resize(style_tensor, (256, 256))

            # Execute Model
            outputs = self.model(tf.constant(content_tensor), tf.constant(style_tensor))
            stylized_img = outputs[0]

            # Convert Tensor back to PIL
            stylized_img = np.array(stylized_img * 255, dtype=np.uint8)
            if len(stylized_img.shape) > 3:
                stylized_img = stylized_img[0]
            
            output_pil = Image.fromarray(stylized_img)

            # Intensity Blending (Optional)
            intensity = self.slider.get()
            if intensity < 1.0:
                base = Image.open(self.content_path).convert("RGB").resize(output_pil.size)
                output_pil = Image.blend(base, output_pil, intensity)

            # Update Display
            self.output_img_ref = ctk.CTkImage(output_pil, size=(300, 300))
            self.display_output.configure(image=self.output_img_ref, text="")
            self.status_var.set("Transfer Complete!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.status_var.set("Error during processing.")
        finally:
            self.btn_run.configure(state="normal")

if __name__ == "__main__":
    app = NSTApp()
    app.mainloop()
