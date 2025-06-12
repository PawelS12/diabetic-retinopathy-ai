import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b2
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np
import os
import math

class ModernButton(tk.Canvas):
    def __init__(self, parent, text, command, bg_color="#4A90E2", hover_color="#357ABD", 
                 text_color="white", width=200, height=45, font=("Segoe UI", 11, "bold")):
        super().__init__(parent, width=width, height=height, highlightthickness=0, 
                         relief='flat', bd=0)
        
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.text = text
        self.font = font
        self.width = width
        self.height = height
        self.disabled_color = "#CCCCCC"
        self.is_disabled = False
        
        self.create_button()
        self.bind_events()
        
    def create_button(self):
        self.delete("all")
        color = self.disabled_color if self.is_disabled else self.bg_color
        
        self.create_rounded_rect(2, 2, self.width-2, self.height-2, 
                                radius=8, fill=color, outline="")
        
        if not self.is_disabled:
            self.create_rounded_rect(0, 4, self.width-2, self.height, 
                                    radius=8, fill="#000000", outline="")
        
        text_color = "#999999" if self.is_disabled else self.text_color
        self.create_text(self.width//2, self.height//2, text=self.text, 
                        fill=text_color, font=self.font)
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = []
        for x, y in [(x1, y1 + radius), (x1, y1), (x1 + radius, y1),
                     (x2 - radius, y1), (x2, y1), (x2, y1 + radius),
                     (x2, y2 - radius), (x2, y2), (x2 - radius, y2),
                     (x1 + radius, y2), (x1, y2), (x1, y2 - radius)]:
            points.extend([x, y])
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def bind_events(self):
        self.bind("<Button-1>", self.on_click)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        
    def on_click(self, event):
        if not self.is_disabled and self.command:
            self.command()
            
    def on_enter(self, event):
        if not self.is_disabled:
            self.delete("all")
            self.create_rounded_rect(2, 2, self.width-2, self.height-2, 
                                    radius=8, fill=self.hover_color, outline="")
            self.create_text(self.width//2, self.height//2, text=self.text, 
                            fill=self.text_color, font=self.font)
            
    def on_leave(self, event):
        self.create_button()
        
    def configure_state(self, state):
        self.is_disabled = (state == 'disabled')
        self.create_button()

class StatusIndicator(tk.Canvas):
    def __init__(self, parent, width=20, height=20):
        super().__init__(parent, width=width, height=height, highlightthickness=0, 
                         relief='flat', bd=0, bg='#f8f9fa')
        self.width = width
        self.height = height
        self.status = "inactive"
        self.draw_indicator()
        
    def draw_indicator(self):
        self.delete("all")
        center_x, center_y = self.width//2, self.height//2
        radius = min(self.width, self.height)//3
        
        colors = {
            "inactive": "#E0E0E0",
            "loading": "#FFA500", 
            "success": "#4CAF50",
            "error": "#F44336"
        }
        
        color = colors.get(self.status, "#E0E0E0")
        
        self.create_oval(center_x-radius, center_y-radius, 
                        center_x+radius, center_y+radius, 
                        fill=color, outline="")
        
        if self.status == "loading":
            self.create_oval(center_x-radius-2, center_y-radius-2, 
                            center_x+radius+2, center_y+radius+2, 
                            fill="", outline=color, width=2)
    
    def set_status(self, status):
        self.status = status
        self.draw_indicator()

class DiabeticRetinopathyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MedVision - Diabetic Retinopathy Analysis")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f8f9fa')
        self.root.resizable(True, True)

        self.setup_styles()

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = "best_model.pt"
        
        self.class_labels = {
            0: "No Diabetic Retinopathy",
            1: "Mild Non-proliferative DR", 
            2: "Moderate Non-proliferative DR",
            3: "Severe Non-proliferative DR",
            4: "Proliferative DR"
        }
        
        self.class_descriptions = {
            0: "Normal retina with no signs of diabetic damage",
            1: "Microaneurysms present - early stage damage",
            2: "More widespread vascular abnormalities",
            3: "Extensive retinal damage with cotton wool spots",
            4: "Advanced stage with new blood vessel growth"
        }
        
        self.severity_colors = {
            0: "#10B981",  # Emerald - Healthy
            1: "#F59E0B",  # Amber - Caution  
            2: "#EF4444",  # Red - Warning
            3: "#DC2626",  # Dark Red - Danger
            4: "#991B1B"   # Very Dark Red - Critical
        }
        
        self.setup_ui()
        self.load_model_automatically()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Medical.TLabelframe', 
                       background='#ffffff',
                       borderwidth=1,
                       relief='solid')
        style.configure('Medical.TLabelframe.Label',
                       background='#ffffff',
                       foreground='#1f2937',
                       font=('Segoe UI', 12, 'bold'))
        
    def setup_ui(self):
        main_container = tk.Frame(self.root, bg='#f8f9fa')
        main_container.pack(fill='both', expand=True)
        
        self.create_header(main_container)
        
        content_frame = tk.Frame(main_container, bg='#f8f9fa')
        content_frame.pack(fill='both', expand=True, padx=30, pady=(0, 30))
        
        left_panel = tk.Frame(content_frame, bg='#ffffff', relief='flat', bd=0)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        shadow_frame = tk.Frame(content_frame, bg='#e5e7eb', height=2)
        shadow_frame.place(in_=left_panel, relx=0, rely=1, relwidth=1, anchor='nw')
        
        self.create_image_section(left_panel)
        right_panel = tk.Frame(content_frame, bg='#ffffff', relief='flat', bd=0, width=420)
        right_panel.pack(side='right', fill='y', padx=(15, 0))
        right_panel.pack_propagate(False)
        
        self.create_results_section(right_panel)
        
        self.current_image_path = None
        self.current_image = None
        
    def create_header(self, parent):
        header_frame = tk.Frame(parent, bg='#1f2937', height=100)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        for i in range(5):
            shade = 31 + i * 3
            grad_frame = tk.Frame(header_frame, bg=f'#{shade:02x}{41+i*2:02x}{55+i*3:02x}', height=2)
            grad_frame.pack(fill='x')
        
        content_header = tk.Frame(header_frame, bg='#1f2937')
        content_header.pack(fill='both', expand=True, padx=30, pady=20)

        title_frame = tk.Frame(content_header, bg='#1f2937')
        title_frame.pack(side='left', fill='both', expand=True)
        
        title_label = tk.Label(
            title_frame,
            text="MedVision AI",
            font=('Segoe UI', 24, 'bold'),
            bg='#1f2937',
            fg='#ffffff'
        )
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(
            title_frame,
            text="Diabetic Retinopathy Classification System",
            font=('Segoe UI', 14),
            bg='#1f2937',
            fg='#9ca3af'
        )
        subtitle_label.pack(anchor='w', pady=(5, 0))
        
        status_frame = tk.Frame(content_header, bg='#1f2937')
        status_frame.pack(side='right', padx=(20, 0))
        
        self.model_status_indicator = StatusIndicator(status_frame)
        self.model_status_indicator.pack(side='left', padx=(0, 10), pady=10)
        
        self.model_status_label = tk.Label(
            status_frame,
            text="Loading AI Model...",
            font=('Segoe UI', 11),
            bg='#1f2937',
            fg='#9ca3af'
        )
        self.model_status_label.pack(side='left', pady=10)
        
    def create_image_section(self, parent):
        upload_frame = tk.Frame(parent, bg='#ffffff')
        upload_frame.pack(fill='x', padx=25, pady=25)
        
        upload_title = tk.Label(
            upload_frame,
            text="Retinal Image Analysis",
            font=('Segoe UI', 14, 'bold'),
            bg='#ffffff',
            fg='#1f2937'
        )
        upload_title.pack(anchor='w', pady=(0, 15))

        self.image_frame = tk.Frame(upload_frame, bg='#f3f4f6', relief='flat', bd=2)
        self.image_frame.pack(pady=(0, 20))
        
        self.image_canvas = tk.Canvas(
            self.image_frame,
            width=400,
            height=400,
            bg='#ffffff',
            relief='flat',
            bd=0,
            highlightthickness=1,
            highlightcolor='#e5e7eb'
        )
        self.image_canvas.pack(padx=3, pady=3)
        
        self.create_image_placeholder()

        button_frame = tk.Frame(upload_frame, bg='#ffffff')
        button_frame.pack(pady=(0, 10))
        
        self.select_btn = ModernButton(
            button_frame,
            "Select Retinal Image",
            self.select_image,
            bg_color="#4F46E5",
            hover_color="#4338CA",
            width=180,
            height=50
        )
        self.select_btn.pack(side='left', padx=(0, 15))
        
        self.analyze_btn = ModernButton(
            button_frame,
            "Analyze Image",
            self.classify_image,
            bg_color="#059669",
            hover_color="#047857",
            width=150,
            height=50
        )
        self.analyze_btn.pack(side='left')
        self.analyze_btn.configure_state('disabled')
        
        instructions = tk.Label(
            upload_frame,
            text="Select a high-quality retinal fundus image (JPG, PNG, TIFF)\nOptimal resolution: 260x260 pixels or higher",
            font=('Segoe UI', 10),
            bg='#ffffff',
            fg='#6b7280',
            justify='center'
        )
        instructions.pack(pady=(15, 0))
        
    def create_image_placeholder(self):
        self.image_canvas.delete("all")

        canvas_width = 400
        canvas_height = 400

        self.image_canvas.create_rectangle(0, 0, canvas_width, canvas_height, 
                                         fill='#f9fafb', outline='')
        
        dash_length = 10
        gap_length = 8
        for i in range(0, canvas_width, dash_length + gap_length):
            self.image_canvas.create_line(i, 0, min(i + dash_length, canvas_width), 0, 
                                        fill='#d1d5db', width=2)
            self.image_canvas.create_line(i, canvas_height-1, min(i + dash_length, canvas_width), 
                                        canvas_height-1, fill='#d1d5db', width=2)
        
        for i in range(0, canvas_height, dash_length + gap_length):
            self.image_canvas.create_line(0, i, 0, min(i + dash_length, canvas_height), 
                                        fill='#d1d5db', width=2)
            self.image_canvas.create_line(canvas_width-1, i, canvas_width-1, 
                                        min(i + dash_length, canvas_height), fill='#d1d5db', width=2)

        center_x, center_y = canvas_width//2, canvas_height//2
        
        self.image_canvas.create_oval(center_x-40, center_y-25, center_x+40, center_y+25, 
                                    outline='#9ca3af', width=3, fill='')
        
        self.image_canvas.create_oval(center_x-15, center_y-15, center_x+15, center_y+15, 
                                    fill='#6b7280', outline='')
        
        self.image_canvas.create_oval(center_x-8, center_y-8, center_x+8, center_y+8, 
                                    fill='#9ca3af', outline='')
        
        self.image_canvas.create_text(center_x, center_y + 60, 
                                    text="Upload Retinal Image", 
                                    font=('Segoe UI', 14, 'bold'), 
                                    fill='#6b7280')
        
        self.image_canvas.create_text(center_x, center_y + 80, 
                                    text="Drag & drop or click to browse", 
                                    font=('Segoe UI', 11), 
                                    fill='#9ca3af')
        
    def create_results_section(self, parent):
        results_header = tk.Frame(parent, bg='#ffffff')
        results_header.pack(fill='x', padx=25, pady=(25, 0))
        
        results_title = tk.Label(
            results_header,
            text="Diagnostic Results",
            font=('Segoe UI', 16, 'bold'),
            bg='#ffffff',
            fg='#1f2937'
        )
        results_title.pack(anchor='w')

        self.diagnosis_card = tk.Frame(parent, bg='#f8f9fa', relief='flat', bd=1)
        self.diagnosis_card.pack(fill='x', padx=25, pady=20)
        diagnosis_content = tk.Frame(self.diagnosis_card, bg='#f8f9fa')
        diagnosis_content.pack(fill='x', padx=25, pady=25)
        
        self.diagnosis_label = tk.Label(
            diagnosis_content,
            text="No Analysis Performed",
            font=('Segoe UI', 16, 'bold'),
            bg='#f8f9fa',
            fg='#6b7280',
            wraplength=330,
            justify='left'
        )
        self.diagnosis_label.pack(anchor='w')
        
        self.confidence_label = tk.Label(
            diagnosis_content,
            text="",
            font=('Segoe UI', 11),
            bg='#f8f9fa',
            fg='#6b7280',
            wraplength=330,
            justify='left'
        )
        self.confidence_label.pack(anchor='w', pady=(5, 0))
        
        self.description_label = tk.Label(
            diagnosis_content,
            text="Upload a retinal image to begin analysis",
            font=('Segoe UI', 10),
            bg='#f8f9fa',
            fg='#9ca3af',
            wraplength=330,
            justify='left'
        )
        self.description_label.pack(anchor='w', pady=(10, 0))

        info_frame = tk.Frame(parent, bg='#ffffff')
        info_frame.pack(fill='x', padx=25, pady=(20, 0))
        
        info_title = tk.Label(
            info_frame,
            text="Classification Levels",
            font=('Segoe UI', 14, 'bold'),
            bg='#ffffff',
            fg='#1f2937'
        )
        info_title.pack(anchor='w', pady=(0, 15))

        for level, (class_name, color) in enumerate(zip(self.class_labels.values(), 
                                                       self.severity_colors.values())):
            level_frame = tk.Frame(info_frame, bg='#ffffff')
            level_frame.pack(fill='x', pady=2)
            
            indicator = tk.Canvas(level_frame, width=16, height=16, 
                                bg='#ffffff', highlightthickness=0)
            indicator.pack(side='left', padx=(0, 10), pady=2)
            indicator.create_oval(2, 2, 14, 14, fill=color, outline='')
            
            level_label = tk.Label(
                level_frame,
                text=class_name,
                font=('Segoe UI', 10),
                bg='#ffffff',
                fg='#374151'
            )
            level_label.pack(side='left', anchor='w')
        
    def load_model_automatically(self):
        if os.path.exists(self.model_path):
            self.model_status_indicator.set_status("loading")
            self.model_status_label.config(text="Loading AI Model...")
            self.root.update()
            
            try:
                self.model = efficientnet_b2(weights=None)
                self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 5)

                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model = self.model.to(self.device)
                self.model.eval()
                
                self.model_status_indicator.set_status("success")
                device_name = "GPU" if self.device.type == "cuda" else "CPU"
                self.model_status_label.config(text=f"AI Model Ready ({device_name})")
                
            except Exception as e:
                self.model_status_indicator.set_status("error")
                self.model_status_label.config(text="Model Loading Failed")
                messagebox.showerror("Model Error", f"Failed to load model:\n{str(e)}")
        else:
            self.model_status_indicator.set_status("error")
            self.model_status_label.config(text="Model File Not Found")
            messagebox.showerror("File Error", f"Model file '{self.model_path}' not found.")
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Retinal Fundus Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG Images", "*.jpg *.jpeg"),
                ("PNG Images", "*.png"),
                ("TIFF Images", "*.tiff *.tif"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                image = Image.open(file_path)
                self.current_image_path = file_path
                self.current_image = image

                self.display_image(image)

                if self.model is not None:
                    self.analyze_btn.configure_state('normal')
                self.diagnosis_label.config(
                    text="Image Loaded Successfully",
                    fg='#059669',
                    bg='#f8f9fa'
                )
                self.confidence_label.config(text="", bg='#f8f9fa')
                self.description_label.config(text="Click 'Analyze Image' to perform classification", bg='#f8f9fa')
                
            except Exception as e:
                messagebox.showerror("Image Error", f"Failed to load image:\n{str(e)}")
    
    def display_image(self, image):
        self.image_canvas.delete("all")

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        canvas_size = (394, 394)
        image.thumbnail(canvas_size, Image.Resampling.LANCZOS)
        
        canvas_width, canvas_height = 400, 400
        x = (canvas_width - image.width) // 2
        y = (canvas_height - image.height) // 2

        photo = ImageTk.PhotoImage(image)

        self.image_canvas.create_rectangle(0, 0, canvas_width, canvas_height, 
                                         fill='#000000', outline='')

        self.image_canvas.create_image(x, y, anchor='nw', image=photo)
        self.image_canvas.image = photo

        self.image_canvas.create_text(10, canvas_height-25, 
                                    text=f"Resolution: {self.current_image.size[0]}x{self.current_image.size[1]}", 
                                    font=('Segoe UI', 9), 
                                    fill='#ffffff', anchor='w')
    
    def classify_image(self):
        if self.model is None:
            messagebox.showwarning("Warning", "AI model is not loaded.")
            return
            
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        
        try:
            self.analyze_btn.configure_state('disabled')
            self.diagnosis_label.config(text="Analyzing...", fg='#f59e0b', bg='#f8f9fa')
            self.confidence_label.config(text="Processing retinal image...", bg='#f8f9fa')
            self.description_label.config(text="AI is examining the image for signs of diabetic retinopathy", bg='#f8f9fa')
            self.root.update()
            
            transform = transforms.Compose([
                transforms.Resize((260, 260)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])

            if self.current_image.mode != 'RGB':
                image = self.current_image.convert('RGB')
            else:
                image = self.current_image

            input_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = predicted.item()
                confidence_score = confidence.item() * 100
                
                class_name = self.class_labels[predicted_class]
                class_description = self.class_descriptions[predicted_class]
                severity_color = self.severity_colors[predicted_class]
                self.diagnosis_card.config(bg='#f8f9fa', relief='solid', bd=1)
                
                self.diagnosis_label.config(
                    text=class_name,
                    fg=severity_color,
                    bg='#f8f9fa'
                )

                confidence_text = f"Confidence: {confidence_score:.1f}%"
                if confidence_score >= 90:
                    confidence_text += " (High)"
                elif confidence_score >= 75:
                    confidence_text += " (Moderate)"
                else:
                    confidence_text += " (Low - Review Needed)"
                
                self.confidence_label.config(
                    text=confidence_text,
                    fg='#374151',
                    bg='#f8f9fa'
                )
                
                self.description_label.config(
                    text=class_description,
                    fg='#6b7280',
                    bg='#f8f9fa'
                )
                
                self.show_detailed_results(probabilities[0])
                self.analyze_btn.configure_state('normal')
                
        except Exception as e:
            self.analyze_btn.configure_state('normal')
            self.diagnosis_label.config(text="Analysis Failed", fg='#ef4444', bg='#f8f9fa')
            self.confidence_label.config(text="", bg='#f8f9fa')
            self.description_label.config(text="An error occurred during analysis", bg='#f8f9fa')
            messagebox.showerror("Analysis Error", f"Classification failed:\n{str(e)}")
    
    def show_detailed_results(self, probabilities):
        detail_window = tk.Toplevel(self.root)
        detail_window.title("Detailed Analysis Results")
        detail_window.geometry("500x600")
        detail_window.configure(bg='#ffffff')
        detail_window.resizable(False, False)

        detail_window.transient(self.root)
        detail_window.grab_set()

        header_frame = tk.Frame(detail_window, bg='#1f2937', height=60)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame,
            text="Probability Distribution Analysis",
            font=('Segoe UI', 16, 'bold'),
            bg='#1f2937',
            fg='#ffffff'
        )
        header_label.pack(expand=True)

        content_frame = tk.Frame(detail_window, bg='#ffffff')
        content_frame.pack(fill='both', expand=True, padx=30, pady=30)

        for i, (class_name, prob) in enumerate(zip(self.class_labels.values(), probabilities)):
            prob_value = prob.item() * 100
            color = self.severity_colors[i]

            class_frame = tk.Frame(content_frame, bg='#ffffff')
            class_frame.pack(fill='x', pady=8)
            
            info_frame = tk.Frame(class_frame, bg='#ffffff')
            info_frame.pack(fill='x', pady=(0, 5))
            
            name_label = tk.Label(
                info_frame,
                text=class_name,
                font=('Segoe UI', 11, 'bold'),
                bg='#ffffff',
                fg='#1f2937'
            )
            name_label.pack(side='left')
            
            percent_label = tk.Label(
                info_frame,
                text=f"{prob_value:.1f}%",
                font=('Segoe UI', 11, 'bold'),
                bg='#ffffff',
                fg=color
            )
            percent_label.pack(side='right')

            bar_frame = tk.Frame(class_frame, bg='#f3f4f6', height=20, relief='flat', bd=1)
            bar_frame.pack(fill='x')
            bar_frame.pack_propagate(False)
            
            fill_width = int((prob_value / 100) * 450)  # Max width
            if fill_width > 0:
                fill_frame = tk.Frame(bar_frame, bg=color, height=18)
                fill_frame.place(x=1, y=1, width=fill_width)

        close_btn = ModernButton(
            content_frame,
            "Close",
            detail_window.destroy,
            bg_color="#6b7280",
            hover_color="#4b5563",
            width=120,
            height=40
        )
        close_btn.pack(pady=(30, 0))

def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    root = tk.Tk()
    
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    app = DiabeticRetinopathyApp(root)
    
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
