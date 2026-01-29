import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from processor import PassportGenerator

class PassportApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Passport Studio")
        self.geometry("1200x800")
        
        self.processor = PassportGenerator()
        self.input_path = None
        self.base_passport_img = None 
        self.enhanced_img = None      
        self.grid_img = None

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Sidebar
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Studio Controls", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=10)

        # 1. Select
        self.select_btn = ctk.CTkButton(self.sidebar, text="1. Select Image", command=self.select_image)
        self.select_btn.pack(pady=10, padx=20)

        # 2. Knobs
        self.knobs_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.knobs_frame.pack(fill="x", pady=5)

        self.add_knob("Brightness", 1.0, 0.5, 1.5)
        self.add_knob("Contrast", 1.0, 0.5, 1.5)
        self.add_knob("Sharpness", 1.0, 0.0, 3.0)
        self.add_knob("Color", 1.0, 0.0, 2.0)

        self.remove_bg_var = ctk.BooleanVar(value=True)
        self.remove_bg_switch = ctk.CTkSwitch(self.sidebar, text="Remove Background", variable=self.remove_bg_var)
        self.remove_bg_switch.pack(pady=5, padx=20)

        self.process_btn = ctk.CTkButton(self.sidebar, text="2. Process Photo", command=self.initial_process)
        self.process_btn.pack(pady=10, padx=20)

        # 3. Grid
        self.count_label = ctk.CTkLabel(self.sidebar, text="Grid Size:")
        self.count_label.pack(pady=(10, 5))
        self.count_seg_btn = ctk.CTkSegmentedButton(self.sidebar, values=["8", "16"])
        self.count_seg_btn.set("8")
        self.count_seg_btn.pack(pady=5, padx=20)

        self.generate_btn = ctk.CTkButton(self.sidebar, text="3. Create Grid (Yes?)", command=self.confirm_and_generate, state="disabled")
        self.generate_btn.pack(pady=15, padx=20)

        self.save_btn = ctk.CTkButton(self.sidebar, text="4. Save Result", command=self.save_image, state="disabled")
        self.save_btn.pack(pady=5, padx=20)

        # Right Preview Area
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Upload a photo to start")
        self.preview_label.pack(expand=True, fill="both")

    def add_knob(self, label, default, from_, to):
        lbl = ctk.CTkLabel(self.knobs_frame, text=label, font=ctk.CTkFont(size=12))
        lbl.pack(padx=20, anchor="w")
        
        slider = ctk.CTkSlider(self.knobs_frame, from_=from_, to=to, number_of_steps=100, command=self.update_enhancements)
        slider.set(default)
        slider.pack(padx=20, pady=(0, 10), fill="x")
        
        setattr(self, f"{label.lower()}_slider", slider)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            self.input_path = file_path
            self.show_preview(file_path)
            self.generate_btn.configure(state="disabled")
            self.save_btn.configure(state="disabled")

    def show_preview(self, path_or_img):
        if isinstance(path_or_img, str):
            img = Image.open(path_or_img)
        else:
            img = path_or_img
            
        img_copy = img.copy()
        img_copy.thumbnail((600, 600))
        ctk_img = ctk.CTkImage(light_image=img_copy, dark_image=img_copy, size=img_copy.size)
        self.preview_label.configure(image=ctk_img, text="")
        self.preview_label.image = ctk_img

    def initial_process(self):
        if not self.input_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return

        try:
            self.base_passport_img = self.processor.process_image(
                self.input_path, 
                remove_bg=self.remove_bg_var.get()
            )
            self.update_enhancements()
            self.generate_btn.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process: {str(e)}")

    def update_enhancements(self, _=None):
        if self.base_passport_img:
            self.enhanced_img = self.processor.enhance_image(
                self.base_passport_img,
                brightness=self.brightness_slider.get(),
                contrast=self.contrast_slider.get(),
                sharpness=self.sharpness_slider.get(),
                color=self.color_slider.get()
            )
            self.show_preview(self.enhanced_img)

    def confirm_and_generate(self):
        if not self.enhanced_img:
            return
            
        if messagebox.askyesno("Confirm", "Continue to generate the grid?"):
            try:
                count = int(self.count_seg_btn.get())
                self.grid_img = self.processor.create_grid(self.enhanced_img, count)
                self.show_preview(self.grid_img)
                self.save_btn.configure(state="normal")
            except Exception as e:
                messagebox.showerror("Error", f"Grid failure: {str(e)}")

    def save_image(self):
        if self.grid_img:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
            )
            if save_path:
                self.grid_img.save(save_path, quality=95)
                messagebox.showinfo("Saved", f"Image saved to {save_path}")

if __name__ == "__main__":
    app = PassportApp()
    app.mainloop()
