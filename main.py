import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from processor import PassportGenerator

class PassportApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Passport Photo Generator")
        self.geometry("800x600")
        
        self.processor = PassportGenerator()
        self.input_path = None
        self.processed_img = None
        self.grid_img = None

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Passport Tool", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20)

        self.select_btn = ctk.CTkButton(self.sidebar, text="Select Image", command=self.select_image)
        self.select_btn.pack(pady=10, padx=20)

        self.count_label = ctk.CTkLabel(self.sidebar, text="Number of Images:")
        self.count_label.pack(pady=(20, 5))
        
        self.count_seg_btn = ctk.CTkSegmentedButton(self.sidebar, values=["8", "16"])
        self.count_seg_btn.set("8")
        self.count_seg_btn.pack(pady=10, padx=20)

        self.generate_btn = ctk.CTkButton(self.sidebar, text="Generate Grid", command=self.generate_passport)
        self.generate_btn.pack(pady=20, padx=20)

        self.save_btn = ctk.CTkButton(self.sidebar, text="Save Result", command=self.save_image, state="disabled")
        self.save_btn.pack(pady=10, padx=20)

        # Right Preview Area
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="No image selected")
        self.preview_label.pack(expand=True, fill="both")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            self.input_path = file_path
            self.show_preview(file_path)

    def show_preview(self, path):
        img = Image.open(path)
        img.thumbnail((500, 500))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.preview_label.configure(image=ctk_img, text="")
        self.preview_label.image = ctk_img

    def generate_passport(self):
        if not self.input_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return

        try:
            # Step 1: Crop face to passport size
            self.processed_img = self.processor.process_image(self.input_path)
            
            # Step 2: Create grid
            count = int(self.count_seg_btn.get())
            self.grid_img = self.processor.create_grid(self.processed_img, count)
            
            # Show preview of the grid
            preview_grid = self.grid_img.copy()
            preview_grid.thumbnail((500, 500))
            ctk_grid = ctk.CTkImage(light_image=preview_grid, dark_image=preview_grid, size=preview_grid.size)
            self.preview_label.configure(image=ctk_grid)
            self.preview_label.image = ctk_grid
            
            self.save_btn.configure(state="normal")
            messagebox.showinfo("Success", "Passport grid generated!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

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
