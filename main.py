import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas, colorchooser
from PIL import Image, ImageTk
import os
from processor import PassportGenerator

class PassportApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Passport Studio")
        self.geometry("1400x900")
        
        self.processor = PassportGenerator()
        self.input_path = None
        self.base_passport_img = None 
        self.enhanced_img = None      
        self.grid_img = None
        
        # Manual alignment state
        self.original_processed_img = None  # Store the original processed image before alignment
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # ROI Selection state
        self.roi_rect_id = None
        self.roi_start_x = None
        self.roi_start_y = None
        self.roi_coords = None # (x1, y1, x2, y2) in canvas coordinates
        self.is_selecting_roi = False
        
        self.bg_color = "#FFFFFF" # Default white background
        
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Sidebar (Scrollable)
        self.sidebar = ctk.CTkScrollableFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Studio Controls", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=10)

        # 1. Select & Reset
        self.btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=10)
        
        self.select_btn = ctk.CTkButton(self.btn_frame, text="1. Select Image", command=self.select_image, width=120)
        self.select_btn.pack(side="left", padx=5, pady=5)
        
        self.reset_btn = ctk.CTkButton(self.btn_frame, text="Reset", command=self.reset_all, width=80, fg_color="#d32f2f", hover_color="#b71c1c")
        self.reset_btn.pack(side="left", padx=5, pady=5)

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

        # Background Color Selection
        self.bg_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.bg_frame.pack(fill="x", pady=5, padx=20)
        
        self.bg_color_label = ctk.CTkLabel(self.bg_frame, text="Background Color:")
        self.bg_color_label.pack(side="left", padx=5)
        
        self.color_preview = ctk.CTkButton(self.bg_frame, text="", width=30, height=30, 
                                          fg_color=self.bg_color, hover_color=self.bg_color,
                                          corner_radius=15, command=self.choose_bg_color)
        self.color_preview.pack(side="left", padx=5)

        # Quick Studio Presets
        self.presets_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.presets_frame.pack(fill="x", pady=5, padx=20)
        
        presets = [
            ("#FFFFFF", "White"), 
            ("#E1F5FE", "Light Blue"), 
            ("#B3E5FC", "Sky Blue"), 
            ("#F5F5F5", "Light Grey"),
            ("#E0E0E0", "Studio Grey")
        ]
        
        for color, name in presets:
            btn = ctk.CTkButton(self.presets_frame, text="", width=25, height=25, 
                                fg_color=color, hover_color=color, corner_radius=12,
                                command=lambda c=color: self.set_preset_color(c))
            btn.pack(side="left", padx=3)

        # Studio Enhancement Sliders
        self.enhancements_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.enhancements_frame.pack(fill="x", pady=5)
        
        self.enhancements_label = ctk.CTkLabel(self.enhancements_frame, text="Studio Enhancements", font=ctk.CTkFont(size=14, weight="bold"))
        self.enhancements_label.pack(pady=5)
        
        self.add_enhancement_slider("Edge Smoothing", 0.5, 0.0, 1.0)
        self.add_enhancement_slider("Skin Smoothing", 0.0, 0.0, 1.0)
        self.add_enhancement_slider("Red-eye Removal", 0.0, 0.0, 1.0)
        self.add_enhancement_slider("Teeth Whitening", 0.0, 0.0, 1.0)
        self.add_enhancement_slider("Eye Brightening", 0.0, 0.0, 1.0)
        self.add_enhancement_slider("Auto White Balance", 0.5, 0.0, 1.0)
        self.add_enhancement_slider("Shadow/Highlight Fix", 0.0, 0.0, 1.0)
        self.add_enhancement_slider("Vignette Removal", 0.0, 0.0, 1.0)

        self.process_btn = ctk.CTkButton(self.sidebar, text="2. Process Photo", command=self.initial_process)
        self.process_btn.pack(pady=10, padx=20)

        # 3. Manual Alignment Controls (initially hidden)
        self.alignment_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.alignment_frame.pack(fill="x", pady=5)
        self.alignment_frame.pack_forget()  # Hidden by default
        
        self.alignment_label = ctk.CTkLabel(self.alignment_frame, text="Manual Alignment", font=ctk.CTkFont(size=14, weight="bold"))
        self.alignment_label.pack(pady=5)
        
        # Zoom controls
        self.zoom_frame = ctk.CTkFrame(self.alignment_frame, fg_color="transparent")
        self.zoom_frame.pack(fill="x", pady=5)
        
        self.zoom_out_btn = ctk.CTkButton(self.zoom_frame, text="-", width=40, command=self.zoom_out)
        self.zoom_out_btn.pack(side="left", padx=5)
        
        self.zoom_label = ctk.CTkLabel(self.zoom_frame, text="Zoom: 100%")
        self.zoom_label.pack(side="left", padx=10)
        
        self.zoom_in_btn = ctk.CTkButton(self.zoom_frame, text="+", width=40, command=self.zoom_in)
        self.zoom_in_btn.pack(side="left", padx=5)
        
        self.reset_zoom_btn = ctk.CTkButton(self.alignment_frame, text="Reset Alignment", command=self.reset_alignment)
        self.reset_zoom_btn.pack(pady=5)
        
        self.apply_alignment_btn = ctk.CTkButton(self.alignment_frame, text="Apply Alignment", command=self.apply_alignment)
        self.apply_alignment_btn.pack(pady=5)
        
        self.alignment_hint = ctk.CTkLabel(self.alignment_frame, text="Drag image to reposition", font=ctk.CTkFont(size=10))
        self.alignment_hint.pack(pady=2)

        # 4. Grid
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
        
        # Canvas for interactive alignment
        self.preview_canvas = Canvas(self.preview_frame, bg="#2b2b2b", highlightthickness=0)
        self.preview_canvas.pack(expand=True, fill="both")
        
        # Bind mouse events for panning
        self.preview_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.preview_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.preview_canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows/Mac
        self.preview_canvas.bind("<Button-4>", self.on_mouse_wheel)     # Linux scroll up
        self.preview_canvas.bind("<Button-5>", self.on_mouse_wheel)     # Linux scroll down
        
        # Initial placeholder text on canvas (centered)
        self.preview_canvas.create_text(
            400, 300,
            text="Upload a photo to start",
            fill="white",
            font=("Helvetica", 16),
            tags="placeholder"
        )

    def add_knob(self, label, default, from_, to):
        lbl = ctk.CTkLabel(self.knobs_frame, text=label, font=ctk.CTkFont(size=12))
        lbl.pack(padx=20, anchor="w")
        
        slider = ctk.CTkSlider(self.knobs_frame, from_=from_, to=to, number_of_steps=100, command=self.update_enhancements)
        slider.set(default)
        slider.pack(padx=20, pady=(0, 10), fill="x")
        
        setattr(self, f"{label.lower()}_slider", slider)

    def add_enhancement_slider(self, label, default, from_, to):
        """Add a studio enhancement slider with label showing percentage"""
        frame = ctk.CTkFrame(self.enhancements_frame, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=2)
        
        lbl = ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=11))
        lbl.pack(side="left")
        
        value_lbl = ctk.CTkLabel(frame, text=f"{int(default * 100)}%", font=ctk.CTkFont(size=11), width=40)
        value_lbl.pack(side="right")
        
        slider = ctk.CTkSlider(self.enhancements_frame, from_=from_, to=to, number_of_steps=100)
        slider.set(default)
        slider.pack(padx=20, pady=(0, 5), fill="x")
        
        # Store reference to value label for updates
        slider.value_label = value_lbl
        slider.label_text = label
        
        # Bind to update value label and enhancements in real-time
        def on_slider_change(value):
            value_lbl.configure(text=f"{int(value * 100)}%")
            self.update_enhancements()
        
        slider.configure(command=on_slider_change)
        
        # Store with sanitized name
        attr_name = label.lower().replace(" ", "_").replace("/", "_").replace("-", "_") + "_slider"
        setattr(self, attr_name, slider)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            self.input_path = file_path
            self.show_preview(file_path)
            self.generate_btn.configure(state="disabled")
            self.save_btn.configure(state="disabled")
            self.alignment_frame.pack_forget()
            self.reset_alignment_state()

    def show_preview(self, path_or_img):
        if isinstance(path_or_img, str):
            img = Image.open(path_or_img)
        else:
            img = path_or_img.copy()
            
        # Clear canvas
        self.preview_canvas.delete("all")
        
        # Store current image for alignment
        self.current_preview_img = img
        
        # Force canvas update to get correct dimensions
        self.preview_canvas.update_idletasks()
        
        # Calculate display size to fit canvas while maintaining aspect ratio
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width < 100:  # Canvas not yet rendered
            canvas_width = 800
            canvas_height = 600
        
        img_copy = img.copy()
        img_copy.thumbnail((canvas_width - 40, canvas_height - 40), Image.Resampling.LANCZOS)
        
        # Store scale and offset for ROI mapping
        self.preview_scale = img.width / img_copy.width
        self.preview_offset_x = (canvas_width - img_copy.width) // 2
        self.preview_offset_y = (canvas_height - img_copy.height) // 2
        
        # Convert to PhotoImage - keep reference to prevent garbage collection
        self.current_tk_img = ImageTk.PhotoImage(img_copy)
        
        # Display on canvas centered
        x = canvas_width // 2
        y = canvas_height // 2
        self.preview_canvas.create_image(x, y, image=self.current_tk_img, anchor="center", tags="image")
        
        # Force canvas refresh
        self.preview_canvas.update()

    def show_aligned_preview(self):
        """Display the image with current zoom and pan applied"""
        if self.original_processed_img is None:
            return
        
        # Clear canvas
        self.preview_canvas.delete("all")
        
        # Get canvas size
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width < 100:
            canvas_width = 800
            canvas_height = 600
        
        # Calculate the crop area based on zoom and pan
        img_width, img_height = self.original_processed_img.size
        
        # Calculate visible area size (inverse of zoom)
        visible_width = int(img_width / self.zoom_level)
        visible_height = int(img_height / self.zoom_level)
        
        # Clamp pan values to keep image within bounds
        max_pan_x = (img_width - visible_width) // 2
        max_pan_y = (img_height - visible_height) // 2
        self.pan_x = max(-max_pan_x, min(max_pan_x, self.pan_x))
        self.pan_y = max(-max_pan_y, min(max_pan_y, self.pan_y))
        
        # Calculate crop coordinates
        center_x = img_width // 2 + self.pan_x
        center_y = img_height // 2 + self.pan_y
        
        left = max(0, center_x - visible_width // 2)
        top = max(0, center_y - visible_height // 2)
        right = min(img_width, left + visible_width)
        bottom = min(img_height, top + visible_height)
        
        # Ensure we have the full visible area
        if right - left < visible_width:
            left = max(0, right - visible_width)
        if bottom - top < visible_height:
            top = max(0, bottom - visible_height)
        
        # Crop and resize to passport size
        cropped = self.original_processed_img.crop((left, top, right, bottom))
        
        # Apply background color in real-time if background was removed
        if cropped.mode == 'RGBA':
            cropped = self.processor.apply_studio_background(cropped, self.bg_color)
            
        self.current_aligned_img = cropped.resize((413, 531), Image.Resampling.LANCZOS)
        
        # Display on canvas
        display_img = self.current_aligned_img.copy()
        display_img.thumbnail((canvas_width - 40, canvas_height - 40), Image.Resampling.LANCZOS)
        
        self.current_tk_img = ImageTk.PhotoImage(display_img)
        x = canvas_width // 2
        y = canvas_height // 2
        self.preview_canvas.create_image(x, y, image=self.current_tk_img, anchor="center", tags="image")
        
        # Draw crop overlay to indicate the passport area
        self.draw_crop_overlay(canvas_width, canvas_height)
        
        # Force canvas update
        self.preview_canvas.update()

    def draw_crop_overlay(self, canvas_width, canvas_height):
        """Draw an overlay showing the passport crop area"""
        # Calculate the display size of the aligned image
        if not hasattr(self, 'current_aligned_img'):
            return
        
        img_w, img_h = self.current_aligned_img.size
        
        # Calculate thumbnail size (same logic as show_aligned_preview)
        display_w = min(img_w, canvas_width - 40)
        display_h = min(img_h, canvas_height - 40)
        
        # Maintain aspect ratio
        scale = min(display_w / img_w, display_h / img_h)
        display_w = int(img_w * scale)
        display_h = int(img_h * scale)
        
        # Center position
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        # Draw border around the image
        x1 = center_x - display_w // 2
        y1 = center_y - display_h // 2
        x2 = center_x + display_w // 2
        y2 = center_y + display_h // 2
        
        self.preview_canvas.create_rectangle(
            x1 - 2, y1 - 2, x2 + 2, y2 + 2,
            outline="#00ff00", width=2, tags="overlay"
        )

    def reset_alignment_state(self):
        """Reset alignment state variables"""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        # Don't reset original_processed_img here - it's set by initial_process
        self.update_zoom_label()

    def update_zoom_label(self):
        """Update the zoom percentage label"""
        self.zoom_label.configure(text=f"Zoom: {int(self.zoom_level * 100)}%")

    def zoom_in(self):
        """Increase zoom level"""
        if self.zoom_level < 3.0:
            self.zoom_level = min(3.0, self.zoom_level + 0.1)
            self.update_zoom_label()
            self.show_aligned_preview()

    def zoom_out(self):
        """Decrease zoom level"""
        if self.zoom_level > 0.5:
            self.zoom_level = max(0.5, self.zoom_level - 0.1)
            self.update_zoom_label()
            self.show_aligned_preview()

    def reset_alignment(self):
        """Reset alignment to default"""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_zoom_label()
        self.show_aligned_preview()

    def apply_alignment(self):
        """Apply the current alignment and proceed"""
        if hasattr(self, 'current_aligned_img'):
            # Store the current state as the base for further enhancements
            # We need to re-process the alignment area but WITHOUT the background applied
            # so that update_enhancements can still swap colors later.
            
            img_width, img_height = self.original_processed_img.size
            visible_width = int(img_width / self.zoom_level)
            visible_height = int(img_height / self.zoom_level)
            
            center_x = img_width // 2 + self.pan_x
            center_y = img_height // 2 + self.pan_y
            
            left = max(0, center_x - visible_width // 2)
            top = max(0, center_y - visible_height // 2)
            right = min(img_width, left + visible_width)
            bottom = min(img_height, top + visible_height)
            
            # This is the transparency-preserved version
            self.base_passport_img = self.original_processed_img.crop((left, top, right, bottom)).resize((413, 531), Image.Resampling.LANCZOS)
            
            self.update_enhancements()
            
            # Show the Generate Grid button
            self.generate_btn.configure(state="normal")
            
            # Hide alignment frame to indicate success
            self.alignment_frame.pack_forget()
            
            messagebox.showinfo("Alignment Applied", "Manual alignment has been applied. You can now adjust background colors and studio enhancements.")

    def on_mouse_down(self, event):
        """Handle mouse button press for panning or ROI selection"""
        if self.input_path and not self.original_processed_img:
            # Start ROI selection if image is loaded but not yet processed
            self.is_selecting_roi = True
            self.roi_start_x = event.x
            self.roi_start_y = event.y
            if self.roi_rect_id:
                self.preview_canvas.delete(self.roi_rect_id)
            self.roi_rect_id = self.preview_canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="yellow", width=2, dash=(4, 4)
            )
        elif self.original_processed_img is not None:
            # Handle panning
            self.is_dragging = True
            self.drag_start_x = event.x
            self.drag_start_y = event.y

    def on_mouse_drag(self, event):
        """Handle mouse drag for panning or ROI selection"""
        if self.is_selecting_roi:
            # Update ROI rectangle
            self.preview_canvas.coords(
                self.roi_rect_id,
                self.roi_start_x, self.roi_start_y,
                event.x, event.y
            )
        elif self.is_dragging and self.original_processed_img is not None:
            # Handle panning
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            
            sensitivity = 2 / self.zoom_level
            self.pan_x -= int(dx * sensitivity)
            self.pan_y -= int(dy * sensitivity)
            
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            
            self.show_aligned_preview()

    def on_mouse_up(self, event):
        """Handle mouse button release"""
        if self.is_selecting_roi:
            self.is_selecting_roi = False
            self.roi_coords = (
                min(self.roi_start_x, event.x),
                min(self.roi_start_y, event.y),
                max(self.roi_start_x, event.x),
                max(self.roi_start_y, event.y)
            )
            # Ensure selection has some size
            if self.roi_coords[2] - self.roi_coords[0] < 10 or self.roi_coords[3] - self.roi_coords[1] < 10:
                self.preview_canvas.delete(self.roi_rect_id)
                self.roi_rect_id = None
                self.roi_coords = None
        
        self.is_dragging = False

    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if self.original_processed_img is None:
            return
        
        # Determine scroll direction
        if hasattr(event, 'delta'):
            # Windows/Mac
            delta = event.delta
        else:
            # Linux
            delta = 120 if event.num == 4 else -120
        
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def reset_all(self):
        """Reset the application state but keep the input image"""
        if self.input_path:
            # Reset all images except the input path
            self.base_passport_img = None
            self.enhanced_img = None
            self.grid_img = None
            self.original_processed_img = None
            
            # Reset alignment state
            self.reset_alignment_state()
            
            # Reset sliders to default values
            self.brightness_slider.set(1.0)
            self.contrast_slider.set(1.0)
            self.sharpness_slider.set(1.0)
            self.color_slider.set(1.0)
            
            # Reset enhancement sliders
            for attr in dir(self):
                if attr.endswith("_slider") and attr != "brightness_slider" and \
                   attr != "contrast_slider" and attr != "sharpness_slider" and attr != "color_slider":
                    slider = getattr(self, attr)
                    if hasattr(slider, 'set'):
                        # Set default based on original defaults
                        if "edge_smoothing" in attr or "white_balance" in attr:
                            slider.set(0.5)
                            if hasattr(slider, 'value_label'):
                                slider.value_label.configure(text="50%")
                        else:
                            slider.set(0.0)
                            if hasattr(slider, 'value_label'):
                                slider.value_label.configure(text="0%")

            # Reset ROI state
            if self.roi_rect_id:
                self.preview_canvas.delete(self.roi_rect_id)
                self.roi_rect_id = None
                self.roi_coords = None
            self.is_selecting_roi = False
            
            # Reset UI elements
            self.generate_btn.configure(state="disabled")
            self.save_btn.configure(state="disabled")
            self.alignment_frame.pack_forget()
            
            # Show the original input image preview again
            self.show_preview(self.input_path)
            
            messagebox.showinfo("Reset Complete", "All changes have been reset. You can now process the image again.")

    def initial_process(self):
        if not self.input_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return

        try:
            # Prepare the source image
            source_img = Image.open(self.input_path)
            
            # Apply ROI crop if selected
            if self.roi_coords:
                x1, y1, x2, y2 = self.roi_coords
                
                # Map canvas coordinates to original image coordinates
                orig_x1 = int((x1 - self.preview_offset_x) * self.preview_scale)
                orig_y1 = int((y1 - self.preview_offset_y) * self.preview_scale)
                orig_x2 = int((x2 - self.preview_offset_x) * self.preview_scale)
                orig_y2 = int((y2 - self.preview_offset_y) * self.preview_scale)
                
                # Clamp coordinates
                orig_x1 = max(0, min(source_img.width, orig_x1))
                orig_y1 = max(0, min(source_img.height, orig_y1))
                orig_x2 = max(0, min(source_img.width, orig_x2))
                orig_y2 = max(0, min(source_img.height, orig_y2))
                
                if orig_x2 - orig_x1 > 10 and orig_y2 - orig_y1 > 10:
                    source_img = source_img.crop((orig_x1, orig_y1, orig_x2, orig_y2))

            # Process the image with larger size for alignment flexibility
            # We'll process at a larger size to allow for zooming and cropping
            self.original_processed_img = self.processor.process_image(
                source_img, 
                output_size=(620, 796),  # 1.5x passport size for alignment flexibility
                remove_bg=self.remove_bg_var.get()
            )
            
            # Clear ROI rectangle if it exists
            if self.roi_rect_id:
                self.preview_canvas.delete(self.roi_rect_id)
                self.roi_rect_id = None
                self.roi_coords = None
                
            # Show alignment controls
            self.alignment_frame.pack(fill="x", pady=5, after=self.process_btn)
            self.sidebar.update()
            
            # Reset alignment state
            self.reset_alignment_state()
            
            # Show the alignment preview
            self.show_aligned_preview()
            
            messagebox.showinfo("Alignment Mode", "Use zoom controls or mouse wheel to zoom. Drag the image to reposition. Click 'Apply Alignment' when satisfied.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process: {str(e)}")

    def choose_bg_color(self):
        color = colorchooser.askcolor(title="Select Background Color", initialcolor=self.bg_color)
        if color[1]:
            self.set_preset_color(color[1])

    def set_preset_color(self, color):
        self.bg_color = color
        self.color_preview.configure(fg_color=self.bg_color, hover_color=self.bg_color)
        
        # If we are in alignment mode, update the alignment preview instead
        if self.original_processed_img and not self.base_passport_img:
            self.show_aligned_preview()
        elif self.base_passport_img:
            self.update_enhancements()

    def update_enhancements(self, _=None):
        if self.base_passport_img:
            # Apply realistic studio background
            if self.base_passport_img.mode == 'RGBA':
                final_img = self.processor.apply_studio_background(self.base_passport_img, self.bg_color)
            else:
                final_img = self.base_passport_img.copy()

            self.enhanced_img = self.processor.enhance_image(
                final_img,
                brightness=self.brightness_slider.get(),
                contrast=self.contrast_slider.get(),
                sharpness=self.sharpness_slider.get(),
                color=self.color_slider.get(),
                edge_smoothing=self.edge_smoothing_slider.get(),
                skin_smoothing=self.skin_smoothing_slider.get(),
                redeye_removal=self.red_eye_removal_slider.get(),
                teeth_whitening=self.teeth_whitening_slider.get(),
                eye_brightening=self.eye_brightening_slider.get(),
                white_balance=self.auto_white_balance_slider.get(),
                shadow_highlight=self.shadow_highlight_fix_slider.get(),
                vignette_removal=self.vignette_removal_slider.get()
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
