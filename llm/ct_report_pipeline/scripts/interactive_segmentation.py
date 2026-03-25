"""
Interactive MedSAM2 Segmentation UI

An interactive GUI application that allows doctors to:
1. Load and browse CT scans (slice by slice)
2. Click on lesions to mark point prompts
3. Use MedSAM2 to segment the lesion based on clicks
4. View and save segmentation results

Usage:
    python interactive_segmentation.py --ct_path <path_to_ct.mhd>
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    load_config, 
    get_medsam2_root, 
    get_medsam2_checkpoint,
    get_llm_config,
    get_ct_window,
)


class InteractiveSegmentationApp:
    """
    Interactive GUI for MedSAM2 point-click segmentation.
    """
    
    def __init__(self, root, ct_path=None, checkpoint_path=None):
        self.root = root
        self.root.title("Interactive MedSAM2 Segmentation")
        self.root.geometry("1200x800")
        
        # Configuration
        self.config = load_config()
        
        # MedSAM2 paths from config
        try:
            self.checkpoint_path = checkpoint_path or str(get_medsam2_checkpoint(self.config))
        except FileNotFoundError:
            self.checkpoint_path = checkpoint_path or ''
        self.medsam2_root = str(get_medsam2_root(self.config))
        
        # Model names from config
        self.seg_model_name = "MEDSAM2"
        llm_config = get_llm_config(self.config)
        self.llm_model_name = llm_config.get('model_name', '')
        # Extract short name from full model path
        self.llm_model_short = self.llm_model_name.split('/')[-1] if '/' in self.llm_model_name else self.llm_model_name
        
        # State variables
        self.ct_volume = None
        self.affine = None
        self.current_slice = 0
        self.total_slices = 0
        
        # Multi-nodule tracking: each foreground point = one nodule
        # Each nodule has: {'id': int, 'prompts': [(x, y, z, label), ...]}
        self.nodule_prompts = []  # List of nodule dicts
        self.current_nodule_id = 0  # Counter for nodule IDs
        self.nodule_masks = []  # List of masks, one per nodule
        self.current_mask = None  # Combined display mask
        
        self.segmenter = None
        self.display_scale = 1.0
        
        # Setup UI
        self.setup_ui()
        
        # Load CT if path provided
        if ct_path:
            self.load_ct(ct_path)
    
    def setup_ui(self):
        """Setup the main UI layout."""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls (with scrollbar)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Create canvas for scrolling
        canvas = tk.Canvas(left_panel, width=200, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=canvas.yview)
        control_frame = ttk.LabelFrame(canvas, text="Controls", padding="5")
        
        # Configure canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create window in canvas
        canvas_window = canvas.create_window((0, 0), window=control_frame, anchor=tk.NW)
        
        # Update scroll region when frame size changes
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        control_frame.bind("<Configure>", on_frame_configure)
        
        # Update canvas width when canvas is resized
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Load CT button
        ttk.Button(control_frame, text="?? Load CT Scan", command=self.open_ct_dialog).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Slice navigation
        ttk.Label(control_frame, text="Slice Navigation:").pack(anchor=tk.W)
        
        self.slice_var = tk.StringVar(value="0 / 0")
        ttk.Label(control_frame, textvariable=self.slice_var, font=('Arial', 12, 'bold')).pack()
        
        slice_nav_frame = ttk.Frame(control_frame)
        slice_nav_frame.pack(fill=tk.X, pady=5)
        ttk.Button(slice_nav_frame, text="?", width=3, command=self.prev_slice).pack(side=tk.LEFT)
        ttk.Button(slice_nav_frame, text="??, width=3, command=self.next_slice).pack(side=tk.RIGHT)
        
        self.slice_slider = ttk.Scale(control_frame, from_=0, to=0, orient=tk.HORIZONTAL, 
                                       command=self.on_slider_change)
        self.slice_slider.pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Point prompt controls
        ttk.Label(control_frame, text="Point Prompts:").pack(anchor=tk.W)
        
        self.prompt_mode_var = tk.StringVar(value="foreground")
        ttk.Radiobutton(control_frame, text="? Foreground (Lesion)", 
                        variable=self.prompt_mode_var, value="foreground").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="? Background", 
                        variable=self.prompt_mode_var, value="background").pack(anchor=tk.W)
        
        self.points_listbox = tk.Listbox(control_frame, height=6, width=25)
        self.points_listbox.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="?? Clear Points", command=self.clear_points).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="??Undo Last Point", command=self.undo_last_point).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Segmentation controls
        ttk.Label(control_frame, text="Segmentation:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        # Model name display (from config)
        ttk.Label(control_frame, text=f"Model: {self.seg_model_name}", 
                  font=('Consolas', 8), foreground='gray').pack(anchor=tk.W)
        
        ttk.Button(control_frame, text="? Run Segmentation", 
                   command=self.run_segmentation, style='Accent.TButton').pack(fill=tk.X, pady=5)
        
        self.show_mask_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show Mask Overlay", 
                        variable=self.show_mask_var, command=self.update_display).pack(anchor=tk.W)
        
        # Mask opacity
        ttk.Label(control_frame, text="Mask Opacity:").pack(anchor=tk.W)
        self.opacity_var = tk.DoubleVar(value=0.5)
        ttk.Scale(control_frame, from_=0.1, to=1.0, variable=self.opacity_var,
                  orient=tk.HORIZONTAL, command=lambda _: self.update_display()).pack(fill=tk.X)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Save controls
        ttk.Button(control_frame, text="? Save Mask", command=self.save_mask).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="?? Save Features", command=self.save_features).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Features display
        ttk.Label(control_frame, text="Lesion Features:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        self.features_text = tk.Text(control_frame, height=10, width=25, wrap=tk.WORD, 
                                      state=tk.DISABLED, font=('Consolas', 9))
        self.features_text.pack(fill=tk.X, pady=5)
        
        # Initialize features storage
        self.lesion_features = None
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Report generation controls
        ttk.Label(control_frame, text="Report Generation:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        self.use_llm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Use Llama LLM", 
                        variable=self.use_llm_var).pack(anchor=tk.W)
        
        # Model name display (from config)
        ttk.Label(control_frame, text=f"Model: {self.llm_model_short}", 
                  font=('Consolas', 8), foreground='gray').pack(anchor=tk.W)
        
        ttk.Button(control_frame, text="?? Generate Report", 
                   command=self.generate_report).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="? Save Report", 
                   command=self.save_report).pack(fill=tk.X, pady=2)
        
        # Initialize report storage
        self.current_report = None
        self.report_generator = None
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Load a CT scan to begin.")
        ttk.Label(control_frame, textvariable=self.status_var, 
                  wraplength=180, foreground='gray').pack(anchor=tk.W, pady=10)
        
        # Right panel - Image display
        image_frame = ttk.LabelFrame(main_frame, text="CT Image", padding="5")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Canvas for image display
        self.canvas = tk.Canvas(image_frame, bg='black', cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        # Keyboard shortcuts
        self.root.bind("<Up>", lambda e: self.next_slice())
        self.root.bind("<Down>", lambda e: self.prev_slice())
        self.root.bind("<space>", lambda e: self.run_segmentation())
        self.root.bind("<Delete>", lambda e: self.clear_points())
        self.root.bind("<Control-z>", lambda e: self.undo_last_point())
    
    def open_ct_dialog(self):
        """Open file dialog to select CT scan."""
        filetypes = [
            ("Medical Images", "*.mhd *.nii *.nii.gz"),
            ("MHD files", "*.mhd"),
            ("NIfTI files", "*.nii *.nii.gz"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            self.load_ct(filepath)
    
    def load_ct(self, ct_path):
        """Load CT volume from file."""
        self.status_var.set(f"Loading: {Path(ct_path).name}...")
        self.root.update()
        
        try:
            ct_path = Path(ct_path)
            
            if ct_path.suffix == '.mhd':
                import SimpleITK as sitk
                sitk_img = sitk.ReadImage(str(ct_path))
                self.ct_volume = sitk.GetArrayFromImage(sitk_img)
                
                # SimpleITK GetSpacing returns (x, y, z) order
                sitk_spacing = sitk_img.GetSpacing()
                origin = sitk_img.GetOrigin()
                
                # Store spacing as (x, y, z) - same as SimpleITK
                self.spacing = np.array([sitk_spacing[0], sitk_spacing[1], sitk_spacing[2]])
                
                # Affine for NIfTI compatibility
                self.affine = np.eye(4)
                self.affine[0, 0] = sitk_spacing[0]
                self.affine[1, 1] = sitk_spacing[1]
                self.affine[2, 2] = sitk_spacing[2]
                self.affine[:3, 3] = origin
                
                print(f"[DEBUG] CT loaded: {ct_path.name}")
                print(f"[DEBUG] Shape (D, H, W): {self.ct_volume.shape}")
                print(f"[DEBUG] Spacing (x, y, z): {self.spacing} mm")
            else:
                import nibabel as nib
                nii_img = nib.load(str(ct_path))
                self.ct_volume = nii_img.get_fdata()
                self.affine = nii_img.affine
                # Extract spacing from NIfTI affine
                self.spacing = np.abs([self.affine[0, 0], self.affine[1, 1], self.affine[2, 2]])
                
                print(f"[DEBUG] NIfTI loaded: {ct_path.name}")
                print(f"[DEBUG] Shape: {self.ct_volume.shape}")
                print(f"[DEBUG] Spacing (x, y, z): {self.spacing} mm")
            
            self.total_slices = self.ct_volume.shape[0]
            self.current_slice = self.total_slices // 2
            self.current_mask = None
            self.point_prompts = []
            
            # Update slider
            self.slice_slider.configure(to=self.total_slices - 1)
            self.slice_slider.set(self.current_slice)
            
            self.update_display()
            self.update_points_list()
            
            self.status_var.set(f"Loaded: {ct_path.name}\nShape: {self.ct_volume.shape}\nSpacing: {self.spacing}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CT:\n{e}")
            self.status_var.set("Failed to load CT.")
    
    def normalize_slice(self, ct_slice):
        """Normalize CT slice to 0-255 for display."""
        # Window/Level for lung CT from config
        lung_window = get_ct_window(self.config, 'lung')
        window_center = lung_window.get('center', -600)
        window_width = lung_window.get('width', 1500)
        
        min_val = window_center - window_width // 2
        max_val = window_center + window_width // 2
        
        normalized = np.clip(ct_slice, min_val, max_val)
        normalized = ((normalized - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        return normalized
    
    def update_display(self):
        """Update the canvas display with current slice and overlays."""
        if self.ct_volume is None:
            return
        
        # Get current slice
        ct_slice = self.ct_volume[self.current_slice]
        normalized = self.normalize_slice(ct_slice)
        
        # Convert to RGB
        rgb_image = np.stack([normalized] * 3, axis=-1)
        
        # Overlay mask if available
        if self.current_mask is not None and self.show_mask_var.get():
            mask_slice = self.current_mask[self.current_slice]
            if mask_slice.any():
                opacity = self.opacity_var.get()
                # Green overlay for mask
                mask_overlay = np.zeros_like(rgb_image)
                mask_overlay[mask_slice > 0] = [0, 255, 0]
                rgb_image = (rgb_image * (1 - opacity * mask_slice[:, :, np.newaxis]) + 
                            mask_overlay * opacity).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Scale to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            scale_x = canvas_width / pil_image.width
            scale_y = canvas_height / pil_image.height
            self.display_scale = min(scale_x, scale_y, 2.0)  # Max 2x zoom
            
            new_width = int(pil_image.width * self.display_scale)
            new_height = int(pil_image.height * self.display_scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear and redraw canvas
        self.canvas.delete("all")
        
        # Center image
        x_offset = (canvas_width - pil_image.width) // 2
        y_offset = (canvas_height - pil_image.height) // 2
        self.image_offset = (x_offset, y_offset)
        
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo)
        
        # Draw point prompts for all nodules
        colors = ['#00FF00', '#00FFFF', '#FFFF00', '#FF00FF', '#FFA500', '#00FF7F']
        for nidx, nodule in enumerate(self.nodule_prompts):
            nodule_color = colors[nidx % len(colors)]  # Cycle through colors for different nodules
            for x, y, z, label in nodule['prompts']:
                if z == self.current_slice:
                    # Convert to display coordinates
                    display_x = x_offset + x * self.display_scale
                    display_y = y_offset + y * self.display_scale
                    
                    color = nodule_color if label == 1 else '#FF0000'  # Red for background
                    radius = 5
                    self.canvas.create_oval(
                        display_x - radius, display_y - radius,
                        display_x + radius, display_y + radius,
                        fill=color, outline='white', width=2
                    )
                    # Draw nodule ID label for foreground points
                    if label == 1:
                        self.canvas.create_text(
                            display_x + 10, display_y - 10,
                            text=f"N{nodule['id']}", fill='white', font=('Arial', 8, 'bold')
                        )
        
        # Update slice label
        self.slice_var.set(f"{self.current_slice + 1} / {self.total_slices}")
    
    def on_canvas_click(self, event):
        """Handle mouse click on canvas to add point prompt.
        
        Each foreground click creates a new nodule.
        Background clicks add to the most recent nodule for refinement.
        """
        if self.ct_volume is None:
            return
        
        # Convert display coordinates to image coordinates
        x_offset, y_offset = self.image_offset
        
        image_x = int((event.x - x_offset) / self.display_scale)
        image_y = int((event.y - y_offset) / self.display_scale)
        
        # Validate coordinates
        if (0 <= image_x < self.ct_volume.shape[2] and 
            0 <= image_y < self.ct_volume.shape[1]):
            
            label = 1 if self.prompt_mode_var.get() == "foreground" else 0
            point = (image_x, image_y, self.current_slice, label)
            
            if label == 1:  # Foreground = new nodule
                self.current_nodule_id += 1
                new_nodule = {
                    'id': self.current_nodule_id,
                    'prompts': [point]
                }
                self.nodule_prompts.append(new_nodule)
                self.status_var.set(f"Created Nodule {self.current_nodule_id} at ({image_x}, {image_y}, z={self.current_slice})")
            else:  # Background = add to current nodule
                if self.nodule_prompts:
                    self.nodule_prompts[-1]['prompts'].append(point)
                    self.status_var.set(f"Added background point to Nodule {self.nodule_prompts[-1]['id']}")
                else:
                    self.status_var.set("No nodule to add background point to. Click foreground first.")
                    return
            
            self.update_display()
            self.update_points_list()
    
    def update_points_list(self):
        """Update the points listbox to show nodules."""
        self.points_listbox.delete(0, tk.END)
        for nodule in self.nodule_prompts:
            nid = nodule['id']
            fg_count = sum(1 for p in nodule['prompts'] if p[3] == 1)
            bg_count = sum(1 for p in nodule['prompts'] if p[3] == 0)
            # Get the foreground point location
            fg_point = next((p for p in nodule['prompts'] if p[3] == 1), None)
            if fg_point:
                loc_str = f"z={fg_point[2]}"
            else:
                loc_str = ""
            self.points_listbox.insert(tk.END, f"? Nodule {nid} ({loc_str}) [FG:{fg_count}, BG:{bg_count}]")
    
    def clear_points(self):
        """Clear all nodule prompts."""
        self.nodule_prompts = []
        self.current_nodule_id = 0
        self.nodule_masks = []
        self.current_mask = None
        self.lesion_features = None
        self.update_display()
        self.update_points_list()
        self.update_features_display()
        self.status_var.set("Cleared all nodules.")
    
    def undo_last_point(self):
        """Remove the last added nodule or background point."""
        if self.nodule_prompts:
            last_nodule = self.nodule_prompts[-1]
            if len(last_nodule['prompts']) > 1:
                # Remove last point from current nodule (likely a background point)
                last_nodule['prompts'].pop()
                self.status_var.set(f"Removed last point from Nodule {last_nodule['id']}")
            else:
                # Only one point (the foreground), remove entire nodule
                self.nodule_prompts.pop()
                self.status_var.set(f"Removed Nodule {last_nodule['id']}")
            self.update_display()
            self.update_points_list()
    
    def on_slider_change(self, value):
        """Handle slice slider change."""
        self.current_slice = int(float(value))
        self.update_display()
    
    def prev_slice(self):
        """Go to previous slice."""
        if self.current_slice > 0:
            self.current_slice -= 1
            self.slice_slider.set(self.current_slice)
            self.update_display()
    
    def next_slice(self):
        """Go to next slice."""
        if self.ct_volume is not None and self.current_slice < self.total_slices - 1:
            self.current_slice += 1
            self.slice_slider.set(self.current_slice)
            self.update_display()
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for slice navigation."""
        if event.delta > 0:
            self.next_slice()
        else:
            self.prev_slice()
    
    def on_canvas_resize(self, event):
        """Handle canvas resize."""
        self.update_display()
    
    def load_segmenter(self):
        """Load MedSAM2 segmenter if not already loaded."""
        if self.segmenter is None:
            self.status_var.set("Loading MedSAM2 model...")
            self.root.update()
            
            import os
            
            # Add MedSAM2 to path BEFORE any imports
            medsam2_path = self.medsam2_root
            if medsam2_path not in sys.path:
                sys.path.insert(0, medsam2_path)
            
            # Change to MedSAM2 directory for proper module resolution
            original_cwd = os.getcwd()
            os.chdir(medsam2_path)
            
            try:
                from segmentation import MedSAM2Segmenter
                
                self.segmenter = MedSAM2Segmenter(
                    checkpoint_path=self.checkpoint_path,
                    medsam2_root=self.medsam2_root
                )
                self.segmenter.load_model()
                self.status_var.set("MedSAM2 model loaded.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load MedSAM2:\n{e}")
                self.status_var.set("Failed to load MedSAM2.")
                return False
            finally:
                os.chdir(original_cwd)
        return True
    
    def run_segmentation(self):
        """Run MedSAM2 segmentation for each nodule separately."""
        if self.ct_volume is None:
            messagebox.showwarning("Warning", "Please load a CT scan first.")
            return
        
        if not self.nodule_prompts:
            messagebox.showwarning("Warning", "Please add at least one nodule (click foreground point).")
            return
        
        # Load segmenter
        if not self.load_segmenter():
            return
        
        self.status_var.set("Running segmentation...")
        self.root.update()
        
        try:
            # Clear previous masks
            self.nodule_masks = []
            
            # Process each nodule separately
            for i, nodule in enumerate(self.nodule_prompts):
                self.status_var.set(f"Segmenting Nodule {nodule['id']} ({i+1}/{len(self.nodule_prompts)})...")
                self.root.update()
                
                # Convert nodule's prompts to MedSAM2 format
                prompts = []
                for x, y, z, label in nodule['prompts']:
                    prompts.append({
                        'coords': (z, y, x),  # (z, y, x) format
                        'label': label
                    })
                
                # Run segmentation for this nodule
                masks = self.segmenter.segment_from_points(
                    self.ct_volume,
                    prompts
                )
                
                if masks:
                    # Combine masks for this nodule (in case of multiple)
                    nodule_mask = np.zeros_like(self.ct_volume, dtype=np.uint8)
                    for mask in masks:
                        nodule_mask = np.maximum(nodule_mask, mask)
                    self.nodule_masks.append({
                        'id': nodule['id'],
                        'mask': nodule_mask
                    })
                else:
                    # Empty mask for this nodule
                    self.nodule_masks.append({
                        'id': nodule['id'],
                        'mask': np.zeros_like(self.ct_volume, dtype=np.uint8)
                    })
            
            # Create combined display mask
            if self.nodule_masks:
                self.current_mask = np.zeros_like(self.ct_volume, dtype=np.uint8)
                for nm in self.nodule_masks:
                    self.current_mask = np.maximum(self.current_mask, nm['mask'])
                
                # Count total segmented voxels
                total_voxels = sum(np.sum(nm['mask'] > 0) for nm in self.nodule_masks)
                self.status_var.set(f"Segmentation complete!\n{len(self.nodule_masks)} nodules, {total_voxels:,} total voxels")
                
                # Extract features for all nodules
                self.extract_lesion_features()
            else:
                self.status_var.set("No segmentation produced.")
            
            self.update_display()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Segmentation failed:\n{e}")
            self.status_var.set("Segmentation failed.")
    
    def extract_lesion_features(self):
        """Extract features for each nodule mask separately."""
        if not self.nodule_masks:
            return
        
        try:
            from scipy import ndimage
            from scipy.spatial.distance import cdist
            
            # Use stored spacing (x, y, z) order
            if hasattr(self, 'spacing') and self.spacing is not None:
                spacing_xyz = self.spacing
            elif self.affine is not None:
                spacing_xyz = np.abs([self.affine[0, 0], self.affine[1, 1], self.affine[2, 2]])
            else:
                spacing_xyz = np.array([1.0, 1.0, 1.0])
            
            spacing_x, spacing_y, spacing_z = spacing_xyz
            
            # Extract features for each nodule
            self.lesion_features = []  # List of feature dicts
            
            for nm in self.nodule_masks:
                nodule_id = nm['id']
                mask = nm['mask'] > 0
                
                if np.sum(mask) == 0:
                    continue  # Skip empty masks
                
                features = self._extract_single_nodule_features(
                    mask, nodule_id, spacing_x, spacing_y, spacing_z
                )
                self.lesion_features.append(features)
            
            # Update display
            self.update_features_display()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Feature extraction error: {e}")
    
    def _extract_single_nodule_features(self, mask, nodule_id, spacing_x, spacing_y, spacing_z):
        """Extract features from a single nodule mask."""
        from scipy import ndimage
        from scipy.spatial.distance import cdist
        
        print(f"[DEBUG] Extracting features for Nodule {nodule_id}")
        
        # Basic measurements
        voxel_count = np.sum(mask)
        voxel_volume_mm3 = spacing_x * spacing_y * spacing_z
        volume_mm3 = voxel_count * voxel_volume_mm3
        volume_cm3 = volume_mm3 / 1000
        
        # Bounding box
        coords = np.where(mask)
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
        n_z = z_max - z_min + 1
        n_y = y_max - y_min + 1
        n_x = x_max - x_min + 1
        
        bbox_x_mm = n_x * spacing_x
        bbox_y_mm = n_y * spacing_y
        bbox_z_mm = n_z * spacing_z
        
        # Center of mass
        center = ndimage.center_of_mass(mask)
        center_mm = (center[2] * spacing_x, center[1] * spacing_y, center[0] * spacing_z)
        
        # Equivalent Sphere Diameter
        equivalent_diameter_mm = 2 * (3 * volume_mm3 / (4 * np.pi)) ** (1/3)
        
        # Boundary analysis
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        boundary_coords = np.column_stack(np.where(boundary))
        
        if len(boundary_coords) > 0:
            boundary_mm = np.zeros_like(boundary_coords, dtype=np.float64)
            boundary_mm[:, 0] = boundary_coords[:, 0] * spacing_z
            boundary_mm[:, 1] = boundary_coords[:, 1] * spacing_y
            boundary_mm[:, 2] = boundary_coords[:, 2] * spacing_x
            
            if len(boundary_mm) > 5000:
                indices = np.random.choice(len(boundary_mm), 5000, replace=False)
                sample_mm = boundary_mm[indices]
            else:
                sample_mm = boundary_mm
            
            distances = cdist(sample_mm, sample_mm)
            longest_axis_mm = np.max(distances)
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(3, len(boundary_mm)))
            pca.fit(boundary_mm)
            projected = pca.transform(boundary_mm)
            
            axis1_extent = projected[:, 0].max() - projected[:, 0].min()
            axis2_extent = projected[:, 1].max() - projected[:, 1].min() if projected.shape[1] > 1 else 0
            axis3_extent = projected[:, 2].max() - projected[:, 2].min() if projected.shape[1] > 2 else 0
            
            short_axis_mm = axis2_extent
        else:
            longest_axis_mm = max(bbox_x_mm, bbox_y_mm, bbox_z_mm)
            short_axis_mm = min(bbox_x_mm, bbox_y_mm, bbox_z_mm)
            axis1_extent = bbox_x_mm
            axis2_extent = bbox_y_mm
            axis3_extent = bbox_z_mm
        
        mean_diameter_mm = (axis1_extent + axis2_extent + axis3_extent) / 3
        
        # Intensity statistics
        lesion_values = self.ct_volume[mask]
        mean_hu = np.mean(lesion_values)
        std_hu = np.std(lesion_values)
        min_hu = np.min(lesion_values)
        max_hu = np.max(lesion_values)
        
        print(f"[DEBUG] Nodule {nodule_id}: ESD={equivalent_diameter_mm:.2f}mm, Volume={volume_mm3:.1f}mm糧, Mean HU={mean_hu:.1f}")
        
        return {
            'nodule_id': nodule_id,
            'voxel_count': int(voxel_count),
            'volume_mm3': float(volume_mm3),
            'volume_cm3': float(volume_cm3),
            'longest_axis_mm': float(longest_axis_mm),
            'short_axis_mm': float(short_axis_mm),
            'mean_diameter_mm': float(mean_diameter_mm),
            'equivalent_diameter_mm': float(equivalent_diameter_mm),
            'bbox_x_mm': float(bbox_x_mm),
            'bbox_y_mm': float(bbox_y_mm),
            'bbox_z_mm': float(bbox_z_mm),
            'bbox': {
                'x_min': int(x_min), 'x_max': int(x_max),
                'y_min': int(y_min), 'y_max': int(y_max),
                'z_min': int(z_min), 'z_max': int(z_max)
            },
            'center_mm': {'x': float(center_mm[0]), 'y': float(center_mm[1]), 'z': float(center_mm[2])},
            'mean_hu': float(mean_hu),
            'std_hu': float(std_hu),
            'min_hu': float(min_hu),
            'max_hu': float(max_hu),
            'spacing_mm': [spacing_x, spacing_y, spacing_z]
        }
    
    def update_features_display(self):
        """Update the features text display for multiple nodules."""
        if not self.lesion_features:
            self.features_text.config(state=tk.NORMAL)
            self.features_text.delete(1.0, tk.END)
            self.features_text.insert(tk.END, "No nodules segmented.")
            self.features_text.config(state=tk.DISABLED)
            return
        
        # Build text for all nodules
        lines = [f"?? {len(self.lesion_features)} Nodule(s)\n"]
        lines.append("=" * 25 + "\n")
        
        for f in self.lesion_features:
            nid = f.get('nodule_id', '?')
            lines.append(f"? Nodule {nid}:\n")
            lines.append(f"  ESD: {f['equivalent_diameter_mm']:.2f} mm\n")
            lines.append(f"  Volume: {f['volume_mm3']:.1f} mm糧\n")
            lines.append(f"  Mean HU: {f['mean_hu']:.1f}\n")
            lines.append(f"  Range: [{f['min_hu']:.0f}, {f['max_hu']:.0f}]\n")
            lines.append("\n")
        
        text = "".join(lines)
        
        self.features_text.config(state=tk.NORMAL)
        self.features_text.delete(1.0, tk.END)
        self.features_text.insert(tk.END, text)
        self.features_text.config(state=tk.DISABLED)
    
    def save_features(self):
        """Save lesion features to JSON file."""
        if self.lesion_features is None:
            messagebox.showwarning("Warning", "No features to save. Run segmentation first.")
            return
        
        import json
        
        filetypes = [
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ]
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.lesion_features, f, indent=2)
                self.status_var.set(f"Saved features: {Path(filepath).name}")
                messagebox.showinfo("Success", f"Features saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save features:\n{e}")
    
    def save_mask(self):
        """Save the current mask to file."""
        if self.current_mask is None:
            messagebox.showwarning("Warning", "No mask to save. Run segmentation first.")
            return
        
        filetypes = [
            ("NIfTI files", "*.nii.gz"),
            ("All files", "*.*")
        ]
        filepath = filedialog.asksaveasfilename(
            defaultextension=".nii.gz",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                import nibabel as nib
                nii_img = nib.Nifti1Image(self.current_mask.astype(np.uint8), self.affine)
                nib.save(nii_img, filepath)
                self.status_var.set(f"Saved mask: {Path(filepath).name}")
                messagebox.showinfo("Success", f"Mask saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save mask:\n{e}")
    
    def generate_report(self):
        """Generate CT report using LLM or simple template."""
        if not self.lesion_features:
            messagebox.showwarning("Warning", "No nodule features. Run segmentation first.")
            return
        
        self.status_var.set("Generating report...")
        self.root.update()
        
        try:
            # Import report generator
            from report_generator import get_report_generator
            
            use_llm = self.use_llm_var.get()
            
            if use_llm:
                self.status_var.set("Loading Llama model (may take a while)...")
                self.root.update()
            
            # Get or create report generator
            if self.report_generator is None or (use_llm and not hasattr(self.report_generator, 'model')):
                self.report_generator = get_report_generator(use_llm=use_llm)
                if use_llm and hasattr(self.report_generator, 'load_model'):
                    try:
                        self.report_generator.load_model()
                    except Exception as e:
                        print(f"Failed to load LLM: {e}")
                        self.status_var.set("LLM load failed. Using simple generator.")
                        self.root.update()
                        self.report_generator = get_report_generator(use_llm=False)
            
            # Generate report
            self.current_report = self.report_generator.generate_report(
                lesion_features=self.lesion_features,
                report_id=f"AUTO_{Path(self.ct_path).stem if hasattr(self, 'ct_path') else 'unknown'}",
            )
            
            # Display report in a new window
            self._show_report_window()
            
            self.status_var.set("Report generated!")
            
        except Exception as e:
            print(f"Report generation error: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to generate report:\n{e}")
            self.status_var.set("Report generation failed.")
    
    def _show_report_window(self):
        """Show report in a new window."""
        if self.current_report is None:
            return
        
        # Create new window
        report_window = tk.Toplevel(self.root)
        report_window.title("Generated CT Report")
        report_window.geometry("800x600")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(report_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text report tab
        text_frame = ttk.Frame(notebook)
        notebook.add(text_frame, text="Text Report")
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
        text_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=text_scrollbar.set)
        
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_content = self.current_report.get("text", "") or ""
        text_widget.insert(tk.END, text_content)
        
        # XML report tab
        xml_frame = ttk.Frame(notebook)
        notebook.add(xml_frame, text="XML Report")
        
        xml_widget = tk.Text(xml_frame, wrap=tk.WORD, font=('Consolas', 9))
        xml_scrollbar = ttk.Scrollbar(xml_frame, orient=tk.VERTICAL, command=xml_widget.yview)
        xml_widget.configure(yscrollcommand=xml_scrollbar.set)
        
        xml_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        xml_widget.pack(fill=tk.BOTH, expand=True)
        xml_content = self.current_report.get("xml", "") or ""
        xml_widget.insert(tk.END, xml_content)
        
        # Buttons
        btn_frame = ttk.Frame(report_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="Save Text", 
                   command=lambda: self._save_report_format("txt")).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save XML", 
                   command=lambda: self._save_report_format("xml")).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", 
                   command=report_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _save_report_format(self, format_type: str):
        """Save report in specified format."""
        if self.current_report is None:
            return
        
        if format_type == "txt":
            filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
            ext = ".txt"
            content = self.current_report.get("text", "")
        else:
            filetypes = [("XML files", "*.xml"), ("All files", "*.*")]
            ext = ".xml"
            content = self.current_report.get("xml", "")
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=filetypes,
            initialfile=f"{self.current_report.get('report_id', 'report')}{ext}"
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Success", f"Report saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report:\n{e}")
    
    def save_report(self):
        """Save report (wrapper for UI button)."""
        if self.current_report is None:
            messagebox.showwarning("Warning", "No report to save. Generate a report first.")
            return
        
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select output directory")
        
        if output_dir:
            try:
                if self.report_generator:
                    saved_files = self.report_generator.save_report(
                        self.current_report,
                        output_dir,
                        formats=["txt", "xml", "json"]
                    )
                    files_list = "\n".join([f"- {k}: {v}" for k, v in saved_files.items()])
                    messagebox.showinfo("Success", f"Report saved:\n{files_list}")
                    self.status_var.set("Report saved!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report:\n{e}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive MedSAM2 Segmentation UI"
    )
    parser.add_argument(
        "--ct_path",
        type=str,
        default=None,
        help="Path to CT scan to load on startup"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to MedSAM2 checkpoint"
    )
    
    args = parser.parse_args()
    
    # Create main window
    root = tk.Tk()
    
    # Style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Create app
    app = InteractiveSegmentationApp(
        root,
        ct_path=args.ct_path,
        checkpoint_path=args.checkpoint
    )
    
    # Run
    root.mainloop()


if __name__ == "__main__":
    main()

