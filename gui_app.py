import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import json
from datetime import datetime
import sys

# Import modules
from e_rara_id_fetcher import search_ids_v2
from e_rara_image_downloader import batch_download_with_target_filtering

class EraraImageMatcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("E-Rara Image Matcher - Search & Match Historical Illustrations")
        self.root.geometry("900x800")
        
        # Variables
        self.target_image_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="e_rara_results")
        self.ids_file_path = tk.StringVar()
        
        # Search variables
        self.author = tk.StringVar()
        self.title = tk.StringVar()
        self.publisher = tk.StringVar()
        self.place = tk.StringVar()
        self.from_date = tk.StringVar()
        self.until_date = tk.StringVar()
        self.max_records = tk.StringVar()
        self.use_website_style = tk.BooleanVar(value=True)
        self.truncation = tk.StringVar(value="on")
        
        # Matching variables
        self.matcher = tk.StringVar(value="superpoint-lg")
        self.device = tk.StringVar(value="cpu")
        self.min_matches = tk.StringVar(value="100")
        self.preprocessing = tk.BooleanVar(value=True)
        self.expand_to_pages = tk.BooleanVar(value=True)
        self.max_workers = tk.StringVar(value="5")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Search Configuration
        search_frame = ttk.Frame(notebook)
        notebook.add(search_frame, text="1. Search Configuration")
        self.setup_search_tab(search_frame)
        
        # Tab 2: Image Matching
        matching_frame = ttk.Frame(notebook)
        notebook.add(matching_frame, text="2. Image Matching")
        self.setup_matching_tab(matching_frame)
        
        # # Tab 3: Pipeline (Combined)
        # pipeline_frame = ttk.Frame(notebook)
        # notebook.add(pipeline_frame, text="3. Complete Pipeline")
        # self.setup_pipeline_tab(pipeline_frame)
        
        # Tab 4: Results & Logs
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="3. Results & Logs")
        self.setup_results_tab(results_frame)
        
    def setup_search_tab(self, parent):
        # Title
        title_label = ttk.Label(parent, text="E-Rara Search Configuration", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Main frame with scrollbar
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Search Parameters Frame
        search_params_frame = ttk.LabelFrame(scrollable_frame, text="Search Parameters", padding=10)
        search_params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        row = 0
        
        # Author
        ttk.Label(search_params_frame, text="Author:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(search_params_frame, textvariable=self.author, width=40).grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(search_params_frame, text='e.g., "Goethe"', foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        
        # Title
        ttk.Label(search_params_frame, text="Title:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(search_params_frame, textvariable=self.title, width=40).grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(search_params_frame, text='e.g., "Historia"', foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        
        # Publisher
        ttk.Label(search_params_frame, text="Publisher:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(search_params_frame, textvariable=self.publisher, width=40).grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(search_params_frame, text='e.g., "Schiller"', foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        
        # Place
        ttk.Label(search_params_frame, text="Place:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(search_params_frame, textvariable=self.place, width=40).grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(search_params_frame, text='e.g., "Bern"', foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        
        # Date range
        ttk.Label(search_params_frame, text="From Date:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(search_params_frame, textvariable=self.from_date, width=40).grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(search_params_frame, text='e.g., "1600"', foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        
        ttk.Label(search_params_frame, text="Until Date:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(search_params_frame, textvariable=self.until_date, width=40).grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(search_params_frame, text='e.g., "1650"', foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        
        # Max records
        ttk.Label(search_params_frame, text="Max Records:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(search_params_frame, textvariable=self.max_records, width=40).grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(search_params_frame, text='Leave empty for all results', foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        
        # Search Options Frame
        options_frame = ttk.LabelFrame(scrollable_frame, text="Search Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Checkbutton(options_frame, text="Use Website Style Search (recommended)", 
                       variable=self.use_website_style).pack(anchor=tk.W)
        
        ttk.Label(options_frame, text="Truncation:").pack(anchor=tk.W, pady=(10,0))
        truncation_frame = ttk.Frame(options_frame)
        truncation_frame.pack(anchor=tk.W)
        ttk.Radiobutton(truncation_frame, text="On", variable=self.truncation, value="on").pack(side=tk.LEFT)
        ttk.Radiobutton(truncation_frame, text="Off", variable=self.truncation, value="off").pack(side=tk.LEFT)
        
        # Buttons Frame
        buttons_frame = ttk.Frame(scrollable_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=20)
        ttk.Button(buttons_frame, text="Search IDs", command=self.search_ids).pack(side=tk.LEFT, padx=5)
        # ttk.Button(buttons_frame, text="Load IDs from File", command=self.load_ids_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Save Search Config", command=self.save_search_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Load Search Config", command=self.load_search_config).pack(side=tk.LEFT, padx=5)
        
        # Status Message Frame
        status_msg_frame = ttk.LabelFrame(scrollable_frame, text="Status", padding=10)
        status_msg_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.search_status_var = tk.StringVar()
        self.search_status_var.set("Ready to search...")
        self.search_status_label = ttk.Label(status_msg_frame, textvariable=self.search_status_var, 
                                           foreground="blue", font=("Arial", 10))
        self.search_status_label.pack(anchor=tk.W)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def setup_matching_tab(self, parent):
        # Title
        title_label = ttk.Label(parent, text="Image Matching Configuration", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Target Image Frame
        target_frame = ttk.LabelFrame(parent, text="Target Image", padding=10)
        target_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(target_frame, text="Target Image:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(target_frame, textvariable=self.target_image_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(target_frame, text="Browse", command=self.browse_target_image).grid(row=0, column=2, padx=5)
        
        # Output Directory Frame
        output_frame = ttk.LabelFrame(parent, text="Output Settings", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(output_frame, textvariable=self.output_dir, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).grid(row=0, column=2, padx=5)
        
        # Matching Parameters Frame
        matching_params_frame = ttk.LabelFrame(parent, text="Matching Parameters", padding=10)
        matching_params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        row = 0
        
        # Matcher selection
        ttk.Label(matching_params_frame, text="Matcher:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        matcher_combo = ttk.Combobox(matching_params_frame, textvariable=self.matcher, 
                                values=["superpoint-lg", "sift-nn"], 
                                state="readonly", width=20)
        matcher_combo.grid(row=row, column=1, padx=5, pady=2)
        # Add callback to update min_matches when matcher changes
        matcher_combo.bind('<<ComboboxSelected>>', self.on_matcher_changed)
        ttk.Label(matching_params_frame, text="Recommended: superpoint-lg (CUDA) or sift-nn (CPU)", foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        
        # Device selection
        ttk.Label(matching_params_frame, text="Device:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        device_combo = ttk.Combobox(matching_params_frame, textvariable=self.device, 
                                  values=["cpu", "cuda"], state="readonly", width=20)
        device_combo.grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(matching_params_frame, text="Use CUDA if available", foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        
        # Min matches
        ttk.Label(matching_params_frame, text="Min Matches:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(matching_params_frame, textvariable=self.min_matches, width=20).grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(matching_params_frame, text="Minimum inliers required", foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        
        # Max workers
        ttk.Label(matching_params_frame, text="Max Workers:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(matching_params_frame, textvariable=self.max_workers, width=20).grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(matching_params_frame, text="Parallel processing threads", foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=5)
        row += 1
        
        # Options
        options_frame = ttk.LabelFrame(parent, text="Processing Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Checkbutton(options_frame, text="Apply preprocessing (border removal)", 
                       variable=self.preprocessing).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Expand record IDs to individual pages", 
                       variable=self.expand_to_pages).pack(anchor=tk.W)
        
        # IDs Input Frame
        ids_frame = ttk.LabelFrame(parent, text="Input IDs", padding=10)
        ids_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(ids_frame, text="IDs File:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(ids_frame, textvariable=self.ids_file_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(ids_frame, text="Browse", command=self.browse_ids_file).grid(row=0, column=2, padx=5)
        
        # Buttons
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill=tk.X, padx=10, pady=20)
        
        ttk.Button(buttons_frame, text="Run Image Matching", command=self.run_image_matching).pack(side=tk.LEFT, padx=5)
        # Status Message Frame for Matching
        matching_status_frame = ttk.LabelFrame(parent, text="Matching Status", padding=10)
        matching_status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.matching_status_var = tk.StringVar()
        self.matching_status_var.set("Ready for image matching...")
        self.matching_status_label = ttk.Label(matching_status_frame, textvariable=self.matching_status_var, 
                                             foreground="blue", font=("Arial", 10))
        self.matching_status_label.pack(anchor=tk.W)
        
        # Progress bar for matching
        self.matching_progress = ttk.Progressbar(parent, mode='determinate')
        self.matching_progress.pack(fill=tk.X, padx=10, pady=5)
    
    def on_matcher_changed(self, event=None):
        """Automatically adjust min_matches based on selected matcher"""
        selected_matcher = self.matcher.get()
        
        if selected_matcher == "superpoint-lg":
            self.min_matches.set("100")
            self.update_matching_status("Matcher changed to SuperPoint-LG. Min matches set to 100.", "blue")
        elif selected_matcher == "sift-nn":
            self.min_matches.set("20")
            self.update_matching_status("Matcher changed to SIFT-NN. Min matches set to 20.", "blue")
        
        self.log_message(f"Matcher changed to {selected_matcher}, min_matches automatically set to {self.min_matches.get()}")

    # def setup_pipeline_tab(self, parent):
    #     # Title
    #     title_label = ttk.Label(parent, text="Complete Pipeline: Search → Match", font=("Arial", 14, "bold"))
    #     title_label.pack(pady=10)
        
    #     # Description
    #     desc_label = ttk.Label(parent, text="This will run the complete pipeline: search for IDs → download and match images", 
    #                           wrap=800)
    #     desc_label.pack(pady=5)
        
    #     # Quick Setup Frame
    #     quick_frame = ttk.LabelFrame(parent, text="Quick Setup", padding=10)
    #     quick_frame.pack(fill=tk.X, padx=10, pady=10)
        
    #     ttk.Button(quick_frame, text="Set Example: Bern 1600-1620", 
    #               command=self.set_bern_example).pack(side=tk.LEFT, padx=5)
    #     ttk.Button(quick_frame, text="Set Example: Basel with Title", 
    #               command=self.set_basel_example).pack(side=tk.LEFT, padx=5)
    #     ttk.Button(quick_frame, text="Clear All", 
    #               command=self.clear_all_fields).pack(side=tk.LEFT, padx=5)
        
    #     # Pipeline Status Frame
    #     status_frame = ttk.LabelFrame(parent, text="Pipeline Status", padding=10)
    #     status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    #     self.status_text = scrolledtext.ScrolledText(status_frame, height=15, wrap=tk.WORD)
    #     self.status_text.pack(fill=tk.BOTH, expand=True)
        
    #     # Progress bar
    #     self.progress = ttk.Progressbar(parent, mode='determinate')
    #     self.progress.pack(fill=tk.X, padx=10, pady=5)
        
    #     # Buttons
    #     buttons_frame = ttk.Frame(parent)
    #     buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
    #     ttk.Button(buttons_frame, text="Run Complete Pipeline", 
    #               command=self.run_complete_pipeline, 
    #               style="Accent.TButton").pack(side=tk.LEFT, padx=5)
    #     ttk.Button(buttons_frame, text="Stop", command=self.stop_pipeline).pack(side=tk.LEFT, padx=5)
    #     ttk.Button(buttons_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        
    def setup_results_tab(self, parent):
        # Title
        title_label = ttk.Label(parent, text="Results & Logs", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(parent, height=25, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Buttons
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(buttons_frame, text="Open Output Folder", command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Save Log", command=self.save_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Clear", command=self.clear_results).pack(side=tk.LEFT, padx=5)
        
    # Event handlers
    def browse_target_image(self):
        filename = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if filename:
            self.target_image_path.set(filename)
            self.update_matching_status(f"🖼️ Target image selected: {os.path.basename(filename)}", "green")
            
    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir.set(dirname)
            
    def browse_ids_file(self):
        filename = filedialog.askopenfilename(
            title="Select IDs File",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.ids_file_path.set(filename)
            
    def set_bern_example(self):
        self.place.set("Bern")
        self.from_date.set("1600")
        self.until_date.set("1620")
        self.max_records.set("50")
        self.log_message("Set example: Bern 1600-1620 with max 50 records")
        
    def set_basel_example(self):
        self.place.set("Basel")
        self.title.set("Historia")
        self.from_date.set("1550")
        self.until_date.set("1650")
        self.max_records.set("30")
        self.log_message("Set example: Basel with 'Historia' in title, 1550-1650, max 30 records")
        
    def clear_all_fields(self):
        for var in [self.author, self.title, self.publisher, self.place, 
                   self.from_date, self.until_date, self.max_records]:
            var.set("")
        self.log_message("Cleared all search fields")
        
    def reset_search_status(self):
        """Reset the search status to ready state"""
        self.search_status_var.set("Ready to search...")
        self.search_status_label.config(foreground="blue")
    
    def update_search_status(self, message, color="blue"):
        """Update the search status message with color"""
        self.search_status_var.set(message)
        self.search_status_label.config(foreground=color)
        self.root.update()
        
    def update_matching_status(self, message, color="blue"):
        """Update the matching status message with color"""
        self.matching_status_var.set(message)
        self.matching_status_label.config(foreground=color)
        self.root.update()

    def search_ids(self):
        def run_search():
            self.update_search_status("🔍 Searching for IDs...", "blue")
            self.log_message("Starting ID search...")
            # self.progress.start()
            
            try:
                filters = self.get_search_filters()
                if not any(filters.values()):
                    self.log_message("ERROR: Please set at least one search parameter")
                    self.update_search_status("❌ ERROR: Please set at least one search parameter", "red")
                    return
                
                self.log_message(f"Search parameters: {filters}")
                
                results, total = search_ids_v2(**filters)
                if results:
                    # Save to ids.txt
                    ids_file = "ids.txt"
                    with open(ids_file, 'w') as f:
                        for record_id in results:
                            f.write(f"{record_id}\n")
                    self.ids_file_path.set(os.path.abspath(ids_file))
                    self.log_message(f"SUCCESS: Found {total} record IDs. Returning the first {len(results)}.")
                    self.log_message(f"IDs saved to: {os.path.abspath(ids_file)}")
                    
                    # Update status message with success
                    self.update_search_status(f"✅ Search complete! Found {total} IDs. Returning first {len(results)}. Ready for image matching.", "green")
                    
                    # Show first few results
                    preview = results[:10] if len(results) > 10 else results
                    self.log_message(f"First {len(preview)} IDs: {preview}")
                    if len(results) > 10:
                        self.log_message(f"... and {len(results) - 10} more")
                else:
                    self.log_message("No results found with current search parameters")
                    self.update_search_status("⚠️ No results found. Try adjusting search parameters.", "orange")
                    
            except Exception as e:
                self.log_message(f"ERROR during search: {str(e)}")
                self.search_status_var.set(f"❌ Error during search: {str(e)}")
                self.search_status_label.config(foreground="red")            
           
        
        threading.Thread(target=run_search, daemon=True).start()
        
    def run_image_matching(self):
        if not self.target_image_path.get():
            messagebox.showerror("Error", "Please select a target image")
            self.update_matching_status("❌ Please select a target image", "red")
            return
            
        if not self.ids_file_path.get():
            messagebox.showerror("Error", "Please select or create an IDs file")
            self.update_matching_status("❌ Please select or create an IDs file", "red")
            return
        
        def run_matching():
            self.update_matching_status("🔍 Starting image matching...", "blue")
            self.log_message("Starting image matching...")
            
            try:
                # Load record IDs
                self.update_matching_status("📄 Loading record IDs...", "blue")
                record_ids = self.load_record_ids()
                if not record_ids:
                    self.log_message("ERROR: No record IDs found")
                    self.update_matching_status("❌ ERROR: No record IDs found", "red")
                    return
                
                self.log_message(f"Loaded {len(record_ids)} record IDs")
                self.update_matching_status(f"⚙️ Loading target image...", "blue")
                
                # Validate target image exists
                if not os.path.exists(self.target_image_path.get()):
                    self.log_message("ERROR: Target image not found")
                    self.update_matching_status("❌ ERROR: Target image not found", "red")
                    return
                
                self.update_matching_status(f"⚙️ Initializing {self.matcher.get()} matcher on {self.device.get()}...", "blue")
                self.log_message(f"Using matcher: {self.matcher.get()} on device: {self.device.get()}")
                
                # Create a progress callback function
                def progress_callback(current, total, record_id=""):
                    progress_percent = (current / total) * 100
                    # Update progress bar
                    self.matching_progress['maximum'] = total
                    self.matching_progress['value'] = current
                    # Update status message
                    self.update_matching_status(f"🔍 Processing image {current}/{total} ({progress_percent:.1f}%) - Record ID: {record_id}", "blue")
                    # Force GUI update
                    self.root.update()
                
                self.update_matching_status(f"🔍 Starting candidate filtering (min {self.min_matches.get()} matches)...", "blue")
                
                # Run matching with progress callback
                results = batch_download_with_target_filtering(
                    record_ids=record_ids,
                    target_image_path=self.target_image_path.get(),
                    output_dir=self.output_dir.get(),
                    matcher=self.matcher.get(),
                    device=self.device.get(),
                    min_matches=int(self.min_matches.get()),
                    expand_to_pages=self.expand_to_pages.get(),
                    max_workers=int(self.max_workers.get()),
                    preprocessing=self.preprocessing.get(),
                    progress_callback=progress_callback  # Add this parameter
                )
                
                if 'error' in results:
                    self.log_message(f"ERROR: {results['error']}")
                    self.update_matching_status(f"❌ Error: {results['error']}", "red")
                    return
                
                # Display results
                summary = results['summary']
                self.log_message("=== MATCHING RESULTS ===")
                self.log_message(f"Input images: {summary['input_images']}")
                self.log_message(f"Candidates found: {summary['candidates_found']}")
                self.log_message(f"Successfully downloaded: {summary['successfully_downloaded']}")
                
                # Update status based on results
                if summary['successfully_downloaded'] > 0:
                    self.update_matching_status(
                        f"✅ Matching complete! Found {summary['candidates_found']} candidates, "
                        f"downloaded {summary['successfully_downloaded']} matches.", 
                        "green"
                    )
                    self.log_message("\nBest matches:")
                    for i, file_info in enumerate(results['download']['downloaded_files'][:5]):
                        self.log_message(f"{i+1}. {file_info['record_id']} ({file_info['good_matches']} matches)")
                else:
                    self.update_matching_status("⚠️ Matching complete, but no candidates found. Try lowering the threshold.", "orange")
                
            except Exception as e:
                self.log_message(f"ERROR during matching: {str(e)}")
                self.update_matching_status(f"❌ Error during matching: {str(e)}", "red")
            finally:
                # Reset progress bar
                self.matching_progress['value'] = 0
        
        threading.Thread(target=run_matching, daemon=True).start()
        
    # def run_complete_pipeline(self):
    #     if not self.target_image_path.get():
    #         messagebox.showerror("Error", "Please select a target image")
    #         return
            
    #     def run_pipeline():
    #         self.log_message("=== STARTING COMPLETE PIPELINE ===")
    #         # self.progress.start()
            
    #         try:
    #             # Step 1: Search for IDs
    #             self.log_message("Step 1: Searching for record IDs...")
    #             filters = self.get_search_filters()
                
    #             if not any(filters.values()):
    #                 self.log_message("ERROR: Please set at least one search parameter")
    #                 return
                
    #             record_ids, total = search_ids_v2(**filters)
                
    #             if not record_ids:
    #                 self.log_message("No record IDs found. Pipeline stopped.")
    #                 return
                
    #             self.log_message(f"Found {total} record IDS. Returning the first {len(record_ids)}.")
                
    #             # Step 2: Run image matching
    #             self.log_message("Step 2: Running image matching...")
                
    #             results = batch_download_with_target_filtering(
    #                 record_ids=record_ids,
    #                 target_image_path=self.target_image_path.get(),
    #                 output_dir=self.output_dir.get(),
    #                 matcher=self.matcher.get(),
    #                 device=self.device.get(),
    #                 min_matches=int(self.min_matches.get()),
    #                 expand_to_pages=self.expand_to_pages.get(),
    #                 max_workers=int(self.max_workers.get()),
    #                 preprocessing=self.preprocessing.get()
    #             )
                
    #             if 'error' in results:
    #                 self.log_message(f"ERROR: {results['error']}")
    #                 return
                
    #             # Display final results
    #             summary = results['summary']
    #             self.log_message("=== PIPELINE COMPLETE ===")
    #             self.log_message(f"Input images processed: {summary['input_images']}")
    #             self.log_message(f"Candidates found: {summary['candidates_found']}")
    #             self.log_message(f"Successfully downloaded: {summary['successfully_downloaded']}")
                
    #             if summary['successfully_downloaded'] > 0:
    #                 self.log_message(f"\nResults saved to: {self.output_dir.get()}")
    #                 self.log_message("Best matches:")
    #                 for i, file_info in enumerate(results['download']['downloaded_files'][:5]):
    #                     self.log_message(f"{i+1}. {file_info['record_id']} ({file_info['good_matches']} matches)")
    #             else:
    #                 self.log_message("No matching images found.")
                
    #         except Exception as e:
    #             self.log_message(f"ERROR in pipeline: {str(e)}")
    #         # finally:
    #         #     self.progress.stop()
        
    #     threading.Thread(target=run_pipeline, daemon=True).start()
        
    def get_search_filters(self):
        filters = {}
        
        if self.author.get():
            filters['author'] = self.author.get()
        if self.title.get():
            filters['title'] = self.title.get()
        if self.publisher.get():
            filters['publisher'] = self.publisher.get()
        if self.place.get():
            filters['place'] = self.place.get()
        if self.from_date.get():
            filters['from_date'] = self.from_date.get()
        if self.until_date.get():
            filters['until_date'] = self.until_date.get()
        if self.max_records.get():
            filters['max_records'] = int(self.max_records.get())
            
        filters['use_website_style'] = self.use_website_style.get()
        filters['truncation'] = self.truncation.get()
        
        return filters
        
    def load_record_ids(self):
        try:
            if self.ids_file_path.get().endswith('.json'):
                with open(self.ids_file_path.get(), 'r') as f:
                    data = json.load(f)
                    return data.get('ids', [])
            else:
                with open(self.ids_file_path.get(), 'r') as f:
                    return [line.strip() for line in f if line.strip()]
        except Exception as e:
            self.log_message(f"Error loading IDs file: {str(e)}")
            return []
    
    # def load_ids_file(self):
    #     self.browse_ids_file()
    #     if self.ids_file_path.get():
    #         record_ids = self.load_record_ids()
    #         self.log_message(f"Loaded {len(record_ids)} record IDs from file")
    #         self.update_matching_status(f"📁 Loaded {len(record_ids)} record IDs from file. Ready to match.", "green")
            
    def save_search_config(self):
        config = {
            'search_filters': self.get_search_filters(),
            'matching_config': {
                'matcher': self.matcher.get(),
                'device': self.device.get(),
                'min_matches': self.min_matches.get(),
                'max_workers': self.max_workers.get(),
                'preprocessing': self.preprocessing.get(),
                'expand_to_pages': self.expand_to_pages.get(),
            },
            'paths': {
                'target_image': self.target_image_path.get(),
                'output_dir': self.output_dir.get(),
            }
        }
        
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            self.log_message(f"Configuration saved to: {filename}")
            
    def load_search_config(self):
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                # Load search filters
                search_filters = config.get('search_filters', {})
                self.author.set(search_filters.get('author', ''))
                self.title.set(search_filters.get('title', ''))
                self.publisher.set(search_filters.get('publisher', ''))
                self.place.set(search_filters.get('place', ''))
                self.from_date.set(search_filters.get('from_date', ''))
                self.until_date.set(search_filters.get('until_date', ''))
                self.max_records.set(str(search_filters.get('max_records', '')))
                self.use_website_style.set(search_filters.get('use_website_style', True))
                self.truncation.set(search_filters.get('truncation', 'on'))
                
                # Load matching config
                matching_config = config.get('matching_config', {})
                self.matcher.set(matching_config.get('matcher', 'superpoint-lg'))
                self.device.set(matching_config.get('device', 'cpu'))
                self.min_matches.set(str(matching_config.get('min_matches', '100')))
                self.max_workers.set(str(matching_config.get('max_workers', '5')))
                self.preprocessing.set(matching_config.get('preprocessing', False))
                self.expand_to_pages.set(matching_config.get('expand_to_pages', True))
                
                # Load paths
                paths = config.get('paths', {})
                self.target_image_path.set(paths.get('target_image', ''))
                self.output_dir.set(paths.get('output_dir', 'e_rara_results'))
                
                self.log_message(f"Configuration loaded from: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
                
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}\n"
        
        # Add to both status text and results text
        if hasattr(self, 'status_text'):
            self.status_text.insert(tk.END, log_msg)
            self.status_text.see(tk.END)
        
        if hasattr(self, 'results_text'):
            self.results_text.insert(tk.END, log_msg)
            self.results_text.see(tk.END)
            
        self.root.update()
        
    def clear_log(self):
        if hasattr(self, 'status_text'):
            self.status_text.delete(1.0, tk.END)
            
    def clear_results(self):
        if hasattr(self, 'results_text'):
            self.results_text.delete(1.0, tk.END)
            
    # def stop_pipeline(self):
    #     self.progress.stop()
    #     self.log_message("Pipeline stopped by user")
        
    def open_output_folder(self):
        output_path = self.output_dir.get()
        if os.path.exists(output_path):
            os.startfile(output_path)  # Windows
        else:
            messagebox.showinfo("Info", f"Output folder does not exist: {output_path}")
            
    def save_log(self):
        if hasattr(self, 'results_text'):
            content = self.results_text.get(1.0, tk.END)
            if content.strip():
                filename = filedialog.asksaveasfilename(
                    title="Save Log",
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
                )
                if filename:
                    with open(filename, 'w') as f:
                        f.write(content)
                    self.log_message(f"Log saved to: {filename}")

def main():
    root = tk.Tk()
    app = EraraImageMatcherGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()