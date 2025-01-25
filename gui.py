import tkinter as tk
from tkinter import ttk, PanedWindow, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import random
from main import MainSystem, SS1, SS2, SS3, SS4
import networkx as nx 
import time 

class SubsystemVisualizer:
    def __init__(self, master):
        self.master = master
        self.main_system = None
        self.thread_ids = {
            "MainSystem": random.randint(1000, 9999),
            "SS1": {"manager": random.randint(1000, 9999), "cores": []},
            "SS2": {"manager": random.randint(1000, 9999), "cores": []},
            "SS3": {"manager": random.randint(1000, 9999), "cores": []},
            "SS4": {"manager": random.randint(1000, 9999), "cores": []}
        }
        self.setup_gui()
        self.update_interval = 1000  # 1 second
        self.running = False

    
    def setup_gui(self):
        self.master.title("Vehicle Electrical System Visualizer")
        self.master.geometry("1400x900")

        # Main horizontal paned window
        main_paned = PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # Left vertical panel (1/5 width)
        left_paned = PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(left_paned)

        # Control buttons
        btn_frame = ttk.Frame(left_paned)
        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.start_simulation)  # Store reference
        self.start_btn.pack(side=tk.LEFT, padx=2)  # Now using self.start_btn
        ttk.Button(btn_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=2)
        left_paned.add(btn_frame)

        # File buttons
        file_btn_frame = ttk.Frame(left_paned)
        ttk.Button(file_btn_frame, text="Load Input", command=self.load_text_input).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_btn_frame, text="Attach .txt File", command=self.load_file).pack(side=tk.LEFT, padx=2)
        self.file_label = ttk.Label(file_btn_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=5)
        left_paned.add(file_btn_frame)

        # Input text box
        input_frame = ttk.Frame(left_paned)
        self.text_input = tk.Text(input_frame, height=10, wrap=tk.WORD)
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_paned.add(input_frame)

        # System Hierarchy Tree
        tree_frame = ttk.Frame(left_paned)
        self.tree = ttk.Treeview(tree_frame)
        self.tree.pack(fill=tk.BOTH, expand=True)
        left_paned.add(tree_frame)

        # Right vertical panel (4/5 width)
        right_paned = PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(right_paned)

        # Subsystem Notebook
        self.notebook = ttk.Notebook(right_paned)
        self.subsystem_tabs = []
        for i in range(4):
            tab = ttk.Frame(self.notebook)
            self.subsystem_tabs.append(tab)
            self.notebook.add(tab, text=f"Subsystem {i+1}")
        right_paned.add(self.notebook)

        # Graphs panel
        graph_paned = PanedWindow(right_paned, orient=tk.HORIZONTAL)
        
        # Resource Allocation Graph
        alloc_frame = ttk.Frame(graph_paned)
        self.alloc_fig = plt.figure(figsize=(6, 4))
        self.alloc_ax = self.alloc_fig.add_subplot(111)
        self.alloc_canvas = FigureCanvasTkAgg(self.alloc_fig, master=alloc_frame)
        self.alloc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        graph_paned.add(alloc_frame)

        # Resource Usage Graph
        usage_frame = ttk.Frame(graph_paned)
        self.usage_fig = plt.figure(figsize=(6, 4))
        self.usage_ax = self.usage_fig.add_subplot(111)
        self.usage_canvas = FigureCanvasTkAgg(self.usage_fig, master=usage_frame)
        self.usage_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        graph_paned.add(usage_frame)

        right_paned.add(graph_paned)

        # Initialize subsystem tabs (keep original tab content)
        for i in range(4):
            self.create_subsystem_tab(i)

    def create_subsystem_tab(self, index):
        tab = self.subsystem_tabs[index]
        
        # Store references to dynamic elements
        tab.core_labels = []
        tab.queue_references = {}
        
        # Thread IDs
        id_frame = ttk.Frame(tab)
        id_frame.pack(fill=tk.X, padx=5, pady=5)
        tab.manager_id = ttk.Label(id_frame, text=f"Managing Thread ID: waiting...")
        tab.manager_id.pack(side=tk.LEFT)
        
        # Resources
        res_frame = ttk.Frame(tab)
        res_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(res_frame, text="Resources:").pack(side=tk.LEFT)
        tab.r1_label = ttk.Label(res_frame, text="R1: -/-")  # Store reference
        tab.r1_label.pack(side=tk.LEFT, padx=10)
        tab.r2_label = ttk.Label(res_frame, text="R2: -/-")  # Store reference
        tab.r2_label.pack(side=tk.LEFT, padx=10)
        
        # Cores
        core_frame = ttk.Frame(tab)
        core_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        for i in range(3 if index == 0 else 2 if index == 1 else 1 if index == 2 else 2):
            frame = ttk.Frame(core_frame)
            frame.pack(fill=tk.X, pady=2)
            
            # Add thread ID label
            id_label = ttk.Label(frame, width=10)
            id_label.pack(side=tk.LEFT)
            
            # Store core status labels
            status_label = ttk.Label(frame, text="Idle", width=20)
            queue_label = ttk.Label(frame, text="Queue: 0")
            
            ttk.Label(frame, text=f"Core {i+1}:").pack(side=tk.LEFT)
            id_label.pack(side=tk.LEFT)
            status_label.pack(side=tk.LEFT)
            queue_label.pack(side=tk.LEFT)
            
            tab.core_labels.append({
                "id_label": id_label,  # Add this line
                "status": status_label,
                "queue": queue_label
        })

         # Queues
        queue_frame = ttk.Frame(tab)
        queue_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Pending Queue
        pending_frame = ttk.Frame(queue_frame)
        pending_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(pending_frame, text="Pending Queue").pack(anchor='w')
        tab.pending_queue = tk.Listbox(pending_frame, width=20, fg="gray")
        tab.pending_queue.pack(fill=tk.BOTH, expand=True)

        # Ready Queue
        ready_frame = ttk.Frame(queue_frame)
        ready_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(ready_frame, text="Ready Queue").pack(anchor='w')
        tab.ready_queue = tk.Listbox(ready_frame, width=20)
        tab.ready_queue.pack(fill=tk.BOTH, expand=True)

        # Waiting Queue
        waiting_frame = ttk.Frame(queue_frame)
        waiting_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(waiting_frame, text="Waiting Queue").pack(anchor='w')
        tab.waiting_queue = tk.Listbox(waiting_frame, width=20)
        tab.waiting_queue.pack(fill=tk.BOTH, expand=True)

        # Completed Tasks
        completed_frame = ttk.Frame(queue_frame)
        completed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(completed_frame, text="Completed Tasks").pack(anchor='w')
        tab.completed_queue = tk.Listbox(completed_frame, width=20, fg="green")
        tab.completed_queue.pack(fill=tk.BOTH, expand=True)

        tab.queue_references = {
            "pending": tab.pending_queue,  
            "ready": tab.ready_queue,
            "waiting": tab.waiting_queue,
            "completed": tab.completed_queue
        }

    def update_allocation_graph(self):
        try:
            if not self.main_system:
                return

            self.alloc_ax.clear()
            G = nx.DiGraph()
            
            # Add resource nodes
            resource_nodes = ['R1', 'R2']
            G.add_nodes_from(resource_nodes, bipartite=0, shape='s')
            
            # Collect tasks and relationships
            task_nodes = []
            assignments = []
            requests = []

            for ss in self.main_system.subsystems:
                # Running tasks
                for core in ss.cores:
                    if core.current_task:
                        task = core.current_task
                        task_id = f"{task.name}\n(SS{self.main_system.subsystems.index(ss)+1})"
                        task_nodes.append(task_id)
                        
                        if task.r1 > 0:
                            assignments.append(('R1', task_id))
                        if task.r2 > 0:
                            assignments.append(('R2', task_id))

                # Waiting tasks
                for task in ss.waiting_queue:
                    task_id = f"{task.name}\n(SS{self.main_system.subsystems.index(ss)+1})"
                    task_nodes.append(task_id)
                    
                    needed_r1 = max(0, task.r1 - ss.available_r1)
                    needed_r2 = max(0, task.r2 - ss.available_r2)
                    
                    if needed_r1 > 0:
                        requests.append((task_id, 'R1'))
                    if needed_r2 > 0:
                        requests.append((task_id, 'R2'))

            # Add nodes and edges
            G.add_nodes_from(task_nodes, bipartite=1, shape='o')
            G.add_edges_from(assignments + requests)

            # Create layout and draw
            pos = nx.bipartite_layout(G, resource_nodes, align='horizontal')
            
            # Draw nodes (using alloc_ax)
            nx.draw_networkx_nodes(G, pos, nodelist=resource_nodes, 
                                node_shape='s', node_size=1500, 
                                node_color='#FFAAAA', edgecolors='red', ax=self.alloc_ax)
            nx.draw_networkx_nodes(G, pos, nodelist=task_nodes,
                                node_shape='o', node_size=1500,
                                node_color='#AAFFAA', edgecolors='green', ax=self.alloc_ax)
            
            # Draw edges (using alloc_ax)
            nx.draw_networkx_edges(G, pos, edgelist=assignments,
                                edge_color='blue', arrowstyle='->', 
                                arrowsize=15, width=1.5, ax=self.alloc_ax)
            nx.draw_networkx_edges(G, pos, edgelist=requests,
                                edge_color='orange', arrowstyle='->',
                                arrowsize=15, width=1.5, style='dashed', ax=self.alloc_ax)
            
            # Draw labels (using alloc_ax)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=self.alloc_ax)
            
            # Add legend (using alloc_ax)
            self.alloc_ax.legend([plt.Line2D([0], [0], color='blue', lw=1.5),
                                plt.Line2D([0], [0], color='orange', lw=1.5, linestyle='--')],
                            ['Resource Held', 'Resource Requested'],
                            loc='upper right')
            
            self.alloc_canvas.draw()

        except Exception as e:
            print(f"Graph update error: {str(e)}")

    def update_usage_graph(self):
        try:
            if not self.main_system:
                return

            self.usage_ax.clear()
            
            # Get synchronized copy of data
            with self.main_system.history_lock:
                history_len = len(self.main_system.resource_history)
                time_steps = range(history_len)
                history_data = list(self.main_system.resource_history)  # Copy while locked
            
            # Plot data
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for ss_idx, ss in enumerate(self.main_system.subsystems):
                total_r1 = ss.r1
                used_r1 = [total_r1 - entry[ss_idx][0] for entry in history_data]
                self.usage_ax.plot(time_steps, used_r1, color=colors[ss_idx], label=f'SS{ss_idx+1} R1')
                
                total_r2 = ss.r2
                used_r2 = [total_r2 - entry[ss_idx][1] for entry in history_data]
                self.usage_ax.plot(time_steps, used_r2, color=colors[ss_idx], linestyle='--', label=f'SS{ss_idx+1} R2')
            
            self.usage_ax.set_title('Resource Usage Over Time')
            self.usage_ax.set_xlabel('Time Steps')
            self.usage_ax.set_ylabel('Resources Used')
            self.usage_ax.legend()
            self.usage_ax.grid(True, alpha=0.3)
            
            self.usage_canvas.draw()

        except Exception as e:
            print(f"Usage graph error: {str(e)}")

    def load_text_input(self):
        input_text = self.text_input.get("1.0", tk.END).strip()
        if input_text:
            input_lines = input_text.split('\n')
            self.initialize_system(input_lines)
        else:
            messagebox.showwarning("Input Error", "Please enter input in the text area or attach a file")

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    input_lines = f.read().splitlines()
                self.file_label.config(text=file_path)
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert(tk.END, '\n'.join(input_lines))
                self.initialize_system(input_lines)
            except Exception as e:
                messagebox.showerror("Error", f"Error loading file: {str(e)}")

    def initialize_system(self, input_lines):
        try:
            self.main_system = MainSystem()
            self.main_system.parse_input(input_lines)
            time.sleep(0.1)  # Give threads time to initialize
            self.main_system.start()  # This will start all threads in proper order
            self.main_system.resource_history = []
            self.populate_tree()
            self.update_gui()
            messagebox.showinfo("Success", "Input parsed successfully!")
        except Exception as e:
            messagebox.showerror("Parse Error", f"Error parsing input: {str(e)}")

    def populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        try:
            # Wait briefly for thread initialization
            self.master.after(100, self._populate_tree_content)
            
        except AttributeError as e:
            print(f"Thread ID Error: {str(e)}")

    def _populate_tree_content(self):
        try:
            # Wait longer for thread initialization
            time.sleep(0.5)  # Give more time for threads to start
            
            main_id = self.main_system.native_id if self.main_system.is_alive() else 'N/A'
            main_item = self.tree.insert("", "end", text=f"Main System (Thread {main_id})")
            
            for i, ss in enumerate(self.main_system.subsystems):
                ss_id = ss.native_id if ss.is_alive() else 'N/A'
                ss_text = f"Subsystem {i+1} ({type(ss).__name__}, Thread {ss_id})"
                ss_node = self.tree.insert(main_item, "end", text=ss_text)
                
                # Wait for core threads to be alive
                for j, core in enumerate(ss.cores):
                    retry = 0
                    while not core.is_alive() and retry < 5:
                        time.sleep(0.1)
                        retry += 1
                    core_id = core.native_id if core.is_alive() else 'N/A'
                    core_text = f"Core {j+1} (Thread {core_id})"
                    self.tree.insert(ss_node, "end", text=core_text)
        except Exception as e:
            print(f"Population error: {str(e)}")

    def start_simulation(self):
        if self.main_system and not self.running:
            self.running = True
            self.start_btn.config(text="Pause", command=self.pause_simulation)
            self.main_system.pause_event.set()  # Resume/start simulation
            self.run_simulation()

    def pause_simulation(self):
        self.running = False
        self.start_btn.config(text="Resume", command=self.start_simulation)
        self.main_system.pause_event.clear()  # Pause simulation

    def reset(self):
        try:
            self.running = False
            if self.main_system:
                # Stop main system first
                self.main_system.pause_event.clear()
                
                # Stop all subsystems and cores
                for ss in self.main_system.subsystems:
                    ss.running = False
                    for core in ss.cores:
                        core.should_run = False
                        core.processing_event.set()  # Wake up any waiting threads
                        core.task_completed.set()    # Wake up any waiting threads
                    ss.process_event.set()  # Wake up subsystem thread
                    ss.cycle_complete.set() # Wake up main thread
                
                # Clear main system events
                self.main_system.stop()
                
                # Wait briefly for threads to stop
                time.sleep(0.2)
                
            # Reset GUI elements
            self.main_system = None
            self.tree.delete(*self.tree.get_children())
            
            # Clear and recreate subsystem tabs
            for tab in self.subsystem_tabs:
                for widget in tab.winfo_children():
                    widget.destroy()
                self.create_subsystem_tab(self.subsystem_tabs.index(tab))
            
            # Reset start button
            self.start_btn.config(text="Start", command=self.start_simulation)
            
            # Clear graphs
            self.alloc_ax.clear()
            self.usage_ax.clear()
            self.alloc_canvas.draw()
            self.usage_canvas.draw()
            
        except Exception as e:
            print(f"Reset error: {str(e)}")

    def run_simulation(self):
        if self.running and self.main_system:
            self.update_gui()
            self.master.after(100, self.run_simulation)

    def update_resource_graphs(self):
        """Update both resource allocation and usage graphs"""
        self.update_allocation_graph()
        self.update_usage_graph()

    def update_gui(self):
        if self.main_system:
            for i, ss in enumerate(self.main_system.subsystems):
                tab = self.subsystem_tabs[i]
                
                # Update thread ID
                if ss.is_alive():
                    tab.manager_id.config(text=f"Managing Thread ID: {ss.native_id}")
                
                # Update resource labels
                tab.r1_label.config(text=f"R1: {ss.available_r1}/{ss.r1}")
                tab.r2_label.config(text=f"R2: {ss.available_r2}/{ss.r2}")
                
                # Update core status and thread IDs
                for j, core in enumerate(ss.cores):
                    if j < len(tab.core_labels):
                        thread_id = core.native_id if core.is_alive() else 'N/A'
                        status_text = f"Status: {core.status}"
                        queue_size = len(core.ready_queue)
                        
                        tab.core_labels[j]["id_label"].config(text=f"TID: {thread_id}")
                        tab.core_labels[j]["status"].config(
                            text=status_text,
                            foreground="green" if core.status == 'running' 
                                      else "red" if core.status == 'error' 
                                      else "orange" if core.status == 'idle'
                                      else "black"
                        )
                        tab.core_labels[j]["queue"].config(text=f"Queue: {queue_size}")
                
                # Update queues
                self._update_queue(tab.queue_references["pending"], ss.pending_tasks, 
                                lambda t: f"{t.name} (Entry: {t.entry_time})")
                
                self._update_queue(tab.queue_references["waiting"], ss.waiting_queue, 
                                lambda t: f"{t.name} (Remaining: {t.remaining_time})")
                
                self._update_queue(tab.queue_references["completed"], ss.completed_tasks,
                                lambda t: f"{t.name} ({t.start_time}-{t.end_time})")
                
                # Update ready queue based on subsystem type
                if isinstance(ss, SS1):
                    ready_tasks = []
                    for core in ss.cores:
                        ready_tasks.extend(core.ready_queue)
                else:
                    ready_tasks = ss.ready_queue
                    
                self._update_queue(tab.queue_references["ready"], ready_tasks,
                                lambda t: f"{t.name} (Remaining: {t.remaining_time})")
                
            self.update_resource_graphs()

    def _update_queue(self, listbox, tasks, format_func):
        listbox.delete(0, tk.END)
        for task in tasks:
            text = format_func(task)
            if hasattr(task, 'failed') and task.failed:
                listbox.insert(tk.END, f"{text} - FAILED: {task.failure_reason}")
                # Set text color to red for failed tasks
                listbox.itemconfig(tk.END, {'fg': 'red'})
            else:
                listbox.insert(tk.END, text)
  

# Adjust the run_gui function definition in gui.py to accept input_lines
def run_gui(input_lines=None):
    root = tk.Tk()
    visualizer = SubsystemVisualizer(root)
    if input_lines:
        visualizer.initialize_system(input_lines)
    root.mainloop()

if __name__ == '__main__':
    run_gui()