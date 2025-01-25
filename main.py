import sys
from collections import deque
import random
import threading
import time

class Task:
    def __init__(self, name, runtime, r1, r2, entry_time, **kwargs):
        self.name = name
        self.original_runtime = runtime
        self.remaining_time = runtime
        self.r1 = r1
        self.r2 = r2
        self.state = 'waiting'
        self.start_time = None
        self.end_time = None
        self.waiting_time = 0
        self.subsystem = None
        self.entry_time = entry_time 
        self.failed = False
        self.failure_reason = None
        for key, value in kwargs.items():
            setattr(self, key, value)
        

    def __repr__(self):
        return f"{self.name} ({self.remaining_time})"

class Core(threading.Thread):
    def __init__(self, subsystem, core_id):
        super().__init__(daemon=True)
        self.subsystem = subsystem
        self.core_id = core_id
        self.ready_queue = deque()
        self.current_task = None
        self.lock = threading.Lock()
        self.processing_event = threading.Event()
        self.task_completed = threading.Event()  # New event for task completion
        self.should_run = True
        self.status = 'idle'

    def run(self):
        while self.should_run:
            try:
                # Only process if we have a task
                if self.current_task:
                    self.status = 'running'
                    self.processing_event.wait()
                    self.processing_event.clear()
                    self.current_task.remaining_time -= 1
                    self.task_completed.set()  # Signal task processing done
                else:
                    self.status = 'idle'
                    # Signal immediately if idle
                    self.task_completed.set()
                time.sleep(0.1)
            except Exception as e:
                print(f"Core {self.core_id} error: {str(e)}")
            
class Subsystem(threading.Thread):
    def __init__(self, r1, r2, num_cores):
        super().__init__(daemon=True)
        self.r1 = r1
        self.r2 = r2
        self.num_cores = num_cores
        self.cores = [Core(self, i+1) for i in range(num_cores)]
        self.tasks = []
        self.pending_tasks = []
        self.waiting_queue = deque()
        self.available_r1 = r1
        self.available_r2 = r2
        self.completed_tasks = []
        self.resource_lock = threading.Lock()
        self.process_event = threading.Event()
        self.cycle_complete = threading.Event()
        self.main_system = None
        self.running = True  # Added missing attribute
        self.borrowed_r1 = 0  # For SS3
        self.borrowed_r2 = 0  # For SS3
        self.borrowed_resources = []  # For SS3
        self.ready_queue = deque()  # For SS2 and SS4
        self.dependency_map = {}  # For SS4


    def attach_main_system(self, main_system):
        """Attach this subsystem to the main system"""
        self.main_system = main_system
        for core in self.cores:
            core.subsystem = self

    def run(self):
        while self.running:
            try:
                # Wait for signal to process
                self.process_event.wait()
                self.process_event.clear()
                
                # Process current timestep
                self.process(self.main_system.current_time)
                
                # Signal active cores to process
                active_cores = [core for core in self.cores if core.current_task or core.ready_queue]
                for core in active_cores:
                    core.processing_event.set()
                
                # Wait for all cores to complete
                for core in active_cores:
                    core.task_completed.wait()
                    core.task_completed.clear()
                
                # Signal completion to MainSystem
                self.cycle_complete.set()

            except Exception as e:
                print(f"Subsystem error: {str(e)}")

    def add_task(self, task):
        with self.resource_lock:
            self.tasks.append(task)
            task.subsystem = self

    def process(self, current_time):
        for task in list(self.pending_tasks):
            if task.entry_time <= current_time:
                self.waiting_queue.append(task)
                self.pending_tasks.remove(task)
        pass

class SS1(Subsystem):
    def __init__(self, r1, r2):
        super().__init__(r1, r2, 3)
        self.last_core = 0 
        self.task_locations = {}
        self.task_history = []  # Add task history tracking

    def load_balance(self):
        total_tasks = sum(len(core.ready_queue) for core in self.cores)
        avg = total_tasks // 3
        for i in range(3):
            while len(self.cores[i].ready_queue) > avg + 1:
                task = self.cores[i].ready_queue.pop()
                min_core = min(enumerate(self.cores), key=lambda x: len(x[1].ready_queue))[0]
                self.cores[min_core].ready_queue.append(task)
            while len(self.cores[i].ready_queue) < avg:
                max_core = max(enumerate(self.cores), key=lambda x: len(x[1].ready_queue))[0]
                if len(self.cores[max_core].ready_queue) > 0:
                    task = self.cores[max_core].ready_queue.pop()
                    self.cores[i].ready_queue.append(task)

    def process(self, current_time):
        with self.resource_lock:
            # Process pending tasks
            pending_to_remove = []
            for task in list(self.pending_tasks):
                if task.entry_time <= current_time:
                    task_core = (task.initial_core - 1) % len(self.cores)
                    if task.r1 <= self.available_r1 and task.r2 <= self.available_r2:
                        self.available_r1 -= task.r1
                        self.available_r2 -= task.r2
                        task.state = 'ready'
                        self.cores[task_core].ready_queue.append(task)
                        pending_to_remove.append(task)
                    else:
                        self.waiting_queue.append(task)
                        pending_to_remove.append(task)

            for task in pending_to_remove:
                self.pending_tasks.remove(task)

            # Check waiting queue
            waiting_to_ready = []
            for task in list(self.waiting_queue):
                if task.r1 <= self.available_r1 and task.r2 <= self.available_r2:
                    task_core = (task.initial_core - 1) % len(self.cores)
                    self.cores[task_core].ready_queue.append(task)
                    self.available_r1 -= task.r1
                    self.available_r2 -= task.r2
                    task.state = 'ready'
                    waiting_to_ready.append(task)

            for task in waiting_to_ready:
                self.waiting_queue.remove(task)

            # Process cores
            for core in self.cores:
                if core.current_task:
                    core.current_task.remaining_time -= 1
                    if core.current_task.remaining_time <= 0:
                        # Task completed
                        self.available_r1 += core.current_task.r1
                        self.available_r2 += core.current_task.r2
                        core.current_task.end_time = current_time
                        core.current_task.state = 'completed'
                        self.completed_tasks.append(core.current_task)
                        core.current_task = None
                        core.status = 'idle'
                        core.processing_event.clear()
                    elif random.random() < 0.3:  # 30% chance to preempt
                        self.waiting_queue.append(core.current_task)
                        self.available_r1 += core.current_task.r1
                        self.available_r2 += core.current_task.r2
                        core.current_task = None
                        core.status = 'idle'
                        core.processing_event.clear()

                if not core.current_task and core.ready_queue:
                    core.current_task = core.ready_queue.popleft()
                    core.current_task.start_time = current_time
                    core.status = 'running'
                    core.processing_event.set()
class SS2(Subsystem):
    def __init__(self, r1, r2):
        super().__init__(r1, r2, 2)
        self.ready_queue = deque()

    def process(self, current_time):
        with self.resource_lock:
            # Process pending tasks
            pending_to_remove = []
            for task in list(self.pending_tasks):
                if task.entry_time <= current_time:
                    if task.r1 <= self.available_r1 and task.r2 <= self.available_r2:
                        task.state = 'ready'
                        self.ready_queue.append(task)
                        pending_to_remove.append(task)
                    else:
                        self.waiting_queue.append(task)
                        pending_to_remove.append(task)

            for task in pending_to_remove:
                self.pending_tasks.remove(task)

            # Try to move waiting tasks to ready
            waiting_to_ready = []
            for task in list(self.waiting_queue):
                if task.r1 <= self.available_r1 and task.r2 <= self.available_r2:
                    task.state = 'ready'
                    self.ready_queue.append(task)
                    waiting_to_ready.append(task)

            for task in waiting_to_ready:
                self.waiting_queue.remove(task)

            # Process cores using SRTF
            for core in self.cores:
                if core.current_task:
                    # If there's a task with shorter remaining time, preempt
                    shortest_task = min(self.ready_queue, key=lambda t: t.remaining_time) if self.ready_queue else None
                    if shortest_task and shortest_task.remaining_time < core.current_task.remaining_time:
                        self.ready_queue.append(core.current_task)
                        core.current_task = shortest_task
                        self.ready_queue.remove(shortest_task)
                        core.status = 'running'
                        core.processing_event.set()

                    core.current_task.remaining_time -= 1
                    if core.current_task.remaining_time <= 0:
                        self.available_r1 += core.current_task.r1
                        self.available_r2 += core.current_task.r2
                        core.current_task.end_time = current_time
                        core.current_task.state = 'completed'
                        self.completed_tasks.append(core.current_task)
                        core.current_task = None
                        core.status = 'idle'
                        core.processing_event.clear()

                if not core.current_task and self.ready_queue:
                    shortest_task = min(self.ready_queue, key=lambda t: t.remaining_time)
                    if self.available_r1 >= shortest_task.r1 and self.available_r2 >= shortest_task.r2:
                        core.current_task = shortest_task
                        self.ready_queue.remove(shortest_task)
                        self.available_r1 -= shortest_task.r1
                        self.available_r2 -= shortest_task.r2
                        core.current_task.start_time = current_time
                        core.status = 'running'
                        core.processing_event.set()

class SS3(Subsystem):
    def __init__(self, r1, r2):
        super().__init__(r1, r2, 1)
        self.ready_queue = deque()
        self.borrowed_r1 = 0
        self.borrowed_r2 = 0
        self.borrowed_resources = []  # Tracks borrowed resources and deadlines

    def process(self, current_time):
        with self.resource_lock:
            # Return any borrowed resources that are past deadline
            borrowed_to_remove = []
            for borrowed in self.borrowed_resources:
                if current_time >= borrowed['deadline']:
                    for ss, r1, r2 in borrowed['sources']:
                        with ss.resource_lock:
                            ss.available_r1 += r1
                            ss.available_r2 += r2
                            self.borrowed_r1 -= r1
                            self.borrowed_r2 -= r2
                    borrowed_to_remove.append(borrowed)
            for borrowed in borrowed_to_remove:
                self.borrowed_resources.remove(borrowed)

            # Process pending tasks
            pending_to_remove = []
            for task in list(self.pending_tasks):
                if task.entry_time <= current_time:
                    self.waiting_queue.append(task)
                    pending_to_remove.append(task)

            for task in pending_to_remove:
                self.pending_tasks.remove(task)

            # Sort waiting queue by period (Rate Monotonic)
            waiting_list = list(self.waiting_queue)
            waiting_list.sort(key=lambda t: t.period)

            core = self.cores[0]  # Single core system
            if not core.current_task:
                for task in waiting_list:
                    borrowed_r1 = 0  # Initialize borrowing variables
                    borrowed_r2 = 0
                    sources = []
                    
                    total_needed_r1 = max(0, task.r1 - self.available_r1)
                    total_needed_r2 = max(0, task.r2 - self.available_r2)
                    
                    if total_needed_r1 > 0 or total_needed_r2 > 0:
                        # Try to borrow resources
                        for ss in self.main_system.subsystems:
                            if ss != self:
                                with ss.resource_lock:
                                    if borrowed_r1 < total_needed_r1:
                                        borrow_r1 = min(ss.available_r1, total_needed_r1 - borrowed_r1)
                                        if borrow_r1 > 0:
                                            ss.available_r1 -= borrow_r1
                                            borrowed_r1 += borrow_r1
                                            sources.append((ss, borrow_r1, 0))
                                            
                                    if borrowed_r2 < total_needed_r2:
                                        borrow_r2 = min(ss.available_r2, total_needed_r2 - borrowed_r2)
                                        if borrow_r2 > 0:
                                            ss.available_r2 -= borrow_r2
                                            borrowed_r2 += borrow_r2
                                            sources.append((ss, 0, borrow_r2))

                        self.borrowed_r1 += borrowed_r1
                        self.borrowed_r2 += borrowed_r2

                    # Check if we have enough resources (own + borrowed)
                    if (self.available_r1 + self.borrowed_r1 >= task.r1 and 
                        self.available_r2 + self.borrowed_r2 >= task.r2):
                        core.current_task = task
                        self.waiting_queue.remove(task)
                        deadline = task.entry_time + task.period * task.num_repetitions
                        if borrowed_r1 > 0 or borrowed_r2 > 0:
                            self.borrowed_resources.append({
                                'sources': sources,
                                'deadline': deadline,
                                'r1': borrowed_r1,
                                'r2': borrowed_r2
                            })
                        core.status = 'running'
                        core.processing_event.set()
                        break

            if core.current_task:
                core.current_task.remaining_time -= 2  # Double speed
                if core.current_task.remaining_time <= 0:
                    core.current_task.end_time = current_time
                    core.current_task.state = 'completed'
                    self.completed_tasks.append(core.current_task)
                    self.available_r1 += min(core.current_task.r1, self.available_r1)
                    self.available_r2 += min(core.current_task.r2, self.available_r2)
                    core.current_task = None
                    core.status = 'idle'
                    core.processing_event.clear()

class SS4(Subsystem):
    def __init__(self, r1, r2):
        super().__init__(r1, r2, 2)
        self.ready_queue = deque()
        self.dependency_map = {}

    def process(self, current_time):
        with self.resource_lock:
            # Process pending tasks
            pending_to_remove = []
            for task in list(self.pending_tasks):
                if task.entry_time <= current_time:
                    # Check if task requires more resources than system has
                    if task.r1 > self.r1 or task.r2 > self.r2:
                        task.failed = True
                        task.failure_reason = "Insufficient Resources"
                        task.end_time = current_time
                        self.completed_tasks.append(task)
                        pending_to_remove.append(task)
                    else:
                        self.waiting_queue.append(task)
                        pending_to_remove.append(task)
                    if hasattr(task, 'dependencies'):
                        self.dependency_map[task.name] = task.dependencies

            for task in pending_to_remove:
                self.pending_tasks.remove(task)

            # Check for dependency failures in waiting queue
            for task in list(self.waiting_queue):
                deps = self.dependency_map.get(task.name, [])
                if deps:
                    for dep_name in deps:
                        # Check if dependency failed or completed
                        dep_failed = False
                        dep_completed = False
                        for completed in self.completed_tasks:
                            if completed.name == dep_name:
                                if completed.failed:
                                    dep_failed = True
                                else:
                                    dep_completed = True
                                break
                        
                        # If dependency failed, fail this task
                        if dep_failed:
                            task.failed = True
                            task.failure_reason = f"Dependency {dep_name} Failed"
                            task.end_time = current_time
                            self.completed_tasks.append(task)
                            self.waiting_queue.remove(task)
                            break
        

            # Process remaining waiting tasks
            waiting_to_ready = []
            for task in list(self.waiting_queue):
                if not task.failed:  # Skip failed tasks
                    deps = self.dependency_map.get(task.name, [])
                    deps_met = True
                    
                    if deps:
                        for dep_name in deps:
                            dep_completed = any(t.name == dep_name and not t.failed 
                                             for t in self.completed_tasks)
                            if not dep_completed:
                                deps_met = False
                                break
                    
                    if deps_met:
                        if task.r1 <= self.available_r1 and task.r2 <= self.available_r2:
                            task.state = 'ready'
                            self.ready_queue.append(task)
                            waiting_to_ready.append(task)

            for task in waiting_to_ready:
                self.waiting_queue.remove(task)

            # Process cores
            for core in self.cores:
                if core.current_task:
                    core.current_task.remaining_time -= 1
                    if core.current_task.remaining_time <= 0:
                        core.current_task.end_time = current_time
                        core.current_task.state = 'completed'
                        self.completed_tasks.append(core.current_task)
                        self.available_r1 += core.current_task.r1
                        self.available_r2 += core.current_task.r2
                        core.current_task = None
                        core.status = 'idle'
                        core.processing_event.clear()

                if not core.current_task and self.ready_queue:
                    task = self.ready_queue.popleft()
                    core.current_task = task
                    self.available_r1 -= task.r1
                    self.available_r2 -= task.r2
                    core.current_task.start_time = current_time
                    core.status = 'running'
                    core.processing_event.set()

class MainSystem(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.subsystems = []
        self.current_time = 0
        self.history_lock = threading.Lock()
        self.resource_history = []
        self.running = True
        self.pause_event = threading.Event()
        self.barrier = threading.Barrier(
            1 +  # Main thread
            4 +  # 4 subsystems
            8    # Total cores (3+2+1+2)
        )

    def start(self):
        # Single place to start all threads
        for ss in self.subsystems:
            # Start subsystem thread
            if not ss.is_alive():
                ss.start()
            # Start core threads
            for core in ss.cores:
                if not core.is_alive():
                    core.start()
        # Start main system thread last
        if not self.is_alive():
            super().start()

    def run(self):
        while self.running:
            try:
                self.pause_event.wait()
                
                # Clear previous cycle flags
                for ss in self.subsystems:
                    ss.cycle_complete.clear()

                # Signal all subsystems to process
                for ss in self.subsystems:
                    ss.process_event.set()

                # Wait for all subsystems to complete
                for ss in self.subsystems:
                    ss.cycle_complete.wait()

                # Increment time after all subsystems finish
                self.current_time += 1
                
                # Update resource history
                self.current_time += 1
                with self.history_lock:
                    current_resources = []
                    for ss in self.subsystems:
                        with ss.resource_lock:
                            current_resources.append((ss.available_r1, ss.available_r2))
                    self.resource_history.append(current_resources)

                if self.check_all_tasks_completed():
                    print("All tasks completed")
                    self.stop()
                    break

                time.sleep(0.1)
            except Exception as e:
                print(f"MainSystem error: {str(e)}")

    def check_all_tasks_completed(self):
        for ss in self.subsystems:
            if (ss.pending_tasks or ss.waiting_queue or 
                any(core.current_task for core in ss.cores) or
                any(core.ready_queue for core in ss.cores) or
                (hasattr(ss, 'ready_queue') and ss.ready_queue)):
                return False
        return True
        
    def stop(self):
        """Stop the main system and all its components"""
        self.running = False
        self.pause_event.set()  # Wake up main thread
        
        # Stop all subsystems
        for ss in self.subsystems:
            ss.running = False
            ss.process_event.set()  # Wake up subsystem thread
            ss.cycle_complete.set() # Release any waiting threads
            
            # Stop all cores
            for core in ss.cores:
                core.should_run = False
                core.processing_event.set()  # Wake up core thread
                core.task_completed.set()    # Release any waiting threads
        
        try:
            # Wait for threads to stop with timeout
            self.join(timeout=0.5)
            for ss in self.subsystems:
                ss.join(timeout=0.5)
                for core in ss.cores:
                    core.join(timeout=0.5)
        except Exception as e:
            print(f"Stop error: {str(e)}")

    def parse_input(self, input_lines):
        # Process lines while ignoring comments
        cleaned_lines = []
        for line in input_lines:
            # Split on '#' and take first part, strip whitespace
            clean_line = line.split('#')[0].strip()
            if clean_line:  # Only keep non-empty lines
                cleaned_lines.append(clean_line)
    
        resource_lines = cleaned_lines[:4]
        ss_resources = []
        for line in resource_lines:
            r1, r2 = map(int, line.split())
            ss_resources.append((r1, r2))

        task_sections = []
        current_section = []
        for line in cleaned_lines[4:]:
            line = line.strip()
            if line == '$':
                if current_section:
                    task_sections.append(current_section)
                    current_section = []
            else:
                current_section.append(line)
        if current_section:
            task_sections.append(current_section)

        for i in range(4):
            r1, r2 = ss_resources[i]
            if i == 0:
                ss = SS1(r1, r2)
            elif i == 1:
                ss = SS2(r1, r2)
            elif i == 2:
                ss = SS3(r1, r2)
            else:
                ss = SS4(r1, r2)
            ss.attach_main_system(self)
            self.subsystems.append(ss)

        for i, section in enumerate(task_sections):
            ss = self.subsystems[i]
            for task_line in section:
                parts = task_line.split()
                name = parts[0]
                runtime = int(parts[1])
                if i == 0:  # SS1: name runtime r1 r2 entry_time target_core
                    r1, r2, entry_time, initial_core = map(int, parts[2:6])
                    task = Task(name, runtime, r1, r2, entry_time, initial_core=initial_core)
                    ss.add_task(task)
                    ss.pending_tasks.append(task)  # Add to pending, not ready queue
                elif i == 1:  # SS2: name runtime r1 r2 entry_time
                    r1, r2, entry_time = map(int, parts[2:5])
                    task = Task(name, runtime, r1, r2, entry_time)
                    ss.add_task(task)
                    ss.pending_tasks.append(task)
                elif i == 2:  # SS3: name runtime r1 r2 entry_time period num_repetitions
                    r1, r2, entry_time, period, num_reps = map(int, parts[2:7])
                    task = Task(name, runtime, r1, r2, entry_time, period=period, num_repetitions=num_reps)
                    ss.add_task(task)
                    ss.pending_tasks.append(task)
                elif i == 3:  # SS4: name runtime r1 r2 entry_time dependencies...
                    r1, r2, entry_time = map(int, parts[2:5])
                    dependencies = parts[5:] if len(parts) > 5 else []
                    dependencies = [d for d in dependencies if d != '-']
                    task = Task(name, runtime, r1, r2, entry_time, dependencies=dependencies)
                    ss.add_task(task)
                    ss.pending_tasks.append(task)

            if i == 3:
                name_to_task = {task.name: task for task in ss.tasks}
                for task_name, deps in ss.dependency_map.items():
                    resolved_deps = []
                    for dep_name in deps:
                        if dep_name in name_to_task:
                            resolved_deps.append(name_to_task[dep_name])
                    ss.dependency_map[task_name] = resolved_deps
                        
    def run_simulation_step(self):
        if self.running and any(any(task.end_time is None for task in ss.tasks) for ss in self.subsystems):
            self.current_time += 1
            for ss in self.subsystems:
                ss.process(self.current_time)
            # Store per-subsystem resources instead of sum
            current_resources = []
            for ss in self.subsystems:
                current_resources.append((ss.available_r1, ss.available_r2))
            self.resource_history.append(current_resources)
            return True
        return False

    # def run_simulation(self):
    #     while any(any(task.end_time is None for task in ss.tasks) for ss in self.subsystems):
    #         self.current_time += 1
    #         for ss in self.subsystems:
    #             ss.process(self.current_time)
    #         self.resource_history.append(
    #             (sum(ss.available_r1 for ss in self.subsystems),
    #              sum(ss.available_r2 for ss in self.subsystems))
    #         )
    #         self.print_status()
    #         return True
    #     self.generate_report()
    #     return False

    def print_status(self):
        print(f"| Time: {self.current_time}    |")
        print("|---|")
        for i, ss in enumerate(self.subsystems):
            print(f"| Sub{i+1}:    |")
            print(f"| Resources: R1: {ss.available_r1} R2: {ss.available_r2}    |")
            if isinstance(ss, SS1):
                print("| Waiting Queue", list(ss.waiting_queue), "   |")
                for j, core in enumerate(ss.cores):
                    print(f"| Core{j+1}:    |")
                    print(f"| Running Task: {core.current_task.name if core.current_task else 'idle'}    |")
                    print(f"| Ready Queue: {list(core.ready_queue)}    |")
            elif isinstance(ss, SS2):
                print("| Ready Queue: ", list(ss.ready_queue), "   |")
                for j, core in enumerate(ss.cores):
                    print(f"| Core{j+1}:    |")
                    print(f"| Running Task: {core.current_task.name if core.current_task else 'idle'}    |")
            elif isinstance(ss, SS3):
                print("| Waiting Queue", list(ss.waiting_queue), "   |")
                print("| Ready Queue: ", list(ss.ready_queue), "   |")
                print(f"| Core1:    |")
                print(f"| Running Task: {ss.cores[0].current_task.name if ss.cores[0].current_task else 'idle'}    |")
            elif isinstance(ss, SS4):
                print("| Waiting Queue", list(ss.waiting_queue), "   |")
                print("| Ready Queue: ", list(ss.ready_queue), "   |")
                for j, core in enumerate(ss.cores):
                    print(f"| Core{j+1}:    |")
                    print(f"| Running Task: {core.current_task.name if core.current_task else 'idle'}    |")
            print("|---|")

    def generate_report(self):
        print("\nFinal Report:")
        for i, ss in enumerate(self.subsystems):
            print(f"Subsystem {i+1}:")
            for task in ss.completed_tasks:
                print(f"Task {task.name}: Entry={task.start_time}, End={task.end_time}, Waiting={task.waiting_time}")


def main():
    input_lines = sys.stdin.read().splitlines()
    main_system = MainSystem()
    main_system.parse_input(input_lines)
    main_system.start()
    
    # For command line operation
    while main_system.is_alive():
        main_system.join(0.1)
    main_system.generate_report()

if __name__ == "__main__":
    import sys
    from gui import run_gui

    input_lines = sys.stdin.read().splitlines()
    run_gui(input_lines)
    #main()