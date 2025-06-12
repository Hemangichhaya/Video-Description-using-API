import json
from datetime import datetime
import os
from typing import Dict, Any

class TaskTracker:
    def __init__(self, data_file: str = "docs/data_record.json"):
        self.data_file = data_file
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.load_data()

    def load_data(self):
        """Load existing data from the JSON file."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.tasks = json.load(f)
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            self.tasks = {}

    def save_data(self):
        """Save current data to the JSON file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.tasks, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {str(e)}")

    def _print_progress_indicator(self, message: str, timing_info: str = None):
        """Print a visual progress indicator with timing information."""
        separator = "=" * 30
        print(f"\n{separator}")
        print(message)
        if timing_info:
            print(timing_info)
        print(f"{separator}\n")

    def _calculate_duration(self, start_time: str, end_time: str = None) -> float:
        """Calculate duration between two timestamps."""
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time) if end_time else datetime.now()
        return (end - start).total_seconds()

    def start_task(self, task_id: str):
        """Initialize a new task with timing and progress data."""
        start_time = datetime.now().isoformat()
        self.tasks[task_id] = {
            "start_time": start_time,
            "steps": {},
            "current_progress": 0,
            "total_steps": 0,
            "status": "in_progress",
            "timing": {
                "start_time": start_time,
                "steps_timing": {}
            }
        }
        self._print_progress_indicator(
            f"Starting new task: {task_id}",
            f"Start Time: {datetime.fromisoformat(start_time).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.save_data()

    def update_progress(self, task_id: str, step_name: str, progress: int):
        """Update progress for a specific step in the task."""
        if task_id not in self.tasks:
            self.start_task(task_id)

        current_time = datetime.now().isoformat()
        
        # Initialize or update step information
        if step_name not in self.tasks[task_id]["steps"]:
            self.tasks[task_id]["steps"][step_name] = {
                "start_time": current_time,
                "progress": progress
            }
            self.tasks[task_id]["timing"]["steps_timing"][step_name] = {
                "start_time": current_time
            }
        else:
            self.tasks[task_id]["steps"][step_name]["progress"] = progress

        # Calculate timing information for the step
        step_timing = self.tasks[task_id]["timing"]["steps_timing"][step_name]
        duration = self._calculate_duration(step_timing["start_time"])
        timing_info = f"Step Duration: {duration:.2f} seconds"

        self._print_progress_indicator(
            f"Progress Update: {step_name} ({progress}%)",
            timing_info
        )

        # Update overall progress
        self.tasks[task_id]["current_progress"] = progress
        self.save_data()

    def complete_step(self, task_id: str, step_name: str):
        """Mark a step as completed and record its completion time."""
        if task_id in self.tasks and step_name in self.tasks[task_id]["steps"]:
            current_time = datetime.now().isoformat()
            
            # Update step completion time
            self.tasks[task_id]["steps"][step_name]["end_time"] = current_time
            self.tasks[task_id]["timing"]["steps_timing"][step_name]["end_time"] = current_time
            
            # Calculate duration
            start_time = self.tasks[task_id]["timing"]["steps_timing"][step_name]["start_time"]
            duration = self._calculate_duration(start_time, current_time)
            self.tasks[task_id]["timing"]["steps_timing"][step_name]["duration_seconds"] = duration

            self._print_progress_indicator(
                f"Completed step: {step_name}",
                f"Step Duration: {duration:.2f} seconds"
            )
            self.save_data()

    def complete_task(self, task_id: str, status: str = "completed"):
        """Mark a task as completed and calculate total duration."""
        if task_id in self.tasks:
            end_time = datetime.now().isoformat()
            self.tasks[task_id]["end_time"] = end_time
            self.tasks[task_id]["status"] = status
            
            # Calculate total duration
            total_duration = self._calculate_duration(
                self.tasks[task_id]["start_time"],
                end_time
            )
            self.tasks[task_id]["timing"]["end_time"] = end_time
            self.tasks[task_id]["timing"]["total_duration_seconds"] = total_duration
            
            # Calculate step durations
            step_durations = {}
            for step, timing in self.tasks[task_id]["timing"]["steps_timing"].items():
                if "end_time" in timing:
                    duration = self._calculate_duration(timing["start_time"], timing["end_time"])
                else:
                    duration = self._calculate_duration(timing["start_time"], end_time)
                step_durations[step] = duration

            # Print final summary
            self._print_progress_indicator(
                f"Task {task_id} {status}",
                self._format_task_summary(task_id, status, total_duration, step_durations)
            )
            
            self.tasks[task_id]["current_progress"] = 100
            self.save_data()

    def _format_task_summary(self, task_id: str, status: str, total_duration: float, step_durations: Dict[str, float]) -> str:
        """Format the task summary with timing information."""
        summary = [
            "Task Summary:",
            f"Status: {status}",
            f"Total Duration: {total_duration:.2f} seconds",
            f"Steps Completed: {len(self.tasks[task_id]['steps'])}",
            f"Final Progress: {self.tasks[task_id]['current_progress']}%",
            "\nStep Durations:"
        ]
        
        for step, duration in step_durations.items():
            summary.append(f"- {step}: {duration:.2f} seconds")
        
        return "\n".join(summary)

# Global instance
task_tracker = TaskTracker() 