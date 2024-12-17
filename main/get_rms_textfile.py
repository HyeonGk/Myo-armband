import myo
import numpy as np
import os
import time
from collections import deque
from threading import Lock

class EmgCollector(myo.DeviceListener):
    def __init__(self, frame_size, hop_length, filename, max_lines=2000, save_interval=500, pause_duration=10):
        self.frame_size = frame_size  
        self.hop_length = hop_length  
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=frame_size)  # Queue to store data up to frame_size
        self.filename = filename  # Filename to save RMS values
        self.max_lines = max_lines  # Maximum lines to save in file
        self.count = 0  # Count to track hop length
        self.write_count = 0  # Track number of writes to file
        self.save_interval = save_interval  # Number of writes before pause
        self.pause_duration = pause_duration  # Pause duration in seconds

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            # Add new EMG data to queue
            self.emg_data_queue.append(event.emg)
            self.count += 1

            # Calculate and save RMS at every hop_length
            if self.count >= self.hop_length and len(self.emg_data_queue) == self.frame_size:
                self.calculate_and_save_rms()
                self.count = 0  # Reset hop length count

    def calculate_and_save_rms(self):
        # Calculate RMS using the latest frame_size data
        emg_array = np.array(self.emg_data_queue)
        rms = np.sqrt(np.mean(np.square(emg_array), axis=0))  # Calculate RMS
        self.save_rms_to_text(rms)

    def save_rms_to_text(self, rms):
        try:
            # Create directory path if it does not exist
            directory = os.path.dirname(self.filename)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Read existing file contents to control line count
            if os.path.exists(self.filename):
                with open(self.filename, "r", encoding='utf-8') as f:
                    lines = f.readlines()

                # Remove oldest lines if line count exceeds max_lines
                if len(lines) >= self.max_lines:
                    lines = lines[-(self.max_lines - 1):] 
            else:
                lines = []

            # Add new RMS data line
            line = ", ".join(f"{value:.8f}" for value in rms) + "\n"
            lines.append(line)

            # Write lines to file
            with open(self.filename, "w", encoding='utf-8') as f:
                f.writelines(lines)

            # Increment write count and print status
            self.write_count += 1
            print(f"RMS data written to {self.filename} successfully. Write count: {self.write_count}")

            # Pause for the defined duration after every save_interval writes
            if self.write_count % self.save_interval == 0:
                print("Pausing data collection for 10 seconds...")
                for remaining in range(self.pause_duration, 0, -1):
                    print(f"Resuming in {remaining} seconds", end='\r')
                    time.sleep(1)
                print("Resuming data collection...")

        except Exception as e:
            print(f"Error occurred while saving data to text file: {e}")

def main():
    myo.init()
    hub = myo.Hub()

    # Set file path relative to current script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_dir, "../for_distingush/rms.txt")

    # Initialize EmgCollector
    listener = EmgCollector(frame_size=40, hop_length=20, filename=filename, max_lines=1000, save_interval=500, pause_duration=10)

    with hub.run_in_background(listener.on_event):
        try:
            while True:
                pass  # Continue collecting data from Myo device
        except KeyboardInterrupt:
            print("Stopping...")

if __name__ == '__main__':
    main()
