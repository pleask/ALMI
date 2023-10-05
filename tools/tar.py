import os
import tarfile
import time
import argparse
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

def add_file_to_tar(tar_filename, file_to_add):
    with tarfile.open(tar_filename, 'a') as tar:
        tar.add(file_to_add, arcname=os.path.basename(file_to_add))

def create_archive_with_existing_files(archive_name, watch_dir):
    with tarfile.open(archive_name, 'w') as tar:
        for filename in os.listdir(watch_dir):
            if filename.endswith('.pickle'):
                tar.add(os.path.join(watch_dir, filename), arcname=filename)

class PickleHandler(PatternMatchingEventHandler):
    patterns = ["*.pickle"]

    def process(self, event):
        add_file_to_tar(args.archive_name, event.src_path)

    def on_created(self, event):
        self.process(event)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor a directory and add .pickle files to a tar archive.')
    parser.add_argument('watch_dir', help='Directory to monitor')
    parser.add_argument('archive_name', help='Name of the archive to add files to')

    args = parser.parse_args()

    # Ensure the watch directory exists
    if not os.path.exists(args.watch_dir):
        raise FileNotFoundError(f"The directory {args.watch_dir} does not exist!")

    # Create archive with existing .pickle files
    create_archive_with_existing_files(args.archive_name, args.watch_dir)

    # setup observer
    observer = Observer()
    observer.schedule(PickleHandler(), path=args.watch_dir)

    # start the observer
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
