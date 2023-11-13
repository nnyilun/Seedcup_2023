import subprocess
import sys

def run_executable(executable_path):
    while True:
        process = subprocess.Popen(executable_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()

        print("Output:\n", stdout.decode())

        if "connect host 0.0.0.0:9999" in stdout.decode():
            break

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_executable>")
        sys.exit(1)

    executable_path = sys.argv[1]
    run_executable(executable_path)
