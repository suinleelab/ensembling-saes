import argparse
import logging
import os
import shlex
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from types import FrameType
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

commands_per_gpu = {}
MAX_WORKERS = 2 # Number of commands to run in parallel on each GPU

# Track all subprocesses to handle termination
running_processes = []

def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-params",
        type=str,
        help="path to yaml file containing SAE parameter values to sweep",
        required=True
    )
    parser.add_argument(
        "--commands",
        type=str,
        help="path to the .txt file containing all the commands to execute",
        required=True
    )
    parser.add_argument(
        "--log-file-path",
        type=str,
        help="path to the file to stream logger output",
        default=None,
        dest="log_file_path"
    )

    return parser.parse_args()

# Handle Ctrl+C to terminate all running processes
def handle_exit(signum: int, frame: Optional[FrameType]) -> None:
    """
    Handles termination signals (Ctrl+C) by stopping all running processes.
    """
    logger.info(f'Signal handler called with signal number {signum} and frame {frame}')
    logger.info("Ctrl+C detected! Terminating all processes...\n")
    for process in running_processes:
        try:
            process.terminate()  # Terminate process
        except Exception as e:
            logger.info(f"Error terminating process: {e}")
    sys.exit(1)  # Exit script

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, handle_exit)

def run_command(command: str) -> None:
    """
    Function to run a command on a particular GPU
    """
    # Properly split command with arguments
    cmd_list = shlex.split(command)  

    # Start subprocess
    process = subprocess.Popen(cmd_list) 
    running_processes.append(process)

    # Wait for script completion
    process.wait()

# Function to execute scripts for a specific GPU
def execute_on_gpu(curr_gpu: int) -> None:
    """
    Executes commands assigned to a specific GPU, running up to MAX_WORKERS commands in parallel.
    """
    # Get the commands to run for the current GPU
    commands = commands_per_gpu.get(curr_gpu, [])

    if not commands:
        logger.info(f"No commands assigned to GPU {curr_gpu}. Skipping...")
        return

    logger.info(f"Starting execution on GPU {curr_gpu}...")
    logger.info(f"Running commands: {commands}")

    # Run multiple scripts in parallel on one GPU
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(run_command, commands)

    logger.info(f"All scripts completed on GPU {curr_gpu}!")

def create_run_command(log_file_dir: str, 
        cmd: str,
        curr_cmd_ind: int, 
        curr_gpu: int) -> str:
    """
    Function to create the training command to run
    """
    log_file_path = os.path.join(log_file_dir, f"cmd_index_{curr_cmd_ind}_gpu_id_{curr_gpu}.txt")
    
    return cmd.strip() \
        + f" --device cuda:{curr_gpu} --log-file-path {log_file_path}"

def main():
    args = _parse_arguments()

    if args.log_file_path is not None:
        logging.basicConfig(level=logging.INFO, filename=args.log_file_path)
    else:
        logging.basicConfig(level=logging.INFO)

    # Load the config parameters file to get the list of GPUs available
    with open(args.config_params) as file:
        cfg_params = yaml.safe_load(file)
    
    # Create list of all commands to run
    with open(args.commands) as file:
        commands_list = file.readlines()
    
    # Make logging director
    experiments_base_dir = os.path.dirname(args.commands)
    log_file_dir = experiments_base_dir + "/logs"
    os.makedirs(log_file_dir, exist_ok=True)

    available_gpus = cfg_params['cuda_available_devices']
    curr_gpu_ind = 0
    curr_cmd_ind = 0

    # Split the train commands among all the gpus
    split_cmds = np.array_split(commands_list, len(available_gpus))

    for curr_gpu_ind in range(len(available_gpus)):
        curr_gpu = available_gpus[curr_gpu_ind]
        commands = []

        # Create the commands list for current gpu
        for cmd in split_cmds[curr_gpu_ind]:
            cmd_to_write = create_run_command(log_file_dir, cmd, curr_cmd_ind, curr_gpu)
            commands.append(cmd_to_write)
            curr_cmd_ind += 1
        
        commands_per_gpu[curr_gpu] = commands

    try:
        # Start processes in parallel across GPUs
        with ProcessPoolExecutor(max_workers=len(available_gpus)) as executor:
            executor.map(execute_on_gpu, available_gpus)
    except KeyboardInterrupt:
        # Ensure cleanup if Ctrl+C is pressed
        handle_exit(None, None) 

    logger.info("All training jobs completed!")

if __name__ == "__main__":
    main()
