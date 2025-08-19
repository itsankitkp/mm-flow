import os
import pickle
import subprocess
import sys
import tempfile
import shutil
import types
from typing import List, Callable

# --- Core Class for the Isolated Python Environment ---

class VirtualPythonEnvironment:
    """
    A class that provides a fully isolated, stateful Python execution environment.

    Each instance creates its own virtual environment (`venv`) in a temporary directory.
    All shell commands and Python code are executed exclusively within this isolated
    environment, ensuring no dependency conflicts. State (variables) is maintained
    between Python executions by pickling the global scope.
    """
    def __init__(self):
        """Initializes the environment, creates a venv, and identifies python/pip paths."""
        self.workdir = tempfile.mkdtemp()
        self.venv_path = os.path.join(self.workdir, "venv")
        self._secrets = {}

        try:
            subprocess.run([sys.executable, "-m", "venv", self.venv_path], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create virtual environment: {e.stderr.decode()}")

        if sys.platform == "win32":
            self.python_executable = os.path.join(self.venv_path, "Scripts", "python.exe")
            self.pip_executable = os.path.join(self.venv_path, "Scripts", "pip.exe")
        else:
            self.python_executable = os.path.join(self.venv_path, "bin", "python")
            self.pip_executable = os.path.join(self.venv_path, "bin", "pip")

        self.state_file = os.path.join(self.workdir, "session_state.pkl")

    def __del__(self):
        """Ensures the entire temporary directory (including the venv) is cleaned up."""
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _resolve_path(self, filename: str) -> str:
        """Resolves a filename to its full, secure path within the working directory."""
        safe_path = os.path.normpath(os.path.join(self.workdir, filename))
        if not safe_path.startswith(self.workdir):
            raise ValueError("File path must be within the designated working directory.")
        return safe_path

    def execute_shell(self, command: str) -> str:
        """Executes a shell command within the virtual environment."""
        if command.strip().startswith("pip "):
            command = command.replace("pip", self.pip_executable, 1)

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=True, cwd=self.workdir
            )
            output = ""
            if result.stdout: output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr: output += f"STDERR:\n{result.stderr}\n"
            return output if output else "Command executed successfully with no output."
        except subprocess.CalledProcessError as e:
            return f"Error executing command '{command}':\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"

    def execute_python(self, code: str) -> str:
        """Executes Python code within the virtual environment, maintaining state."""
        runner_script = f"""
import pickle, sys, types

state_file = r'{self.state_file}'
_globals = {{}}

try:
    with open(state_file, 'rb') as f:
        _globals = pickle.load(f)
except FileNotFoundError:
    pass

_globals.update({self._secrets})

user_code = '''
{code}
'''

try:
    exec(user_code, _globals)
except Exception as e:
    import traceback
    print(traceback.format_exc(), file=sys.stderr)

# Improved cleanup: Remove modules, functions, and other non-serializable types
for key in list(_globals.keys()):
    if key.startswith('__') or isinstance(_globals[key], (types.ModuleType, types.FunctionType)):
        del _globals[key]

try:
    with open(state_file, 'wb') as f:
        pickle.dump(_globals, f)
except Exception as e:
    print(f"State saving error: {{e}}", file=sys.stderr)
"""
        try:
            result = subprocess.run(
                [self.python_executable, "-c", runner_script],
                capture_output=True, text=True, check=True, cwd=self.workdir
            )
            output = ""
            if result.stdout: output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr: output += f"STDERR:\n{result.stderr}\n"
            return output if output else "Execution successful with no output."
        except subprocess.CalledProcessError as e:
            return f"An error occurred during Python execution:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"

    def save_code(self, filename: str, code: str) -> str:
        """Saves a string of code to a file inside the working directory."""
        try:
            full_path = self._resolve_path(filename)
            with open(full_path, 'w') as f:
                f.write(code)
            return f"Code successfully saved to '{filename}' in the working directory."
        except Exception as e:
            return f"An error occurred while saving the file: {e}"

    def run_script(self, filename: str) -> str:
        """Reads and executes a script from the working directory."""
        try:
            full_path = self._resolve_path(filename)
            with open(full_path, 'r') as f:
                script_code = f.read()
            return self.execute_python(script_code)
        except FileNotFoundError:
            return f"Error: The file '{filename}' was not found in the working directory."
        except Exception as e:
            return f"An error occurred: {e}"

    def list_files(self) -> str:
        """Lists all files in the current working directory, excluding the venv."""
        try:
            files = [f for f in os.listdir(self.workdir) if f != 'venv']
            if not files:
                return "The working directory is empty."
            return f"Files in the working directory: {files}"
        except Exception as e:
            return f"An error occurred while listing files: {e}"

    def set_secret(self, key: str, value: str) -> str:
        """Stores a secret in memory to be injected into the Python environment."""
        if not key.isidentifier():
            return f"Error: The secret key '{key}' is not a valid Python identifier."
        self._secrets[key] = value
        return f"Secret '{key}' has been set for the current session."

    def list_secrets(self) -> str:
        """Lists the keys of the secrets that have been set."""
        if not self._secrets: return "No secrets are currently set."
        return f"Available secrets (keys only): {list(self._secrets.keys())}"

# --- Create a single, persistent instance for the entire workflow ---
isolated_env = VirtualPythonEnvironment()

# --- Tool Helper Functions for the LLM Agent ---

def execute_shell_command_in_env(command: str) -> str:
    """
    Executes a shell command within a dedicated, isolated environment.
    Use this for installing dependencies (e.g., 'pip install pandas'), managing files,
    or running any other command-line tool. All installations are temporary and
    isolated to the current session.

    Args:
        command (str): The shell command to execute (e.g., 'pip install requests').

    Returns:
        str: The standard output and standard error from the command.
    """
    return isolated_env.execute_shell(command)

def execute_python_code_in_env(code: str) -> str:
    """
    Executes a block of Python code in a persistent, stateful, and isolated environment.
    Data-type variables (strings, lists, dicts, etc.) created in one execution will be
    available in the next. This environment is separate from the host; libraries must be
    installed first using `execute_shell_command_in_env`.

    Args:
        code (str): A string of valid Python code to execute.

    Returns:
        str: The standard output and standard error from the execution.
    """
    return isolated_env.execute_python(code)

def save_code_to_file_in_env(filename: str, code: str) -> str:
    """
    Saves a string of Python code to a script file in the session's working directory.
    This is useful for creating reusable scripts that can be executed later with
    `run_python_script_in_env`.

    Args:
        filename (str): The name of the file to save (e.g., 'my_script.py').

    Returns:
        str: A confirmation message.
    """
    return isolated_env.save_code(filename, code)

def run_python_script_in_env(filename: str) -> str:
    """
    Reads and executes a Python script from the session's working directory.
    Use `list_files_in_workdir` to see available scripts.

    Args:
        filename (str): The name of the Python script file to be executed.

    Returns:
        str: The standard output and standard error from the script's execution.
    """
    return isolated_env.run_script(filename)

def list_files_in_workdir() -> str:
    """
    Lists all the files currently in the session's temporary working directory.
    This allows the agent to see what scripts or data files it has created.

    Returns:
        str: A string containing the list of filenames.
    """
    return isolated_env.list_files()

def set_secret_variable_in_env(key: str, value: str) -> str:
    """
    Stores a secret (e.g., API key) as a variable for use in Python code execution.
    The secret is only stored in memory and is injected into the Python environment
    at runtime, making it available as a global variable.

    Args:
        key (str): The name of the variable the secret will be assigned to.
        value (str): The actual secret value.

    Returns:
        str: A confirmation message.
    """
    return isolated_env.set_secret(key, value)

def list_available_secrets() -> str:
    """
    Lists the names of secrets that have been set for the current session.
    This function does NOT display the actual secret values.

    Returns:
        str: A string containing the list of available secret keys.
    """
    return isolated_env.list_secrets()

# --- List of Tools for Agent Integration ---

tools: List[Callable[..., str]] = [
    execute_shell_command_in_env,
    execute_python_code_in_env,
    save_code_to_file_in_env,
    run_python_script_in_env,
    list_files_in_workdir,
    set_secret_variable_in_env,
    list_available_secrets,
]