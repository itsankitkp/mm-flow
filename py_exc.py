import os
import pickle
import subprocess
import sys
import tempfile
import shutil
import types
import inspect
import json
from typing import List, Callable, Any
from pathlib import Path


class VirtualPythonEnvironment:
    """
    A class that provides a fully isolated, stateful Python execution environment.
    """

    def __init__(self):
        """Initializes the environment, creates a venv, and identifies python/pip paths."""
        self.workdir = tempfile.mkdtemp()
        self.venv_path = os.path.join(self.workdir, "venv")
        self._secrets = {}

        # Directory for injected modules
        self.modules_dir = os.path.join(self.workdir, "injected_modules")
        os.makedirs(self.modules_dir, exist_ok=True)

        try:
            # Create the virtual environment
            subprocess.run(
                [sys.executable, "-m", "venv", self.venv_path],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create virtual environment: {e.stderr}")

        # Define platform-specific executable paths
        if sys.platform == "win32":
            self.python_executable = os.path.join(
                self.venv_path, "Scripts", "python.exe"
            )
            self.pip_executable = os.path.join(self.venv_path, "Scripts", "pip.exe")
        else:
            self.python_executable = os.path.join(self.venv_path, "bin", "python")
            self.pip_executable = os.path.join(self.venv_path, "bin", "pip")

        # Upgrade pip and install common packages
        try:
            subprocess.run(
                [self.pip_executable, "install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
                text=True,
            )
            common_packages = ["requests", "pandas", "numpy"]
            subprocess.run(
                [self.pip_executable, "install"] + common_packages,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(self.workdir, ignore_errors=True)
            raise RuntimeError(f"Failed to pre-install packages: {e.stderr}")

        self.state_file = os.path.join(self.workdir, "session_state.pkl")
        self.imports_cache = os.path.join(self.workdir, "imports_cache.json")

    def __del__(self):
        """Ensures the entire temporary directory (including the venv) is cleaned up."""
        shutil.rmtree(self.workdir, ignore_errors=True)

    def add_module(self, module):
        """
        Inject a Python module into the virtual environment.

        Args:
            module: A Python module object or path to a .py file
        """
        if isinstance(module, str) and os.path.isfile(module):
            # It's a file path
            module_name = os.path.splitext(os.path.basename(module))[0]
            target_path = os.path.join(self.modules_dir, f"{module_name}.py")
            shutil.copy2(module, target_path)
            return f"Module '{module_name}' added from file"

        if not inspect.ismodule(module):
            raise ValueError("Argument must be a module object or file path")

        module_name = module.__name__.split(".")[
            -1
        ]  # Get the last part of the module name
        module_file = os.path.join(self.modules_dir, f"{module_name}.py")

        try:
            # Try to get the source code
            source = inspect.getsource(module)
            with open(module_file, "w") as f:
                f.write(source)
        except (OSError, TypeError):
            # If we can't get source (e.g., built-in module), try to get from file
            if hasattr(module, "__file__") and module.__file__:
                if module.__file__.endswith(".py"):
                    shutil.copy2(module.__file__, module_file)
                else:
                    raise ValueError(f"Cannot extract source from module {module_name}")
            else:
                raise ValueError(f"Cannot extract source from module {module_name}")

        return f"Module '{module_name}' injected successfully"

    def _resolve_path(self, filename: str) -> str:
        """Resolves a filename to its full, secure path within the working directory."""
        safe_path = os.path.normpath(os.path.join(self.workdir, filename))
        if not safe_path.startswith(self.workdir):
            raise ValueError("File path must be within the working directory.")
        return safe_path

    def execute_shell(self, command: str) -> str:
        """Executes a shell command within the virtual environment."""
        if command.strip().startswith("pip "):
            command = command.replace("pip", self.pip_executable, 1)

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.modules_dir}:{self.workdir}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.workdir,
                env=env,
            )
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            return output if output else "Command executed successfully with no output."
        except subprocess.CalledProcessError as e:
            return f"Error executing command '{command}':\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"

    def execute_python(self, code: str) -> str:
        """
        Executes Python code within the virtual environment, maintaining state.
        """
        # Add the modules directory to sys.path in the prelude
        prelude_code = f"""
import sys
sys.path.insert(0, r'{self.modules_dir}')
sys.path.insert(0, r'{self.workdir}')

import pandas as pd
import numpy as np
import requests
import os

# Load cached imports
import json
try:
    with open(r'{self.imports_cache}', 'r') as f:
        _imports = json.load(f).get('imports', [])
        for imp in _imports:
            try:
                exec(imp)
            except:
                pass
except:
    pass
"""

        runner_script_template = """
import pickle, sys, types
import json
import ast

state_file = r'{state_file_path}'
imports_cache = r'{imports_cache_path}'
_globals = {{}}

# Load previous state (variables only, not modules)
try:
    with open(state_file, 'rb') as f:
        _saved = pickle.load(f)
        for k, v in _saved.items():
            if not isinstance(v, (types.ModuleType, types.FunctionType, type)):
                _globals[k] = v
except:
    pass

_globals.update({secrets})

# Execute prelude
exec({prelude}, _globals)

# Track imports in user code
user_code = {user_code_repr}
_new_imports = []
try:
    tree = ast.parse(user_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                stmt = f"import {{alias.name}}"
                if alias.asname:
                    stmt += f" as {{alias.asname}}"
                _new_imports.append(stmt)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                items = ", ".join([
                    f"{{n.name}}" + (f" as {{n.asname}}" if n.asname else "")
                    for n in node.names
                ])
                _new_imports.append(f"from {{node.module}} import {{items}}")
except:
    pass

# Execute user code
try:
    exec(user_code, _globals)
except Exception as e:
    import traceback
    print(traceback.format_exc(), file=sys.stderr)

# Save new imports
if _new_imports:
    try:
        with open(imports_cache, 'r') as f:
            cache = json.load(f)
    except:
        cache = {{'imports': []}}
    
    for imp in _new_imports:
        if imp not in cache['imports']:
            cache['imports'].append(imp)
    
    with open(imports_cache, 'w') as f:
        json.dump(cache, f)

# Save state (only serializable non-module objects)
serializable_globals = {{}}
for key, value in _globals.items():
    if key.startswith('_') or isinstance(value, (types.ModuleType, types.FunctionType, type)):
        continue
    try:
        pickle.dumps(value)
        serializable_globals[key] = value
    except:
        continue

try:
    with open(state_file, 'wb') as f:
        pickle.dump(serializable_globals, f)
except Exception as e:
    print(f"Warning: Could not save state: {{e}}", file=sys.stderr)
"""

        runner_script = runner_script_template.format(
            state_file_path=self.state_file,
            imports_cache_path=self.imports_cache,
            secrets=self._secrets,
            prelude=repr(prelude_code),
            user_code_repr=repr(code),
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.modules_dir}:{self.workdir}"

        try:
            result = subprocess.run(
                [self.python_executable, "-c", runner_script],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.workdir,
                env=env,
            )
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            return output if output else "Execution successful with no output."
        except subprocess.CalledProcessError as e:
            return f"An error occurred during Python execution:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"

    def save_code(self, filename: str, code: str) -> str:
        """Saves a string of code to a file inside the working directory."""
        try:
            full_path = self._resolve_path(filename)
            with open(full_path, "w") as f:
                f.write(code)
            return f"Code successfully saved to '{filename}' in the working directory."
        except Exception as e:
            return f"An error occurred while saving the file: {e}"

    def run_script(self, filename: str) -> str:
        """Reads and executes a script from the working directory."""
        try:
            full_path = self._resolve_path(filename)
            with open(full_path, "r") as f:
                script_code = f.read()
            return self.execute_python(script_code)
        except FileNotFoundError:
            return (
                f"Error: The file '{filename}' was not found in the working directory."
            )
        except Exception as e:
            return f"An error occurred: {e}"

    def list_files(self) -> str:
        """Lists all files in the current working directory."""
        try:
            files = [
                f
                for f in os.listdir(self.workdir)
                if f
                not in [
                    "venv",
                    "injected_modules",
                    "session_state.pkl",
                    "imports_cache.json",
                ]
            ]
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
        if not self._secrets:
            return "No secrets are currently set."
        return f"Available secrets (keys only): {list(self._secrets.keys())}"


# --- Create a single, persistent instance ---
isolated_env = VirtualPythonEnvironment()


# --- Tool Helper Functions ---
def execute_shell_command_in_env(command: str) -> str:
    """
    Execute a shell command within the isolated virtual environment.
    Use this for installing packages (e.g., 'pip install package_name'),

    Args:
        command (str): The shell command to execute (e.g., 'pip install requests', 'ls -la')
    Returns:
        str: Combined stdout and stderr output from the command execution

    Examples:
        >>> execute_shell_command_in_env("pip install beautifulsoup4")
    """
    return isolated_env.execute_shell(command)


def execute_python_code_in_env(code: str) -> str:
    """
    Execute Python code in a persistent, stateful, isolated environment.
    Variables, imports, and objects created in one execution persist to the next.

    Args:
        code (str): Python code to execute as a string

    Returns:
        str: Combined stdout and stderr from the code execution

    Examples:
        >>> execute_python_code_in_env("from serv import OAuthCallbackServer"
    """
    return isolated_env.execute_python(code)


def save_code_to_file_in_env(filename: str, code: str) -> str:
    """
    Save code to a file in the isolated environment's working directory.
    Args:
        filename (str): Name of the file to create
        code (str): Content to write to the file

    Returns:
        str: Confirmation message or error description

    Examples:
        >>> save_code_to_file_in_env("utils.py", "def greet(name): return f'Hello {name}'")
    """
    return isolated_env.save_code(filename, code)


def run_python_script_in_env(filename: str) -> str:
    """
    Execute a Python script file from the environment's working directory.

    Reads and executes a .py file that was previously saved using save_code_to_file_in_env().
    The script runs in the same persistent environment, accessing all variables and imports.

    Args:
        filename (str): Name of the Python script file to execute

    Returns:
        str: Combined stdout and stderr from script execution

    Examples:
        >>> save_code_to_file_in_env("test.py", "print('Hello from script')")
        >>> run_python_script_in_env("test.py")
    """
    return isolated_env.run_script(filename)


def list_files_in_workdir() -> str:
    """
    List all files in the environment's working directory.

    Shows files created by the agent, excluding system files like the venv,
    state files, and module directories. Useful for checking what scripts
    or data files have been created.

    Returns:
        str: List of filenames or a message if directory is empty

    Examples:
        >>> list_files_in_workdir()
        'Files in the working directory: ["script.py", "data.json", "output.txt"]'
    """
    return isolated_env.list_files()


def set_secret_variable_in_env(key: str, value: str) -> str:
    """
    Store a secret (e.g., API key, password) as a variable in the environment.

    Secrets are injected as global variables in Python executions, allowing
    secure use of sensitive data without hardcoding. Secrets are only stored
    in memory for this session.

    Args:
        key (str): Variable name for the secret (must be valid Python identifier)
        value (str): The secret value to store

    Returns:
        str: Confirmation message or error if key is invalid

    Examples:
        >>> set_secret_variable_in_env("API_KEY", "sk-abc123...")
        >>> execute_python_code_in_env("print(f'Using key: {API_KEY[:10]}...')")
    """
    return isolated_env.set_secret(key, value)


def list_available_secrets() -> str:
    """
    List the names of all secrets stored in the environment.

    Only shows the variable names, not the actual secret values.
    Useful for checking what secrets are available for use in code.

    Returns:
        str: List of secret variable names or message if none are set

    Examples:
        >>> set_secret_variable_in_env("API_KEY", "secret123")
        >>> list_available_secrets()
        'Available secrets (keys only): ["API_KEY"]'
    """
    return isolated_env.list_secrets()


import serv

isolated_env.add_module(serv)


# List of tools for agent
tools: List[Callable[..., str]] = [
    execute_shell_command_in_env,
    execute_python_code_in_env,
    save_code_to_file_in_env,
    run_python_script_in_env,
    list_files_in_workdir,
    set_secret_variable_in_env,
    list_available_secrets,
]
