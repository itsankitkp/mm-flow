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
    A class that provides a fully isolated, stateful Python execution environment
    with complete persistence of variables, classes, functions, and imports.
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
            common_packages = ["requests", "pandas", "numpy", "dill"]
            subprocess.run(
                [self.pip_executable, "install"] + common_packages,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(self.workdir, ignore_errors=True)
            raise RuntimeError(f"Failed to pre-install packages: {e.stderr}")

        # Persistence files
        self.state_file = os.path.join(self.workdir, "session_state.pkl")
        self.imports_cache = os.path.join(self.workdir, "imports_cache.json")
        self.definitions_cache = os.path.join(self.workdir, "definitions_cache.json")
        self.definitions_py_cache = os.path.join(self.workdir, "definitions_cache.py")

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
        Executes Python code within the virtual environment, maintaining state
        including variables, classes, functions, and imports.
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
import dill
"""

        runner_script_template = """
import pickle, sys, types
import json
import ast
import os
import dill  # Better serialization for dynamic classes

state_file = r'{state_file_path}'
imports_cache = r'{imports_cache_path}'
definitions_cache = r'{definitions_cache_path}'
definitions_py_cache = r'{definitions_py_cache_path}'
_globals = {{}}

# CRITICAL FIX: Dynamic classes need a module context for pickling
# We create a fake module '__dynamic__' and register all dynamically
# defined classes to it. This allows dill/pickle to properly serialize
# and deserialize instances of these classes across sessions.
import types as _types
_dynamic_module = _types.ModuleType('__dynamic__')
sys.modules['__dynamic__'] = _dynamic_module

def register_classes_to_dynamic_module(globals_dict):
    \"\"\"Register all classes in globals to the dynamic module for serialization.\"\"\"
    for name, obj in globals_dict.items():
        if isinstance(obj, type) and not name.startswith('_'):
            setattr(_dynamic_module, name, obj)
            obj.__module__ = '__dynamic__'

# Execute prelude FIRST
exec({prelude}, _globals)

# Add secrets EARLY
_globals.update({secrets})

# Load cached imports SECOND
try:
    with open(imports_cache, 'r') as f:
        cache = json.load(f)
        for imp in cache.get('imports', []):
            try:
                exec(imp, _globals)
            except:
                pass
except FileNotFoundError:
    pass
except Exception as e:
    print(f"Warning: Could not load imports: {{e}}", file=sys.stderr)

# Load and execute previous definitions THIRD (BEFORE loading state!)
definitions_loaded_successfully = False
try:
    with open(definitions_py_cache, 'r') as f:
        definitions_code = f.read()
        if definitions_code.strip():
            # Execute in globals with better error handling
            try:
                exec(definitions_code, _globals)
                # IMMEDIATELY register classes to dynamic module
                register_classes_to_dynamic_module(_globals)
                definitions_loaded_successfully = True
                print(f"Debug: Loaded definitions successfully", file=sys.stderr)
            except Exception as def_exec_error:
                print(f"Error executing definitions: {{def_exec_error}}", file=sys.stderr)
                print(f"Definitions code that failed: {{definitions_code[:200]}}...", file=sys.stderr)
except FileNotFoundError:
    print(f"Debug: No definitions cache file found", file=sys.stderr)
    pass
except Exception as e:
    print(f"Warning: Could not load definitions: {{e}}", file=sys.stderr)

# NOW load previous state FOURTH (after classes are defined and registered)
# But ONLY if definitions loaded successfully, otherwise state loading will fail
if definitions_loaded_successfully or not os.path.exists(definitions_py_cache):
    try:
        with open(state_file, 'rb') as f:
            # Use dill for better dynamic object support
            _saved = dill.load(f)
            print(f"Debug: Successfully loaded {{len(_saved)}} variables from state", file=sys.stderr)
            for k, v in _saved.items():
                _globals[k] = v
                print(f"Debug: Loaded variable '{{k}}' of type {{type(v).__name__}}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Debug: No state file found", file=sys.stderr)
        pass
    except Exception as e:
        print(f"Warning: Could not load state: {{e}}", file=sys.stderr)
        print(f"Debug: Available classes in __dynamic__: {{dir(_dynamic_module)}}", file=sys.stderr)
else:
    print(f"Warning: Skipping state loading because definitions failed to load", file=sys.stderr)

# Parse user code to extract imports and definitions
user_code = {user_code_repr}
_new_imports = []
_new_definitions = []

def extract_definition_source(node, source_lines):
    \"\"\"Extract the complete source code for a definition node.\"\"\"
    # Get the indentation of the definition
    start_line = node.lineno - 1
    if start_line >= len(source_lines):
        return None
        
    # Find the actual start (including decorators)
    actual_start = start_line
    if hasattr(node, 'decorator_list') and node.decorator_list:
        if node.decorator_list[0].lineno > 0:
            actual_start = node.decorator_list[0].lineno - 1
    
    # For the end, we need to find where the next non-indented line starts
    # or use the AST end_lineno if available
    if hasattr(node, 'end_lineno'):
        end_line = node.end_lineno
    else:
        # Fallback: scan for the end of the indented block
        end_line = start_line + 1
        if start_line < len(source_lines):
            base_indent = len(source_lines[start_line]) - len(source_lines[start_line].lstrip())
            while end_line < len(source_lines):
                line = source_lines[end_line]
                if line.strip():  # Non-empty line
                    line_indent = len(line) - len(line.lstrip())
                    if line_indent <= base_indent and not line.lstrip().startswith(('def ', 'class ', '@')):
                        break
                end_line += 1
    
    # Extract the lines
    definition_lines = source_lines[actual_start:end_line]
    return '\\n'.join(definition_lines)

try:
    tree = ast.parse(user_code)
    source_lines = user_code.split('\\n')
    
    # Extract imports
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
    
    # Extract class and function definitions from top level only
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            definition_code = extract_definition_source(node, source_lines)
            if definition_code:
                _new_definitions.append({{
                    'name': node.name,
                    'type': type(node).__name__,
                    'code': definition_code
                }})
                
except SyntaxError as e:
    print(f"Warning: Could not parse code for analysis: {{e}}", file=sys.stderr)
except Exception as e:
    print(f"Warning: Error parsing code: {{e}}", file=sys.stderr)

# Execute user code - THIS IS THE KEY PART
try:
    exec(user_code, _globals)
    # IMMEDIATELY register any new classes with the dynamic module
    register_classes_to_dynamic_module(_globals)
except Exception as e:
    import traceback
    print(traceback.format_exc(), file=sys.stderr)
    # Don't exit on error, continue to save state

# Save new imports
if _new_imports:
    try:
        try:
            with open(imports_cache, 'r') as f:
                cache = json.load(f)
        except:
            cache = {{'imports': []}}
        
        for imp in _new_imports:
            if imp not in cache['imports']:
                cache['imports'].append(imp)
        
        with open(imports_cache, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save imports: {{e}}", file=sys.stderr)

# Save new definitions
if _new_definitions:
    try:
        # Read existing definitions from JSON
        try:
            with open(definitions_cache, 'r') as f:
                existing_definitions = json.load(f)
        except:
            existing_definitions = {{'definitions': []}}
        
        # Update with new definitions (replace if same name exists)
        existing_names = {{d['name']: i for i, d in enumerate(existing_definitions.get('definitions', []))}}
        
        for new_def in _new_definitions:
            if new_def['name'] in existing_names:
                # Replace existing definition
                existing_definitions['definitions'][existing_names[new_def['name']]] = new_def
            else:
                # Add new definition
                existing_definitions['definitions'].append(new_def)
        
        # Save as JSON for tracking
        with open(definitions_cache, 'w') as f:
            json.dump(existing_definitions, f, indent=2)
        
        # Save as executable Python for next run
        with open(definitions_py_cache, 'w') as f:
            for defn in existing_definitions['definitions']:
                f.write(defn['code'] + '\\n\\n')
                
    except Exception as e:
        print(f"Warning: Could not save definitions: {{e}}", file=sys.stderr)

# Save state (all serializable objects including instances)
serializable_globals = {{}}
for key, value in _globals.items():
    # Skip private variables, modules, functions, and classes
    if key.startswith('_'):
        continue
    
    # Skip modules and built-in types
    if isinstance(value, types.ModuleType):
        continue
    
    # Skip functions and classes (they're saved as definitions)
    if isinstance(value, (types.FunctionType, type)):
        continue
    
    # Add all other values (dill handles more types than pickle)
    serializable_globals[key] = value

try:
    with open(state_file, 'wb') as f:
        dill.dump(serializable_globals, f)
except Exception as e:
    print(f"Warning: Could not save state: {{e}}", file=sys.stderr)
"""

        runner_script = runner_script_template.format(
            state_file_path=self.state_file,
            imports_cache_path=self.imports_cache,
            definitions_cache_path=self.definitions_cache,
            definitions_py_cache_path=self.definitions_py_cache,
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
                    "definitions_cache.json",
                    "definitions_cache.py",
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

    def clear_cache(self) -> str:
        """Clear all cached state, definitions, and imports. Start fresh."""
        try:
            files_to_remove = [
                self.state_file,
                self.imports_cache,
                self.definitions_cache,
                self.definitions_py_cache,
            ]
            for file in files_to_remove:
                if os.path.exists(file):
                    os.remove(file)
            return "Cache cleared successfully. Environment reset to initial state."
        except Exception as e:
            return f"Error clearing cache: {e}"

    def get_state_info(self) -> str:
        """Get information about the current state of the environment."""
        info = []

        # Check state file
        if os.path.exists(self.state_file):
            try:
                # Try importing dill first (might not be in main env)
                try:
                    import dill

                    with open(self.state_file, "rb") as f:
                        state = dill.load(f)
                        info.append(f"Variables in state: {list(state.keys())}")
                except ImportError:
                    # Fallback to pickle if dill not available
                    with open(self.state_file, "rb") as f:
                        state = pickle.load(f)
                        info.append(f"Variables in state: {list(state.keys())}")
            except:
                info.append("State file exists but couldn't be read")

        # Check definitions
        if os.path.exists(self.definitions_cache):
            try:
                with open(self.definitions_cache, "r") as f:
                    defs = json.load(f)
                    def_names = [d["name"] for d in defs.get("definitions", [])]
                    info.append(f"Defined classes/functions: {def_names}")
            except:
                info.append("Definitions file exists but couldn't be read")

        # Check imports
        if os.path.exists(self.imports_cache):
            try:
                with open(self.imports_cache, "r") as f:
                    imps = json.load(f)
                    info.append(
                        f"Cached imports: {len(imps.get('imports', []))} statements"
                    )
            except:
                info.append("Imports file exists but couldn't be read")

        return (
            "\n".join(info)
            if info
            else "Environment is in initial state (no cached data)"
        )


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
    Variables, imports, classes, functions, and objects created in one execution persist to the next.

    Args:
        code (str): Python code to execute as a string

    Returns:
        str: Combined stdout and stderr from the code execution

    Examples:
        >>> execute_python_code_in_env("class MyClass: pass")
        >>> execute_python_code_in_env("obj = MyClass()")  # MyClass is still available!
    """
    return isolated_env.execute_python(code)


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


# Import serv module
import serv

isolated_env.add_module(serv)


# List of tools for agent
tools: List[Callable[..., str]] = [
    execute_shell_command_in_env,
    execute_python_code_in_env,
    set_secret_variable_in_env,
    list_available_secrets,
]
