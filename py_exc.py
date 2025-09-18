import inspect
import os
import pickle
import subprocess
import sys
import tempfile
import shutil
import json
from pathlib import Path
from code_manager import CodeManager


class VirtualPythonEnvironment:
    """Python execution environment with code management."""

    def __init__(self):
        self.workdir = tempfile.mkdtemp()
        self.venv_path = os.path.join(self.workdir, "venv")
        self._secrets = {}

        self.code_manager = CodeManager(self.workdir)

        self.modules_dir = os.path.join(self.workdir, "injected_modules")
        os.makedirs(self.modules_dir, exist_ok=True)

        self._setup_virtual_environment()

        self.state_file = os.path.join(self.workdir, "session_state.pkl")
        self.imports_cache = os.path.join(self.workdir, "imports_cache.json")

    def _setup_virtual_environment(self):
        """Set up the virtual environment and install basic packages."""
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", self.venv_path],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create virtual environment: {e.stderr}")

        if sys.platform == "win32":
            self.python_executable = os.path.join(
                self.venv_path, "Scripts", "python.exe"
            )
            self.pip_executable = os.path.join(self.venv_path, "Scripts", "pip.exe")
        else:
            self.python_executable = os.path.join(self.venv_path, "bin", "python")
            self.pip_executable = os.path.join(self.venv_path, "bin", "pip")

        try:
            subprocess.run(
                [self.pip_executable, "install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
                text=True,
            )
            common_packages = ["requests", "pandas", "numpy", "beautifulsoup4", "lxml"]
            subprocess.run(
                [self.pip_executable, "install"] + common_packages,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(self.workdir, ignore_errors=True)
            raise RuntimeError(f"Failed to install packages: {e.stderr}")

    def __del__(self) -> None:
        """Ensures the entire temporary directory (including the venv) is cleaned up."""
        shutil.rmtree(self.workdir, ignore_errors=True)

    def _get_file_content_raw(self, filename: str) -> str:
        """Get raw file content without formatting."""
        full_path = self.code_manager._resolve_path(filename)
        with open(full_path, "r") as f:
            return f.read()

    def install_package(self, package_name: str) -> str:
        """Install a Python package in the virtual environment."""
        try:
            result = subprocess.run(
                [self.pip_executable, "install", package_name],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.workdir,
            )
            return f"Successfully installed '{package_name}'\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Failed to install '{package_name}': {e.stderr}"

    def execute_python(self, code: str) -> str:
        """Execute Python code in the persistent environment."""
        prelude_code = f"""
import sys
sys.path.insert(0, r'{self.modules_dir}')
sys.path.insert(0, r'{self.workdir}')

import pandas as pd
import numpy as np
import requests
import os
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
import traceback

state_file = r'{state_file_path}'
imports_cache = r'{imports_cache_path}'
_globals = {{}}

try:
    with open(state_file, 'rb') as f:
        _saved = pickle.load(f)
        for k, v in _saved.items():
            if not isinstance(v, (types.ModuleType, types.FunctionType, type)):
                _globals[k] = v
except:
    pass

_globals.update({secrets})

exec({prelude}, _globals)

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

try:
    exec(user_code, _globals)
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}")
    traceback.print_exc()

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
    print(f"Warning: Could not save state: {{e}}")
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
                cwd=self.workdir,
                env=env,
                timeout=60,
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"

            return output if output else "Execution completed with no output"

        except subprocess.TimeoutExpired:
            return "Code execution timed out after 60 seconds"
        except subprocess.CalledProcessError as e:
            return f"Execution failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"

    def run_script(self, filename: str) -> str:
        """Execute a Python script file."""
        try:
            full_path = self.code_manager._resolve_path(filename)
            if not os.path.exists(full_path):
                return f"File '{filename}' not found"

            with open(full_path, "r") as f:
                script_code = f.read()
            return self.execute_python(script_code)
        except Exception as e:
            return f"Error running script '{filename}': {e}"

    def list_files(self) -> str:
        """List all files in the environment."""
        try:
            all_files = [
                f
                for f in os.listdir(self.workdir)
                if f
                not in [
                    "venv",
                    "injected_modules",
                    "session_state.pkl",
                    "imports_cache.json",
                    "code_registry.json",
                ]
            ]

            if not all_files:
                return "The working directory is empty"

            py_files = [f for f in all_files if f.endswith(".py")]
            other_files = [f for f in all_files if not f.endswith(".py")]

            result = []
            if py_files:
                result.append("Python files:")
                for f in py_files:
                    size = os.path.getsize(os.path.join(self.workdir, f))
                    result.append(f"  {f} ({size} bytes)")

            if other_files:
                result.append("Other files:")
                for f in other_files:
                    size = os.path.getsize(os.path.join(self.workdir, f))
                    result.append(f"  {f} ({size} bytes)")

            return "\n".join(result)
        except Exception as e:
            return f"Error listing files: {e}"

    def add_module(self, module) -> str:
        """
        Inject a Python module into the virtual environment.

        Args:
            module: A Python module object or path to a .py file

        Returns:
            str: Success message indicating module was injected
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
