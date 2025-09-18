from typing import List, Callable
from py_exc import VirtualPythonEnvironment


# Create a single, persistent instance
env = VirtualPythonEnvironment()
import serv

env.add_module(serv)


def file_exists(filename: str) -> str:
    """Check if a file exists in the environment.

    Args:
        filename (str): Name of the file to check

    Returns:
        str: Status message indicating if file exists with version and size info
    """
    if env.code_manager.file_exists(filename):
        try:
            registry = env.code_manager._load_registry()
            info = registry.get(filename, {})
            size = len(env._get_file_content_raw(filename))
            return f"File '{filename}' exists (v{info.get('version', 1)}, {size} chars)"
        except:
            return f"File '{filename}' exists"
    else:
        return f"File '{filename}' does not exist"


def create_file(filename: str, code: str, description: str = "") -> str:
    """Create a new file or update existing file with code content.

    Args:
        filename (str): Name of the file to create or update
        code (str): Python code content to write to the file
        description (str, optional): Description of what this code does. Defaults to "".

    Returns:
        str: Status message indicating file creation/update with version info
    """
    return env.code_manager.create_or_update_file(filename, code, description)


def edit_code(filename: str, search_text: str, replace_text: str) -> str:
    """Edit a file by replacing search_text with replace_text.

    Args:
        filename (str): Name of the file to edit
        search_text (str): Text to search for in the file
        replace_text (str): Text to replace the search_text with

    Returns:
        str: Status message indicating success or failure of the edit operation
    """
    return env.code_manager.edit_file(filename, search_text, replace_text)


def read_file(filename: str) -> str:
    """Read the content of a file.

    Args:
        filename (str): Name of the file to read

    Returns:
        str: File content with header, or error message if file doesn't exist
    """
    if not env.code_manager.file_exists(filename):
        return f"File '{filename}' does not exist"

    try:
        content = env._get_file_content_raw(filename)
        return f"Content of {filename}:\n\n{content}"
    except Exception as e:
        return f"Error reading '{filename}': {e}"


def run_code(filename: str) -> str:
    """Execute a Python script file.

    Args:
        filename (str): Name of the Python script file to execute

    Returns:
        str: Execution output including stdout, stderr, or error messages
    """
    return env.run_script(filename)


def list_files() -> str:
    """List all files in the environment.

    Returns:
        str: Formatted list of all files with their sizes, separated by Python and other files
    """
    return env.list_files()


def execute_python(code: str) -> str:
    """Execute Python code in the persistent environment.

    Args:
        code (str): Python code to execute as a string

    Returns:
        str: Execution output including stdout, stderr, and any error messages
    """
    return env.execute_python(code)


def install_package(package_name: str) -> str:
    """Install a Python package in the virtual environment.

    Args:
        package_name (str): Name of the Python package to install (e.g., 'requests', 'pandas')

    Returns:
        str: Installation result message including success/failure and any output
    """
    return env.install_package(package_name)


def set_secret(key: str, value: str) -> str:
    """Store a secret variable that will be available in code execution.

    Args:
        key (str): Variable name for the secret (must be valid Python identifier)
        value (str): The secret value to store (e.g., API key, password)

    Returns:
        str: Confirmation message or error if key is not a valid identifier
    """
    if not key.isidentifier():
        return f"'{key}' is not a valid Python identifier"
    env._secrets[key] = value
    return f"Secret '{key}' has been set"


def list_secrets() -> str:
    """List the names of all stored secret variables.

    Returns:
        str: List of secret variable names, or message if no secrets are set
    """
    if not env._secrets:
        return "No secrets are currently set"
    return f"Available secrets: {list(env._secrets.keys())}"


# Tools for agents - one function per job
TOOLS: List[Callable[..., str]] = [
    file_exists,
    create_file,
    read_file,
    edit_code,
    run_code,
    list_files,
    execute_python,
    install_package,
    set_secret,
    list_secrets,
]
