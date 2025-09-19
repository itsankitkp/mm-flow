import uuid
import json
from typing import Dict, List, Optional, Protocol
from dataclasses import dataclass, asdict
from datetime import datetime
from abc import ABC, abstractmethod
from langchain_core.tools import tool


@dataclass
class Todo:
    """Todo item data structure"""

    id: str
    title: str
    description: str = ""
    completed: bool = False
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class TodoStorage(ABC):
    """Abstract base class for todo storage backends"""

    @abstractmethod
    def save_todo(self, todo: Todo) -> None:
        pass

    @abstractmethod
    def get_todo(self, todo_id: str) -> Optional[Todo]:
        pass

    @abstractmethod
    def list_todos(self) -> List[Todo]:
        pass

    @abstractmethod
    def delete_todo(self, todo_id: str) -> bool:
        pass

    @abstractmethod
    def update_todo(self, todo: Todo) -> None:
        pass


class InMemoryTodoStorage(TodoStorage):
    """In-memory storage for todos (default)"""

    def __init__(self):
        self._todos: Dict[str, Todo] = {}

    def save_todo(self, todo: Todo) -> None:
        self._todos[todo.id] = todo

    def get_todo(self, todo_id: str) -> Optional[Todo]:
        return self._todos.get(todo_id)

    def list_todos(self) -> List[Todo]:
        return list(self._todos.values())

    def delete_todo(self, todo_id: str) -> bool:
        if todo_id in self._todos:
            del self._todos[todo_id]
            return True
        return False

    def update_todo(self, todo: Todo) -> None:
        self._todos[todo.id] = todo


class FileTodoStorage(TodoStorage):
    """File-based storage for todos"""

    def __init__(self, file_path: str = "todos.json"):
        self.file_path = file_path
        self._load_todos()

    def _load_todos(self):
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                self._todos = {
                    todo_id: Todo(**todo_data) for todo_id, todo_data in data.items()
                }
        except (FileNotFoundError, json.JSONDecodeError):
            self._todos = {}

    def _save_todos(self):
        with open(self.file_path, "w") as f:
            data = {todo_id: asdict(todo) for todo_id, todo in self._todos.items()}
            json.dump(data, f, indent=2)

    def save_todo(self, todo: Todo) -> None:
        self._todos[todo.id] = todo
        self._save_todos()

    def get_todo(self, todo_id: str) -> Optional[Todo]:
        return self._todos.get(todo_id)

    def list_todos(self) -> List[Todo]:
        return list(self._todos.values())

    def delete_todo(self, todo_id: str) -> bool:
        if todo_id in self._todos:
            del self._todos[todo_id]
            self._save_todos()
            return True
        return False

    def update_todo(self, todo: Todo) -> None:
        self._todos[todo.id] = todo
        self._save_todos()


class TodoManager:
    """Central manager for todo operations"""

    def __init__(self, storage: Optional[TodoStorage] = None):
        self.storage = storage or InMemoryTodoStorage()

    def create_todo(self, title: str, description: str = "") -> Todo:
        todo_id = str(uuid.uuid4())[:8]  # Short UUID
        todo = Todo(id=todo_id, title=title, description=description)
        self.storage.save_todo(todo)
        return todo

    def get_todo(self, todo_id: str) -> Optional[Todo]:
        return self.storage.get_todo(todo_id)

    def list_todos(self) -> List[Todo]:
        return self.storage.list_todos()

    def delete_todo(self, todo_id: str) -> bool:
        return self.storage.delete_todo(todo_id)

    def mark_completed(self, todo_id: str, completed: bool = True) -> Optional[Todo]:
        todo = self.storage.get_todo(todo_id)
        if todo:
            todo.completed = completed
            self.storage.update_todo(todo)
            return todo
        return None


# Global todo manager instance (can be configured)
_todo_manager = None


def get_todo_manager() -> TodoManager:
    """Get the global todo manager instance"""
    global _todo_manager
    if _todo_manager is None:
        _todo_manager = TodoManager()
    return _todo_manager


def configure_todo_storage(storage: TodoStorage):
    """Configure the global todo storage backend"""
    global _todo_manager
    _todo_manager = TodoManager(storage)


def create_todo(title: str, description: str = "") -> str:
    """Create a new todo item with title and optional description.

    Args:
        title: The title/summary of the todo task
        description: Optional detailed description of the task

    Returns:
        Success message with the created todo ID
    """
    manager = get_todo_manager()
    todo = manager.create_todo(title, description)
    return f"✅ Created todo '{title}' with ID: {todo.id}"


def list_todos() -> str:
    """List all todos with their current status.

    Returns:
        Formatted list of all todos showing ID, title, status, and description
    """
    manager = get_todo_manager()
    todos = manager.list_todos()

    if not todos:
        return "📝 No todos found. Create some todos to get started!"

    todo_list = []
    for todo in todos:
        status = "✅ DONE" if todo.completed else "⏳ TODO"
        desc = f" - {todo.description}" if todo.description else ""
        todo_list.append(f"[{todo.id}] {status} {todo.title}{desc}")

    return "📋 **Current Todos:**\n" + "\n".join(todo_list)


def delete_todo(todo_id: str) -> str:
    """Delete a todo by its ID.

    Args:
        todo_id: The ID of the todo to delete

    Returns:
        Success or error message
    """
    manager = get_todo_manager()
    todo = manager.get_todo(todo_id)

    if not todo:
        return f"❌ Todo with ID '{todo_id}' not found"

    success = manager.delete_todo(todo_id)
    if success:
        return f"🗑️ Deleted todo: '{todo.title}'"
    else:
        return f"❌ Failed to delete todo with ID '{todo_id}'"


def check_todo(todo_id: str) -> str:
    """Mark a todo as completed.

    Args:
        todo_id: The ID of the todo to mark as completed

    Returns:
        Success or error message
    """
    manager = get_todo_manager()
    todo = manager.mark_completed(todo_id, True)

    if todo:
        return f"✅ Marked '{todo.title}' as completed!"
    else:
        return f"❌ Todo with ID '{todo_id}' not found"


def uncheck_todo(todo_id: str) -> str:
    """Mark a todo as not completed.

    Args:
        todo_id: The ID of the todo to mark as not completed

    Returns:
        Success or error message
    """
    manager = get_todo_manager()
    todo = manager.mark_completed(todo_id, False)

    if todo:
        return f"⏳ Marked '{todo.title}' as not completed"
    else:
        return f"❌ Todo with ID '{todo_id}' not found"


TODO_TOOLS = [
    create_todo,
    list_todos,
    delete_todo,
    check_todo,
    uncheck_todo,
]
