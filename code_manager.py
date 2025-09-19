import os
import json
import difflib
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class CodeManager:
    """
    Enhanced code management with file tracking, editing, and error recovery.
    """

    def __init__(self, workdir: str):
        self.workdir = workdir
        self.code_files_registry = os.path.join(workdir, "code_registry.json")
        self.ensure_registry()

    def ensure_registry(self):
        """Initialize or load the code files registry."""
        if not os.path.exists(self.code_files_registry):
            self._save_registry({})

    def _load_registry(self) -> Dict:
        """Load the code files registry."""
        try:
            with open(self.code_files_registry, "r") as f:
                return json.load(f)
        except:
            return {}

    def _save_registry(self, registry: Dict):
        """Save the code files registry."""
        with open(self.code_files_registry, "w") as f:
            json.dump(registry, f, indent=2)

    def _resolve_path(self, filename: str) -> str:
        """Resolve filename to full path within workdir."""
        safe_path = os.path.normpath(os.path.join(self.workdir, filename))
        if not safe_path.startswith(self.workdir):
            raise ValueError("File path must be within the working directory.")
        return safe_path

    def file_exists(self, filename: str) -> bool:
        """Check if a code file exists."""
        try:
            full_path = self._resolve_path(filename)
            return os.path.exists(full_path)
        except:
            return False

    def create_or_update_file(
        self, filename: str, code: str, description: str = ""
    ) -> str:
        """
        Create new file or update existing one. Tracks changes and provides diff.
        """
        full_path = self._resolve_path(filename)
        registry = self._load_registry()

        # Check if file exists
        file_existed = os.path.exists(full_path)
        old_content = ""

        if file_existed:
            with open(full_path, "r") as f:
                old_content = f.read()

        # Write new content
        with open(full_path, "w") as f:
            f.write(code)

        # Update registry
        if filename not in registry:
            registry[filename] = {
                "created_at": str(Path(full_path).stat().st_mtime),
                "description": description,
                "version": 1,
                "edit_history": [],
            }
        else:
            registry[filename]["version"] += 1
            if file_existed and old_content != code:
                # Generate diff for significant changes
                diff = list(
                    difflib.unified_diff(
                        old_content.splitlines(keepends=True),
                        code.splitlines(keepends=True),
                        fromfile=f"{filename} (old)",
                        tofile=f"{filename} (new)",
                    )
                )
                registry[filename]["edit_history"].append(
                    {
                        "version": registry[filename]["version"],
                        "diff_lines": len(
                            [
                                line
                                for line in diff
                                if line.startswith(("+", "-"))
                                and not line.startswith(("+++", "---"))
                            ]
                        ),
                        "timestamp": str(Path(full_path).stat().st_mtime),
                    }
                )

        self._save_registry(registry)

        if file_existed:
            return f"✅ Updated '{filename}' (v{registry[filename]['version']})"
        else:
            return (
                f"✅ Created new file '{filename}' (v{registry[filename]['version']})"
            )

    def edit_file(self, filename: str, search_text: str, replace_text: str) -> str:
        """
        Edit a specific part of a file by searching and replacing text.
        """
        if not self.file_exists(filename):
            return f"❌ File '{filename}' does not exist. Use create_or_update_file instead."

        full_path = self._resolve_path(filename)
        with open(full_path, "r") as f:
            content = f.read()

        if search_text not in content:
            return f"❌ Search text not found in '{filename}'. No changes made."

        new_content = content.replace(search_text, replace_text)

        # Update file
        return self.create_or_update_file(
            filename, new_content, f"Edited: replaced text"
        )

    def fix_code_error(self, filename: str, error_message: str, fixed_code: str) -> str:
        """
        Fix code errors by replacing the entire file content with corrected version.
        """
        if not self.file_exists(filename):
            return f"❌ File '{filename}' does not exist."

        description = f"Error fix: {error_message[:100]}..."
        return self.create_or_update_file(filename, fixed_code, description)

    def list_all_code_files(self) -> str:
        """
        List all tracked code files with their details.
        """
        registry = self._load_registry()

        if not registry:
            return "📝 No code files have been created yet."

        result = ["📂 **Generated Code Files:**\n"]

        for filename, info in registry.items():
            if self.file_exists(filename):
                full_path = self._resolve_path(filename)
                file_size = os.path.getsize(full_path)

                result.append(f"📄 **{filename}**")
                result.append(f"   • Version: {info['version']}")
                result.append(f"   • Size: {file_size} bytes")
                result.append(
                    f"   • Description: {info.get('description', 'No description')}"
                )

                if info.get("edit_history"):
                    result.append(f"   • Edits: {len(info['edit_history'])} revisions")

                result.append("")

        return "\n".join(result)

    def get_file_content(self, filename: str) -> str:
        """
        Get the content of a specific file.
        """
        if not self.file_exists(filename):
            return f"❌ File '{filename}' does not exist."

        full_path = self._resolve_path(filename)
        try:
            with open(full_path, "r") as f:
                content = f.read()
            return f"📄 **Content of {filename}:**\n```python\n{content}\n```"
        except Exception as e:
            return f"❌ Error reading '{filename}': {e}"

    def get_file_history(self, filename: str) -> str:
        """
        Get the edit history of a specific file.
        """
        registry = self._load_registry()

        if filename not in registry:
            return f"❌ File '{filename}' not found in registry."

        info = registry[filename]
        result = [f"📋 **Edit History for {filename}:**"]
        result.append(f"Current Version: {info['version']}")

        if info.get("edit_history"):
            for edit in info["edit_history"]:
                result.append(
                    f"  • v{edit['version']}: {edit['diff_lines']} lines changed"
                )
        else:
            result.append("  • No edit history (file created once)")

        return "\n".join(result)

    def backup_file(self, filename: str) -> str:
        """
        Create a backup of a file before major changes.
        """
        if not self.file_exists(filename):
            return f"❌ File '{filename}' does not exist."

        backup_name = f"{filename}.backup"
        full_path = self._resolve_path(filename)
        backup_path = self._resolve_path(backup_name)

        try:
            with open(full_path, "r") as src, open(backup_path, "w") as dst:
                dst.write(src.read())
            return f"✅ Backup created: '{backup_name}'"
        except Exception as e:
            return f"❌ Error creating backup: {e}"
