"""
Filesystem tools for Pori agents.

Provides safe file and directory operations with built-in security measures.
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import mimetypes
import hashlib

from pydantic import BaseModel, Field, validator
from ..tools import tool_registry

Registry = tool_registry()
logger = logging.getLogger("pori.filesystem_tools")

# Security configuration
DEFAULT_ALLOWED_EXTENSIONS = {
    '.txt', '.md', '.json', '.csv', '.log', '.py', '.js', '.html', '.css',
    '.xml', '.yml', '.yaml', '.ini', '.cfg', '.conf', '.gitignore'
}

DEFAULT_FORBIDDEN_DIRS = {
    'System32', 'Windows', 'Program Files', 'Program Files (x86)',
    '.git', 'node_modules', '__pycache__', '.env'
}

class FilesystemConfig:
    """Configuration for filesystem operations."""
    
    def __init__(self):
        # Get user's home and current working directory for safe operations
        self.home_dir = Path.home()
        self.current_dir = Path.cwd()
        self.allowed_extensions = DEFAULT_ALLOWED_EXTENSIONS.copy()
        self.forbidden_dirs = DEFAULT_FORBIDDEN_DIRS.copy()
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit
        self.max_files_per_operation = 100

    def is_path_safe(self, path: Union[str, Path]) -> bool:
        """Check if a path is safe to access."""
        try:
            path = Path(path).resolve()
            
            # Check if path is within allowed areas (home or current project)
            home_relative = self.home_dir in path.parents or path == self.home_dir
            current_relative = self.current_dir in path.parents or path == self.current_dir
            
            if not (home_relative or current_relative):
                return False
            
            # Check forbidden directory names
            for part in path.parts:
                if part in self.forbidden_dirs:
                    return False
            
            return True
        except (OSError, ValueError):
            return False

    def is_extension_allowed(self, path: Union[str, Path]) -> bool:
        """Check if file extension is allowed."""
        extension = Path(path).suffix.lower()
        return extension in self.allowed_extensions or not extension

# Global config instance
fs_config = FilesystemConfig()

# ============= Parameter Models =============

class ReadFileParams(BaseModel):
    file_path: str = Field(..., description="Path to the file to read")
    encoding: str = Field("utf-8", description="File encoding (default: utf-8)")
    max_lines: Optional[int] = Field(None, description="Maximum number of lines to read")

class WriteFileParams(BaseModel):
    file_path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write to the file")
    encoding: str = Field("utf-8", description="File encoding (default: utf-8)")
    append: bool = Field(False, description="Append to file instead of overwriting")

class ListDirectoryParams(BaseModel):
    directory_path: str = Field(".", description="Directory path to list (default: current directory)")
    show_hidden: bool = Field(False, description="Show hidden files and directories")
    recursive: bool = Field(False, description="List recursively")
    max_depth: int = Field(3, description="Maximum recursion depth")

class FileInfoParams(BaseModel):
    file_path: str = Field(..., description="Path to the file or directory")

class CreateDirectoryParams(BaseModel):
    directory_path: str = Field(..., description="Path of the directory to create")
    parents: bool = Field(True, description="Create parent directories if they don't exist")

class DeleteParams(BaseModel):
    path: str = Field(..., description="Path to the file or directory to delete")
    recursive: bool = Field(False, description="Delete directories recursively")

class CopyParams(BaseModel):
    source_path: str = Field(..., description="Source file or directory path")
    destination_path: str = Field(..., description="Destination path")

class MoveParams(BaseModel):
    source_path: str = Field(..., description="Source file or directory path")
    destination_path: str = Field(..., description="Destination path")

class SearchFilesParams(BaseModel):
    directory_path: str = Field(".", description="Directory to search in")
    pattern: str = Field("*", description="File name pattern (supports wildcards)")
    content_search: Optional[str] = Field(None, description="Search for content within files")
    max_results: int = Field(50, description="Maximum number of results")

class ExecuteScriptParams(BaseModel):
    script_path: str = Field(..., description="Path to the script file to execute")
    args: List[str] = Field(default_factory=list, description="Arguments to pass to the script")
    working_dir: Optional[str] = Field(None, description="Working directory for execution")

# ============= Helper Functions =============

def safe_path_operation(func):
    """Decorator to ensure path operations are safe."""
    def wrapper(params, context: Dict[str, Any]):
        try:
            # Extract path from params
            if hasattr(params, 'file_path'):
                path = params.file_path
            elif hasattr(params, 'directory_path'):
                path = params.directory_path
            elif hasattr(params, 'path'):
                path = params.path
            elif hasattr(params, 'source_path'):
                # For operations with multiple paths, check all
                if not fs_config.is_path_safe(params.source_path):
                    return {"success": False, "error": f"Source path not allowed: {params.source_path}"}
                if hasattr(params, 'destination_path') and not fs_config.is_path_safe(params.destination_path):
                    return {"success": False, "error": f"Destination path not allowed: {params.destination_path}"}
                path = params.source_path
            else:
                return {"success": False, "error": "No valid path found in parameters"}

            if not fs_config.is_path_safe(path):
                return {"success": False, "error": f"Path not allowed: {path}"}

            return func(params, context)
        except Exception as e:
            logger.error(f"Path operation failed: {e}")
            return {"success": False, "error": str(e)}
    return wrapper

def get_file_info(path: Path) -> Dict[str, Any]:
    """Get comprehensive file information."""
    try:
        stat = path.stat()
        return {
            "name": path.name,
            "path": str(path.absolute()),
            "size": stat.st_size,
            "size_human": format_bytes(stat.st_size),
            "type": "directory" if path.is_dir() else "file",
            "extension": path.suffix.lower() if path.is_file() else None,
            "mime_type": mimetypes.guess_type(str(path))[0] if path.is_file() else None,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "readable": os.access(path, os.R_OK),
            "writable": os.access(path, os.W_OK),
            "executable": os.access(path, os.X_OK),
        }
    except Exception as e:
        return {"error": str(e)}

def format_bytes(bytes_count: int) -> str:
    """Format bytes in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f}{unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f}PB"

# ============= Tool Implementations =============

@Registry.tool(name="read_file", param_model=ReadFileParams, description="Read the contents of a text file")
@safe_path_operation
def read_file_tool(params: ReadFileParams, context: Dict[str, Any]):
    """Read file contents with safety checks."""
    try:
        path = Path(params.file_path)
        
        if not path.exists():
            return {"success": False, "error": f"File does not exist: {path}"}
        
        if not path.is_file():
            return {"success": False, "error": f"Path is not a file: {path}"}
        
        if not fs_config.is_extension_allowed(path):
            return {"success": False, "error": f"File extension not allowed: {path.suffix}"}
        
        # Check file size
        if path.stat().st_size > fs_config.max_file_size:
            return {"success": False, "error": f"File too large (max {format_bytes(fs_config.max_file_size)})"}
        
        # Read file
        with open(path, 'r', encoding=params.encoding) as f:
            if params.max_lines:
                lines = []
                for i, line in enumerate(f):
                    if i >= params.max_lines:
                        break
                    lines.append(line.rstrip('\n\r'))
                content = '\n'.join(lines)
                truncated = i >= params.max_lines - 1
            else:
                content = f.read()
                truncated = False
        
        result = {
            "content": content,
            "file_info": get_file_info(path),
            "truncated": truncated
        }
        
        if truncated:
            result["message"] = f"Content truncated to {params.max_lines} lines"
        
        logger.info(f"Read file: {path} ({format_bytes(len(content.encode()))})")
        return result
        
    except UnicodeDecodeError:
        return {"success": False, "error": f"Cannot decode file with encoding {params.encoding}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@Registry.tool(name="write_file", param_model=WriteFileParams, description="Write content to a file")
@safe_path_operation
def write_file_tool(params: WriteFileParams, context: Dict[str, Any]):
    """Write content to a file with safety checks."""
    try:
        path = Path(params.file_path)
        
        if not fs_config.is_extension_allowed(path):
            return {"success": False, "error": f"File extension not allowed: {path.suffix}"}
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        mode = 'a' if params.append else 'w'
        with open(path, mode, encoding=params.encoding) as f:
            f.write(params.content)
        
        file_info = get_file_info(path)
        action = "Appended to" if params.append else "Wrote"
        
        logger.info(f"{action} file: {path} ({format_bytes(len(params.content.encode()))})")
        return {
            "message": f"{action} file successfully",
            "file_info": file_info,
            "bytes_written": len(params.content.encode())
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@Registry.tool(name="list_directory", param_model=ListDirectoryParams, description="List files and directories")
@safe_path_operation
def list_directory_tool(params: ListDirectoryParams, context: Dict[str, Any]):
    """List directory contents."""
    try:
        path = Path(params.directory_path)
        
        if not path.exists():
            return {"success": False, "error": f"Directory does not exist: {path}"}
        
        if not path.is_dir():
            return {"success": False, "error": f"Path is not a directory: {path}"}
        
        items = []
        
        def scan_directory(dir_path: Path, current_depth: int = 0):
            if current_depth > params.max_depth:
                return
            
            try:
                for item in sorted(dir_path.iterdir()):
                    # Skip hidden files unless requested
                    if not params.show_hidden and item.name.startswith('.'):
                        continue
                    
                    # Security check
                    if not fs_config.is_path_safe(item):
                        continue
                    
                    item_info = get_file_info(item)
                    item_info["depth"] = current_depth
                    item_info["relative_path"] = str(item.relative_to(path))
                    items.append(item_info)
                    
                    # Recurse into subdirectories
                    if params.recursive and item.is_dir() and current_depth < params.max_depth:
                        scan_directory(item, current_depth + 1)
                        
            except PermissionError:
                # Skip directories we can't access
                pass
        
        scan_directory(path)
        
        # Summary statistics
        files = [item for item in items if item["type"] == "file"]
        directories = [item for item in items if item["type"] == "directory"]
        total_size = sum(item["size"] for item in files)
        
        logger.info(f"Listed directory: {path} ({len(files)} files, {len(directories)} directories)")
        
        return {
            "directory": str(path.absolute()),
            "items": items,
            "summary": {
                "total_items": len(items),
                "files": len(files),
                "directories": len(directories),
                "total_size": total_size,
                "total_size_human": format_bytes(total_size)
            }
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@Registry.tool(name="file_info", param_model=FileInfoParams, description="Get detailed information about a file or directory")
@safe_path_operation
def file_info_tool(params: FileInfoParams, context: Dict[str, Any]):
    """Get file or directory information."""
    try:
        path = Path(params.file_path)
        
        if not path.exists():
            return {"success": False, "error": f"Path does not exist: {path}"}
        
        info = get_file_info(path)
        
        # Add additional info for files
        if path.is_file() and path.stat().st_size > 0:
            try:
                # Try to get file hash for small files
                if path.stat().st_size < 1024 * 1024:  # 1MB
                    with open(path, 'rb') as f:
                        content = f.read()
                        info["md5_hash"] = hashlib.md5(content).hexdigest()
                        info["sha256_hash"] = hashlib.sha256(content).hexdigest()
                
                # Try to detect if it's a text file
                if fs_config.is_extension_allowed(path):
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        sample = f.read(1000)
                        info["is_text"] = True
                        info["line_count"] = sample.count('\n') + 1
                        info["sample_content"] = sample[:200] + "..." if len(sample) > 200 else sample
                        
            except Exception:
                # If we can't read the file, that's okay
                pass
        
        logger.info(f"Got file info: {path}")
        return info
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@Registry.tool(name="create_directory", param_model=CreateDirectoryParams, description="Create a new directory")
@safe_path_operation
def create_directory_tool(params: CreateDirectoryParams, context: Dict[str, Any]):
    """Create a directory."""
    try:
        path = Path(params.directory_path)
        
        if path.exists():
            return {"success": False, "error": f"Path already exists: {path}"}
        
        path.mkdir(parents=params.parents, exist_ok=False)
        
        logger.info(f"Created directory: {path}")
        return {
            "message": f"Directory created successfully: {path}",
            "directory_info": get_file_info(path)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@Registry.tool(name="search_files", param_model=SearchFilesParams, description="Search for files by name pattern or content")
@safe_path_operation
def search_files_tool(params: SearchFilesParams, context: Dict[str, Any]):
    """Search for files matching criteria."""
    try:
        path = Path(params.directory_path)
        
        if not path.exists():
            return {"success": False, "error": f"Directory does not exist: {path}"}
        
        if not path.is_dir():
            return {"success": False, "error": f"Path is not a directory: {path}"}
        
        results = []
        
        # Use glob for pattern matching
        pattern = params.pattern
        if not pattern.startswith('*') and not pattern.startswith('?'):
            pattern = f"*{pattern}*"  # Make it more flexible
        
        for match in path.rglob(pattern):
            if not fs_config.is_path_safe(match):
                continue
            
            if len(results) >= params.max_results:
                break
            
            result_item = get_file_info(match)
            result_item["relative_path"] = str(match.relative_to(path))
            
            # Content search for text files
            if params.content_search and match.is_file() and fs_config.is_extension_allowed(match):
                try:
                    with open(match, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if params.content_search.lower() in content.lower():
                            # Find matching lines
                            lines = content.split('\n')
                            matching_lines = [
                                (i+1, line.strip()) for i, line in enumerate(lines)
                                if params.content_search.lower() in line.lower()
                            ]
                            result_item["content_matches"] = matching_lines[:5]  # First 5 matches
                            results.append(result_item)
                except Exception:
                    # Skip files we can't read
                    continue
            else:
                results.append(result_item)
        
        logger.info(f"Search completed: {len(results)} results for pattern '{params.pattern}'")
        
        return {
            "search_params": {
                "directory": str(path.absolute()),
                "pattern": params.pattern,
                "content_search": params.content_search
            },
            "results": results,
            "total_found": len(results),
            "truncated": len(results) >= params.max_results
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@Registry.tool(name="copy_file", param_model=CopyParams, description="Copy a file or directory to another location")
@safe_path_operation
def copy_file_tool(params: CopyParams, context: Dict[str, Any]):
    """Copy files or directories."""
    try:
        source = Path(params.source_path)
        destination = Path(params.destination_path)
        
        if not source.exists():
            return {"success": False, "error": f"Source does not exist: {source}"}
        
        # If destination is a directory, copy into it
        if destination.is_dir():
            destination = destination / source.name
        
        if destination.exists():
            return {"success": False, "error": f"Destination already exists: {destination}"}
        
        # Create parent directories if needed
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        if source.is_file():
            shutil.copy2(source, destination)
            operation = "File copied"
        else:
            shutil.copytree(source, destination)
            operation = "Directory copied"
        
        logger.info(f"Copied {source} to {destination}")
        return {
            "message": f"{operation} successfully",
            "source_info": get_file_info(source),
            "destination_info": get_file_info(destination)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@Registry.tool(name="move_file", param_model=MoveParams, description="Move or rename a file or directory")
@safe_path_operation
def move_file_tool(params: MoveParams, context: Dict[str, Any]):
    """Move or rename files or directories."""
    try:
        source = Path(params.source_path)
        destination = Path(params.destination_path)
        
        if not source.exists():
            return {"success": False, "error": f"Source does not exist: {source}"}
        
        # If destination is a directory, move into it
        if destination.is_dir():
            destination = destination / source.name
        
        if destination.exists():
            return {"success": False, "error": f"Destination already exists: {destination}"}
        
        # Create parent directories if needed
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Get source info before moving
        source_info = get_file_info(source)
        source_type = "File" if source.is_file() else "Directory"
        
        shutil.move(str(source), str(destination))
        
        logger.info(f"Moved {source} to {destination}")
        return {
            "message": f"{source_type} moved successfully",
            "old_path": str(source),
            "new_path": str(destination),
            "destination_info": get_file_info(destination)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@Registry.tool(name="delete_file", param_model=DeleteParams, description="Delete a file or directory (use with caution!)")
@safe_path_operation
def delete_file_tool(params: DeleteParams, context: Dict[str, Any]):
    """Delete files or directories with safety checks."""
    try:
        path = Path(params.path)
        
        if not path.exists():
            return {"success": False, "error": f"Path does not exist: {path}"}
        
        # Additional safety check - prevent deletion of important directories
        dangerous_paths = {'/', 'C:\\', str(Path.home()), '.'}
        if str(path.resolve()) in dangerous_paths:
            return {"success": False, "error": f"Cannot delete system path: {path}"}
        
        # Get info before deletion
        file_info = get_file_info(path)
        
        if path.is_file():
            path.unlink()
            operation = "File deleted"
        elif path.is_dir():
            if not params.recursive:
                # Only delete if empty
                try:
                    path.rmdir()
                    operation = "Empty directory deleted"
                except OSError:
                    return {"success": False, "error": f"Directory not empty. Use recursive=True to delete non-empty directories"}
            else:
                shutil.rmtree(path)
                operation = "Directory deleted recursively"
        
        logger.warning(f"Deleted: {path} ({file_info.get('type', 'unknown')})")
        return {
            "message": f"{operation} successfully",
            "deleted_info": file_info
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def register_filesystem_tools(registry=None):
    """Register all filesystem tools. Tools auto-register on import."""
    logger.info(f"Filesystem tools registered: {len([t for t in Registry.tools.keys() if any(name in t for name in ['read_file', 'write_file', 'list_directory'])])} tools")
    return None