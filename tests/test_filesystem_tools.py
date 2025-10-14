"""
Tests for filesystem tools.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock

from pori.tools_builtin.filesystem_tools import (
    read_file_tool, write_file_tool, list_directory_tool, 
    file_info_tool, create_directory_tool, search_files_tool,
    ReadFileParams, WriteFileParams, ListDirectoryParams,
    FileInfoParams, CreateDirectoryParams, SearchFilesParams,
    fs_config
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_files(temp_dir):
    """Create sample files for testing."""
    # Create some test files
    (temp_dir / "test1.txt").write_text("Hello world!", encoding="utf-8")
    (temp_dir / "test2.py").write_text("print('Python file')", encoding="utf-8")
    (temp_dir / "data.json").write_text('{"key": "value"}', encoding="utf-8")
    
    # Create a subdirectory
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("Nested file content", encoding="utf-8")
    
    return temp_dir


class TestFilesystemTools:
    """Test filesystem tool functionality."""

    def test_read_file_success(self, sample_files):
        """Test successful file reading."""
        test_file = sample_files / "test1.txt"
        params = ReadFileParams(file_path=str(test_file))
        context = {}
        
        # Temporarily allow this path
        original_is_safe = fs_config.is_path_safe
        fs_config.is_path_safe = lambda x: True
        
        try:
            result = read_file_tool(params, context)
            assert result["content"] == "Hello world!"
            assert result["file_info"]["name"] == "test1.txt"
            assert not result["truncated"]
        finally:
            fs_config.is_path_safe = original_is_safe

    def test_read_file_with_max_lines(self, sample_files):
        """Test file reading with line limit."""
        # Create a multi-line file
        test_file = sample_files / "multiline.txt"
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        test_file.write_text(content, encoding="utf-8")
        
        params = ReadFileParams(file_path=str(test_file), max_lines=3)
        context = {}
        
        # Temporarily allow this path
        original_is_safe = fs_config.is_path_safe
        fs_config.is_path_safe = lambda x: True
        
        try:
            result = read_file_tool(params, context)
            lines = result["content"].split("\n")
            assert len(lines) == 3
            assert result["truncated"]
        finally:
            fs_config.is_path_safe = original_is_safe

    def test_write_file_success(self, temp_dir):
        """Test successful file writing."""
        test_file = temp_dir / "new_file.txt"
        params = WriteFileParams(
            file_path=str(test_file),
            content="Hello from test!"
        )
        context = {}
        
        # Temporarily allow this path
        original_is_safe = fs_config.is_path_safe
        fs_config.is_path_safe = lambda x: True
        
        try:
            result = write_file_tool(params, context)
            assert "successfully" in result["message"]
            assert test_file.exists()
            assert test_file.read_text(encoding="utf-8") == "Hello from test!"
        finally:
            fs_config.is_path_safe = original_is_safe

    def test_write_file_append(self, sample_files):
        """Test appending to existing file."""
        test_file = sample_files / "test1.txt"
        original_content = test_file.read_text(encoding="utf-8")
        
        params = WriteFileParams(
            file_path=str(test_file),
            content="\\nAppended content",
            append=True
        )
        context = {}
        
        # Temporarily allow this path
        original_is_safe = fs_config.is_path_safe
        fs_config.is_path_safe = lambda x: True
        
        try:
            result = write_file_tool(params, context)
            assert "Appended" in result["message"]
            new_content = test_file.read_text(encoding="utf-8")
            assert new_content == original_content + "\\nAppended content"
        finally:
            fs_config.is_path_safe = original_is_safe

    def test_list_directory(self, sample_files):
        """Test directory listing."""
        params = ListDirectoryParams(directory_path=str(sample_files))
        context = {}
        
        # Temporarily allow this path
        original_is_safe = fs_config.is_path_safe
        fs_config.is_path_safe = lambda x: True
        
        try:
            result = list_directory_tool(params, context)
            items = result["items"]
            
            # Should find our test files
            file_names = [item["name"] for item in items]
            assert "test1.txt" in file_names
            assert "test2.py" in file_names
            assert "data.json" in file_names
            assert "subdir" in file_names
            
            # Check summary
            summary = result["summary"]
            assert summary["files"] >= 3
            assert summary["directories"] >= 1
        finally:
            fs_config.is_path_safe = original_is_safe

    def test_list_directory_recursive(self, sample_files):
        """Test recursive directory listing."""
        params = ListDirectoryParams(
            directory_path=str(sample_files),
            recursive=True,
            max_depth=2
        )
        context = {}
        
        # Temporarily allow this path
        original_is_safe = fs_config.is_path_safe
        fs_config.is_path_safe = lambda x: True
        
        try:
            result = list_directory_tool(params, context)
            items = result["items"]
            
            # Should find nested file
            file_names = [item["name"] for item in items]
            assert "nested.txt" in file_names
        finally:
            fs_config.is_path_safe = original_is_safe

    def test_file_info(self, sample_files):
        """Test getting file information."""
        test_file = sample_files / "test1.txt"
        params = FileInfoParams(file_path=str(test_file))
        context = {}
        
        # Temporarily allow this path
        original_is_safe = fs_config.is_path_safe
        fs_config.is_path_safe = lambda x: True
        
        try:
            result = file_info_tool(params, context)
            assert result["name"] == "test1.txt"
            assert result["type"] == "file"
            assert result["extension"] == ".txt"
            assert result["size"] > 0
            assert "created" in result
            assert "modified" in result
        finally:
            fs_config.is_path_safe = original_is_safe

    def test_create_directory(self, temp_dir):
        """Test directory creation."""
        new_dir = temp_dir / "new_directory"
        params = CreateDirectoryParams(directory_path=str(new_dir))
        context = {}
        
        # Temporarily allow this path
        original_is_safe = fs_config.is_path_safe
        fs_config.is_path_safe = lambda x: True
        
        try:
            result = create_directory_tool(params, context)
            assert "successfully" in result["message"]
            assert new_dir.exists()
            assert new_dir.is_dir()
        finally:
            fs_config.is_path_safe = original_is_safe

    def test_search_files_pattern(self, sample_files):
        """Test file search by pattern."""
        params = SearchFilesParams(
            directory_path=str(sample_files),
            pattern="*.py"
        )
        context = {}
        
        # Temporarily allow this path
        original_is_safe = fs_config.is_path_safe
        fs_config.is_path_safe = lambda x: True
        
        try:
            result = search_files_tool(params, context)
            results = result["results"]
            
            # Should find Python files
            assert len(results) >= 1
            py_files = [r for r in results if r["extension"] == ".py"]
            assert len(py_files) >= 1
        finally:
            fs_config.is_path_safe = original_is_safe

    def test_search_files_content(self, sample_files):
        """Test file search by content."""
        params = SearchFilesParams(
            directory_path=str(sample_files),
            pattern="*",
            content_search="Hello"
        )
        context = {}
        
        # Temporarily allow this path
        original_is_safe = fs_config.is_path_safe
        fs_config.is_path_safe = lambda x: True
        
        try:
            result = search_files_tool(params, context)
            results = result["results"]
            
            # Should find files containing "Hello"
            hello_files = [r for r in results if "content_matches" in r]
            assert len(hello_files) >= 1
        finally:
            fs_config.is_path_safe = original_is_safe

    def test_path_safety_restrictions(self, temp_dir):
        """Test that path safety restrictions work."""
        # Try to access a forbidden path
        params = ReadFileParams(file_path="/etc/passwd")
        context = {}
        
        result = read_file_tool(params, context)
        assert not result.get("success", True)
        assert "not allowed" in result["error"]

    def test_extension_restrictions(self, temp_dir):
        """Test file extension restrictions."""
        # Try to read a forbidden file type
        exe_file = temp_dir / "test.exe"
        exe_file.write_bytes(b"binary content")
        
        params = ReadFileParams(file_path=str(exe_file))
        context = {}
        
        # Temporarily allow the path but not the extension
        original_is_safe = fs_config.is_path_safe
        fs_config.is_path_safe = lambda x: True
        
        try:
            result = read_file_tool(params, context)
            assert not result.get("success", True)
            assert "extension not allowed" in result["error"]
        finally:
            fs_config.is_path_safe = original_is_safe


@pytest.mark.integration
def test_filesystem_tools_integration():
    """Integration test using filesystem tools together."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Temporarily allow this path for all operations
        original_is_safe = fs_config.is_path_safe
        fs_config.is_path_safe = lambda x: tmp_path in Path(x).parents or Path(x) == tmp_path
        
        try:
            context = {}
            
            # 1. Create a directory
            new_dir = tmp_path / "test_integration"
            create_result = create_directory_tool(
                CreateDirectoryParams(directory_path=str(new_dir)), context
            )
            assert "successfully" in create_result["message"]
            
            # 2. Write a file in the new directory
            test_file = new_dir / "integration_test.txt"
            write_result = write_file_tool(
                WriteFileParams(
                    file_path=str(test_file),
                    content="Integration test content"
                ), context
            )
            assert "successfully" in write_result["message"]
            
            # 3. Read the file back
            read_result = read_file_tool(
                ReadFileParams(file_path=str(test_file)), context
            )
            assert read_result["content"] == "Integration test content"
            
            # 4. List directory contents
            list_result = list_directory_tool(
                ListDirectoryParams(directory_path=str(new_dir)), context
            )
            file_names = [item["name"] for item in list_result["items"]]
            assert "integration_test.txt" in file_names
            
            # 5. Search for the file
            search_result = search_files_tool(
                SearchFilesParams(
                    directory_path=str(tmp_path),
                    pattern="integration_test.txt"
                ), context
            )
            assert len(search_result["results"]) == 1
            
        finally:
            fs_config.is_path_safe = original_is_safe