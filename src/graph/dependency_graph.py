"""
Module for extracting dependency graphs from Python repositories.
Uses AST to analyze import statements and build a dependency graph.
"""

import ast
import os
from pathlib import Path
from typing import Dict, Set, List, Optional
import networkx as nx


class DependencyGraphExtractor:
    """Extracts dependency graph from Python source code."""
    
    def __init__(self, repo_path: str):
        """
        Initialize the dependency graph extractor.
        
        Args:
            repo_path: Path to the repository root directory
        """
        self.repo_path = Path(repo_path).resolve()
        self.graph = nx.DiGraph()
        self.module_files: Dict[str, Path] = {}
        
    def extract(self) -> nx.DiGraph:
        """
        Extract dependency graph from the repository.
        
        Returns:
            NetworkX DiGraph representing module dependencies
        """
        # First pass: discover all Python modules
        self._discover_modules()
        
        # Second pass: analyze imports and build graph
        for module_name, file_path in self.module_files.items():
            self._analyze_file(module_name, file_path)
        
        return self.graph
    
    def _discover_modules(self):
        """Discover all Python modules in the repository."""
        for root, dirs, files in os.walk(self.repo_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    module_name = self._get_module_name(file_path)
                    self.module_files[module_name] = file_path
                    self.graph.add_node(module_name, path=str(file_path.relative_to(self.repo_path)))
    
    def _get_module_name(self, file_path: Path) -> str:
        """
        Convert file path to module name.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Module name (e.g., 'src.graph.dependency_graph')
        """
        relative_path = file_path.relative_to(self.repo_path)
        parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        
        # Remove __init__ from the end if present
        if parts[-1] == '__init__':
            parts = parts[:-1]
        
        return '.'.join(parts) if parts else relative_path.stem
    
    def _analyze_file(self, module_name: str, file_path: Path):
        """
        Analyze a Python file for imports and add edges to the graph.
        
        Args:
            module_name: Name of the module being analyzed
            file_path: Path to the Python file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            imports = self._extract_imports(tree)
            
            for imported_module in imports:
                # Try to resolve the import to a local module
                resolved = self._resolve_import(imported_module, module_name)
                
                if resolved and resolved in self.module_files:
                    # Add edge from current module to imported module
                    self.graph.add_edge(module_name, resolved, import_type='local')
                else:
                    # External dependency
                    if imported_module not in self.graph:
                        self.graph.add_node(imported_module, external=True)
                    self.graph.add_edge(module_name, imported_module, import_type='external')
                    
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """
        Extract all import statements from an AST.
        
        Args:
            tree: AST of the Python file
            
        Returns:
            Set of imported module names
        """
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
                elif node.level > 0:
                    # Relative import
                    imports.add('.' * node.level)
        
        return imports
    
    def _resolve_import(self, imported_module: str, current_module: str) -> Optional[str]:
        """
        Resolve an import to a module name in the repository.
        
        Args:
            imported_module: The imported module name
            current_module: The module doing the importing
            
        Returns:
            Resolved module name or None if not found
        """
        # Handle relative imports
        if imported_module.startswith('.'):
            parts = current_module.split('.')
            level = len(imported_module) - len(imported_module.lstrip('.'))
            
            if level <= len(parts):
                base_parts = parts[:-level] if level > 0 else parts
                return '.'.join(base_parts)
            return None
        
        # Check if it's a direct match
        if imported_module in self.module_files:
            return imported_module
        
        # Check for submodules
        for module in self.module_files.keys():
            if module.startswith(imported_module + '.'):
                return imported_module
        
        return None
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the dependency graph.
        
        Returns:
            Dictionary with graph statistics
        """
        local_nodes = [n for n, d in self.graph.nodes(data=True) if not d.get('external', False)]
        external_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('external', False)]
        
        return {
            'total_modules': self.graph.number_of_nodes(),
            'local_modules': len(local_nodes),
            'external_modules': len(external_nodes),
            'total_dependencies': self.graph.number_of_edges(),
            'local_dependencies': len([e for e in self.graph.edges(data=True) if e[2].get('import_type') == 'local']),
            'external_dependencies': len([e for e in self.graph.edges(data=True) if e[2].get('import_type') == 'external']),
        }
