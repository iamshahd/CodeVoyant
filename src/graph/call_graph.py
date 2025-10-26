"""
Module for extracting call graphs from Python repositories.
Uses pyan3 to analyze function and method calls.
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, Set, Optional
import networkx as nx
from pyan.analyzer import CallGraphVisitor as Pyan3Visitor
from pyan.visgraph import VisualGraph


class CallGraphExtractor:
    """Extracts call graph from Python source code using pyan3."""
    
    def __init__(self, repo_path: str, entry_points: Optional[list] = None):
        """
        Initialize the call graph extractor.
        
        Args:
            repo_path: Path to the repository root directory
            entry_points: List of entry point files (e.g., main.py, __main__.py)
                         If None, will search for common entry points
        """
        self.repo_path = Path(repo_path).resolve()
        self.entry_points = entry_points or self._find_entry_points()
        self.graph = nx.DiGraph()
        
    def _find_entry_points(self) -> list:
        """
        Find potential entry points in the repository.
        
        Returns:
            List of entry point file paths
        """
        entry_points = []
        common_names = ['__main__.py', 'main.py', 'app.py', 'run.py']
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env', 'node_modules']]
            
            for file in files:
                if file in common_names:
                    entry_points.append(str(Path(root) / file))
        
        # If no entry points found, use all Python files in the root
        if not entry_points:
            for file in os.listdir(self.repo_path):
                if file.endswith('.py') and not file.startswith('_'):
                    entry_points.append(str(self.repo_path / file))
        
        return entry_points
    
    def extract(self) -> nx.DiGraph:
        """
        Extract call graph from the repository using pyan3.
        
        Returns:
            NetworkX DiGraph representing function/method calls
        """
        if not self.entry_points:
            print("Warning: No entry points found. Call graph may be incomplete.")
            return self.graph
        
        try:
            # Generate call graph using pyan3
            self._run_pyan3()
            
        except Exception as e:
            print(f"Error extracting call graph with pyan3: {e}")
            # Fallback to AST-based simple analysis
            print("Falling back to AST-based analysis...")
            self._fallback_ast_analysis()
        
        return self.graph
    
    def _run_pyan3(self):
        """
        Run pyan3 to generate call graph and populate the NetworkX graph.
        """
        # Collect all Python files to analyze
        python_files = []
        for root, dirs, files in os.walk(self.repo_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(str(Path(root) / file))
        
        if not python_files:
            print("Warning: No Python files found in repository")
            return
        
        # Create pyan3 visual graph
        vis_graph = VisualGraph()
        
        # Analyze each Python file
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the file
                tree = ast.parse(content, filename=py_file)
                
                # Create visitor and analyze
                visitor = Pyan3Visitor(py_file, vis_graph)
                visitor.visit(tree)
                
            except Exception as e:
                print(f"Warning: Could not analyze {py_file}: {e}")
                continue
        
        # Convert pyan3 graph to NetworkX
        self._convert_pyan3_to_networkx(vis_graph)
    
    def _convert_pyan3_to_networkx(self, vis_graph: VisualGraph):
        """
        Convert pyan3 VisualGraph to NetworkX DiGraph.
        
        Args:
            vis_graph: pyan3 VisualGraph object
        """
        # Add nodes
        for node in vis_graph.nodes:
            node_id = node.get_annotated_name()
            
            # Parse node information
            node_info = {
                'full_name': node_id,
                'function_name': node.name,
                'namespace': node.namespace,
                'type': 'method' if node.namespace else 'function',
                'flavor': node.flavor  # 'function', 'method', 'class', etc.
            }
            
            if node.namespace:
                node_info['class_name'] = node.namespace.split('.')[-1]
            
            self.graph.add_node(node_id, **node_info)
        
        # Add edges (calls)
        for edge in vis_graph.edges:
            source = edge[0].get_annotated_name()
            target = edge[1].get_annotated_name()
            
            # Ensure both nodes exist
            if not self.graph.has_node(source):
                self.graph.add_node(source, function_name=edge[0].name)
            if not self.graph.has_node(target):
                self.graph.add_node(target, function_name=edge[1].name)
            
            self.graph.add_edge(source, target, call_type='direct')
    
    def _parse_function_name(self, full_name: str) -> Dict:
        """
        Parse a fully qualified function name.
        
        Args:
            full_name: Fully qualified name (e.g., 'module.Class.method')
            
        Returns:
            Dictionary with parsed information
        """
        parts = full_name.split('.')
        
        info = {
            'full_name': full_name,
            'function_name': parts[-1] if parts else full_name,
        }
        
        if len(parts) > 1:
            info['module'] = '.'.join(parts[:-1])
        
        # Check if it's a method (has class in the path)
        if len(parts) > 2:
            info['type'] = 'method'
            info['class_name'] = parts[-2]
        else:
            info['type'] = 'function'
        
        return info
    
    def _fallback_ast_analysis(self):
        """
        Fallback to AST-based analysis if pyan3 fails.
        Uses simple AST walking to find function calls.
        """
        import ast
        
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    self._analyze_file_ast(file_path)
    
    def _analyze_file_ast(self, file_path: Path):
        """
        Analyze a file using AST to extract function definitions and calls.
        
        Args:
            file_path: Path to the Python file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            # Get module name
            relative_path = file_path.relative_to(self.repo_path)
            module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
            if module_parts[-1] == '__init__':
                module_parts = module_parts[:-1]
            module_name = '.'.join(module_parts) if module_parts else relative_path.stem
            
            # Extract functions and their calls
            visitor = CallGraphVisitor(module_name, self.graph)
            visitor.visit(tree)
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the call graph.
        
        Returns:
            Dictionary with graph statistics
        """
        functions = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'function']
        methods = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'method']
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'functions': len(functions),
            'methods': len(methods),
            'avg_calls_per_function': self.graph.number_of_edges() / max(1, self.graph.number_of_nodes()),
        }


class CallGraphVisitor(ast.NodeVisitor):
    """AST visitor to extract function definitions and calls."""
    
    def __init__(self, module_name: str, graph: nx.DiGraph):
        self.module_name = module_name
        self.graph = graph
        self.current_function = None
        self.current_class = None
    
    def visit_ClassDef(self, node):
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        # Create full function name
        if self.current_class:
            full_name = f"{self.module_name}.{self.current_class}.{node.name}"
            func_type = 'method'
        else:
            full_name = f"{self.module_name}.{node.name}"
            func_type = 'function'
        
        # Add node if not exists
        if not self.graph.has_node(full_name):
            self.graph.add_node(
                full_name,
                function_name=node.name,
                module=self.module_name,
                type=func_type,
                class_name=self.current_class if self.current_class else None,
                lineno=node.lineno
            )
        
        # Visit function body
        old_function = self.current_function
        self.current_function = full_name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Call(self, node):
        """Visit function call."""
        if self.current_function:
            called_name = self._get_call_name(node)
            if called_name:
                # Add edge from current function to called function
                if not self.graph.has_node(called_name):
                    self.graph.add_node(called_name, function_name=called_name)
                
                self.graph.add_edge(self.current_function, called_name, call_type='direct')
        
        self.generic_visit(node)
    
    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract the name of a called function."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None
