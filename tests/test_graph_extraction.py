"""
Basic tests for the graph extraction modules.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
from src.graph import (
    DependencyGraphExtractor,
    CallGraphExtractor,
    CodeAnalyzer,
    GraphSerializer
)


class TestDependencyGraphExtractor(unittest.TestCase):
    """Tests for dependency graph extraction."""
    
    def setUp(self):
        """Create a temporary test repository."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create test files
        (self.test_path / "module_a.py").write_text("""
import module_b
from module_c import something

def func_a():
    pass
""")
        
        (self.test_path / "module_b.py").write_text("""
import os
import sys

def func_b():
    pass
""")
        
        (self.test_path / "module_c.py").write_text("""
def something():
    pass
""")
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)
    
    def test_extraction(self):
        """Test that dependency graph is extracted."""
        extractor = DependencyGraphExtractor(self.test_dir)
        graph = extractor.extract()
        
        self.assertIsNotNone(graph)
        self.assertGreater(graph.number_of_nodes(), 0)
    
    def test_module_discovery(self):
        """Test that all modules are discovered."""
        extractor = DependencyGraphExtractor(self.test_dir)
        graph = extractor.extract()
        
        # Check that our test modules are in the graph
        nodes = list(graph.nodes())
        self.assertTrue(any('module_a' in str(n) for n in nodes))
        self.assertTrue(any('module_b' in str(n) for n in nodes))
        self.assertTrue(any('module_c' in str(n) for n in nodes))
    
    def test_statistics(self):
        """Test that statistics are generated correctly."""
        extractor = DependencyGraphExtractor(self.test_dir)
        graph = extractor.extract()
        stats = extractor.get_statistics()
        
        self.assertIn('total_modules', stats)
        self.assertIn('local_modules', stats)
        self.assertIn('external_modules', stats)
        self.assertGreater(stats['total_modules'], 0)


class TestCallGraphExtractor(unittest.TestCase):
    """Tests for call graph extraction."""
    
    def setUp(self):
        """Create a temporary test repository."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create test files
        (self.test_path / "main.py").write_text("""
def main():
    helper()
    
def helper():
    pass

if __name__ == '__main__':
    main()
""")
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)
    
    def test_extraction(self):
        """Test that call graph is extracted."""
        extractor = CallGraphExtractor(self.test_dir)
        graph = extractor.extract()
        
        self.assertIsNotNone(graph)
        # Graph might be empty if PyCG fails, but should at least exist
    
    def test_statistics(self):
        """Test that statistics are generated."""
        extractor = CallGraphExtractor(self.test_dir)
        graph = extractor.extract()
        stats = extractor.get_statistics()
        
        self.assertIn('total_nodes', stats)
        self.assertIn('total_edges', stats)


class TestGraphSerializer(unittest.TestCase):
    """Tests for graph serialization."""
    
    def setUp(self):
        """Create a test graph."""
        import networkx as nx
        self.graph = nx.DiGraph()
        self.graph.add_node("module_a", type="module")
        self.graph.add_node("module_b", type="module")
        self.graph.add_edge("module_a", "module_b", import_type="local")
        
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)
    
    def test_json_serialization(self):
        """Test JSON serialization."""
        output_path = Path(self.test_dir) / "test_graph.json"
        GraphSerializer.to_json(self.graph, str(output_path))
        
        self.assertTrue(output_path.exists())
    
    def test_cytoscape_format(self):
        """Test Cytoscape format."""
        output_path = Path(self.test_dir) / "test_cytoscape.json"
        GraphSerializer.to_json(self.graph, str(output_path), format_type='cytoscape')
        
        self.assertTrue(output_path.exists())
    
    def test_d3_format(self):
        """Test D3 format."""
        output_path = Path(self.test_dir) / "test_d3.json"
        GraphSerializer.to_json(self.graph, str(output_path), format_type='d3')
        
        self.assertTrue(output_path.exists())
    
    def test_graphml_serialization(self):
        """Test GraphML serialization."""
        output_path = Path(self.test_dir) / "test_graph.graphml"
        GraphSerializer.to_graphml(self.graph, str(output_path))
        
        self.assertTrue(output_path.exists())


class TestCodeAnalyzer(unittest.TestCase):
    """Tests for the main CodeAnalyzer."""
    
    def setUp(self):
        """Create a temporary test repository."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Create a simple test file
        (self.test_path / "test_module.py").write_text("""
import os

def test_function():
    pass
""")
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir)
    
    def test_analyzer_creation(self):
        """Test that analyzer can be created."""
        analyzer = CodeAnalyzer(self.test_dir)
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.repo_path, Path(self.test_dir).resolve())
    
    def test_analyze(self):
        """Test that analysis runs without errors."""
        analyzer = CodeAnalyzer(self.test_dir)
        graphs = analyzer.analyze(extract_dependencies=True, extract_calls=True)
        
        self.assertIn('dependency_graph', graphs)
        self.assertIsNotNone(graphs['dependency_graph'])
    
    def test_save_graphs(self):
        """Test that graphs can be saved."""
        analyzer = CodeAnalyzer(self.test_dir)
        analyzer.analyze(extract_dependencies=True, extract_calls=False)
        output_files = analyzer.save_graphs(formats=['json'])
        
        self.assertGreater(len(output_files), 0)


if __name__ == '__main__':
    unittest.main()
