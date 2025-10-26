"""
Main analyzer module that orchestrates graph extraction and serialization.
"""

from pathlib import Path
from typing import Dict, Optional, List
import networkx as nx

from .dependency_graph import DependencyGraphExtractor
from .call_graph import CallGraphExtractor
from .serializer import GraphSerializer


class CodeAnalyzer:
    """
    Main analyzer that extracts and saves call graphs and dependency graphs.
    """
    
    def __init__(self, repo_path: str, output_dir: Optional[str] = None):
        """
        Initialize the code analyzer.
        
        Args:
            repo_path: Path to the repository to analyze
            output_dir: Directory where output files will be saved
                       (defaults to repo_path/codevoyant_output)
        """
        self.repo_path = Path(repo_path).resolve()
        
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = self.repo_path / 'codevoyant_output'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dependency_graph: Optional[nx.DiGraph] = None
        self.call_graph: Optional[nx.DiGraph] = None
    
    def analyze(self, extract_dependencies: bool = True, 
                extract_calls: bool = True,
                entry_points: Optional[List[str]] = None) -> Dict[str, nx.DiGraph]:
        """
        Analyze the repository and extract graphs.
        
        Args:
            extract_dependencies: Whether to extract dependency graph
            extract_calls: Whether to extract call graph
            entry_points: List of entry point files for call graph analysis
            
        Returns:
            Dictionary containing the extracted graphs
        """
        results = {}
        
        if extract_dependencies:
            print("Extracting dependency graph...")
            self.dependency_graph = self._extract_dependency_graph()
            results['dependency_graph'] = self.dependency_graph
            
            # Print statistics
            stats = self._get_dependency_stats()
            print("\nDependency Graph Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        if extract_calls:
            print("\nExtracting call graph...")
            self.call_graph = self._extract_call_graph(entry_points)
            results['call_graph'] = self.call_graph
            
            # Print statistics
            stats = self._get_call_stats()
            print("\nCall Graph Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        return results
    
    def save_graphs(self, formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Save the extracted graphs to files.
        
        Args:
            formats: List of output formats ('json', 'graphml', 'gexf', 'pickle', 'cytoscape', 'd3')
                    Defaults to ['json', 'graphml']
            
        Returns:
            Dictionary mapping graph types to their output file paths
        """
        if formats is None:
            formats = ['json', 'graphml']
        
        output_files = {}
        
        if self.dependency_graph:
            print("\nSaving dependency graph...")
            dep_files = self._save_graph(
                self.dependency_graph,
                'dependency_graph',
                formats
            )
            output_files.update(dep_files)
        
        if self.call_graph:
            print("\nSaving call graph...")
            call_files = self._save_graph(
                self.call_graph,
                'call_graph',
                formats
            )
            output_files.update(call_files)
        
        # Save summary report
        self._save_summary_report()
        
        return output_files
    
    def _extract_dependency_graph(self) -> nx.DiGraph:
        """Extract dependency graph from the repository."""
        extractor = DependencyGraphExtractor(str(self.repo_path))
        return extractor.extract()
    
    def _extract_call_graph(self, entry_points: Optional[List[str]] = None) -> nx.DiGraph:
        """Extract call graph from the repository."""
        extractor = CallGraphExtractor(str(self.repo_path), entry_points)
        return extractor.extract()
    
    def _get_dependency_stats(self) -> Dict:
        """Get statistics about the dependency graph."""
        if not self.dependency_graph:
            return {}
        
        extractor = DependencyGraphExtractor(str(self.repo_path))
        extractor.graph = self.dependency_graph
        return extractor.get_statistics()
    
    def _get_call_stats(self) -> Dict:
        """Get statistics about the call graph."""
        if not self.call_graph:
            return {}
        
        extractor = CallGraphExtractor(str(self.repo_path))
        extractor.graph = self.call_graph
        return extractor.get_statistics()
    
    def _save_graph(self, graph: nx.DiGraph, graph_type: str, formats: List[str]) -> Dict[str, str]:
        """
        Save a graph in multiple formats.
        
        Args:
            graph: Graph to save
            graph_type: Type of graph ('dependency_graph' or 'call_graph')
            formats: List of output formats
            
        Returns:
            Dictionary mapping format names to file paths
        """
        output_files = {}
        
        for fmt in formats:
            if fmt == 'json':
                output_path = self.output_dir / f"{graph_type}.json"
                GraphSerializer.to_json(graph, str(output_path), format_type='node_link')
                output_files[f'{graph_type}_json'] = str(output_path)
                
            elif fmt == 'cytoscape':
                output_path = self.output_dir / f"{graph_type}_cytoscape.json"
                GraphSerializer.to_json(graph, str(output_path), format_type='cytoscape')
                output_files[f'{graph_type}_cytoscape'] = str(output_path)
                
            elif fmt == 'd3':
                output_path = self.output_dir / f"{graph_type}_d3.json"
                GraphSerializer.to_json(graph, str(output_path), format_type='d3')
                output_files[f'{graph_type}_d3'] = str(output_path)
                
            elif fmt == 'graphml':
                output_path = self.output_dir / f"{graph_type}.graphml"
                GraphSerializer.to_graphml(graph, str(output_path))
                output_files[f'{graph_type}_graphml'] = str(output_path)
                
            elif fmt == 'gexf':
                output_path = self.output_dir / f"{graph_type}.gexf"
                GraphSerializer.to_gexf(graph, str(output_path))
                output_files[f'{graph_type}_gexf'] = str(output_path)
                
            elif fmt == 'pickle':
                output_path = self.output_dir / f"{graph_type}.pkl"
                GraphSerializer.to_pickle(graph, str(output_path))
                output_files[f'{graph_type}_pickle'] = str(output_path)
        
        return output_files
    
    def _save_summary_report(self):
        """Save a summary report of the analysis."""
        summary_path = self.output_dir / 'analysis_summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("CodeVoyant Analysis Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Repository: {self.repo_path}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            if self.dependency_graph:
                f.write("Dependency Graph Analysis\n")
                f.write("-" * 70 + "\n")
                stats = self._get_dependency_stats()
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                # Save detailed dependency graph summary
                dep_summary_path = self.output_dir / 'dependency_graph_summary.txt'
                GraphSerializer.to_summary(self.dependency_graph, str(dep_summary_path))
            
            if self.call_graph:
                f.write("Call Graph Analysis\n")
                f.write("-" * 70 + "\n")
                stats = self._get_call_stats()
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                # Save detailed call graph summary
                call_summary_path = self.output_dir / 'call_graph_summary.txt'
                GraphSerializer.to_summary(self.call_graph, str(call_summary_path))
        
        print(f"\nSummary report saved to {summary_path}")


def analyze_repository(repo_path: str, 
                       output_dir: Optional[str] = None,
                       formats: Optional[List[str]] = None,
                       entry_points: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Convenience function to analyze a repository and save graphs.
    
    Args:
        repo_path: Path to the repository to analyze
        output_dir: Directory where output files will be saved
        formats: List of output formats (defaults to ['json', 'graphml'])
        entry_points: List of entry point files for call graph analysis
        
    Returns:
        Dictionary mapping graph types to their output file paths
    """
    analyzer = CodeAnalyzer(repo_path, output_dir)
    analyzer.analyze(extract_dependencies=True, extract_calls=True, entry_points=entry_points)
    return analyzer.save_graphs(formats=formats)
