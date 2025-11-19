pip freeze | % { pip install --upgrade ($_ -split '==')[0] 2>$null }import os
import ast
import hashlib
import argparse
from typing import Dict, List
from pathlib import Path


def get_function_hashes(filepath: str) -> Dict[str, List[dict]]:
    """Extract all functions (and async ones) with their block context."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        print(f"⚠️  Error reading {filepath}: {e}")
        return {}

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"⚠️  Syntax error in {filepath}: {e}")
        return {}

    functions = {}
    block_stack = []

    class FunctionVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            block_stack.append(node.name)
            
            # Create hash from AST dump
            func_hash = hashlib.md5(ast.dump(node).encode("utf-8")).hexdigest()
            
            # Copy block_stack to avoid reference issues
            env = " → ".join(block_stack[:-1]) if len(block_stack) > 1 else "global"
            
            functions.setdefault(func_hash, []).append({
                "filepath": filepath,
                "name": node.name,
                "env": env,
                "line": node.lineno,
                "end_line": node.end_lineno
            })
            
            self.generic_visit(node)
            block_stack.pop()

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

        def visit_ClassDef(self, node):
            block_stack.append(f"class {node.name}")
            self.generic_visit(node)
            block_stack.pop()

    FunctionVisitor().visit(tree)
    return functions


def find_duplicate_functions(target: str, min_lines: int = 1) -> Dict[str, List[dict]]:
    """Find duplicate functions including their context (block env)."""
    hashes = {}
    
    target_path = Path(target)
    
    if target_path.is_file():
        if target_path.suffix != ".py":
            print(f"⚠️  {target} is not a Python file")
            return {}
        files = [str(target_path)]
    elif target_path.is_dir():
        files = [str(p) for p in target_path.rglob("*.py")]
    else:
        print(f"❌ {target} does not exist")
        return {}

    print(f"📁 Scanning {len(files)} Python file(s)...\n")

    for f in files:
        func_data = get_function_hashes(f)
        for h, infos in func_data.items():
            hashes.setdefault(h, []).extend(infos)

    # Filter hashes with multiple functions and apply min_lines filter
    duplicates = {}
    for h, funcs in hashes.items():
        if len(funcs) > 1:
            # Filter by minimum line count if specified
            if min_lines > 1:
                filtered = [f for f in funcs if (f.get('end_line', f['line']) - f['line'] + 1) >= min_lines]
                if len(filtered) > 1:
                    duplicates[h] = filtered
            else:
                duplicates[h] = funcs
    
    return duplicates


def format_output(dupes: Dict[str, List[dict]], show_stats: bool = False):
    """Format and print duplicate function results."""
    if not dupes:
        print("✅ No duplicates found.")
        return
    
    total_funcs = sum(len(funcs) for funcs in dupes.values())
    total_groups = len(dupes)
    
    print(f"🔍 Found {total_groups} duplicate function group(s) ({total_funcs} total instances):\n")
    
    # Sort by number of duplicates (descending) then by function name
    sorted_dupes = sorted(dupes.items(), key=lambda x: (-len(x[1]), x[1][0]['name']))
    
    for idx, (h, funcs) in enumerate(sorted_dupes, 1):
        # Sort functions by filepath and line number
        sorted_funcs = sorted(funcs, key=lambda x: (x['filepath'], x['line']))
        
        func_name = sorted_funcs[0]['name']
        line_count = sorted_funcs[0].get('end_line', sorted_funcs[0]['line']) - sorted_funcs[0]['line'] + 1
        
        print(f"[{idx}] Function '{func_name}' ({len(funcs)} instances, ~{line_count} lines)")
        print(f"    Hash: {h}")
        
        for f in sorted_funcs:
            rel_path = Path(f['filepath']).name
            print(f"    📄 {rel_path}:{f['line']} | Env: {f['env']}")
        print()
    
    if show_stats:
        print(f"📊 Statistics:")
        print(f"   - Total duplicate groups: {total_groups}")
        print(f"   - Total duplicate instances: {total_funcs}")
        print(f"   - Potential duplicates to review: {total_funcs - total_groups}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find duplicate Python functions with environment context.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s myfile.py
  %(prog)s ./src --min-lines 5 --stats
  %(prog)s . --min-lines 10
        """
    )
    parser.add_argument("target", help="File or directory to scan")
    parser.add_argument("--min-lines", type=int, default=1, 
                        help="Minimum number of lines for a function to be considered (default: 1)")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics summary")
    
    args = parser.parse_args()

    dupes = find_duplicate_functions(args.target, min_lines=args.min_lines)
    format_output(dupes, show_stats=args.stats)