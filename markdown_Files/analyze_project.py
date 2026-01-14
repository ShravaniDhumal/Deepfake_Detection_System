#!/usr/bin/env python3
"""
Comprehensive project analysis script
Identifies issues, bugs, and improvement opportunities
"""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict

class ProjectAnalyzer:
    def __init__(self):
        self.issues = defaultdict(list)
        self.improvements = defaultdict(list)
        self.warnings = defaultdict(list)
        
    def analyze_file(self, filepath, category):
        """Analyze a Python file for issues"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
            # Try to parse as AST
            try:
                tree = ast.parse(content)
                self.analyze_ast(tree, filepath, category)
            except SyntaxError as e:
                self.issues[category].append(f"{filepath}: Syntax error - {e}")
            
            # Line-by-line analysis
            self.analyze_code_lines(lines, filepath, category)
            
        except Exception as e:
            self.warnings[category].append(f"{filepath}: Could not analyze - {e}")
    
    def analyze_ast(self, tree, filepath, category):
        """Analyze AST for code quality issues"""
        for node in ast.walk(tree):
            # Check for bare except
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                self.issues[category].append(f"{filepath}: Bare except clause (line {node.lineno})")
            
            # Check for print statements (should use logging)
            if isinstance(node, ast.Print):
                self.improvements[category].append(f"{filepath}: Consider using logging instead of print (line {node.lineno})")
    
    def analyze_code_lines(self, lines, filepath, category):
        """Analyze code lines for patterns"""
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for common issues
            if 'except:' in stripped and 'except Exception:' not in stripped:
                if 'except:' in stripped:
                    self.issues[category].append(f"{filepath}: Bare except clause (line {i})")
            
            if 'import *' in stripped:
                self.improvements[category].append(f"{filepath}: Wildcard import (line {i})")
            
            if 'TODO' in stripped or 'FIXME' in stripped:
                self.warnings[category].append(f"{filepath}: TODO/FIXME found (line {i})")
    
    def check_training_script(self):
        """Analyze training script"""
        filepath = "pytorch/train.py"
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        issues = []
        improvements = []
        
        # Check for validation loop
        if 'val_loader' in content and 'val' not in content.lower() or 'validation' not in content.lower():
            improvements.append("Training script loads validation data but doesn't use it for evaluation")
        
        # Check for model saving
        if 'torch.save' in content:
            if 'model.state_dict()' in content:
                improvements.append("Consider saving full model or including optimizer state for resuming training")
        
        # Check for error handling
        if 'try:' not in content and 'except' not in content:
            improvements.append("Add error handling for file operations and model loading")
        
        # Check for progress tracking
        if 'tqdm' not in content:
            improvements.append("Consider adding tqdm for better progress visualization")
        
        # Check for early stopping
        if 'early' not in content.lower() or 'stop' not in content.lower():
            improvements.append("Consider implementing early stopping to prevent overfitting")
        
        # Check for learning rate scheduling
        if 'scheduler' not in content.lower():
            improvements.append("Consider adding learning rate scheduling for better convergence")
        
        # Check for gradient clipping
        if 'clip_grad' not in content.lower():
            improvements.append("Consider adding gradient clipping for training stability")
        
        self.improvements['Training'].extend(improvements)
        self.issues['Training'].extend(issues)
    
    def check_dataset(self):
        """Analyze dataset class"""
        filepath = "pytorch/dataset.py"
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        issues = []
        improvements = []
        
        # Check for error handling
        if 'Image.open' in content and 'try' not in content:
            issues.append("Dataset doesn't handle corrupted image files - will crash on bad images")
        
        # Check for file filtering
        if 'os.listdir' in content:
            improvements.append("Dataset should filter out non-image files (.gitkeep, .DS_Store, etc.)")
        
        # Check for data augmentation
        if 'Random' not in content and 'augment' not in content.lower():
            improvements.append("Consider adding data augmentation for better generalization")
        
        # Check for caching
        if 'cache' not in content.lower():
            improvements.append("Consider implementing image caching for faster training")
        
        self.issues['Dataset'].extend(issues)
        self.improvements['Dataset'].extend(improvements)
    
    def check_model(self):
        """Analyze model architecture"""
        filepath = "pytorch/models/xception.py"
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        issues = []
        improvements = []
        
        # Check model name vs implementation
        if 'xception' in filepath.lower() and 'mobilenet' in content.lower():
            issues.append("Model file named 'xception.py' but uses MobileNetV2 - naming mismatch")
        
        # Check for dropout
        if 'dropout' not in content.lower():
            improvements.append("Consider adding dropout layers for regularization")
        
        # Check for model documentation
        if '"""' not in content and "'''" not in content:
            improvements.append("Add docstrings to model functions")
        
        self.issues['Model'].extend(issues)
        self.improvements['Model'].extend(improvements)
    
    def check_webcam_detector(self):
        """Analyze webcam detector"""
        filepath = "tensorflow/webcam_detector.py"
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        issues = []
        improvements = []
        
        # Check for bare except
        if 'except:' in content and 'except Exception:' not in content:
            issues.append("Bare except clause catches all exceptions including KeyboardInterrupt")
        
        # Check for resource cleanup
        if 'cap.release()' in content:
            improvements.append("Good: Proper resource cleanup")
        
        # Check for error handling
        if 'VideoCapture' in content and 'isOpened' not in content:
            improvements.append("Check if camera is opened before using it")
        
        # Check for FPS limiting
        if 'time.sleep' not in content and 'fps' not in content.lower():
            improvements.append("Consider limiting FPS to reduce CPU usage")
        
        # Check for confidence scores
        if 'confidence' not in content.lower():
            improvements.append("Display prediction confidence scores for better UX")
        
        self.issues['Webcam Detector'].extend(issues)
        self.improvements['Webcam Detector'].extend(improvements)
    
    def check_temporal_logic(self):
        """Analyze temporal logic"""
        filepath = "tensorflow/temporal_logic.py"
        if not os.path.exists(filepath):
            return
        
        improvements = []
        
        improvements.append("Consider adding confidence-weighted averaging instead of simple majority")
        improvements.append("Add method to reset buffer for new video sequences")
        improvements.append("Consider exponential moving average for smoother transitions")
        
        self.improvements['Temporal Logic'].extend(improvements)
    
    def check_evaluation(self):
        """Analyze evaluation script"""
        filepath = "evaluation/evaluate_model.py"
        if not os.path.exists(filepath):
            return
        
        issues = []
        improvements = []
        
        # Check for hardcoded paths
        if 'y_true.npy' in open(filepath).read() and '../' not in open(filepath).read():
            improvements.append("Hardcoded file paths - consider using command-line arguments")
        
        # Check for visualization
        if 'plot' not in open(filepath).read().lower() and 'matplotlib' not in open(filepath).read():
            improvements.append("Add visualization of confusion matrix and ROC curve")
        
        # Check for metrics file
        if 'save' not in open(filepath).read().lower():
            improvements.append("Save evaluation results to file for tracking")
        
        self.improvements['Evaluation'].extend(improvements)
        self.issues['Evaluation'].extend(issues)
    
    def check_project_structure(self):
        """Check project structure issues"""
        issues = []
        improvements = []
        
        # Check for empty files
        empty_files = [
            "pytorch/utils.py",
            "tensorflow/inference.py",
            "evaluation/metrics.py"
        ]
        
        for filepath in empty_files:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    if len(f.read().strip()) == 0:
                        issues.append(f"{filepath} is empty - should contain utility functions")
        
        # Check for missing __init__.py
        python_dirs = ["pytorch/models", "evaluation", "tensorflow"]
        for dirpath in python_dirs:
            if os.path.exists(dirpath) and "__init__.py" not in os.listdir(dirpath):
                improvements.append(f"Add __init__.py to {dirpath} for proper Python package structure")
        
        # Check for .gitignore
        if not os.path.exists(".gitignore"):
            improvements.append("Add .gitignore to exclude venv, __pycache__, .pyc files, etc.")
        
        # Check for requirements versioning
        if os.path.exists("requirements.txt"):
            with open("requirements.txt", 'r') as f:
                reqs = f.read()
                if '==' not in reqs and '>=' not in reqs:
                    improvements.append("Pin package versions in requirements.txt for reproducibility")
        
        self.issues['Structure'].extend(issues)
        self.improvements['Structure'].extend(improvements)
    
    def run_analysis(self):
        """Run complete analysis"""
        print("=" * 80)
        print("PROJECT ANALYSIS REPORT")
        print("=" * 80)
        
        # Analyze specific components
        self.check_training_script()
        self.check_dataset()
        self.check_model()
        self.check_webcam_detector()
        self.check_temporal_logic()
        self.check_evaluation()
        self.check_project_structure()
        
        # Print results
        print("\n" + "=" * 80)
        print("CRITICAL ISSUES")
        print("=" * 80)
        if self.issues:
            for category, items in self.issues.items():
                if items:
                    print(f"\n{category}:")
                    for item in items:
                        print(f"  ❌ {item}")
        else:
            print("✅ No critical issues found!")
        
        print("\n" + "=" * 80)
        print("IMPROVEMENTS & RECOMMENDATIONS")
        print("=" * 80)
        if self.improvements:
            for category, items in self.improvements.items():
                if items:
                    print(f"\n{category}:")
                    for item in items:
                        print(f"  💡 {item}")
        else:
            print("✅ All good!")
        
        print("\n" + "=" * 80)
        print("WARNINGS")
        print("=" * 80)
        if self.warnings:
            for category, items in self.warnings.items():
                if items:
                    print(f"\n{category}:")
                    for item in items:
                        print(f"  ⚠️  {item}")
        else:
            print("✅ No warnings!")
        
        # Summary
        total_issues = sum(len(items) for items in self.issues.values())
        total_improvements = sum(len(items) for items in self.improvements.values())
        total_warnings = sum(len(items) for items in self.warnings.values())
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Critical Issues: {total_issues}")
        print(f"Improvements: {total_improvements}")
        print(f"Warnings: {total_warnings}")
        print("\n" + "=" * 80)

if __name__ == "__main__":
    analyzer = ProjectAnalyzer()
    analyzer.run_analysis()
