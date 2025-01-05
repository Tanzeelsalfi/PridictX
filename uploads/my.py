import pandas as pd
import radon.metrics as radon_metrics
import radon.complexity as radon_complexity
import lizard
from datetime import datetime

# Constants
CURRENT_UTC = "2025-01-02 15:46:39"
CURRENT_USER = "tabish090"

def extract_code_metrics(file_path):
    try:
        with open(file_path, 'r') as file:
            code = file.read()
        
        # Create DataFrame with all specified metrics
        metrics_df = pd.DataFrame({
            # Metadata
            'timestamp': [CURRENT_UTC],
            'analyzed_by': [CURRENT_USER],
            
            # Basic Code Metrics
            'loc': [0],              # Will be updated with actual count
            'v(g)': [0],            # Cyclomatic Complexity
            'ev(g)': [0],           # Essential Complexity
            'iv(g)': [0],           # Design Complexity
            
            # Halstead Metrics
            'n': [0],               # Program vocabulary
            'v': [0],               # Program volume
            'l': [0],               # Program level
            'd': [0],               # Program difficulty
            'i': [0],               # Intelligence content
            'e': [0],               # Effort to implement
            'b': [0],               # Number of delivered bugs
            't': [0],               # Time to implement
            
            # Line Count Metrics
            'lOCode': [0],          # Lines of code
            'lOComment': [0],       # Lines of comments
            'lOBlank': [0],         # Blank lines
            'locCodeAndComment': [0], # Lines with both code and comments
            
            # Operator/Operand Metrics
            'uniq_Op': [0],         # Unique operators
            'uniq_Opnd': [0],       # Unique operands
            'total_Op': [0],        # Total operators
            'total_Opnd': [0],      # Total operands
            
            # Control Flow
            'branchCount': [0]      # Number of branches
        })
        
        # Calculate actual metrics
        lines = code.splitlines()
        
        # Basic line counts
        metrics_df.at[0, 'loc'] = len(lines)
        metrics_df.at[0, 'lOBlank'] = len([line for line in lines if not line.strip()])
        metrics_df.at[0, 'lOComment'] = len([line for line in lines if line.strip().startswith('#')])
        metrics_df.at[0, 'lOCode'] = metrics_df.at[0, 'loc'] - metrics_df.at[0, 'lOBlank'] - metrics_df.at[0, 'lOComment']
        
        # Try to get Halstead metrics using radon
        try:
            h_visit = radon_metrics.h_visit(code)
            if h_visit:
                # Update Halstead metrics if available
                metrics_df.at[0, 'n'] = h_visit.total_operators + h_visit.total_operands
                metrics_df.at[0, 'v'] = h_visit.volume
                metrics_df.at[0, 'd'] = h_visit.difficulty
                metrics_df.at[0, 'e'] = h_visit.effort
                metrics_df.at[0, 'uniq_Op'] = h_visit.unique_operators
                metrics_df.at[0, 'uniq_Opnd'] = h_visit.unique_operands
                metrics_df.at[0, 'total_Op'] = h_visit.total_operators
                metrics_df.at[0, 'total_Opnd'] = h_visit.total_operands
        except Exception as e:
            print(f"Warning: Could not calculate Halstead metrics: {e}")
        
        # Try to get complexity metrics using lizard
        try:
            lizard_analysis = lizard.analyze_file(file_path)
            if lizard_analysis:
                metrics_df.at[0, 'v(g)'] = sum(func.cyclomatic_complexity for func in lizard_analysis.function_list)
                metrics_df.at[0, 'branchCount'] = len([func for func in lizard_analysis.function_list if func.cyclomatic_complexity > 1])
        except Exception as e:
            print(f"Warning: Could not calculate complexity metrics: {e}")
        
        return metrics_df
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return pd.DataFrame()

# File path
file_path = 'my.py'

# Extract metrics and display
df = extract_code_metrics(file_path)

if not df.empty:
    print("\nMetrics Analysis Results:")
    print("\nFile analyzed:", file_path)
    print("-" * 100)
    # Display metrics grouped by category
    print("\nBasic Code Metrics:")
    print(df[['loc', 'v(g)', 'ev(g)', 'iv(g)']].to_string(index=False))
    print("\nHalstead Metrics:")
    print(df[['n', 'v', 'l', 'd', 'i', 'e', 'b', 't']].to_string(index=False))
    print("\nLine Count Metrics:")
    print(df[['lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment']].to_string(index=False))
    print("\nOperator/Operand Metrics:")
    print(df[['uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd']].to_string(index=False))
    print("\nControl Flow Metrics:")
    print(df[['branchCount']].to_string(index=False))
    print("-" * 100)
else:
    print("\nNo metrics were generated.")
