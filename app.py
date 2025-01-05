# PART 1: Imports, Configuration, and Setup
import nest_asyncio
from flask import Flask, render_template_string, request, send_file, flash, redirect, url_for, render_template, jsonify, make_response
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import logging
import radon.metrics as radon_metrics
import radon.complexity as radon_complexity
import lizard
from datetime import datetime
import tokenize
from io import StringIO
import joblib
import pandas as pd
import ast
import mccabe
import astroid
from pylint import lint
from pylint.reporters import JSONReporter
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
from typing import List, Dict, Any
import symtable
import re
import json
from io import StringIO
import sys
import pdfkit
from datetime import datetime
import tempfile
import os
from jinja2 import Template
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from io import BytesIO
import textwrap
from datetime import datetime
import logging

# Apply the nest_asyncio patch
nest_asyncio.apply()

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Configure logging and security
logging.basicConfig(level=logging.DEBUG)
app.secret_key = 'your_secret_key_here'  # Required for flash messages

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'py'}

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load ML models and data
try:
    model = joblib.load('bug_detector_model.pkl')
    scaler = joblib.load('bug_detector_scaler.pkl')
    feature_names = joblib.load('bug_detector_features.pkl')
except Exception as e:
    logging.error(f"Error loading ML models: {e}")
    raise

# Global variables for timestamp and user
CURRENT_UTC = "2025-01-04 12:24:18"  # Your specified timestamp
CURRENT_USER = "Tanzeelsalfi"  # Your specified username

# Helper function for file validation
def allowed_file(filename):
    """
    Check if uploaded file has an allowed extension
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ================== END OF PART 1 ==================
# Insert Part 2 below this line (Metric Calculation Functions)
# PART 2: Metric Calculation Functions

def count_operators_and_operands(code):
    """
    Analyze code to count unique and total operators and operands
    Returns: tuple (unique_operators, unique_operands, total_operators, total_operands)
    """
    operators = set()
    operands = set()
    total_operators = 0
    total_operands = 0

    python_operators = {
        '+', '-', '*', '/', '//', '%', '**',
        '==', '!=', '<', '>', '<=', '>=',
        'and', 'or', 'not',
        '&', '|', '^', '~', '<<', '>>',
        '=', '+=', '-=', '*=', '/=', '//=',
        '%=', '**=', '&=', '|=', '^=',
        '(', ')', '[', ']', '{', '}',
        ',', '.', ':', ';'
    }

    try:
        tokens = list(tokenize.generate_tokens(StringIO(code).readline))
        for token in tokens:
            if token.string in python_operators:
                operators.add(token.string)
                total_operators += 1
            elif token.type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING):
                if token.string not in ('and', 'or', 'not', 'in', 'is'):
                    operands.add(token.string)
                    total_operands += 1
    except Exception as e:
        logging.warning(f"Error in counting operators and operands: {e}")

    return (len(operators), len(operands), total_operators, total_operands)

def extract_code_metrics(file_path):
    """
    Extract comprehensive code metrics from a Python file
    Returns: pandas DataFrame containing all metrics
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code = file.read()

        # Basic code analysis
        uniq_op, uniq_opnd, total_op, total_opnd = count_operators_and_operands(code)
        lines = code.splitlines()
        loc = len(lines)
        blank_lines = len([line for line in lines if not line.strip()])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        code_lines = loc - blank_lines - comment_lines

        # Calculate Halstead metrics
        n = uniq_op + uniq_opnd
        N = total_op + total_opnd
        V = N * (n.bit_length()) if n > 0 else 0
        D = (uniq_op * total_opnd) / (2 * uniq_opnd) if uniq_opnd > 0 else 0
        E = D * V
        T = E / 18 if E > 0 else 0
        B = V / 3000 if V > 0 else 0

        # Calculate complexity metrics
        complexity_results = radon_complexity.cc_visit(code)
        cyclomatic_complexity = sum([result.complexity for result in complexity_results]) if complexity_results else 0
        essential_complexity = sum([result.complexity - 1 for result in complexity_results]) if complexity_results else 0
        design_complexity = sum([result.complexity for result in complexity_results]) if complexity_results else 0

        # Create metrics DataFrame with updated timestamp
        metrics_df = pd.DataFrame({
            'timestamp': ["2025-01-04 12:24:42"],  # Updated timestamp
            'analyzed_by': [CURRENT_USER],
            'loc': [loc],
            'v(g)': [cyclomatic_complexity],
            'ev(g)': [essential_complexity],
            'iv(g)': [design_complexity],
            'n': [n],
            'v': [V],
            'l': [1/D if D > 0 else 0],
            'd': [D],
            'i': [V],
            'e': [E],
            'b': [B],
            't': [T],
            'lOCode': [code_lines],
            'lOComment': [comment_lines],
            'lOBlank': [blank_lines],
            'locCodeAndComment': [code_lines + comment_lines],
            'uniq_Op': [uniq_op],
            'uniq_Opnd': [uniq_opnd],
            'total_Op': [total_op],
            'total_Opnd': [total_opnd],
            'branchCount': [0]
        })

        # Add lizard metrics
        try:
            lizard_analysis = lizard.analyze_file(file_path)
            if lizard_analysis:
                metrics_df.at[0, 'branchCount'] = len([func for func in lizard_analysis.function_list
                                                      if func.cyclomatic_complexity > 1])
        except Exception as e:
            logging.warning(f"Could not calculate additional complexity metrics: {e}")

        return metrics_df

    except Exception as e:
        logging.error(f"Error analyzing file: {e}")
        return pd.DataFrame()

# ================== END OF PART 2 ==================
# Insert Part 3 below this line (Routes and Main Execution)
# PART 3: Routes and Main Execution

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and analysis"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Generate metrics
            df = extract_code_metrics(filepath)

            if not df.empty:
                # Generate filenames with current timestamp
                timestamp_str = "2025-01-04_12:25:19"  # Updated timestamp
                csv_filename = f'metrics_analysis_{timestamp_str}.csv'
                txt_filename = f'metrics_analysis_{timestamp_str}.txt'

                csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
                txt_path = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)

                # Save CSV
                df.to_csv(csv_path, index=False)

                # Generate detailed report
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("Code Metrics Analysis Report\n")
                    f.write(f"Generated on: {CURRENT_UTC}\n")
                    f.write(f"Analyzed by: {CURRENT_USER}\n")
                    f.write(f"File analyzed: {filename}\n")
                    f.write("=" * 80 + "\n\n")

                    # Write metrics sections
                    sections = [
                        ("Basic Code Metrics", ['loc', 'v(g)', 'ev(g)', 'iv(g)']),
                        ("Halstead Metrics", ['n', 'v', 'l', 'd', 'i', 'e', 'b', 't']),
                        ("Line Count Metrics", ['lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment']),
                        ("Operator/Operand Metrics", ['uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd']),
                        ("Control Flow Metrics", ['branchCount'])
                    ]

                    for section_title, metrics in sections:
                        f.write(f"{section_title}:\n" + "-" * 40 + "\n")
                        f.write(df[metrics].to_string(index=False) + "\n\n")

                    f.write("=" * 80 + "\n")
                    f.write(f"End of Analysis Report - {CURRENT_UTC}\n")

                return render_template('results.html',
                                    metrics=df.to_dict('records')[0],
                                    csv_file=csv_filename,
                                    txt_file=txt_filename)
            else:
                flash('Error analyzing file')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a Python file.')
            return redirect(request.url)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    """Handle file downloads"""
    if '..' in filename or filename.startswith('/'):
        return "Invalid filename", 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    return send_file(filepath, as_attachment=True)

@app.route('/inn')
def inn():
    """Render main page with features"""
    return render_template('inn.html', features=feature_names)

@app.route('/help_info')
def help_info():
    """Render help page"""
    return render_template('help.html')

@app.route('/credit')
def credit():
    """Render credits page"""
    return render_template('credit.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle single prediction requests"""
    try:
        input_data = request.get_json()
        df = pd.DataFrame([input_data])

        # Validate features
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            return jsonify({'error': f'Missing required features: {missing_features}'}), 400

        # Process and predict
        df = df[feature_names]
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        return jsonify({
            'prediction': int(predictions[0]),
            'probability': float(probabilities[0])
        })
    except Exception as e:
        logging.error(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """Handle batch predictions from CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({'error': 'Invalid or no file selected'}), 400

        # Process CSV file
        df = pd.read_csv(file)
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            return jsonify({'error': f'Missing required features: {missing_features}'}), 400

        # Prepare and predict
        df = df[feature_names]
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        # Format results
        results = [{'prediction': int(pred), 'probability': float(prob)}
                  for pred, prob in zip(predictions, probabilities)]

        return jsonify({
            'message': 'CSV file processed successfully',
            'results': results
        })
    except Exception as e:
        logging.error(f"Error in /predict_csv: {e}")
        return jsonify({'error': str(e)}), 500
    

class CodeAnalyzer:
    def __init__(self, code: str):
        self.code = code
        self.issues = []
        self.metrics = {}
    
    def analyze_all(self) -> Dict[str, Any]:
        """Run all analysis checks and return comprehensive results."""
        if not self.code.strip():
            return {"error": "No code provided", "has_issues": True}

        try:
            self._check_syntax()
            self._check_complexity()
            self._check_variables()
            self._check_security()
            self._check_performance()
            self._check_best_practices()
            self._calculate_metrics()
            
            return {
                "issues": self.issues,
                "metrics": self.metrics,
                "has_issues": bool(self.issues)
            }
        except Exception as e:
            return {"error": str(e), "has_issues": True}

    def _check_syntax(self):
        """Check for syntax errors and basic structural issues."""
        try:
            ast.parse(self.code)
        except SyntaxError as e:
            self.issues.append({
                "type": "syntax",
                "severity": "high",
                "message": f"Syntax Error: {str(e)}",
                "line": e.lineno,
                "offset": e.offset
            })

    def _check_complexity(self):
        """Analyze code complexity using various metrics."""
        try:
            # Cyclomatic complexity
            for block in radon_cc.cc_visit(self.code):
                if block.complexity > 10:
                    self.issues.append({
                        "type": "complexity",
                        "severity": "medium",
                        "message": f"High complexity ({block.complexity}) in function '{block.name}'",
                        "line": block.lineno
                    })
                
            # Cognitive complexity
            cognitive_complexity = radon_metrics.h_visit(self.code)
            if cognitive_complexity > 15:
                self.issues.append({
                    "type": "complexity",
                    "severity": "medium",
                    "message": f"High cognitive complexity: {cognitive_complexity}"
                })
        except:
            pass

    def _check_variables(self):
        """Analyze variable usage and potential issues."""
        class VariableAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.defined = set()
                self.used = set()
                self.unused = set()
                self.undefined = set()
                self.reassigned = set()

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    if node.id in self.defined:
                        self.reassigned.add(node.id)
                    self.defined.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    self.used.add(node.id)
                    if node.id not in self.defined and node.id not in __builtins__:
                        self.undefined.add(node.id)
                self.generic_visit(node)

        try:
            tree = ast.parse(self.code)
            analyzer = VariableAnalyzer()
            analyzer.visit(tree)
            
            # Report undefined variables
            for var in analyzer.undefined:
                self.issues.append({
                    "type": "variable",
                    "severity": "high",
                    "message": f"Undefined variable: '{var}'"
                })
            
            # Report unused variables
            unused = analyzer.defined - analyzer.used
            for var in unused:
                if not var.startswith('_'):  # Skip variables starting with underscore
                    self.issues.append({
                        "type": "variable",
                        "severity": "low",
                        "message": f"Unused variable: '{var}'"
                    })
        except:
            pass

    def _check_security(self):
        """Check for common security issues."""
        security_patterns = {
            r"eval\(": "Use of eval() is potentially dangerous",
            r"exec\(": "Use of exec() is potentially dangerous",
            r"__import__\(": "Dynamic imports could be security risk",
            r"subprocess\.": "Subprocess calls should be carefully reviewed",
            r"os\.system\(": "Direct system calls are potentially dangerous",
            r"pickle\.loads?\(": "Pickle usage could be a security risk",
            r"input\(": "Unvalidated input could be dangerous"
        }

        for pattern, message in security_patterns.items():
            if re.search(pattern, self.code):
                self.issues.append({
                    "type": "security",
                    "severity": "high",
                    "message": message
                })

    def _check_performance(self):
        """Analyze potential performance issues."""
        class PerformanceAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.issues = []

            def visit_For(self, node):
                # Check for list operations in loops
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name):
                        if node.iter.func.id == 'range' and len(node.iter.args) == 1:
                            if isinstance(node.iter.args[0], ast.Call):
                                if isinstance(node.iter.args[0].func, ast.Name):
                                    if node.iter.args[0].func.id == 'len':
                                        self.issues.append({
                                            "message": "Consider using enumerate() instead of range(len())",
                                            "line": node.lineno
                                        })
                self.generic_visit(node)

        try:
            tree = ast.parse(self.code)
            analyzer = PerformanceAnalyzer()
            analyzer.visit(tree)
            
            for issue in analyzer.issues:
                self.issues.append({
                    "type": "performance",
                    "severity": "medium",
                    "message": issue["message"],
                    "line": issue.get("line")
                })
        except:
            pass

    def _check_best_practices(self):
        """Check for Python best practices and common anti-patterns."""
        class BestPracticesAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.issues = []

            def visit_FunctionDef(self, node):
                # Check for function length
                if len(node.body) > 20:
                    self.issues.append({
                        "message": f"Function '{node.name}' is too long ({len(node.body)} lines)",
                        "line": node.lineno
                    })
                
                # Check for number of parameters
                args_count = len(node.args.args)
                if args_count > 5:
                    self.issues.append({
                        "message": f"Function '{node.name}' has too many parameters ({args_count})",
                        "line": node.lineno
                    })
                self.generic_visit(node)

        try:
            tree = ast.parse(self.code)
            analyzer = BestPracticesAnalyzer()
            analyzer.visit(tree)
            
            for issue in analyzer.issues:
                self.issues.append({
                    "type": "best_practice",
                    "severity": "low",
                    "message": issue["message"],
                    "line": issue.get("line")
                })
        except:
            pass

    def _calculate_metrics(self):
        """Calculate various code metrics."""
        try:
            self.metrics = {
                "loc": len(self.code.splitlines()),
                "lloc": radon_metrics.sloc(self.code),
                "complexity": radon_cc.average_complexity(self.code),
                "maintainability": radon_metrics.mi_visit(self.code, True),
                "difficulty": radon_metrics.h_visit(self.code)
            }
        except:
            pass

@app.route('/code', methods=['GET', 'POST'])
def code():
    if request.method == 'POST':
        code = request.form.get('code', '')
        analyzer = CodeAnalyzer(code)
        results = analyzer.analyze_all()
        return render_template('index.html',
                             code=code,
                             results=results["issues"],
                             metrics=results["metrics"],
                             has_issues=results["has_issues"])
    return render_template('code_analyse.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_api():
    """API endpoint for code analysis."""
    code = request.json.get('code', '')
    analyzer = CodeAnalyzer(code)
    return jsonify(analyzer.analyze_all())

@app.route('/api/analyze-file', methods=['POST'])
def analyze_file_api():
    """API endpoint for analyzing uploaded Python files."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read the file content
            content = file.read().decode('utf-8')
            
            # Use the existing CodeAnalyzer class to analyze the code
            analyzer = CodeAnalyzer(content)
            results = analyzer.analyze_all()
            
            # Add file information to the results
            results['filename'] = secure_filename(file.filename)
            results['timestamp'] = "2025-01-05 09:14:57"  # Use your current timestamp
            results['analyzed_by'] = "Tanzeelsalfi"  # Use your current user
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'error': f'Error analyzing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a Python file.'}), 400

@app.route('/api/export-pdf', methods=['POST'])
def export_pdf_api():
    """API endpoint for exporting analysis results as PDF."""
    try:
        data = request.json
        if not data:
            logging.error("No data provided")
            return jsonify({'error': 'No data provided'}), 400

        # Create BytesIO object to store PDF
        buffer = BytesIO()
        
        # Create the PDF document using reportlab
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        heading2_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Create custom styles
        code_style = ParagraphStyle(
            'CodeStyle',
            parent=styles['Code'],
            fontSize=8,
            leading=10,
            fontName='Courier'
        )
        
        # Create the document content
        content = []
        
        # Add title
        content.append(Paragraph("Code Analysis Report", title_style))
        content.append(Spacer(1, 12))
        
        # Add metadata
        content.append(Paragraph(f"Generated on: {CURRENT_UTC}", normal_style))
        content.append(Paragraph(f"Analyzed by: {CURRENT_USER}", normal_style))
        content.append(Spacer(1, 20))
        
        # Add code section if exists
        if data.get('code'):
            content.append(Paragraph("Analyzed Code", heading2_style))
            content.append(Spacer(1, 12))
            
            # Wrap code to prevent overflow
            wrapped_code = textwrap.fill(data['code'], width=80)
            content.append(Paragraph(wrapped_code.replace('<', '&lt;').replace('>', '&gt;'), code_style))
            content.append(Spacer(1, 20))
        
        # Add metrics section
        if data.get('metrics'):
            content.append(Paragraph("Metrics", heading2_style))
            content.append(Spacer(1, 12))
            
            # Create metrics table
            metrics_data = [[Paragraph("Metric", normal_style), Paragraph("Value", normal_style)]]
            for key, value in data['metrics'].items():
                formatted_key = key.replace('_', ' ').title()
                metrics_data.append([
                    Paragraph(formatted_key, normal_style),
                    Paragraph(str(value), normal_style)
                ])
            
            metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            content.append(metrics_table)
            content.append(Spacer(1, 20))
        
        # Add issues section
        if data.get('issues'):
            content.append(Paragraph("Issues Found", heading2_style))
            content.append(Spacer(1, 12))
            
            for issue in data['issues']:
                # Create a colored box based on severity
                severity_colors = {
                    'high': colors.red,
                    'medium': colors.orange,
                    'low': colors.yellow
                }
                issue_color = severity_colors.get(issue['severity'], colors.grey)
                
                content.append(Paragraph(
                    f"<font color='{issue_color.hexval()}'>"
                    f"<strong>{issue['type'].title()}</strong> ({issue['severity'].title()})"
                    f"{'<br>Line ' + str(issue['line']) if 'line' in issue else ''}"
                    f"</font><br/>{issue['message']}",
                    normal_style
                ))
                content.append(Spacer(1, 12))

        # Build the PDF
        doc.build(content)

        # Create the response
        response = make_response(buffer.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=code_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        response.mimetype = 'application/pdf'

        buffer.close()
        return response

    except Exception as e:
        logging.error(f"PDF generation failed: {str(e)}")
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500


if __name__ == '__main__':
    logging.info(f"Starting application at {CURRENT_UTC}")
    logging.info(f"Current user: {CURRENT_USER}")
    app.run(debug=False, use_reloader=False)

# ================== END OF APP.PY ==================