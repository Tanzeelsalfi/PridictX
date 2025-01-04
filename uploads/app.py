# PART 1: Imports, Configuration, and Setup
import nest_asyncio
from flask import Flask, render_template_string, request, send_file, flash, redirect, url_for, render_template, jsonify
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
    return render_template('index.html', features=feature_names)

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

if __name__ == '__main__':
    logging.info(f"Starting application at {CURRENT_UTC}")
    logging.info(f"Current user: {CURRENT_USER}")
    app.run(debug=False, use_reloader=False)

# ================== END OF APP.PY ==================
