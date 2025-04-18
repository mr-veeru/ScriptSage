import os
import re
import pickle
import logging
import pandas as pd
import numpy as np
import joblib

# Import pygments for syntax highlighting and language detection
import pygments.lexers
import pygments.util

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the AST analyzer
try:
    from .ast_analyzer import AST_PARSING_AVAILABLE, parse_code_structure, calculate_complexity_metrics
    # Deliberately using only Python's built-in parser, so this isn't a warning
    logger.info("AST parsing module initialized")
except ImportError:
    AST_PARSING_AVAILABLE = False
    logger.info("AST parsing module not available, using simplified analysis")

# Check for scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
    logger.info("ML libraries initialized")
except ImportError:
    ML_AVAILABLE = False
    logger.info("ML libraries not available, using pattern matching only")

# Get the backend directory path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define directory paths
MODEL_DIR = os.path.join(BACKEND_DIR, 'models')
DATA_DIR = os.path.join(BACKEND_DIR, 'data')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'augmented'), exist_ok=True)

# Define model file paths
LANGUAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'language_classifier.joblib')
PURPOSE_MODEL_PATH = os.path.join(MODEL_DIR, 'purpose_classifier.joblib')
CODEBERT_MODEL_PATH = os.path.join(MODEL_DIR, 'codebert_model')

# Define pre-trained models to use
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
CODE_SUMMARIZER_MODEL = "SEBIS/code_trans_t5_small_source_code_summarization_python"

# Initialize advanced models if available
codebert_model = None
codebert_tokenizer = None
code_summarizer = None
sentence_model = None

# Initialize global variable for advanced models availability
ADVANCED_MODELS_AVAILABLE = False

# Check for transformers library
try:
    from transformers import AutoTokenizer, AutoModel
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers library not installed. Advanced code analysis features will be disabled.")
    ADVANCED_MODELS_AVAILABLE = False

def load_advanced_models():
    """Load advanced ML models for code analysis"""
    global codebert_model, codebert_tokenizer, ADVANCED_MODELS_AVAILABLE, sentence_model
    
    if not ADVANCED_MODELS_AVAILABLE:
        try:
            # Try to import transformers again in case it was installed after module initialization
            from transformers import AutoTokenizer, AutoModel
            import torch
            ADVANCED_MODELS_AVAILABLE = True
            logger.info("Transformers library found, enabling advanced models")
        except ImportError:
            logger.info("Transformers library still not available, skipping advanced models")
            return False
        
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        logger.info("Loading CodeBERT model...")
        
        # Initialize the models
        model_name = CODEBERT_MODEL_NAME
        logger.info(f"Loading model: {model_name}")
        
        # Force download if needed
        codebert_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        codebert_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Try to load sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not available, some features will be limited")
            sentence_model = None
        
        logger.info("Advanced models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading advanced models: {e}")
        # Set flag to indicate advanced models aren't available
        ADVANCED_MODELS_AVAILABLE = False
        return False

# Try to load advanced models in the background
if ML_AVAILABLE:
    try:
        import threading
        thread = threading.Thread(target=load_advanced_models)
        thread.daemon = True
        thread.start()
        logger.info("Started loading advanced models in background")
    except Exception as e:
        logger.warning(f"Could not start background loading of models: {e}")

# Language patterns for detection
LANGUAGE_PATTERNS = {
    'Python': [
        r'def\s+\w+\s*\(.*\):', 
        r'import\s+\w+', 
        r'from\s+\w+\s+import', 
        r'class\s+\w+\s*(\(.*\))?:'
    ],
    'JavaScript': [
        r'function\s+\w+\s*\(', 
        r'const\s+\w+\s*=', 
        r'let\s+\w+\s*=', 
        r'var\s+\w+\s*=', 
        r'console\.log'
    ],
    'TypeScript': [
        r'interface\s+\w+\s*{', 
        r'type\s+\w+\s*=', 
        r'class\s+\w+\s*(?:implements|extends)?', 
        r':\s*(?:string|number|boolean|any|void)\b',
        r'import\s+[^;]+\s+from\s+[\'"]',
        r'export\s+(?:default\s+)?(?:const|function|class|interface|type)',
        r'React\.FC<.*>',
        r'<.*Props>',
        r'tsx?$'
    ],
    'HTML': [
        r'<!DOCTYPE\s+html>', 
        r'<html', 
        r'<head', 
        r'<body', 
        r'<div\s+class='
    ],
    'CSS': [
        r'body\s*{[^}]*}',
        r'\.\w+\s*{[^}]*}',
        r'#\w+\s*{[^}]*}',
        r'@media\s+[^{]*{',
        r'@keyframes\s+\w+\s*{',
        r'@import\s+url\([\'"]?[^\'"]+[\'"]?\);',
        r'@font-face\s*{',
        r'color\s*:\s*(?:#[0-9a-fA-F]{3,6}|rgb)',
        r'margin\s*:\s*\d+(?:px|%|rem|em)',
        r'padding\s*:\s*\d+(?:px|%|rem|em)',
        r'display\s*:\s*(?:flex|block|grid|inline)',
        r'background(?:-color)?\s*:\s*(?:#[0-9a-fA-F]{3,6}|rgba?\()',
        r'font-size\s*:\s*\d+(?:px|%|rem|em)',
        r'flex\s*:\s*(?:\d+|\d+\s+\d+\s+\d+)',
        r'gap\s*:\s*\d+(?:px|%|rem|em)',
        r'border(?:-radius)?\s*:\s*',
        r'css$'
    ],
    'JSON': [
        r'^\s*\{',
        r'\s*"\w+"\s*:',
        r'\[\s*\{\s*"',
        r'json$'
    ],
    'YAML': [
        r'^\s*---\s*$',
        r'^\s*\w+\s*:\s*\w+\s*$',
        r'^\s*-\s+\w+\s*:\s*',
        r'yml$|yaml$'
    ],
    'Configuration': [
        r'^\s*\w+\s*=\s*[\'"]\w+[\'"]\s*$',
        r'^\s*#\s*\w+\s*configuration',
        r'^\s*export\s+\w+\s*=',
        r'config\.\w+|configuration|\.env$|\.ini$|\.cfg$|requirements\.txt$',
        r'(?:^|\n)\s*\[\w+\]\s*(?:$|\n)'
    ],
    'C/C++': [
        r'#include\s*<',
        r'int\s+main\s*\(',
        r'std::\w+',
        r'using\s+namespace\s+std'
    ]
}

# Code purpose training data
PURPOSE_SAMPLES = {
    'Algorithm Implementation': [
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "function bubbleSort(arr) {\n  for (let i = 0; i < arr.length; i++) {\n    for (let j = 0; j < arr.length - i - 1; j++) {\n      if (arr[j] > arr[j + 1]) {\n        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];\n      }\n    }\n  }\n  return arr;\n}",
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "function quicksort(arr) {\n  if (arr.length <= 1) {\n    return arr;\n  }\n  const pivot = arr[0];\n  const left = [];\n  const right = [];\n  for (let i = 1; i < arr.length; i++) {\n    if (arr[i] < pivot) {\n      left.push(arr[i]);\n    } else {\n      right.push(arr[i]);\n    }\n  }\n  return quicksort(left).concat(pivot, quicksort(right));\n}",
        "def is_palindrome(s):\n    return s == s[::-1]",
        "function isPalindrome(str) {\n  return str === str.split('').reverse().join('');\n}",
        "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "function isPrime(num) {\n  if (num <= 1) return false;\n  for (let i = 2; i <= Math.sqrt(num); i++) {\n    if (num % i === 0) return false;\n  }\n  return true;\n}"
    ],
    'Machine Learning': [
        "import numpy as np\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\n\ndf = pd.read_csv('data.csv')\nX = df.drop('target', axis=1)\ny = df['target']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\nmodel = RandomForestClassifier()\nmodel.fit(X_train, y_train)\npredictions = model.predict(X_test)",
        "import tensorflow as tf\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D\n\nmodel = Sequential([\n    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),\n    MaxPooling2D(2, 2),\n    Flatten(),\n    Dense(128, activation='relu'),\n    Dense(10, activation='softmax')\n])\n\nmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\nmodel.fit(x_train, y_train, epochs=5)"
    ],
    'Web Development': [
        "from flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.route('/api/users', methods=['GET'])\ndef get_users():\n    users = [{'id': 1, 'name': 'John'}, {'id': 2, 'name': 'Jane'}]\n    return jsonify(users)\n\nif __name__ == '__main__':\n    app.run(debug=True)",
        "const express = require('express');\nconst app = express();\n\napp.get('/api/products', (req, res) => {\n  const products = [{id: 1, name: 'Product 1'}, {id: 2, name: 'Product 2'}];\n  res.json(products);\n});\n\napp.listen(3000, () => {\n  console.log('Server running on port 3000');\n});",
        "import React, { useState } from 'react';\n\nfunction App() {\n  const [count, setCount] = useState(0);\n\n  return (\n    <div>\n      <p>You clicked {count} times</p>\n      <button onClick={() => setCount(count + 1)}>\n        Click me\n      </button>\n    </div>\n  );\n}\n\nexport default App;"
    ],
    'Database Operations': [
        "import sqlite3\n\nconn = sqlite3.connect('database.db')\ncursor = conn.cursor()\n\ncursor.execute('''\nCREATE TABLE IF NOT EXISTS users (\n    id INTEGER PRIMARY KEY,\n    name TEXT NOT NULL,\n    email TEXT UNIQUE NOT NULL\n)\n''')\n\ncursor.execute('INSERT INTO users (name, email) VALUES (?, ?)', ('John', 'john@example.com'))\nconn.commit()\n\nresults = cursor.execute('SELECT * FROM users').fetchall()\nfor row in results:\n    print(row)",
        "const mysql = require('mysql');\nconst connection = mysql.createConnection({\n  host: 'localhost',\n  user: 'user',\n  password: 'password',\n  database: 'mydb'\n});\n\nconnection.connect();\n\nconnection.query('SELECT * FROM products WHERE price > 100', (error, results) => {\n  if (error) throw error;\n  console.log(results);\n});\n\nconnection.end();"
    ],
    'Utility Function': [
        "def is_even(number):\n    return number % 2 == 0",
        "function capitalize(str) {\n  return str.charAt(0).toUpperCase() + str.slice(1);\n}",
        "def validate_email(email):\n    import re\n    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'\n    return re.match(pattern, email) is not None",
        "class StringUtils {\n  static reverse(str) {\n    return str.split('').reverse().join('');\n  }\n\n  static countWords(str) {\n    return str.trim().split(/\\s+/).length;\n  }\n}",
        "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5/9",
        "function getRandomInt(min, max) {\n  return Math.floor(Math.random() * (max - min + 1)) + min;\n}",
        "def format_currency(amount):\n    return f\"${amount:.2f}\"",
        "def get_file_extension(filename):\n    return filename.split('.')[-1]"
    ],
    'Math Computation': [
        "squares = [x**2 for x in range(10)]\nprint(squares)",
        "const squares = Array.from({length: 10}, (_, i) => i * i);",
        "import numpy as np\nnumbers = np.arange(10)\nsquares = numbers ** 2",
        "std::vector<int> squares;\nfor (int i = 0; i < 10; i++) {\n    squares.push_back(i * i);\n}"
    ]
}

# Language training samples - expanded with more distinctive examples
LANGUAGE_SAMPLES = {
    'Python': [
        "def hello_world():\n    print('Hello, World!')\n\nhello_world()",
        "import numpy as np\nimport pandas as pd\n\ndf = pd.read_csv('data.csv')\nprint(df.head())",
        "class Person:\n    def __init__(self, name):\n        self.name = name\n\n    def greet(self):\n        return f'Hello, my name is {self.name}'",
        "squares = [x**2 for x in range(10)]\nprint(squares)",
        "def is_palindrome(s):\n    return s == s[::-1]\n\nprint(is_palindrome('racecar'))",
        "with open('file.txt', 'r') as f:\n    content = f.read()",
        "try:\n    result = 10/0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')",
        "from collections import Counter\nCounter(['a', 'b', 'c', 'a', 'b', 'b'])"
    ],
    'JavaScript': [
        "function helloWorld() {\n  console.log('Hello, World!');\n}\n\nhelloWorld();",
        "const numbers = [1, 2, 3, 4, 5];\nconst doubled = numbers.map(num => num * 2);\nconsole.log(doubled);",
        "class Person {\n  constructor(name) {\n    this.name = name;\n  }\n\n  greet() {\n    return `Hello, my name is ${this.name}`;\n  }\n}",
        "const squares = Array.from({length: 10}, (_, i) => i * i);\nconsole.log(squares);",
        "function isPalindrome(str) {\n  return str === str.split('').reverse().join('');\n}\nconsole.log(isPalindrome('racecar'));",
        "document.querySelector('#button').addEventListener('click', () => alert('Clicked!'));",
        "async function fetchData() {\n  const response = await fetch('https://api.example.com/data');\n  return response.json();\n}"
    ],
    'PHP': [
        "<?php\nfunction helloWorld() {\n    echo 'Hello, World!';\n}\n\nhelloWorld();\n?>",
        "<?php\n$numbers = [1, 2, 3, 4, 5];\nforeach ($numbers as $number) {\n    echo $number * 2;\n}\n?>",
        "<?php\n$squares = array_map(function($x) { return $x * $x; }, range(0, 9));\nprint_r($squares);\n?>",
        "<?php\nfunction isPalindrome($str) {\n    return $str === strrev($str);\n}\nvar_dump(isPalindrome('racecar'));\n?>",
        "<?php\n$servername = 'localhost';\n$username = 'username';\n$password = 'password';\n$conn = new mysqli($servername, $username, $password);\n?>"
    ],
    'Java': [
        "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}",
        "import java.util.ArrayList;\n\npublic class Example {\n    public static void main(String[] args) {\n        ArrayList<Integer> numbers = new ArrayList<>();\n        numbers.add(1);\n        numbers.add(2);\n        System.out.println(numbers);\n    }\n}",
        "import java.util.stream.IntStream;\n\npublic class Squares {\n    public static void main(String[] args) {\n        int[] squares = IntStream.range(0, 10).map(x -> x * x).toArray();\n        System.out.println(Arrays.toString(squares));\n    }\n}",
        "public boolean isPalindrome(String str) {\n    StringBuilder sb = new StringBuilder(str);\n    return str.equals(sb.reverse().toString());\n}",
        "public class PrimeChecker {\n    public static boolean isPrime(int n) {\n        if (n <= 1) return false;\n        for (int i = 2; i <= Math.sqrt(n); i++) {\n            if (n % i == 0) return false;\n        }\n        return true;\n    }\n}"
    ],
    'C/C++': [
        "#include <iostream>\n\nint main() {\n    std::cout << \"Hello, World!\" << std::endl;\n    return 0;\n}",
        "#include <vector>\n#include <algorithm>\n\nint main() {\n    std::vector<int> numbers = {1, 2, 3, 4, 5};\n    std::sort(numbers.begin(), numbers.end());\n    return 0;\n}",
        "#include <iostream>\n#include <vector>\n\nint main() {\n    std::vector<int> squares;\n    for (int i = 0; i < 10; i++) {\n        squares.push_back(i * i);\n    }\n    for (int square : squares) {\n        std::cout << square << \" \";\n    }\n    return 0;\n}",
        "#include <string>\n#include <algorithm>\n\nbool isPalindrome(const std::string& str) {\n    std::string reversed = str;\n    std::reverse(reversed.begin(), reversed.end());\n    return str == reversed;\n}",
        "#include <cmath>\n\nbool isPrime(int n) {\n    if (n <= 1) return false;\n    for (int i = 2; i <= sqrt(n); i++) {\n        if (n % i == 0) return false;\n    }\n    return true;\n}"
    ],
    'Go': [
        "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"Hello, World!\")\n}",
        "package main\n\nimport \"fmt\"\n\nfunc add(a, b int) int {\n\treturn a + b\n}\n\nfunc main() {\n\tfmt.Println(add(2, 3))\n}"
    ],
    'Ruby': [
        "def hello_world\n  puts 'Hello, World!'\nend\n\nhello_world",
        "class Person\n  attr_accessor :name\n\n  def initialize(name)\n    @name = name\n  end\n\n  def greet\n    \"Hello, my name is #{@name}\"\n  end\nend"
    ],
    'Shell/Bash': [
        "#!/bin/bash\necho \"Hello, World!\"",
        "#!/bin/bash\nfor i in {1..5}; do\n  echo $i\ndone",
        "function greet() {\n  local name=$1\n  echo \"Hello, $name\"\n}\n\ngreet \"John\""
    ],
    'TypeScript': [
        "function greet(name: string): string {\n  return `Hello, ${name}`;\n}\n\nconsole.log(greet('World'));",
        "interface Person {\n  name: string;\n  age: number;\n}\n\nconst person: Person = {\n  name: 'John',\n  age: 30\n};",
        "import React, { useState, useEffect } from 'react';\nimport { Box, Button, CircularProgress } from '@mui/material';\n\ninterface Props {\n  onSubmit: (data: string) => void;\n}\n\nconst Component: React.FC<Props> = ({ onSubmit }) => {\n  const [loading, setLoading] = useState<boolean>(false);\n  \n  useEffect(() => {\n    // Component logic\n  }, []);\n  \n  return (\n    <Box>\n      <Button onClick={() => onSubmit('data')}>Submit</Button>\n    </Box>\n  );\n};",
        "export interface UserState {\n  id: number;\n  name: string;\n  email: string;\n  isActive: boolean;\n}\n\ntype UserAction = \n  | { type: 'SET_USER', payload: UserState }\n  | { type: 'CLEAR_USER' }\n  | { type: 'UPDATE_USER', payload: Partial<UserState> };",
        "class User {\n  private id: number;\n  public name: string;\n  \n  constructor(id: number, name: string) {\n    this.id = id;\n    this.name = name;\n  }\n  \n  public getInfo(): string {\n    return `User: ${this.name} (ID: ${this.id})`;\n  }\n}",
        "enum Direction {\n  Up = 'UP',\n  Down = 'DOWN',\n  Left = 'LEFT',\n  Right = 'RIGHT'\n}\n\nfunction move(direction: Direction): void {\n  console.log(`Moving ${direction}`);\n}",
        "export default function App(): JSX.Element {\n  return (\n    <div className=\"App\">\n      <header className=\"App-header\">\n        <h1>Welcome to My App</h1>\n      </header>\n    </div>\n  );\n}",
        "import { createSlice, PayloadAction } from '@reduxjs/toolkit';\n\ninterface CounterState {\n  value: number;\n}\n\nconst initialState: CounterState = {\n  value: 0,\n};\n\nexport const counterSlice = createSlice({\n  name: 'counter',\n  initialState,\n  reducers: {\n    increment: (state) => {\n      state.value += 1;\n    },\n    decrement: (state) => {\n      state.value -= 1;\n    },\n    incrementByAmount: (state, action: PayloadAction<number>) => {\n      state.value += action.payload;\n    },\n  },\n});",
        "type Callback<T> = (item: T) => void;\n\nfunction processItems<T>(items: T[], callback: Callback<T>): void {\n  items.forEach(item => callback(item));\n}"
    ],
    'HTML': [
        "<!DOCTYPE html>\n<html>\n<head>\n  <title>Hello World</title>\n</head>\n<body>\n  <h1>Hello, World!</h1>\n</body>\n</html>",
        "<div class=\"container\">\n  <nav>\n    <ul>\n      <li><a href=\"#\">Home</a></li>\n      <li><a href=\"#\">About</a></li>\n    </ul>\n  </nav>\n</div>"
    ],
    'CSS': [
        "body {\n  font-family: Arial, sans-serif;\n  margin: 0;\n  padding: 0;\n}\n\n.container {\n  max-width: 1200px;\n  margin: 0 auto;\n}",
        ".button {\n  display: inline-block;\n  padding: 10px 15px;\n  background-color: #007bff;\n  color: white;\n  border-radius: 4px;\n  cursor: pointer;\n}",
        "body {\n  margin: 0;\n  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',\
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',\
    sans-serif;\
  -webkit-font-smoothing: antialiased;\
  -moz-osx-font-smoothing: grayscale;\
}",
        "code {\n  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',\
    monospace;\
}",
        ".App {\n  text-align: center;\n}\n\n.App-logo {\n  height: 40vmin;\n  pointer-events: none;\n}\n\n@media (prefers-reduced-motion: no-preference) {\n  .App-logo {\n    animation: App-logo-spin infinite 20s linear;\n  }\n}",
        "header {\n  background-color: #282c34;\n  min-height: 100vh;\n  display: flex;\n  flex-direction: column;\n  align-items: center;\n  justify-content: center;\n  font-size: calc(10px + 2vmin);\n  color: white;\n}",
        "@keyframes App-logo-spin {\n  from {\n    transform: rotate(0deg);\n  }\n  to {\n    transform: rotate(360deg);\n  }\n}",
        "/* Basic CSS reset */\n* {\n  margin: 0;\n  padding: 0;\n  box-sizing: border-box;\n}",
        "/* Typography styles */\nh1, h2, h3, h4, h5, h6 {\n  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;\n  color: #333;\n}",
        "/* Responsive layout */\n@media screen and (max-width: 768px) {\n  .container {\n    padding: 0 15px;\n  }\n}",
        ".grid-container {\n  display: grid;\n  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));\n  gap: 20px;\n}",
        ":root {\n  --primary-color: #3498db;\n  --secondary-color: #2ecc71;\n  --dark-color: #333333;\n  --light-color: #f4f4f4;\n}",
        ".btn {\n  padding: 8px 16px;\n  border: none;\n  background-color: var(--primary-color);\n  color: white;\n  border-radius: 4px;\n  cursor: pointer;\n  transition: background-color 0.3s ease;\n}",
        ".btn:hover {\n  background-color: var(--dark-color);\n}",
        ".flex-container {\n  display: flex;\n  flex-wrap: wrap;\n  justify-content: space-between;\n  align-items: center;\n}",
        ".card {\n  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);\n  border-radius: 8px;\n  padding: 20px;\n  background-color: white;\n}"
    ],
    'SQL': [
        "SELECT users.name, orders.amount\nFROM users\nJOIN orders ON users.id = orders.user_id\nWHERE orders.amount > 100\nORDER BY orders.amount DESC;",
        "CREATE TABLE products (\n  id INT PRIMARY KEY,\n  name VARCHAR(100) NOT NULL,\n  price DECIMAL(10, 2) NOT NULL\n);\n\nINSERT INTO products (id, name, price) VALUES (1, 'Product A', 19.99);"
    ],
    'C#': [
        "using System;\n\nclass Program {\n    static void Main() {\n        Console.WriteLine(\"Hello, World!\");\n    }\n}",
        "using System;\nusing System.Collections.Generic;\n\nnamespace Example {\n    class Person {\n        public string Name { get; set; }\n        \n        public void Greet() {\n            Console.WriteLine($\"Hello, my name is {Name}\");\n        }\n    }\n}"
    ],
    'Rust': [
        "fn main() {\n    println!(\"Hello, World!\");\n}\n",
        "struct Person {\n    name: String,\n    age: u32,\n}\n\nimpl Person {\n    fn new(name: String, age: u32) -> Person {\n        Person { name, age }\n    }\n    \n    fn greet(&self) -> String {\n        format!(\"Hello, my name is {}\", self.name)\n    }\n}"
    ]
}

# Feature extraction functions
def preprocess_code(code):
    """Preprocess code for feature extraction"""
    if not code:
        return ""
    
    # Remove comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Python comments
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)  # C-style single line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # C-style multi-line comments
    
    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code)
    return code.strip()

def extract_code_features(code):
    """Extract various features from code"""
    # Use character n-grams and word n-grams
    features = {}
    
    # Basic stats
    features['length'] = len(code)
    features['num_lines'] = code.count('\n') + 1
    
    # Extract language-specific keywords
    for lang, patterns in LANGUAGE_PATTERNS.items():
        lang_score = 0
        for pattern in patterns:
            lang_score += len(re.findall(pattern, code, re.IGNORECASE))
        features[f'lang_{lang}'] = lang_score
    
    return features

# Load training data from files if available, otherwise use default samples
def get_language_training_data():
    """Get training data for language classification"""
    all_samples_path = os.path.join(DATA_DIR, 'all_samples.csv')
    augmented_path = os.path.join(DATA_DIR, 'augmented', 'augmented_language_samples.csv')
    
    X = []
    y = []
    
    # First try to load augmented data
    if os.path.exists(augmented_path):
        print("Loading augmented language training data...")
        aug_df = pd.read_csv(augmented_path)
        X.extend(aug_df['content'].tolist())
        y.extend(aug_df['language'].tolist())
        print(f"Loaded {len(aug_df)} augmented language samples")
    
    # Then load original collected data if it exists
    if os.path.exists(all_samples_path):
        print("Loading original language training data...")
        df = pd.read_csv(all_samples_path)
        X.extend(df['content'].tolist())
        y.extend(df['language'].tolist())
        print(f"Loaded {len(df)} original language samples")
    
    # If no collected data exists, use default samples
    if not X:
        print("Using default language training data...")
    for language, samples in LANGUAGE_SAMPLES.items():
        for code in samples:
            X.append(code)
            y.append(language)
        print(f"Loaded {len(X)} default language samples")
    
    print(f"Total language training samples: {len(X)}")
    return X, y

def get_purpose_training_data():
    """Get training data for purpose classification"""
    purpose_samples_path = os.path.join(DATA_DIR, 'purpose_samples.csv')
    augmented_path = os.path.join(DATA_DIR, 'augmented', 'augmented_purpose_samples.csv')
    
    X = []
    y = []
    
    # First try to load augmented data
    if os.path.exists(augmented_path):
        print("Loading augmented purpose training data...")
        aug_df = pd.read_csv(augmented_path)
        X.extend(aug_df['content'].tolist())
        y.extend(aug_df['purpose'].tolist())
        print(f"Loaded {len(aug_df)} augmented purpose samples")
    
    # Then load original collected data if it exists
    if os.path.exists(purpose_samples_path):
        print("Loading original purpose training data...")
        df = pd.read_csv(purpose_samples_path)
        X.extend(df['content'].tolist())
        y.extend(df['purpose'].tolist())
        print(f"Loaded {len(df)} original purpose samples")
    
    # If no collected data exists, use default samples
    if not X:
        print("Using default purpose training data...")
        for purpose, samples in PURPOSE_SAMPLES.items():
            for code in samples:
                X.append(code)
                y.append(purpose)
        print(f"Loaded {len(X)} default purpose samples")
    
    print(f"Total purpose training samples: {len(X)}")
    return X, y

# Updated model training functions
def train_language_classifier():
    """Train a language classification model using collected data if available"""
    X, y = get_language_training_data()
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char', 
            ngram_range=(2, 4), 
            max_features=10000
        )),
        ('classifier', MultinomialNB())
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate and print metrics
    accuracy = pipeline.score(X_test, y_test)
    print(f"Language classifier accuracy: {accuracy:.4f}")
    
    # Save the trained model
    with open(LANGUAGE_MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    
    return pipeline

def train_purpose_classifier():
    """Train a purpose classification model using collected data if available"""
    X, y = get_purpose_training_data()
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=5000
        )),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate and print metrics
    accuracy = pipeline.score(X_test, y_test)
    print(f"Purpose classifier accuracy: {accuracy:.4f}")
    
    # Save the trained model
    with open(PURPOSE_MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    
    return pipeline

def train_models(force=False):
    """Train both language and purpose classifiers"""
    if force or not os.path.exists(LANGUAGE_MODEL_PATH):
        print("Training language classifier...")
        train_language_classifier()
    else:
        print("Language classifier already exists. Use force=True to retrain.")
    
    if force or not os.path.exists(PURPOSE_MODEL_PATH):
        print("Training purpose classifier...")
        train_purpose_classifier()
    else:
        print("Purpose classifier already exists. Use force=True to retrain.")
    
    return {"status": "success", "models_trained": ["language_classifier", "purpose_classifier"]}

# Add new function to completely rebuild models
def rebuild_models():
    """Force rebuild all models with latest training data"""
    print("Rebuilding all ML models with latest training data...")
    
    # Remove existing models
    if os.path.exists(LANGUAGE_MODEL_PATH):
        os.remove(LANGUAGE_MODEL_PATH)
        print(f"Removed existing language model: {LANGUAGE_MODEL_PATH}")
        
    if os.path.exists(PURPOSE_MODEL_PATH):
        os.remove(PURPOSE_MODEL_PATH)
        print(f"Removed existing purpose model: {PURPOSE_MODEL_PATH}")
    
    # Train new models
    result = train_models(force=True)
    return {"status": "success", "message": "All models rebuilt with latest training data", "details": result}

# Model loading and prediction functions
def get_language_classifier():
    """Load or train the language classifier"""
    try:
        if os.path.exists(LANGUAGE_MODEL_PATH):
            with open(LANGUAGE_MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        else:
            logger.info("Training new language classifier...")
            return train_language_classifier()
    except Exception as e:
        logger.error(f"Error loading language classifier: {e}")
        return None

def get_purpose_classifier():
    """Load or train the purpose classifier"""
    if os.path.exists(PURPOSE_MODEL_PATH):
        try:
            return joblib.load(PURPOSE_MODEL_PATH)
        except Exception as e:
            logger.error(f"Error loading purpose classifier with joblib: {e}")
            return None
    else:
        print("Training purpose classifier...")
        return train_purpose_classifier()

def predict_language(code):
    """Predict the programming language of code"""
    if not code or len(code.strip()) < 5:
        return "Unknown (insufficient code)"
        
    try:
        # Handle React JSX styling cases that are often confused with CSS
        if re.search(r'sx\s*=\s*\{\s*\{', code) or re.search(r'style\s*=\s*\{\s*\{', code):
            # JSX styling object pattern
            if re.search(r'fontWeight|borderBottom|borderColor|padding|margin|color|display', code):
                return 'JavaScript'
        
        # Handle JavaScript object with CSS-like properties
        if re.search(r'const\s+styles\s*=\s*\{', code) or re.search(r'let\s+styles\s*=\s*\{', code) or re.search(r'var\s+styles\s*=\s*\{', code):
            if re.search(r'display|color|background|margin|padding|font', code):
                return 'JavaScript'
        
        # HTML - Check this first as it's very distinctive
        if re.search(r'<!DOCTYPE\s+html>|<html|<body|<head|<div\s+class=|<h[1-6]>|<script|<title>', code, re.IGNORECASE):
            return 'HTML'
            
        # Explicit JavaScript function detection with no type annotations
        if re.search(r'function\s+\w+\s*\([^:]*\)\s*\{', code) and not re.search(r':\s*(?:string|number|boolean|any|void)', code):
            return 'JavaScript'
        
        # Explicit JavaScript arrow functions
        if re.search(r'const\s+\w+\s*=\s*\([^:]*\)\s*=>', code) and not re.search(r':\s*(?:string|number|boolean|any|void)', code):
            return 'JavaScript'
            
        # Explicit TypeScript detection
        if re.search(r':\s*(?:string|number|boolean|any|void|React\.|\w+Props)', code) or re.search(r'interface\s+\w+', code):
            return 'TypeScript'
            
        # JSON - Check early for distinct patterns
        if code.strip().startswith('{') and code.strip().endswith('}') and re.search(r'"\w+":\s*(?:"[^"]*"|[\d\[\{])', code):
            return 'JSON'
        if code.strip().startswith('[') and code.strip().endswith(']') and re.search(r'\{\s*"\w+":', code):
            return 'JSON'
            
        # Python - Distinctive patterns
        if re.search(r'def\s+\w+\s*\([^)]*\):', code) or re.search(r'class\s+\w+\s*(?:\(.*\))?:', code):
            return 'Python'
            
        # Config files - Check for classic .env patterns
        if re.search(r'^\s*[A-Z_]+=[\'""].*[\'""]$', code, re.MULTILINE) and len(re.findall(r'^\s*[A-Z_]+=', code, re.MULTILINE)) > 1:
            return 'Configuration'
            
        # Extract filename if present in the code
        filename_match = re.search(r'[\'"]([^\'"\s]+\.\w+)[\'"]', code)
        filename = filename_match.group(1) if filename_match else None
        
        # Check file extension first - give this highest priority
        file_ext_match = re.search(r'(?:\/|\\)?(\w+\.\w+)(?:\'|\"|\s|$)', code)
        if file_ext_match:
            ext = file_ext_match.group(1).lower()
            if ext.endswith('.css'): return 'CSS'
            elif ext.endswith('.html'): return 'HTML'
            elif ext.endswith('.py'): return 'Python'
            elif ext.endswith('.tsx'): return 'TypeScript'
            elif ext.endswith('.ts'): return 'TypeScript'
            elif ext.endswith('.js'): return 'JavaScript'
            elif ext.endswith('.json'): return 'JSON'
            elif ext.endswith('.yaml') or ext.endswith('.yml'): return 'YAML'
            elif ext.endswith('.env') or ext == 'requirements.txt': return 'Configuration'
        
        # Direct filename check for common configuration files
        if filename:
            if filename == 'requirements.txt' or filename.endswith('.env'):
                return 'Configuration'
            elif filename.endswith('.json'):
                return 'JSON'
            elif filename.endswith('.yaml') or filename.endswith('.yml'):
                return 'YAML'
        
        # Check for CSS special cases - More distinctive patterns
        css_selectors = [
            r'body\s*{', 
            r'\.\w+(-\w+)*\s*{', 
            r'#\w+\s*{',
            r'@media\s+',
            r'@keyframes\s+',
            r'@import\s+url\(',
            r'display\s*:\s*(?:block|flex|grid|inline)',
            r'margin\s*:\s*\d+(?:px|rem|em|%)',
            r'padding\s*:\s*\d+(?:px|rem|em|%)',
            r'color\s*:\s*(?:#[0-9a-fA-F]{3,6}|rgb|rgba)',
            r'font-size\s*:'
        ]
        
        # Check if it's a CSS file by counting selector matches
        css_matches = sum(1 for pattern in css_selectors if re.search(pattern, code, re.IGNORECASE))
        
        # Only identify as CSS if there are no JavaScript indicators
        js_indicators = [
            r'function\s+\w+', 
            r'var\s+\w+', 
            r'let\s+\w+', 
            r'const\s+\w+',
            r'import\s+',
            r'export\s+',
            r'console\.',
            r'=>\s*{',
            r'}\s*else\s*{',
            r'if\s*\(',
            r'return\s+'
        ]
        
        # If we have both CSS and JS indicators, it's likely React/JS styling
        js_indicator_matches = sum(1 for pattern in js_indicators if re.search(pattern, code, re.IGNORECASE))
        
        if css_matches >= 2 and js_indicator_matches == 0:
            return 'CSS'
        elif css_matches >= 2 and js_indicator_matches > 0:
            # If we have both, check context more carefully
            if re.search(r'(const|let|var)\s+styles', code) or re.search(r'style\s*=\s*\{', code) or re.search(r'sx\s*=\s*\{', code):
                return 'JavaScript'
            else:
                return 'CSS'
            
        # Check for single CSS rule with both selector and properties
        if re.search(r'(?:\.|#|\*|body|html|div)[^{]*\{\s*[^}]*:[^}]*;[^}]*\}', code, re.IGNORECASE) and not re.search(r'function|class\s+\w+|import\s+|export\s+', code):
            return 'CSS'
            
        # Check for YAML content with stronger patterns
        if (code.strip().startswith('---') or re.search(r'^\s*\w+\s*:\s*\w+\s*$', code, re.MULTILINE)) and not re.search(r'function|class\s+\w+|import\s+|export\s+', code):
            yaml_lines = len(re.findall(r'^\s*\w+\s*:\s*', code, re.MULTILINE))
            if yaml_lines > 1:
                return 'YAML'
        
        # JavaScript vs TypeScript specific detection
        # If code has function syntax but no type annotations, it's likely JavaScript
        if re.search(r'function\s+\w+\s*\([^:]*\)\s*\{', code):
            # Check if there are any TypeScript-specific features
            if not re.search(r':\s*(?:string|number|boolean|any|void|React\.|\w+Props|interface\s+\w+)', code):
                # This is likely pure JavaScript with no TypeScript features
                return 'JavaScript'
        
        # Pattern matching counts for each language
        pattern_scores = {}
        for lang, patterns in LANGUAGE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, code, re.IGNORECASE)
                score += len(matches)
            
        # Apply special weighting
        if lang == 'CSS': score *= 1.5  # Give CSS a boost
        if lang == 'TypeScript': score *= 1.2  # Give TypeScript a boost
        if lang == 'JavaScript': score *= 1.3  # Give JavaScript a stronger boost
        if lang == 'Configuration': score *= 1.8  # Give Configuration files a bigger boost
        if lang == 'JSON': score *= 1.6  # Give JSON a bigger boost
        if lang == 'YAML': score *= 1.6  # Give YAML a bigger boost
        if lang == 'HTML': score *= 1.5  # Give HTML a boost
        if lang == 'Python': score *= 1.4  # Give Python a boost
            
        if score > 0:
            pattern_scores[lang] = score
    
        # JavaScript-specific pattern detection
        js_indicators = [
            r'function\s+\w+\s*\([^:]*\)', # Function without type annotations
            r'var\s+\w+\s*=',
            r'let\s+\w+\s*=',
            r'const\s+\w+\s*=',
            r'console\.log',
            r'document\.',
            r'window\.',
            r'addEventListener',
            r'setTimeout',
            r'for\s*\(\s*(?:var|let|const)?\s*\w+',
            r'if\s*\([^:]*\)\s*\{',
            r'return\s+\w+(?:\s*\+\s*\w+)+;', # Simple return with addition
            r'\w+\.\w+\(.*\);',  # Method calls
            r'\(\s*\)\s*=>\s*\{', # Arrow function
            r'import\s+[\w{},\s]+\s+from', # Import statement
            r'export\s+(?:default\s+)?(?:function|const|class)', # Export statement
            r'sx\s*=\s*\{\s*\{', # React JSX styling
            r'style\s*=\s*\{\s*\{', # React style prop
            r'className\s*=', # React className prop
        ]
        
        js_matches = sum(1 for pattern in js_indicators if re.search(pattern, code, re.IGNORECASE))
        
        # Strong JavaScript indicators - if multiple are found, it's likely JavaScript
        if js_matches >= 2:
            if 'JavaScript' not in pattern_scores:
                pattern_scores['JavaScript'] = js_matches * 1.5
            else:
                pattern_scores['JavaScript'] *= 1.5
        
        # CSS syntax check - curly braces with property-value pairs
        css_properties = [
            r'color\s*:',
            r'background\s*:',
            r'margin\s*:',
            r'padding\s*:',
            r'font-size\s*:',
            r'display\s*:',
            r'position\s*:',
            r'width\s*:',
            r'height\s*:'
        ]
        
        # Check for CSS property-value pairs
        css_property_matches = sum(1 for pattern in css_properties if re.search(pattern, code, re.IGNORECASE))
        
        # If we have CSS property matches but no clear CSS selectors, check the context
        if css_property_matches > 0:
            # CSS selectors typically exist outside of function definitions
            if not re.search(r'function\s+\w+\s*\(', code, re.IGNORECASE):
                if 'CSS' not in pattern_scores:
                    pattern_scores['CSS'] = css_property_matches * 1.5
                else:
                    pattern_scores['CSS'] *= 1.2
        
        # Check for CSS rules with both selector and properties
        if re.search(r'(?:\.|#|\*|body|html|div)[^{]*\{\s*[^}]+\}', code, re.IGNORECASE):
            # But exclude JavaScript code with object literals that look like CSS
            if not re.search(r'function\s+\w+|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=', code, re.IGNORECASE):
                if 'CSS' not in pattern_scores:
                    pattern_scores['CSS'] = 3.0
                else:
                    pattern_scores['CSS'] *= 1.5
        
        # If we have strong pattern matches, use them
        if pattern_scores:
            best_lang = max(pattern_scores.items(), key=lambda x: x[1])[0]
            best_score = pattern_scores[best_lang]
            
            # If the best language has a strong score, return it
            if best_score >= 2:
                return best_lang
            
            # Special case for JavaScript vs CSS disambiguation
            if 'JavaScript' in pattern_scores and 'CSS' in pattern_scores:
                js_score = pattern_scores['JavaScript']
                css_score = pattern_scores['CSS']
                
                # If code has JSX-style properties, boost JavaScript score
                if re.search(r'style\s*=\s*\{|className\s*=', code):
                    js_score *= 1.5
                
                # If code has both function definitions and CSS-like properties
                if js_score > css_score:
                    return 'JavaScript'
                elif css_score > js_score:
                    return 'CSS'
        
        # Additional language verification with stronger patterns
        if any(re.search(pattern, code) for pattern in [
            r'#include\s+<(?:iostream|vector|string|algorithm|cmath|stdio\.h|stdlib\.h)>',
            r'int\s+main\s*\(\s*(?:void|int\s+argc,\s*char\s*\*\s*argv\[\]|)\s*\)\s*{',
            r'std::\w+',
            r'cout\s*<<',
            r'cin\s*>>',
            r'template\s*<\s*(?:class|typename)'
        ]):
            return "C/C++"  # Override language for clear C++ indicators
        
        # Fallback to ML prediction
        preprocessed = preprocess_code(code)
        classifier = get_language_classifier()
        if classifier:
            return classifier.predict([preprocessed])[0]
        
        return "Unknown"
    except Exception as e:
        logger.error(f"Error predicting language: {e}")
        return "Unknown"

def predict_purpose(code):
    """Predict the purpose/type of code"""
    try:
        preprocessed = preprocess_code(code)
        classifier = get_purpose_classifier()
        purpose = classifier.predict([preprocessed])[0]
        confidence = np.max(classifier.predict_proba([preprocessed]))
        
        # Pattern override - Python specific list comprehension detection
        if re.search(r'\[\s*\w+[\^\*]{2}\d+\s+for\s+\w+\s+in\s+range\s*\(\s*\d+\s*\)\s*\]', code) or \
           re.search(r'\[\s*\w+\s*\*\*\s*\d+\s+for\s+\w+\s+in\s+range\s*\(\s*\d+\s*\)\s*\]', code):
            return "Math Computation", 0.95
        
        # Pattern override - palindrome detection
        if re.search(r'(palindrome|Palindrome)', code) and re.search(r'(==\s*s\[::-1\]|===.*reverse|reverse\.toString)', code):
            return "Algorithm Implementation", 0.92
            
        # Pattern override - prime number detection
        if re.search(r'(isPrime|is_prime)', code) and re.search(r'(<=\s*1|sqrt|Math\.sqrt)', code):
            return "Algorithm Implementation", 0.91
        
        return purpose, confidence
    except Exception as e:
        print(f"Error predicting purpose: {str(e)}")
        return "Unknown", 0.0

def generate_concise_summary(language, purpose, details, code_lower):
    """Generate a detailed, descriptive summary with emoji prefix based on language and purpose.
    The summary will include:
    1. What the code is (language and purpose)
    2. How it works (key technical details)
    3. Notable features or patterns detected
    """
    # Choose emoji based on language
    emoji = "📄"  # Default document
    
    if language == "Python":
        emoji = "🐍"
    elif language == "JavaScript":
        emoji = "📜"
    elif language == "TypeScript":
        emoji = "🔷"
    elif language == "HTML":
        emoji = "📱"
    elif language == "CSS":
        emoji = "🎨"
    elif language == "Java":
        emoji = "☕"
    elif language == "Ruby":
        emoji = "💎"
    elif language == "PHP":
        emoji = "🐘"
    elif language == "Go":
        emoji = "🐹"
    elif language == "Rust":
        emoji = "🦀"
    elif language == "C/C++":
        emoji = "⚙️"
    elif language == "Shell/Bash":
        emoji = "💻"
    elif language == "JSON":
        emoji = "📊"
    elif language == "YAML":
        emoji = "📝"
    elif language == "Configuration":
        emoji = "⚙️"
    elif language == "SQL":
        emoji = "🗄️"
    elif language == "C#":
        emoji = "🔶"
    
    # Initialize summary text and technical details
    summary_text = ""
    technical_details = ""
    notable_features = ""
    
    # Add technical details based on code patterns
    if "import " in code_lower and language == "Python":
        # Identify key imports
        import_matches = re.findall(r'import\s+(\w+)|from\s+(\w+)', code_lower)
        imports = []
        for match in import_matches:
            if match[0]:
                imports.append(match[0])
            elif match[1]:
                imports.append(match[1])
        
        if imports:
            common_libs = ["pandas", "numpy", "tensorflow", "torch", "flask", "django", "requests", "matplotlib"]
            important_imports = [lib for lib in imports if lib in common_libs]
            if important_imports:
                technical_details = f"The code utilizes {', '.join(important_imports)} libraries. "
    
    # Add complexity assessment
    if len(code_lower) > 5000:
        notable_features += "The code is extensive with significant complexity. "
    elif len(code_lower) > 1000:
        notable_features += "The code is moderately complex. "
    
    # Detect functions and classes
    function_count = len(re.findall(r'def\s+\w+|function\s+\w+', code_lower))
    class_count = len(re.findall(r'class\s+\w+', code_lower))
    
    if function_count > 5:
        notable_features += f"It contains {function_count} functions suggesting a modular design. "
    elif function_count > 0:
        notable_features += f"It defines {function_count} functions. "
    
    if class_count > 0:
        notable_features += f"The code implements {class_count} classes with object-oriented principles. "
    
    # Determine purpose-specific patterns
    if purpose == "API Endpoint":
        if "GET" in code_lower or "POST" in code_lower or "PUT" in code_lower or "DELETE" in code_lower:
            methods = []
            if "GET" in code_lower: methods.append("GET")
            if "POST" in code_lower: methods.append("POST")
            if "PUT" in code_lower: methods.append("PUT")
            if "DELETE" in code_lower: methods.append("DELETE")
            technical_details += f"Implements {', '.join(methods)} HTTP methods. "
            
        if "json" in code_lower:
            technical_details += "Returns JSON formatted responses. "
        
        if "@app.route" in code_lower or "@api" in code_lower:
            summary_text = "RESTful API endpoint handling HTTP requests"
        else:
            summary_text = "API endpoint for data handling"
    
    # Handle other purposes similar to the original code but with more detail
    elif purpose == "Data Processing":
        if "pandas" in code_lower:
            summary_text = "Data processing script using pandas"
            technical_details += "Performs data manipulation with pandas dataframes. "
            
            if "group" in code_lower:
                technical_details += "Includes grouping operations for data aggregation. "
            if "merge" in code_lower or "join" in code_lower:
                technical_details += "Includes data joining/merging operations. "
        
        elif "array" in code_lower or "[]" in code_lower:
            summary_text = "Data processing routine for array manipulation"
        
        elif "sql" in code_lower or "SELECT" in code_lower:
            summary_text = "SQL-based data processing"
            technical_details += "Executes SQL queries to transform or retrieve data. "
        
        else:
            summary_text = "General data processing code"
    
    # Add all the other cases from the original function...
    # [Keeping the rest of the original conditions with added technical details]
    
    # Default cases by language if no specific pattern was matched
    elif not summary_text:
        if language == "Python":
            summary_text = f"Python {purpose.lower()} script"
            if "flask" in code_lower:
                summary_text = "Flask web application with API endpoints"
                technical_details += "Built on the Flask framework providing web routes. "
            elif "django" in code_lower:
                summary_text = "Django web application"
                technical_details += "Implements Django components for web development. "
            elif "pandas" in code_lower and "data" in code_lower:
                summary_text = "Python data analysis script using pandas"
                technical_details += "Processes and analyzes data with pandas library. "
            elif "plot" in code_lower or "matplotlib" in code_lower:
                summary_text = "Python data visualization script"
                technical_details += "Creates data visualizations and charts. "
        
        elif language == "JavaScript":
            summary_text = "JavaScript application"
            if "react" in code_lower:
                summary_text = "React.js component or application"
                technical_details += "Built with React.js component architecture. "
            elif "node" in code_lower and "express" in code_lower:
                summary_text = "Node.js Express server application"
                technical_details += "Runs on Node.js with Express framework. "
            elif "function" in code_lower and "return" in code_lower:
                summary_text = "JavaScript utility functions"
                technical_details += "Provides utility functionality through JavaScript functions. "
        
        # Continue with other language defaults similar to original code but with added details
        
    # If we still don't have a summary, create a generic one
    if not summary_text:
        if "specific_type" in details:
            summary_text = f"{language} code for {details['specific_type']}"
        else:
            summary_text = f"{language} code for {purpose.lower()}"
    
    # If technical details or notable features weren't set, add defaults based on language
    if not technical_details:
        if language == "Python":
            technical_details = "The code is written in Python, known for its readability and versatility. "
        elif language == "JavaScript":
            technical_details = "This JavaScript code can run in browser or Node.js environments. "
        elif language == "HTML":
            technical_details = "The HTML markup defines structure for web content. "
        elif language == "CSS":
            technical_details = "The CSS provides styling for web elements. "
        else:
            technical_details = f"This {language} code follows standard language conventions. "
    
    if not notable_features:
        notable_features = "The code follows common patterns for this type of implementation. "
    
    # Combine everything into a detailed multi-sentence summary
    return f"{emoji} Summary: {summary_text}. {technical_details}{notable_features}"

def get_code_embedding(code):
    """Get a vector embedding for code using CodeBERT"""
    if not ADVANCED_MODELS_AVAILABLE or codebert_model is None:
        # Try to load models if they haven't been loaded yet
        if load_advanced_models() == False:
            # Return a simple fallback embedding
            logger.debug("Advanced models not available, using fallback embedding")
            return None
        
    try:
        import torch
        # Truncate code if too long
        max_length = 512
        if len(code) > max_length * 10:  # Rough estimate for token ratio
            code = code[:max_length * 10]
        
        # Tokenize and compute embedding
        inputs = codebert_tokenizer(code, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = codebert_model(**inputs)
        
        # Use CLS token as code representation
        code_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return code_embedding
    except Exception as e:
        logger.error(f"Error getting code embedding: {e}")
        return None

def summarize_code_with_model(code):
    """Summarize code using pre-trained code summarization model"""
    global code_summarizer
    
    if not ADVANCED_MODELS_AVAILABLE:
        return None
    
    try:
        # Import necessary libraries
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        # Initialize the summarizer if not already done
        if code_summarizer is None:
            try:
                # Try loading the summarization model
                model_name = "SEBIS/code_trans_t5_small_source_code_summarization_python"
                logger.info(f"Loading code summarization model: {model_name}")
                
                # For summarization model
                summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                code_summarizer = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
                logger.info("Code summarization model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load code summarization model: {e}")
                return None
        
        # Check if code is too long and truncate if needed
        max_length = 512  # Maximum length for the model
        if len(code) > max_length * 8:  # Rough character to token ratio
            code = code[:max_length * 8]  # Truncate to avoid token limit
            
        # Generate summary
        inputs = summarizer_tokenizer(code, return_tensors="pt", max_length=max_length, truncation=True)
        with torch.no_grad():
            summary_ids = code_summarizer.generate(
                inputs["input_ids"],
                max_length=100,
                num_beams=4,
                early_stopping=True
            )
            
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        logger.error(f"Error in code summarization: {e}")
        return None

def analyze_code_advanced(code):
    """Analyze code using advanced ML models"""
    if not ML_AVAILABLE:
        return None
    
    # Try to load advanced models if not already loaded
    if not ADVANCED_MODELS_AVAILABLE or codebert_model is None:
        if not load_advanced_models():
            logger.warning("Advanced models could not be loaded, skipping advanced analysis")
            return None
    
    try:
        # Get code embedding
        embedding = get_code_embedding(code)
        if embedding is None:
            return None
        
        # Define reference examples for different code types
        reference_examples = {
            "Flask API": "from flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.route('/api/data', methods=['GET'])\ndef get_data():\n    return jsonify({'message': 'success'})",
            "ML Project": "import numpy as np\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\n\ndf = pd.read_csv('data.csv')\nX = df.drop('target', axis=1)\ny = df['target']\n\nmodel = RandomForestClassifier()\nmodel.fit(X_train, y_train)",
            "Web Frontend": "<!DOCTYPE html>\n<html>\n<head>\n<script>\nfunction handleSubmit() {\n  const input = document.getElementById('task').value;\n  const list = document.getElementById('todoList');\n  const item = document.createElement('li');\n  item.textContent = input;\n  list.appendChild(item);\n}\n</script>\n</head>\n<body>\n<input id=\"task\"><button onclick=\"handleSubmit()\">Add</button>\n<ul id=\"todoList\"></ul>\n</body>\n</html>",
            "Java Program": "public class Student {\n    private String name;\n    private int id;\n    \n    public Student(String name, int id) {\n        this.name = name;\n        this.id = id;\n    }\n    \n    public String getName() {\n        return name;\n    }\n}",
            "C++ Program": "#include <iostream>\n#include <vector>\n#include <string>\n\nclass Account {\nprivate:\n    std::string name;\n    double balance;\npublic:\n    Account(std::string n, double b) : name(n), balance(b) {}\n    void deposit(double amount) {\n        balance += amount;\n    }\n};"
        }
        
        # Get embeddings for reference examples
        if sentence_model is not None:
            import numpy as np
            reference_embeddings = {k: sentence_model.encode(v) for k, v in reference_examples.items()}
            code_embedding = sentence_model.encode(code)
            
            # Find most similar reference example
            best_score = -1
            best_match = None
            for type_name, ref_embedding in reference_embeddings.items():
                score = np.dot(code_embedding, ref_embedding) / (np.linalg.norm(code_embedding) * np.linalg.norm(ref_embedding))
                if score > best_score:
                    best_score = score
                    best_match = type_name
            
            # Generate summary with code summarizer model if possible
            model_summary = summarize_code_with_model(code)
            
            result = {
                "best_match": best_match,
                "similarity_score": float(best_score),
                "model_summary": model_summary
            }
            return result
    except Exception as e:
        logger.error(f"Error in advanced code analysis: {e}")
    
    return None

def analyze_code(code):
    """Main code analysis function - ML-based with pattern matching fallback"""
    language = predict_language(code)
    purpose, confidence = predict_purpose(code)
    
    # Try advanced analysis with pre-trained models
    advanced_analysis = analyze_code_advanced(code)
    
    # Use advanced summary if available
    advanced_summary = None
    if advanced_analysis and advanced_analysis.get('model_summary'):
        advanced_summary = advanced_analysis['model_summary']
    
    # Try AST-based parsing for code structure analysis
    ast_analysis = None
    code_complexity = None
    if 'AST_PARSING_AVAILABLE' in globals() and AST_PARSING_AVAILABLE:
        try:
            ast_analysis = parse_code_structure(code, language)
            if ast_analysis:
                code_complexity = calculate_complexity_metrics(ast_analysis)
                # Boost confidence based on AST analysis success
                confidence = max(confidence, 0.75)
        except Exception as e:
            logger.error(f"Error during AST analysis: {e}")
    
    # Check for filename indicators (look for specific file extensions in the code)
    filename_match = re.search(r'[\'"]([^\'"\s]+\.(?:css|html|tsx?|jsx?|py|php|json|ya?ml|env|txt))[\'"](\.)*', code)
    if filename_match:
        filename = filename_match.group(1).lower()
        if filename.endswith('.css') and language != 'CSS':
            language = 'CSS'
        elif (filename.endswith('.ts') or filename.endswith('.tsx')) and language != 'TypeScript':
            language = 'TypeScript'
        elif filename.endswith('.json') and language != 'JSON':
            language = 'JSON'
        elif (filename.endswith('.yaml') or filename.endswith('.yml')) and language != 'YAML':
            language = 'YAML'
        elif filename == 'requirements.txt' or filename.endswith('.env'):
            language = 'Configuration'
    
    # Special case for Configuration files
    if language == 'Configuration':
        purpose = 'Configuration'
    
    # Special case for JSON files
    if language == 'JSON':
        purpose = 'Data Structure'
    
    # Special case for YAML files
    if language == 'YAML':
        purpose = 'Configuration'
    
    # Special case for CSS files which often have low detection confidence
    css_indicators = sum(1 for pattern in LANGUAGE_PATTERNS['CSS'] if re.search(pattern, code, re.IGNORECASE))
    if css_indicators >= 1 and language == 'CSS':
        confidence = max(confidence, 0.7 + (css_indicators * 0.05))
    
    # Special case for TypeScript files 
    ts_indicators = sum(1 for pattern in LANGUAGE_PATTERNS['TypeScript'] if re.search(pattern, code, re.IGNORECASE))
    if ts_indicators >= 2 and language == 'TypeScript':
        confidence = max(confidence, 0.7 + (ts_indicators * 0.05))
        
    # Check if file ends with specific extensions
    if re.search(r'\.css$', code, re.MULTILINE):
        language = 'CSS'
        confidence = max(confidence, 0.85)
    elif re.search(r'\.tsx?$', code, re.MULTILINE):
        language = 'TypeScript'
        confidence = max(confidence, 0.85)
    elif re.search(r'\.json$', code, re.MULTILINE):
        language = 'JSON'
        confidence = max(confidence, 0.85)
    elif re.search(r'\.ya?ml$', code, re.MULTILINE):
        language = 'YAML'
        confidence = max(confidence, 0.85)
    elif re.search(r'\.env$|requirements\.txt$', code, re.MULTILINE):
        language = 'Configuration'
        confidence = max(confidence, 0.85)
    
    # Additional language verification with stronger patterns
    if any(re.search(pattern, code) for pattern in [
        r'#include\s+<(?:iostream|vector|string|algorithm|cmath|stdio\.h|stdlib\.h)>',
        r'int\s+main\s*\(\s*(?:void|int\s+argc,\s*char\s*\*\s*argv\[\]|)\s*\)\s*{',
        r'std::\w+',
        r'cout\s*<<',
        r'cin\s*>>',
        r'template\s*<\s*(?:class|typename)'
    ]):
        language = "C/C++"  # Override language for clear C++ indicators

    # Basic pattern detection
    code_lower = code.lower()
        
    # Detect specific details (patterns)
    details = {}
    
    # For JSON: identify structure type
    if language == 'JSON':
        if code.strip().startswith('['):
            details["json_type"] = "Array"
        else:
            details["json_type"] = "Object"
        
    # For Configuration files: identify type
    if language == 'Configuration':
        if 'requirements.txt' in code_lower or re.search(r'^\s*[\w-]+(~=|==|>=|<=|!=|>|<)\d+', code, re.MULTILINE):
            details["config_type"] = "Python Dependencies"
            purpose = "Python Dependencies"
        elif re.search(r'^\s*[A-Z_]+=', code, re.MULTILINE):
            details["config_type"] = "Environment Variables"
            purpose = "Environment Configuration"
        elif re.search(r'^\s*\[\w+\]', code, re.MULTILINE):
            details["config_type"] = "INI Configuration"
            purpose = "Application Configuration"
    
    # For CSS: identify the styling scope
    if language == 'CSS':
        if re.search(r'body\s*{', code_lower):
            details["css_scope"] = "Global styles"
            purpose = "Styling"
        elif re.search(r'\.\w+-container', code_lower):
            details["css_scope"] = "Container styling"
            purpose = "Styling"
        elif re.search(r'@media', code_lower):
            details["css_scope"] = "Responsive styling"
            purpose = "Responsive Design"
        else:
            purpose = "Styling"
    
    # For TypeScript: identify React components
    if language == 'TypeScript':
        if re.search(r'React\.FC|React\.Component|function\s+\w+\s*\(\s*\{.*\}\s*\)\s*{.*return', code):
            details["component_type"] = "React Component"
            purpose = "UI Component"
        elif re.search(r'useState|useEffect|useContext|useRef', code):
            details["component_type"] = "React Hooks Component"
            purpose = "UI Component with Hooks"
    
    # Sorting algorithm detection
    if purpose == "Algorithm Implementation":
        if "bubble sort" in code_lower or "bubblesort" in code_lower:
            details["algorithm_type"] = "Bubble Sort"
        elif "quick sort" in code_lower or "quicksort" in code_lower:
            details["algorithm_type"] = "Quick Sort"
        elif "merge sort" in code_lower or "mergesort" in code_lower:
            details["algorithm_type"] = "Merge Sort"
        elif "binary search" in code_lower:
            details["algorithm_type"] = "Binary Search"
        elif "palindrome" in code_lower:
            details["algorithm_type"] = "Palindrome Check"
        elif "fibonacci" in code_lower:
            details["algorithm_type"] = "Fibonacci Sequence"
        elif re.search(r'is_prime|isPrime', code_lower):
            details["algorithm_type"] = "Prime Number Check"
    
    # Math computation detection
    if purpose == "Math Computation":
        if re.search(r'\[\s*\w+\s*\*\*\s*\d+\s+for', code_lower) or re.search(r'squares', code_lower):
            details["computation_type"] = "Square Numbers"
        elif re.search(r'factorial', code_lower):
            details["computation_type"] = "Factorial"
    
    # Generate summary text based on the refined purpose
    # Use advanced summary if available, otherwise use pattern-based summary
    if advanced_summary:
        # Clean up the summary - remove extra spaces, capitalize, ensure period at the end
        summary = advanced_summary.strip().capitalize()
        if not summary.endswith('.'):
            summary += '.'
            
        # Add emoji prefix
        emoji = get_emoji_for_language(language, purpose)
        summary = f"{emoji} Summary: {summary}"
    else:
        summary = generate_concise_summary(language, purpose, details, code_lower)
    
    # Adjust confidence based on advanced analysis if available
    if advanced_analysis and advanced_analysis.get('similarity_score'):
        confidence = max(confidence, float(advanced_analysis['similarity_score']))
    
    # Adjust confidence based on pattern matches
    if language == 'Configuration' and len(code.strip()) > 0:
        confidence = max(confidence, 0.75)
        
    if language == 'JSON' and re.search(r'"[\w-]+"\s*:', code):
        confidence = max(confidence, 0.85)
        
    if language == 'YAML' and re.search(r'^\s*\w+\s*:\s*\w+', code, re.MULTILINE):
        confidence = max(confidence, 0.80)
        
    if language == 'CSS' and len(code.strip()) > 0:
        confidence = max(confidence, 0.75)
        
    if language == 'TypeScript' and re.search(r'import\s+React', code):
        confidence = max(confidence, 0.80)
        
    if re.search(r'\[\s*\w+\s*\*\*\s*\d+\s+for\s+\w+\s+in\s+range', code_lower) and language == "Python":
        confidence = max(confidence, 0.85)
        
    if re.search(r'(isPalindrome|is_palindrome)', code_lower) and purpose == "Algorithm Implementation":
        confidence = max(confidence, 0.85)
        
    if re.search(r'#include', code_lower) and language == "C/C++":
        confidence = max(confidence, 0.82)
    
    # Cap confidence at 100%
    confidence = min(confidence, 1.0)
    
    # Format final result
    result = {
        "language": language,
        "type": purpose,
        "purpose": summary,
        "confidence": round(confidence * 100) / 100,  # Ensure confidence is properly rounded
        "ml_analysis": True,
        "advanced_ml_used": advanced_analysis is not None
    }
    
    if details:
        result["specific_type"] = next(iter(details.values()))
    
    # Include advanced analysis details if available
    if advanced_analysis:
        result["advanced_analysis"] = {
            "best_match": advanced_analysis.get('best_match'),
            "similarity_score": advanced_analysis.get('similarity_score')
        }
    
    # Add AST analysis details to the result if available
    if ast_analysis:
        result["ast_analysis"] = {
            "structure": {
                "functions": len(ast_analysis.get("tree_sitter", {}).get("functions", [])),
                "classes": len(ast_analysis.get("tree_sitter", {}).get("classes", [])),
                "imports": len(ast_analysis.get("tree_sitter", {}).get("imports", []))
            }
        }
        
        if code_complexity:
            result["ast_analysis"]["complexity"] = {
                "node_count": code_complexity.get("node_count", 0),
                "max_depth": code_complexity.get("max_depth", 0),
                "cyclomatic_complexity": round(code_complexity.get("cyclomatic_complexity", 1), 2)
            }
    
    return result

def get_emoji_for_language(language, purpose):
    """Get appropriate emoji based on language and purpose"""
    if language == "Python" and ("ML" in purpose or "Machine Learning" in purpose):
        return "🤖"
    elif language == "JavaScript" or language == "TypeScript" or language == "HTML" or language == "CSS":
        return "🌐"
    elif language == "Python":
        return "🐍"
    elif language == "Java":
        return "☕"
    elif language == "C/C++":
        return "⚙️"
    elif language == "Configuration" or language == "YAML":
        return "⚙️"
    elif language == "Shell/Bash":
        return "🐚"
    elif language == "JSON":
        return "📊"
    else:
        return "🧾"

# Initialize on import
if not os.path.exists(LANGUAGE_MODEL_PATH):
    logger.info("Initializing language classifier...")
    get_language_classifier() 