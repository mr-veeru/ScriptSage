import React, { useState, useEffect } from 'react';
import { Box, Button, CircularProgress, Paper, Typography, useTheme, TextField } from '@mui/material';
import CodeEditor from '@uiw/react-textarea-code-editor';

interface CodeInputProps {
  onAnalysisComplete: (result: any) => void;
}

// Language patterns for syntax highlighting
const LANGUAGE_PATTERNS = {
  css: [
    /body\s*{[^}]*}/i,
    /\.\w+\s*{[^}]*}/i,
    /#\w+\s*{[^}]*}/i,
    /@media/i,
    /@import\s+url/i,
    /@keyframes/i,
    /@font-face/i,
    /margin\s*:/i,
    /padding\s*:/i,
    /color\s*:/i,
    /background\s*:/i,
    /display\s*:\s*(?:flex|block|grid|inline)/i
  ],
  python: [
    /def\s+\w+\s*\(.*\):/,
    /import\s+\w+/,
    /from\s+\w+\s+import/,
    /class\s+\w+\s*(\(.*\))?:/
  ],
  javascript: [
    /function\s+\w+\s*\(/,
    /const\s+\w+\s*=/,
    /let\s+\w+\s*=/,
    /var\s+\w+\s*=/,
    /console\.log/
  ],
  typescript: [
    /interface\s+\w+\s*{/,
    /type\s+\w+\s*=/,
    /class\s+\w+\s*(?:implements|extends)?/,
    /:\s*(?:string|number|boolean|any|void)\b/,
    /import\s+[^;]+\s+from\s+['"]/,
    /export\s+(?:default\s+)?(?:const|function|class|interface|type)/
  ],
  html: [
    /<!DOCTYPE\s+html>/,
    /<html/,
    /<head/,
    /<body/,
    /<div\s+class=/
  ]
};

const detectLanguage = (code: string): string => {
  if (!code) return 'js';
  
  // Check for file extensions first
  const fileExtMatch = code.match(/(?:\/|\\)?(\w+\.\w+)(?:['"]\s|$)/i);
  if (fileExtMatch) {
    const ext = fileExtMatch[1].toLowerCase();
    if (ext.endsWith('.css')) return 'css';
    if (ext.endsWith('.html')) return 'html';
    if (ext.endsWith('.py')) return 'python';
    if (ext.endsWith('.ts')) return 'typescript';
    if (ext.endsWith('.js')) return 'javascript';
  }
  
  // Pattern matching for each language
  const scores: Record<string, number> = {};
  
  for (const [lang, patterns] of Object.entries(LANGUAGE_PATTERNS)) {
    scores[lang] = 0;
    for (const pattern of patterns) {
      if (pattern.test(code)) {
        scores[lang]++;
      }
    }
  }
  
  // Get language with highest score
  let bestLang = 'js';
  let maxScore = 0;
  
  for (const [lang, score] of Object.entries(scores)) {
    if (score > maxScore) {
      maxScore = score;
      bestLang = lang;
    }
  }
  
  // Return detected language if we have strong matches
  return maxScore >= 2 ? bestLang : 'js';
};

const CodeInput: React.FC<CodeInputProps> = ({ onAnalysisComplete }) => {
  const [code, setCode] = useState('');
  const [filename, setFilename] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [language, setLanguage] = useState('js');
  const theme = useTheme();

  useEffect(() => {
    if (code.length > 10) {
      const detectedLang = detectLanguage(code);
      setLanguage(detectedLang);

      // Try to detect filename from import statements or comments
      const filenameMatch = code.match(/["'`]([\w.-]+\.[a-z]+)["'`]/i);
      if (filenameMatch && !filename) {
        setFilename(filenameMatch[1]);
      }
    }
  }, [code, filename]);

  // Update language when filename changes
  useEffect(() => {
    if (filename) {
      const ext = filename.split('.').pop()?.toLowerCase();
      if (ext === 'css') setLanguage('css');
      else if (ext === 'html') setLanguage('html');
      else if (ext === 'py') setLanguage('python');
      else if (ext === 'ts' || ext === 'tsx') setLanguage('typescript');
      else if (ext === 'js') setLanguage('javascript');
    }
  }, [filename]);

  const handleAnalyze = async () => {
    if (!code.trim()) {
      setError('Please enter some code to analyze.');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          code,
          filename 
        }),
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please try again.');
      }

      const result = await response.json();
      result.filename = filename;
      onAnalysisComplete(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box>
      <Paper 
        elevation={2} 
        sx={{ 
          p: 1, 
          mb: 3, 
          border: '1px solid',
          borderColor: theme.palette.divider,
          borderRadius: 1,
          overflow: 'hidden',
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1, px: 1 }}>
          <Typography variant="subtitle2" color="textSecondary">
            Paste your code here
          </Typography>
          
          <TextField 
            size="small"
            placeholder="Optional filename (e.g. style.css)"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
            sx={{ width: '250px' }}
          />
        </Box>
        
        <Box sx={{ 
          position: 'relative',
          overflow: 'hidden',
          borderRadius: 1,
          '&:hover': {
            boxShadow: '0 0 0 2px rgba(0, 120, 212, 0.3)'
          }
        }}>
          <CodeEditor
            value={code}
            language={language}
            placeholder="Paste your code here..."
            onChange={(evn: React.ChangeEvent<HTMLTextAreaElement>) => setCode(evn.target.value)}
            padding={15}
            style={{
              fontSize: 14,
              backgroundColor: theme.palette.mode === 'dark' ? '#1e1e1e' : '#f5f5f5',
              fontFamily: 'ui-monospace,SFMono-Regular,SF Mono,Menlo,Consolas,Liberation Mono,monospace',
              borderRadius: '4px',
              minHeight: '300px',
            }}
            data-color-mode={theme.palette.mode}
          />
        </Box>
      </Paper>

      {error && (
        <Typography color="error" variant="body2" gutterBottom>
          {error}
        </Typography>
      )}

      <Box sx={{ display: 'flex', justifyContent: 'center' }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleAnalyze}
          disabled={isLoading}
          sx={{ 
            minWidth: '150px',
            textTransform: 'none',
            fontWeight: 'bold',
            boxShadow: 2,
            '&:hover': {
              boxShadow: 4,
            }
          }}
        >
          {isLoading ? <CircularProgress size={24} color="inherit" /> : 'Analyze Code'}
        </Button>
      </Box>
    </Box>
  );
};

export default CodeInput; 