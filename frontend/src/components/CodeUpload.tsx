import React, { useState } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { Box, Button, Typography, CircularProgress, Alert } from '@mui/material';

interface CodeUploadProps {
  onAnalysisComplete: (results: any) => void;
}

const CodeUpload: React.FC<CodeUploadProps> = ({ onAnalysisComplete }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    acceptedFiles.forEach(file => {
      formData.append('files', file);
    });
    
    try {
      const response = await axios.post('http://localhost:5000/api/analyze-files', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      // Process the results to include filenames
      const results = response.data;
      if (Array.isArray(results)) {
        // If multiple files were analyzed
        results.forEach((result, index) => {
          if (index < acceptedFiles.length) {
            result.filename = acceptedFiles[index].name;
          }
        });
      } else if (results && acceptedFiles.length > 0) {
        // If a single file was analyzed
        results.filename = acceptedFiles[0].name;
      }
      
      onAnalysisComplete(results);
    } catch (err) {
      console.error('Error analyzing files:', err);
      setError('Failed to analyze files. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });
  
  return (
    <Box>
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      
      <Box 
        {...getRootProps()} 
        sx={{
          border: '2px dashed #ccc',
          borderRadius: 2,
          p: 4,
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: isDragActive ? '#f0f8ff' : 'transparent',
        }}
      >
        <input {...getInputProps()} />
        
        {loading ? (
          <CircularProgress />
        ) : (
          <>
            <Typography variant="h6" gutterBottom>
              Drag and drop code files here
            </Typography>
            <Typography variant="body2" color="textSecondary">
              or click to select files
            </Typography>
          </>
        )}
      </Box>
      
      <Box sx={{ mt: 2, textAlign: 'center' }}>
        <Button
          variant="contained"
          color="primary"
          component="label"
          disabled={loading}
        >
          Browse Files
          <input
            type="file"
            hidden
            multiple
            onChange={(e) => {
              if (e.target.files?.length) {
                onDrop(Array.from(e.target.files));
              }
            }}
          />
        </Button>
      </Box>
    </Box>
  );
};

export default CodeUpload; 