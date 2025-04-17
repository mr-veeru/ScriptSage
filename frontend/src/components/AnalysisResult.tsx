import React from 'react';
import { 
  Box, 
  Typography, 
  Chip, 
  Tooltip, 
  IconButton, 
  useTheme,
  Paper,
  Button,
  LinearProgress
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import CodeIcon from '@mui/icons-material/Code';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import MemoryIcon from '@mui/icons-material/Memory';
import FunctionsIcon from '@mui/icons-material/Functions';
import CategoryIcon from '@mui/icons-material/Category';
import DownloadIcon from '@mui/icons-material/Download';
import AttachFileIcon from '@mui/icons-material/AttachFile';

interface AnalysisResultProps {
  result: {
    filename?: string;
    language: string;
    analysis: {
      language: string;
      type: string;
      purpose: string;
      confidence?: number;
      specific_type?: string;
      ml_analysis?: boolean;
      computation_type?: string;
      algorithm_type?: string;
      ast_analysis?: any;
      advanced_ml_used?: boolean;
    };
    ml_powered?: boolean;
  };
}

const AnalysisResult: React.FC<AnalysisResultProps> = ({ result }) => {
  const theme = useTheme();
  // const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // Get confidence as percentage
  // const confidence = result.analysis.confidence 
  //   ? `${Math.round(result.analysis.confidence * 100)}%` 
  //   : undefined;
  
  // Get confidence color based on value - only return valid LinearProgress colors
  const getConfidenceColor = (confidenceValue?: number): "primary" | "secondary" | "success" | "error" | "warning" | "info" => {
    if (!confidenceValue) return "primary";
    if (confidenceValue >= 0.8) return "success";
    if (confidenceValue >= 0.6) return "primary";
    if (confidenceValue >= 0.4) return "warning";
    return "error";
  };
  
  const confidenceColor = getConfidenceColor(result.analysis.confidence);
  const confidenceValue = result.analysis.confidence || 0;
  
  // Add this component to display AST analysis data
  const ASTAnalysisSection = ({ astAnalysis }: { astAnalysis: any }) => {
    if (!astAnalysis) return null;
    
    return (
      <Box mt={3}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CodeIcon fontSize="small" />
          Code Structure Analysis
        </Typography>
        
        {astAnalysis.structure && (
          <Box mt={1} sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            <Chip 
              icon={<FunctionsIcon />} 
              label={`Functions: ${astAnalysis.structure.functions}`} 
              color="primary" 
              variant="outlined" 
            />
            <Chip 
              icon={<CategoryIcon />} 
              label={`Classes: ${astAnalysis.structure.classes}`} 
              color="primary" 
              variant="outlined" 
            />
            <Chip 
              icon={<DownloadIcon />} 
              label={`Imports: ${astAnalysis.structure.imports}`} 
              color="primary" 
              variant="outlined" 
            />
          </Box>
        )}
        
        {astAnalysis.complexity && (
          <Box mt={2}>
            <Typography variant="subtitle2">Complexity Metrics:</Typography>
            <Box mt={1} sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              <Chip 
                label={`Node Count: ${astAnalysis.complexity.node_count}`} 
                color="secondary" 
                variant="outlined" 
                size="small"
              />
              <Chip 
                label={`Max Depth: ${astAnalysis.complexity.max_depth}`} 
                color="secondary" 
                variant="outlined"
                size="small" 
              />
              <Chip 
                label={`Cyclomatic Complexity: ${astAnalysis.complexity.cyclomatic_complexity}`} 
                color="secondary" 
                variant="outlined"
                size="small" 
              />
            </Box>
          </Box>
        )}
      </Box>
    );
  };

  return (
    <Paper elevation={3} sx={{ p: 3, mt: 2, borderRadius: 2, backgroundColor: theme.palette.background.paper }}>
      {/* Display filename if available */}
      {result.filename && (
        <Box sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
          <AttachFileIcon fontSize="small" sx={{ mr: 1, color: theme.palette.text.secondary }} />
          <Typography variant="subtitle1" fontWeight="medium">
            {result.filename}
          </Typography>
        </Box>
      )}
      
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
        <Box flexGrow={1}>
          <Box>
            {/* Language Badge */}
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Chip
                icon={
                  <Box component="span" sx={{ display: 'flex', alignItems: 'center' }}>
                    <CodeIcon fontSize="small" />
                  </Box>
                }
                label={`Language: ${result.language}`}
                color="primary"
                sx={{ 
                  borderRadius: 1,
                  mr: 1
                }}
              />
              
              {/* Type Badge */}
              <Chip
                icon={
                  <Box component="span" sx={{ display: 'flex', alignItems: 'center' }}>
                    <CategoryIcon fontSize="small" />
                  </Box>
                }
                label={`Type: ${result.analysis.type}`}
                color="secondary"
                sx={{ 
                  borderRadius: 1
                }}
              />
            </Box>
            
            {/* Confidence Bar */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" component="div" sx={{ mb: 0.5 }}>
                Confidence:
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Box sx={{ width: '100%', mr: 1, maxWidth: '250px' }}>
                  <LinearProgress 
                    variant="determinate" 
                    value={confidenceValue * 100} 
                    color={confidenceColor}
                    sx={{ 
                      height: 10, 
                      borderRadius: 5,
                      '& .MuiLinearProgress-bar': {
                        borderRadius: 5
                      }
                    }}
                  />
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {`${Math.round(confidenceValue * 100)}%`}
                </Typography>
              </Box>
            </Box>

            {/* Purpose */}
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6">Purpose</Typography>
              <Typography>{result.analysis.purpose}</Typography>
            </Box>

            {/* Classification */}
            {result.analysis.specific_type && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="h6">Classification</Typography>
                <Box sx={{ mt: 1 }}>
                  <Chip 
                    icon={<CheckCircleIcon />} 
                    label={result.analysis.specific_type} 
                    color="success" 
                    variant="outlined" 
                  />
                </Box>
              </Box>
            )}
          </Box>
        </Box>

        <Box sx={{ minWidth: { xs: '100%', md: '45%' } }}>
          {result.analysis.ml_analysis && (
            <Box sx={{ p: 2, border: '1px solid rgba(0,0,0,0.1)', borderRadius: 2, bgcolor: 'rgba(0,0,0,0.02)' }}>
              <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <MemoryIcon fontSize="small" />
                Machine Learning Analysis
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Our ML model has analyzed your code based on patterns and structures common to different programming languages and applications.
              </Typography>
              
              {confidenceValue >= 0.8 ? (
                <Typography variant="body2" color="success.main" sx={{ mt: 1 }}>
                  High confidence detection based on distinctive code patterns.
                </Typography>
              ) : confidenceValue >= 0.6 ? (
                <Typography variant="body2" color="primary.main" sx={{ mt: 1 }}>
                  Medium confidence detection with supporting pattern evidence.
                </Typography>
              ) : (
                <Typography variant="body2" color="warning.main" sx={{ mt: 1 }}>
                  Low confidence score. Consider providing more code context.
                </Typography>
              )}
              
              <Button
                variant="contained"
                color="primary"
                size="small"
                startIcon={<MemoryIcon />}
                sx={{ mt: 2, textTransform: 'none' }}
              >
                ML-Powered Analysis
              </Button>
              
              {result.analysis.advanced_ml_used && (
                <Tooltip title="Advanced machine learning models were used for this analysis">
                  <IconButton size="small" color="primary" sx={{ ml: 1 }}>
                    <InfoIcon />
                  </IconButton>
                </Tooltip>
              )}
            </Box>
          )}
        </Box>
      </Box>
      
      {/* Add AST Analysis Section */}
      {result.analysis?.ast_analysis && (
        <ASTAnalysisSection astAnalysis={result.analysis.ast_analysis} />
      )}
      
    </Paper>
  );
};

export default AnalysisResult; 