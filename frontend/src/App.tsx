import React, { useState } from 'react';
import CodeUpload from './components/CodeUpload';
import CodeInput from './components/CodeInput';
import AnalysisResult from './components/AnalysisResult';
import { Container, Typography, Box, Paper, Tabs, Tab, ThemeProvider, createTheme, CssBaseline, useMediaQuery, AppBar } from '@mui/material';
import CodeIcon from '@mui/icons-material/Code';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import MemoryIcon from '@mui/icons-material/Memory';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = useState(0);
  const [analysisResults, setAnalysisResults] = useState<any[]>([]);
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');

  const theme = React.useMemo(
    () =>
      createTheme({
        palette: {
          mode: prefersDarkMode ? 'dark' : 'light',
          primary: {
            main: '#3f51b5',
          },
          secondary: {
            main: '#f50057',
          },
          background: {
            default: prefersDarkMode ? '#121212' : '#f5f7fa',
            paper: prefersDarkMode ? '#1e1e1e' : '#ffffff',
          },
        },
        typography: {
          fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
          h3: {
            fontWeight: 700,
          },
          h5: {
            fontWeight: 500,
          },
        },
        shape: {
          borderRadius: 8,
        },
        components: {
          MuiButton: {
            styleOverrides: {
              root: {
                textTransform: 'none',
                borderRadius: 8,
              },
            },
          },
          MuiPaper: {
            styleOverrides: {
              root: {
                borderRadius: 8,
              },
            },
          },
        },
      }),
    [prefersDarkMode],
  );

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleAnalysisResults = (results: any) => {
    setAnalysisResults(Array.isArray(results) ? results : [results]);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        minHeight: '100vh', 
        bgcolor: 'background.default',
        pt: 2,
        pb: 6
      }}>
        <Container maxWidth="lg">
          <Box sx={{ 
            my: 4, 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center'
          }}>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              mb: 1
            }}>
              <MemoryIcon sx={{ fontSize: 40, mr: 1, color: 'primary.main' }} />
              <Typography 
                variant="h3" 
                component="h1" 
                align="center" 
                gutterBottom
                sx={{ fontWeight: 'bold', color: 'text.primary' }}
              >
                ScriptSage
              </Typography>
            </Box>
            
            <Typography 
              variant="h5" 
              align="center" 
              color="text.secondary" 
              paragraph
              sx={{ mb: 4, maxWidth: '750px' }}
            >
              Upload or paste code to identify the language and get a summary
            </Typography>

            <Paper 
              elevation={3} 
              sx={{ 
                width: '100%',
                overflow: 'hidden',
                mb: 4,
                borderRadius: 2,
              }}
            >
              <AppBar position="static" color="default" elevation={0} sx={{ borderRadius: '8px 8px 0 0' }}>
                <Tabs 
                  value={tabValue} 
                  onChange={handleTabChange} 
                  centered
                  indicatorColor="primary"
                  textColor="primary"
                  variant="fullWidth"
                  sx={{
                    '& .MuiTab-root': {
                      fontWeight: 'medium',
                      fontSize: '0.95rem',
                      py: 2
                    }
                  }}
                >
                  <Tab icon={<UploadFileIcon />} iconPosition="start" label="Upload Files" />
                  <Tab icon={<CodeIcon />} iconPosition="start" label="Paste Code" />
                </Tabs>
              </AppBar>

              <TabPanel value={tabValue} index={0}>
                <CodeUpload onAnalysisComplete={handleAnalysisResults} />
              </TabPanel>
              
              <TabPanel value={tabValue} index={1}>
                <CodeInput onAnalysisComplete={handleAnalysisResults} />
              </TabPanel>
            </Paper>

            {analysisResults.length > 0 && (
              <Box sx={{ width: '100%' }}>
                <Typography 
                  variant="h5" 
                  gutterBottom 
                  sx={{ 
                    fontWeight: 'bold',
                    borderBottom: '2px solid',
                    borderColor: 'primary.main',
                    pb: 1,
                    mb: 3,
                    display: 'inline-block'
                  }}
                >
                  Analysis Results
                </Typography>
                
                {analysisResults.map((result, index) => (
                  <AnalysisResult key={index} result={result} />
                ))}
              </Box>
            )}
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
