import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { ragAPI } from '../services/api';
import Aurora from '../components/Aurora';
import {
  Container,
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Divider,
  Stepper,
  Step,
  StepLabel,
  Grid,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Send as SendIcon,
  Description as FileIcon,
  Chat as ChatIcon,
  CheckCircle as CheckIcon,
  TrendingUp as TrendingIcon,
} from '@mui/icons-material';

const RAGCoach = () => {
  const [step, setStep] = useState(0);
  const [resumeFile, setResumeFile] = useState(null);
  const [resumeFileName, setResumeFileName] = useState('');
  const [jobDescFile, setJobDescFile] = useState(null);
  const [jobDescFileName, setJobDescFileName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [processedResult, setProcessedResult] = useState(null);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [conversation, setConversation] = useState([]);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleResumeChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      // Validate file
      if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
        setError('Resume must be a PDF file');
        return;
      }
      if (selectedFile.size === 0) {
        setError('Resume file is empty');
        return;
      }
      if (selectedFile.size > 10 * 1024 * 1024) { // 10MB limit
        setError('Resume file is too large (max 10MB)');
        return;
      }
      
      console.log('Resume file selected:', selectedFile.name, selectedFile.size, 'bytes');
      setResumeFile(selectedFile);
      setResumeFileName(selectedFile.name);
      setError('');
    }
  };

  const handleJobDescChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      // Validate file
      if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
        setError('Job description must be a PDF file');
        return;
      }
      if (selectedFile.size === 0) {
        setError('Job description file is empty');
        return;
      }
      if (selectedFile.size > 10 * 1024 * 1024) { // 10MB limit
        setError('Job description file is too large (max 10MB)');
        return;
      }
      
      console.log('Job desc file selected:', selectedFile.name, selectedFile.size, 'bytes');
      setJobDescFile(selectedFile);
      setJobDescFileName(selectedFile.name);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!resumeFile || !jobDescFile) {
      setError('Please upload both resume and job description');
      return;
    }

    setUploading(true);
    setProcessing(true);
    setError('');

    try {
      console.log('=== UPLOAD DEBUG START ===');
      console.log('Resume File Object:', resumeFile);
      console.log('  - name:', resumeFile.name);
      console.log('  - type:', resumeFile.type);
      console.log('  - size:', resumeFile.size);
      console.log('  - lastModified:', resumeFile.lastModified);
      
      console.log('Job Desc File Object:', jobDescFile);
      console.log('  - name:', jobDescFile.name);
      console.log('  - type:', jobDescFile.type);
      console.log('  - size:', jobDescFile.size);
      console.log('  - lastModified:', jobDescFile.lastModified);

      // Create FormData - CRITICAL: Don't specify filename, let File object provide it
      const formData = new FormData();
      formData.append('files', resumeFile);  // File object already has name, type, data
      formData.append('files', jobDescFile); // File object already has name, type, data
      formData.append('process_resume_job', 'true');

      // Debug FormData
      console.log('\nFormData entries:');
      for (let [key, value] of formData.entries()) {
        if (value instanceof File) {
          console.log(`  ${key}:`, value.name, value.type, value.size, 'bytes');
        } else {
          console.log(`  ${key}:`, value);
        }
      }

      console.log('\nSending POST request using AXIOS (not fetch)...');
      
      // Use axios instead of fetch - it handles multipart/form-data better
      const token = localStorage.getItem('token');
      const response = await axios.post(
        'http://127.0.0.1:8000/rag-coach/upload',
        formData,
        {
          headers: {
            // Do NOT set Content-Type - axios will set it correctly with boundary
            ...(token && { 'Authorization': `Bearer ${token}` }),
          },
        }
      );

      console.log('\nResponse received:');
      console.log('  - status:', response.status);
      console.log('  - statusText:', response.statusText);
      console.log('  - data:', response.data);
      
      if (response.status !== 200) {
        console.error('❌ Upload failed!');
        console.error('Error detail:', response.data);
        throw new Error(response.data.detail || JSON.stringify(response.data));
      }

      const data = response.data;
      console.log('✅ Upload successful!');
      console.log('=== UPLOAD DEBUG END ===');
      
      // Poll for processed results
      let attempts = 0;
      const maxAttempts = 30; // 30 seconds max wait
      const pollInterval = setInterval(async () => {
        attempts++;
        try {
          const resultResponse = await ragAPI.getProcessedResult();
          if (resultResponse.data && resultResponse.data.result) {
            clearInterval(pollInterval);
            setProcessedResult(resultResponse.data.result);
            setProcessing(false);
            setStep(1);
          }
        } catch (pollErr) {
          if (attempts >= maxAttempts) {
            clearInterval(pollInterval);
            setProcessing(false);
            setError('Processing timeout. Please try again.');
          }
        }
      }, 1000);

    } catch (err) {
      console.error('Error uploading files:', err);
      setError(err.message || 'Failed to upload files. Please try again.');
      setProcessing(false);
    } finally {
      setUploading(false);
    }
  };

  const handleSubmitQuestion = async (e) => {
    e.preventDefault();
    
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    const userMessage = { type: 'user', text: question };
    setConversation(prev => [...prev, userMessage]);
    setQuestion('');
    setLoading(true);
    setError('');

    try {
      const response = await ragAPI.query(question);
      const aiMessage = {
        type: 'ai',
        text: response.data.answer || response.data.response,
        sources: response.data.sources || [],
      };
      setConversation(prev => [...prev, aiMessage]);
    } catch (err) {
      console.error('Error querying RAG:', err);
      setError(err.response?.data?.detail || 'Failed to get answer. Please try again.');
      const errorMessage = {
        type: 'error',
        text: 'Sorry, I encountered an error processing your request. Please try again.',
      };
      setConversation(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setStep(0);
    setResumeFile(null);
    setResumeFileName('');
    setJobDescFile(null);
    setJobDescFileName('');
    setProcessedResult(null);
    setConversation([]);
    setQuestion('');
    setError('');
  };

  const steps = ['Upload Documents', 'View Analysis & Ask Questions'];

  return (
    <>
      {/* Aurora Animated Background */}
      <Aurora
        colorStops={['#dc2626', '#f59e0b', '#000000']}
        blend={0.8}
        amplitude={1.5}
        speed={0.4}
      />
    
    <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1, pb: 8 }}>
      <Box sx={{ mb: 5 }}>
        <Typography 
          variant="h2" 
          gutterBottom 
          sx={{ 
            fontWeight: 700, 
            background: 'linear-gradient(135deg, #8b5cf6 0%, #10b981 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            mb: 2,
            fontFamily: 'Space Grotesk',
            color: 'white',
          }}
        >
          RAG Coach
        </Typography>
        <Typography variant="h6" sx={{ fontWeight: 500, color: 'rgba(255, 255, 255, 0.8)', textShadow: '0 2px 4px rgba(0,0,0,0.8)' }}>
          Upload documents and get contextual answers using Retrieval-Augmented Generation
        </Typography>
      </Box>

      {/* Stepper */}
      <Paper 
        elevation={0} 
        className="glass-card"
        sx={{ 
          p: 4, 
          mb: 4,
          borderRadius: 4,
          border: '1px solid rgba(16, 185, 129, 0.2)',
          background: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(20px)',
        }}
      >
        <Stepper 
          activeStep={step}
          sx={{
            '& .MuiStepLabel-root .Mui-completed': {
              color: 'success.main',
            },
            '& .MuiStepLabel-root .Mui-active': {
              color: 'success.main',
            },
          }}
        >
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
      </Paper>

      {/* Step 0: Upload Documents */}
      {step === 0 && (
        <Paper 
          elevation={0} 
          className="glass-card"
          sx={{ 
            p: 6,
            borderRadius: 4,
            border: '1px solid rgba(16, 185, 129, 0.2)',
            background: 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(20px)',
          }}
        >
          <Box sx={{ textAlign: 'center', mb: 5 }}>
            <Box sx={{
              display: 'inline-flex',
              p: 3,
              borderRadius: 4,
              background: 'linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(59,130,246,0.1) 100%)',
              mb: 3,
              animation: 'float 3s ease-in-out infinite',
            }}>
              <FileIcon sx={{ fontSize: 100, color: 'success.main' }} />
            </Box>
            <Typography variant="h4" gutterBottom sx={{ fontWeight: 800, mb: 2 }}>
              Upload Resume & Job Description
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ fontSize: '1.1rem' }}>
              Upload both files to get AI-powered analysis and skill gap insights
            </Typography>
          </Box>

          <Grid container spacing={3}>
            {/* Resume Upload */}
            <Grid item xs={12} md={6}>
              <Paper 
                variant="outlined" 
                className="glass-card"
                sx={{ 
                  p: 4, 
                  textAlign: 'center',
                  borderRadius: 3,
                  border: '2px dashed',
                  borderColor: resumeFileName ? 'success.main' : 'rgba(16, 185, 129, 0.3)',
                  background: resumeFileName ? 'rgba(16, 185, 129, 0.05)' : 'rgba(16, 185, 129, 0.02)',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    borderColor: 'success.main',
                    background: 'rgba(16, 185, 129, 0.08)',
                    transform: 'translateY(-4px)',
                  },
                }}
              >
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 800 }}>
                  📄 Your Resume
                </Typography>
                <input
                  accept=".pdf"
                  style={{ display: 'none' }}
                  id="resume-file"
                  type="file"
                  onChange={handleResumeChange}
                />
                <label htmlFor="resume-file">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<UploadIcon />}
                    fullWidth
                    sx={{ 
                      py: 2.5, 
                      mb: 2,
                      borderRadius: 2,
                      borderWidth: 2,
                      fontWeight: 600,
                      '&:hover': {
                        borderWidth: 2,
                      },
                    }}
                  >
                    {resumeFileName || 'Choose Resume (PDF)'}
                  </Button>
                </label>
                {resumeFileName && (
                  <Box 
                    sx={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      animation: 'scaleIn 0.3s ease-out',
                    }}
                  >
                    <CheckIcon sx={{ mr: 1, color: 'success.main' }} />
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{resumeFileName}</Typography>
                  </Box>
                )}
              </Paper>
            </Grid>

            {/* Job Description Upload */}
            <Grid item xs={12} md={6}>
              <Paper 
                variant="outlined" 
                className="glass-card"
                sx={{ 
                  p: 4, 
                  textAlign: 'center',
                  borderRadius: 3,
                  border: '2px dashed',
                  borderColor: jobDescFileName ? 'success.main' : 'rgba(16, 185, 129, 0.3)',
                  background: jobDescFileName ? 'rgba(16, 185, 129, 0.05)' : 'rgba(16, 185, 129, 0.02)',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    borderColor: 'success.main',
                    background: 'rgba(16, 185, 129, 0.08)',
                    transform: 'translateY(-4px)',
                  },
                }}
              >
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 800 }}>
                  📋 Job Description
                </Typography>
                <input
                  accept=".pdf"
                  style={{ display: 'none' }}
                  id="job-desc-file"
                  type="file"
                  onChange={handleJobDescChange}
                />
                <label htmlFor="job-desc-file">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<UploadIcon />}
                    fullWidth
                    sx={{ 
                      py: 2.5, 
                      mb: 2,
                      borderRadius: 2,
                      borderWidth: 2,
                      fontWeight: 600,
                      '&:hover': {
                        borderWidth: 2,
                      },
                    }}
                  >
                    {jobDescFileName || 'Choose Job Description (PDF)'}
                  </Button>
                </label>
                {jobDescFileName && (
                  <Box 
                    sx={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      animation: 'scaleIn 0.3s ease-out',
                    }}
                  >
                    <CheckIcon sx={{ mr: 1, color: 'success.main' }} />
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{jobDescFileName}</Typography>
                  </Box>
                )}
              </Paper>
            </Grid>
          </Grid>

          {error && (
            <Alert severity="error" sx={{ mt: 3 }}>
              {error}
            </Alert>
          )}

          {processing && (
            <Box sx={{ mt: 4 }}>
              <LinearProgress 
                sx={{
                  height: 8,
                  borderRadius: 4,
                  background: 'rgba(16, 185, 129, 0.1)',
                  '& .MuiLinearProgress-bar': {
                    background: 'linear-gradient(90deg, #10b981 0%, #3b82f6 100%)',
                  },
                }} 
              />
              <Typography variant="body1" color="text.secondary" sx={{ mt: 2, textAlign: 'center', fontWeight: 600 }}>
                Processing your documents... This may take up to 30 seconds
              </Typography>
            </Box>
          )}

          <Button
            variant="contained"
            size="large"
            fullWidth
            onClick={handleUpload}
            disabled={!resumeFile || !jobDescFile || uploading || processing}
            sx={{ 
              py: 2, 
              mt: 4,
              borderRadius: 3,
              fontSize: '1.1rem',
              fontWeight: 700,
              textTransform: 'none',
              background: 'linear-gradient(135deg, #10b981 0%, #3b82f6 100%)',
              boxShadow: '0 4px 14px rgba(16, 185, 129, 0.4)',
              '&:hover': {
                background: 'linear-gradient(135deg, #059669 0%, #2563eb 100%)',
                boxShadow: '0 6px 20px rgba(16, 185, 129, 0.5)',
                transform: 'translateY(-2px)',
              },
            }}
          >
            {uploading ? <CircularProgress size={24} sx={{ color: 'white' }} /> : '✨ Analyze Documents'}
          </Button>
        </Paper>
      )}

      {/* Step 1: Analysis Results & Ask Questions */}
      {step === 1 && (
        <>
          {/* Uploaded Files Info */}
          <Paper elevation={3} sx={{ p: 2, mb: 3, background: '#f5f5f5' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <FileIcon color="primary" />
                <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                  {resumeFileName} & {jobDescFileName}
                </Typography>
              </Box>
              <Button size="small" onClick={handleReset}>
                Upload New
              </Button>
            </Box>
          </Paper>

          {/* Analysis Results */}
          {processedResult && processedResult.similarity_metrics && (
            <Box sx={{ mb: 3 }}>
              {/* Match Percentage */}
              <Paper elevation={3} sx={{ p: 4, mb: 3 }}>
                <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold' }}>
                  📊 Skill Match Analysis
                </Typography>
                <Box sx={{ mb: 3 }}>
                  <LinearProgress
                    variant="determinate"
                    value={processedResult.similarity_metrics.match_percentage}
                    sx={{ height: 12, borderRadius: 6, mb: 1 }}
                  />
                  <Typography variant="h6" color="text.primary">
                    {processedResult.similarity_metrics.match_percentage.toFixed(1)}% Match
                  </Typography>
                </Box>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          Job Skills Required
                        </Typography>
                        <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                          {processedResult.similarity_metrics.total_jd_skills}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Card variant="outlined" sx={{ background: '#e8f5e9' }}>
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          Skills Matched
                        </Typography>
                        <Typography variant="h5" sx={{ fontWeight: 'bold', color: 'success.main' }}>
                          {processedResult.similarity_metrics.matched_skills_count}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Card variant="outlined" sx={{ background: '#fff3e0' }}>
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          Skills to Add
                        </Typography>
                        <Typography variant="h5" sx={{ fontWeight: 'bold', color: 'warning.main' }}>
                          {processedResult.similarity_metrics.missing_skills_count}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Paper>

              {/* Matched Skills */}
              {processedResult.similarity_metrics.matched_skills && processedResult.similarity_metrics.matched_skills.length > 0 && (
                <Paper elevation={3} sx={{ p: 4, mb: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                    ✅ Matched Skills
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {processedResult.similarity_metrics.matched_skills.map((skill, index) => (
                      <Chip
                        key={index}
                        label={skill}
                        color="success"
                        icon={<CheckIcon />}
                      />
                    ))}
                  </Box>
                </Paper>
              )}

              {/* Missing Skills */}
              {processedResult.similarity_metrics.missing_skills && processedResult.similarity_metrics.missing_skills.length > 0 && (
                <Paper elevation={3} sx={{ p: 4, mb: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                    💡 Skills to Develop
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    These skills from the job description are missing from your resume
                  </Typography>
                  <List>
                    {processedResult.similarity_metrics.missing_skills.map((skill, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <TrendingIcon color="warning" />
                        </ListItemIcon>
                        <ListItemText primary={skill} />
                      </ListItem>
                    ))}
                  </List>
                </Paper>
              )}

              {/* Formatted Output */}
              {processedResult.formatted && (
                <Paper elevation={3} sx={{ p: 4, mb: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                    📝 Detailed Analysis
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  <Box sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '0.9rem' }}>
                    {processedResult.formatted}
                  </Box>
                </Paper>
              )}
            </Box>
          )}

          <Paper
            elevation={3}
            sx={{
              height: '500px',
              mb: 3,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            <Box
              sx={{
                flexGrow: 1,
                overflow: 'auto',
                p: 3,
                background: 'linear-gradient(to bottom, #ffffff, #f9f9f9)',
              }}
            >
              {conversation.length === 0 ? (
                <Box sx={{ textAlign: 'center', mt: 8 }}>
                  <ChatIcon sx={{ fontSize: 80, color: 'success.main', mb: 2 }} />
                  <Typography variant="h5" gutterBottom>
                    Ask Questions About Your Document
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    I'll provide answers based on the content you uploaded
                  </Typography>
                </Box>
              ) : (
                conversation.map((msg, index) => (
                  <Box
                    key={index}
                    sx={{
                      display: 'flex',
                      justifyContent: msg.type === 'user' ? 'flex-end' : 'flex-start',
                      mb: 2,
                    }}
                  >
                    <Card
                      sx={{
                        maxWidth: '70%',
                        background: msg.type === 'user'
                          ? 'linear-gradient(135deg, #43a047 0%, #66bb6a 100%)'
                          : msg.type === 'error'
                          ? '#ffebee'
                          : '#f5f5f5',
                        color: msg.type === 'user' ? 'white' : 'inherit',
                      }}
                    >
                      <CardContent>
                        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                          {msg.text}
                        </Typography>
                        {msg.sources && msg.sources.length > 0 && (
                          <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid #ddd' }}>
                            <Typography variant="caption" color="text.secondary">
                              Sources: {msg.sources.length} reference(s)
                            </Typography>
                          </Box>
                        )}
                      </CardContent>
                    </Card>
                  </Box>
                ))
              )}
              {loading && (
                <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
                  <Card sx={{ maxWidth: '70%', background: '#f5f5f5' }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <CircularProgress size={20} />
                        <Typography variant="body2">Searching document...</Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Box>
              )}
              <div ref={messagesEndRef} />
            </Box>

            <Divider />

            <Box component="form" onSubmit={handleSubmitQuestion} sx={{ p: 2, background: 'white' }}>
              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {error}
                </Alert>
              )}
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  fullWidth
                  placeholder="Ask a question about the document..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  disabled={loading}
                  variant="outlined"
                  size="small"
                />
                <Button
                  type="submit"
                  variant="contained"
                  disabled={loading || !question.trim()}
                  endIcon={<SendIcon />}
                  color="success"
                >
                  Ask
                </Button>
              </Box>
            </Box>
          </Paper>
        </>
      )}
    </Container>
    </>
  );
};

export default RAGCoach;
