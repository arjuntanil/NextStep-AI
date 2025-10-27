import React, { useState, useEffect, useRef } from 'react';
import { ragAPI } from '../services/api';
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
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Send as SendIcon,
  Description as FileIcon,
  Chat as ChatIcon,
} from '@mui/icons-material';

const RAGCoach = () => {
  const [step, setStep] = useState(0);
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
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

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    setUploading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await ragAPI.uploadDocument(formData);
      setSessionId(response.data.session_id);
      setStep(1);
    } catch (err) {
      console.error('Error uploading file:', err);
      setError(err.response?.data?.detail || 'Failed to upload file. Please try again.');
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
      const response = await ragAPI.query(question, sessionId);
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
    setFile(null);
    setFileName('');
    setSessionId(null);
    setConversation([]);
    setQuestion('');
    setError('');
  };

  const steps = ['Upload Document', 'Ask Questions'];

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          🤖 RAG Coach
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Upload documents and get contextual answers using Retrieval-Augmented Generation
        </Typography>
      </Box>

      {/* Stepper */}
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Stepper activeStep={step}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
      </Paper>

      {/* Step 0: Upload Document */}
      {step === 0 && (
        <Paper elevation={3} sx={{ p: 4 }}>
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <FileIcon sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
            <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold' }}>
              Upload Your Document
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Supported formats: PDF, DOCX, TXT
            </Typography>
          </Box>

          <Box>
            <input
              accept=".pdf,.doc,.docx,.txt"
              style={{ display: 'none' }}
              id="rag-file"
              type="file"
              onChange={handleFileChange}
            />
            <label htmlFor="rag-file">
              <Button
                variant="outlined"
                component="span"
                startIcon={<UploadIcon />}
                fullWidth
                sx={{ py: 2, mb: 2 }}
              >
                {fileName || 'Choose Document'}
              </Button>
            </label>
            {fileName && (
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
                <FileIcon sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="body2">{fileName}</Typography>
              </Box>
            )}

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            <Button
              variant="contained"
              size="large"
              fullWidth
              onClick={handleUpload}
              disabled={!file || uploading}
              sx={{ py: 1.5 }}
            >
              {uploading ? <CircularProgress size={24} /> : 'Upload & Process'}
            </Button>
          </Box>
        </Paper>
      )}

      {/* Step 1: Ask Questions */}
      {step === 1 && (
        <>
          <Paper elevation={3} sx={{ p: 2, mb: 3, background: '#f5f5f5' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <FileIcon color="primary" />
                <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                  Document: {fileName}
                </Typography>
              </Box>
              <Button size="small" onClick={handleReset}>
                Upload New
              </Button>
            </Box>
          </Paper>

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
  );
};

export default RAGCoach;
