import React, { useState, useEffect, useRef } from 'react';
import { careerAPI } from '../services/api';
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
  Chip,
  Grid,
} from '@mui/material';
import {
  Send as SendIcon,
  Psychology as BrainIcon,
  SmartToy as AIIcon,
} from '@mui/icons-material';

const CareerAdvisor = () => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [conversation, setConversation] = useState([]);
  const [modelStatus, setModelStatus] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetchModelStatus();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchModelStatus = async () => {
    try {
      const response = await careerAPI.getModelStatus();
      setModelStatus(response.data);
    } catch (err) {
      console.error('Error fetching model status:', err);
    }
  };

  const handleSubmit = async (e) => {
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
      const response = await careerAPI.queryCareerPath(question);
      // Fix: Access the correct field from the response
      const aiMessage = {
        type: 'ai',
        text: response.data.generative_advice || response.data.advice || response.data.response || 'No response available',
        model: 'AI Career Advisor',
      };
      setConversation(prev => [...prev, aiMessage]);
    } catch (err) {
      console.error('Error getting career advice:', err);
      // Fix: Better error handling
      const errorMessageText = err.response?.data?.detail || 
                              (err.response?.data ? JSON.stringify(err.response.data) : 'Failed to get career advice. Please try again.');
      setError(errorMessageText);
      const errorMessage = {
        type: 'error',
        text: `Sorry, I encountered an error: ${errorMessageText}`,
      };
      setConversation(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const quickQuestions = [
    "How do I transition from marketing to tech?",
    "What skills should I learn for data science?",
    "How can I negotiate a better salary?",
    "What are the best career paths in AI?",
  ];

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 5 }}>
        <Typography 
          variant="h2" 
          gutterBottom 
          sx={{ 
            fontWeight: 700, 
            background: 'linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            mb: 2,
            fontFamily: 'Space Grotesk',
          }}
        >
          Career Advisor
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ fontWeight: 500 }}>
          Get personalized career guidance from our AI
        </Typography>
      </Box>

      {/* Split Layout - Input on Left, Conversation on Right */}
      <Grid container spacing={3}>
        {/* LEFT SIDE - Input & Quick Questions */}
        <Grid item xs={12} lg={4}>
          <Paper
            elevation={0}
            sx={{
              p: 3,
              position: 'sticky',
              top: 80,
              borderRadius: 3,
              border: '2px solid #1e3a8a',
              background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
              position: 'relative',
              overflow: 'hidden',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                zIndex: 0,
                opacity: 0.03,
                backgroundImage: 'radial-gradient(circle at 70% 30%, rgba(14, 165, 233, 0.3) 0%, transparent 40%)',
                animation: 'floatUpDown 18s ease-in-out infinite',
              },
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 3, fontFamily: 'Space Grotesk', color: '#1e3a8a', position: 'relative', zIndex: 1 }}>
              Ask a Question
            </Typography>
            
            <Box component="form" onSubmit={handleSubmit} sx={{ position: 'relative', zIndex: 1 }}>
              <TextField
                fullWidth
                multiline
                rows={4}
                placeholder="Type your career question here..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={loading}
                variant="outlined"
                sx={{
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 2,
                    fontFamily: 'Space Grotesk',
                    '&:hover fieldset': {
                      borderColor: '#1e3a8a',
                      borderWidth: 2,
                    },
                  },
                }}
              />
              {error && (
                <Alert severity="error" sx={{ mb: 2, fontSize: '0.85rem' }}>
                  {error}
                </Alert>
              )}
              <Button
                type="submit"
                variant="contained"
                fullWidth
                disabled={loading || !question.trim()}
                endIcon={<SendIcon />}
                sx={{
                  py: 1.5,
                  borderRadius: 2,
                  fontWeight: 700,
                  fontFamily: 'Space Grotesk',
                  bgcolor: '#1e3a8a',
                  '&:hover': {
                    bgcolor: '#1e40af',
                  },
                }}
              >
                {loading ? 'Processing...' : 'Send Question'}
              </Button>
            </Box>

            {/* Quick Questions */}
            <Box sx={{ mt: 4 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2, color: '#1e3a8a', fontFamily: 'Space Grotesk' }}>
                💡 Try These Questions:
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {quickQuestions.map((q, index) => (
                  <Chip
                    key={index}
                    label={q}
                    onClick={() => setQuestion(q)}
                    clickable
                    sx={{
                      py: 2,
                      height: 'auto',
                      justifyContent: 'flex-start',
                      textAlign: 'left',
                      fontSize: '0.85rem',
                      fontWeight: 500,
                      border: '1px solid #3b82f6',
                      background: 'rgba(59, 130, 246, 0.05)',
                      transition: 'all 0.3s ease',
                      '& .MuiChip-label': {
                        whiteSpace: 'normal',
                        padding: '8px',
                      },
                      '&:hover': {
                        background: '#1e3a8a',
                        color: 'white',
                        borderColor: '#1e3a8a',
                        transform: 'translateX(4px)',
                      },
                    }}
                  />
                ))}
              </Box>
            </Box>

            {/* Model Status */}
            {modelStatus && (
              <Box sx={{ mt: 3, p: 2, bgcolor: 'rgba(30, 58, 138, 0.05)', borderRadius: 2 }}>
                <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5, color: '#1e3a8a' }}>
                  {modelStatus.finetuned_career_advisor?.loaded ? '✅ AI Model Ready' : '⏳ Loading Model'}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Fine-tuned Career Advisor
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* RIGHT SIDE - Conversation Display */}
        <Grid item xs={12} lg={8}>
          <Paper
            elevation={0}
            sx={{
              height: '700px',
              display: 'flex',
              flexDirection: 'column',
              borderRadius: 3,
              border: '2px solid #0ea5e9',
              background: 'rgba(255, 255, 255, 0.95)',
              overflow: 'hidden',
              position: 'relative',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                zIndex: 0,
                opacity: 0.02,
                backgroundImage: 'radial-gradient(circle at 80% 20%, rgba(59, 130, 246, 0.3) 0%, transparent 50%)',
                animation: 'floatUpDown 25s ease-in-out infinite',
              },
            }}
          >
            <Box sx={{ p: 2.5, borderBottom: '1px solid #e5e7eb', bgcolor: '#f8fafc', position: 'relative', zIndex: 1 }}>
              <Typography variant="h6" sx={{ fontWeight: 700, fontFamily: 'Space Grotesk', color: '#0284c7' }}>
                Conversation
              </Typography>
            </Box>
            <Box
              sx={{
                flexGrow: 1,
                overflow: 'auto',
                p: 3,
                background: 'linear-gradient(to bottom, #f0f9ff, #ffffff)',
                position: 'relative',
                zIndex: 1,
              }}
            >
              {conversation.length === 0 ? (
                <Box sx={{ textAlign: 'center', mt: 10, animation: 'fadeIn 0.5s ease-out' }}>
                  <Box
                    sx={{
                      width: 100,
                      height: 100,
                      mx: 'auto',
                      mb: 3,
                      borderRadius: '50%',
                      background: 'linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      animation: 'floatUpDown 3s ease-in-out infinite',
                    }}
                  >
                    <AIIcon sx={{ fontSize: 50, color: 'white' }} />
                  </Box>
                  <Typography variant="h5" gutterBottom sx={{ fontWeight: 700, mb: 2, fontFamily: 'Space Grotesk' }}>
                    Welcome to AI Career Advisor!
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    Ask me anything about your career journey
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
                      animation: `${msg.type === 'user' ? 'slideInRight' : 'slideInLeft'} 0.3s ease-out`,
                    }}
                  >
                    <Card
                      elevation={0}
                      sx={{
                        maxWidth: '80%',
                        background: msg.type === 'user'
                          ? 'linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%)'
                          : msg.type === 'error'
                          ? 'rgba(239, 68, 68, 0.1)'
                          : 'rgba(30, 58, 138, 0.05)',
                        color: msg.type === 'user' ? 'white' : 'inherit',
                        borderRadius: 3,
                        border: msg.type === 'user' ? 'none' : '1px solid #cbd5e1',
                        boxShadow: msg.type === 'user' ? '0 4px 12px rgba(30, 58, 138, 0.3)' : 'none',
                      }}
                    >
                      <CardContent sx={{ p: 2.5 }}>
                        {msg.type === 'ai' && msg.model && (
                          <Chip
                            label={msg.model}
                            size="small"
                            sx={{
                              mb: 1.5,
                              bgcolor: '#1e3a8a',
                              color: 'white',
                              fontWeight: 600,
                            }}
                          />
                        )}
                        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.7, fontFamily: 'Space Grotesk' }}>
                          {msg.text}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Box>
                ))
              )}
              {loading && (
                <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2, animation: 'fadeIn 0.3s ease-out' }}>
                  <Card elevation={0} sx={{ background: 'rgba(30, 58, 138, 0.05)', border: '1px solid #cbd5e1', borderRadius: 3 }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <CircularProgress size={20} sx={{ color: '#1e3a8a' }} />
                        <Typography variant="body1" sx={{ fontWeight: 600, fontFamily: 'Space Grotesk' }}>AI is thinking...</Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Box>
              )}
              <div ref={messagesEndRef} />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default CareerAdvisor;