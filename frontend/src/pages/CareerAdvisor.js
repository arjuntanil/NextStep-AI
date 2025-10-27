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
  Divider,
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
      const aiMessage = {
        type: 'ai',
        text: response.data.advice || response.data.response,
        model: response.data.model_used || 'AI',
      };
      setConversation(prev => [...prev, aiMessage]);
    } catch (err) {
      console.error('Error getting career advice:', err);
      setError(err.response?.data?.detail || 'Failed to get career advice. Please try again.');
      const errorMessage = {
        type: 'error',
        text: 'Sorry, I encountered an error processing your request. Please try again.',
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
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          💬 Career Advisor
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Get personalized career guidance from our AI
        </Typography>
      </Box>

      {/* Model Status */}
      {modelStatus && (
        <Paper elevation={2} sx={{ p: 2, mb: 3, background: '#f5f5f5' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <BrainIcon color="primary" />
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                Model Status: {modelStatus.model_loaded ? '✅ Ready' : '⏳ Loading'}
              </Typography>
              {modelStatus.model_name && (
                <Typography variant="caption" color="text.secondary">
                  Using: {modelStatus.model_name}
                </Typography>
              )}
            </Box>
          </Box>
        </Paper>
      )}

      {/* Conversation Area */}
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
              <AIIcon sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
              <Typography variant="h5" gutterBottom>
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
                }}
              >
                <Card
                  sx={{
                    maxWidth: '70%',
                    background: msg.type === 'user'
                      ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                      : msg.type === 'error'
                      ? '#ffebee'
                      : '#f5f5f5',
                    color: msg.type === 'user' ? 'white' : 'inherit',
                  }}
                >
                  <CardContent>
                    {msg.type === 'ai' && msg.model && (
                      <Chip
                        label={msg.model}
                        size="small"
                        sx={{ mb: 1 }}
                        color="primary"
                      />
                    )}
                    <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                      {msg.text}
                    </Typography>
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
                    <Typography variant="body2">Thinking...</Typography>
                  </Box>
                </CardContent>
              </Card>
            </Box>
          )}
          <div ref={messagesEndRef} />
        </Box>

        <Divider />

        {/* Input Area */}
        <Box component="form" onSubmit={handleSubmit} sx={{ p: 2, background: 'white' }}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          <Box sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              placeholder="Ask me about your career..."
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
            >
              Send
            </Button>
          </Box>
        </Box>
      </Paper>

      {/* Quick Questions */}
      {conversation.length === 0 && (
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
            💡 Try asking:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
            {quickQuestions.map((q, index) => (
              <Chip
                key={index}
                label={q}
                onClick={() => setQuestion(q)}
                clickable
                color="primary"
                variant="outlined"
              />
            ))}
          </Box>
        </Paper>
      )}
    </Container>
  );
};

export default CareerAdvisor;
