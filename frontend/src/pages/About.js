import React from 'react';
import {
  Container,
  Box,
  Typography,
  Grid,
  Paper,
  Card,
  CardContent,
  Chip,
  Avatar,
} from '@mui/material';
import {
  Code as CodeIcon,
  Storage as StorageIcon,
  Cloud as CloudIcon,
  Psychology as AIIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';

const About = () => {
  const technologies = [
    {
      category: 'Frontend',
      icon: <CodeIcon sx={{ fontSize: 40 }} />,
      color: '#1e3a8a',
      items: ['React.js', 'Material-UI', 'React Router', 'Axios'],
    },
    {
      category: 'Backend',
      icon: <StorageIcon sx={{ fontSize: 40 }} />,
      color: '#3b82f6',
      items: ['Python', 'FastAPI', 'PostgreSQL', 'SQLAlchemy'],
    },
    {
      category: 'AI/ML',
      icon: <AIIcon sx={{ fontSize: 40 }} />,
      color: '#0ea5e9',
      items: ['OpenAI GPT', 'LangChain', 'Fine-tuned Models', 'RAG'],
    },
    {
      category: 'Infrastructure',
      icon: <CloudIcon sx={{ fontSize: 40 }} />,
      color: '#06b6d4',
      items: ['Docker', 'RESTful APIs', 'JWT Auth', 'CORS'],
    },
  ];

  const features = [
    {
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning algorithms analyze your resume and provide intelligent job recommendations.',
      icon: <AIIcon sx={{ fontSize: 50, color: '#1e3a8a' }} />,
    },
    {
      title: 'Secure & Private',
      description: 'Your data is encrypted and securely stored. We prioritize your privacy and data protection.',
      icon: <SecurityIcon sx={{ fontSize: 50, color: '#3b82f6' }} />,
    },
    {
      title: 'Fast Processing',
      description: 'Get instant results with our optimized processing pipeline and efficient algorithms.',
      icon: <SpeedIcon sx={{ fontSize: 50, color: '#0ea5e9' }} />,
    },
  ];

  return (
    <Container maxWidth="lg">
      {/* Hero Section */}
      <Box sx={{ mb: 8, textAlign: 'center' }}>
        <Typography 
          variant="h2" 
          sx={{ 
            fontWeight: 700, 
            mb: 2,
            background: 'linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            fontFamily: 'Space Grotesk',
          }}
        >
          About NextStepAI
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 800, mx: 'auto', lineHeight: 1.8 }}>
          Your intelligent career companion powered by cutting-edge AI technology to help you navigate your professional journey
        </Typography>
      </Box>

      {/* What is NextStepAI */}
      <Paper sx={{ p: 5, mb: 6, bgcolor: '#eff6ff', border: '1px solid #bfdbfe' }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 3, color: '#1e3a8a', fontFamily: 'Space Grotesk' }}>
          What is NextStepAI?
        </Typography>
        <Typography variant="body1" sx={{ mb: 2, lineHeight: 1.8, fontSize: '1.05rem' }}>
          NextStepAI is an advanced career navigation platform that leverages artificial intelligence to help professionals make informed career decisions. Our system combines multiple AI technologies including GPT-4, custom fine-tuned models, and Retrieval-Augmented Generation (RAG) to provide personalized career guidance.
        </Typography>
        <Typography variant="body1" sx={{ lineHeight: 1.8, fontSize: '1.05rem' }}>
          Whether you're looking to analyze your resume, get career advice, or understand skill gaps for your dream job, NextStepAI provides intelligent, data-driven insights to accelerate your professional growth.
        </Typography>
      </Paper>

      {/* How It Works */}
      <Box sx={{ mb: 8 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 4, textAlign: 'center' }}>
          How It Works
        </Typography>
        <Grid container spacing={4}>
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%', border: '2px solid #ff6b35' }}>
              <CardContent sx={{ p: 4 }}>
                <Box sx={{ 
                  width: 60, 
                  height: 60, 
                  borderRadius: '50%', 
                  bgcolor: '#fff7ed',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mb: 2,
                }}>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: '#ff6b35' }}>1</Typography>
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Upload Your Data
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                  Upload your resume or job description. Our system securely processes your documents using advanced NLP techniques.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%', border: '2px solid #f7931e' }}>
              <CardContent sx={{ p: 4 }}>
                <Box sx={{ 
                  width: 60, 
                  height: 60, 
                  borderRadius: '50%', 
                  bgcolor: '#fff7ed',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mb: 2,
                }}>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: '#f7931e' }}>2</Typography>
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  AI Analysis
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                  Our AI models analyze your skills, experience, and career goals to provide personalized recommendations.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%', border: '2px solid #22c55e' }}>
              <CardContent sx={{ p: 4 }}>
                <Box sx={{ 
                  width: 60, 
                  height: 60, 
                  borderRadius: '50%', 
                  bgcolor: '#f0fdf4',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mb: 2,
                }}>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: '#22c55e' }}>3</Typography>
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Get Insights
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                  Receive actionable insights, job matches, skill recommendations, and personalized career guidance.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>

      {/* Technologies */}
      <Box sx={{ mb: 8 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 4, textAlign: 'center' }}>
          Technologies We Use
        </Typography>
        <Grid container spacing={3}>
          {technologies.map((tech, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card sx={{ height: '100%', textAlign: 'center', p: 3 }}>
                <Box sx={{ color: tech.color, mb: 2 }}>
                  {tech.icon}
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, color: tech.color }}>
                  {tech.category}
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {tech.items.map((item, idx) => (
                    <Chip 
                      key={idx} 
                      label={item} 
                      size="small"
                      sx={{ 
                        bgcolor: `${tech.color}15`,
                        color: tech.color,
                        fontWeight: 600,
                      }}
                    />
                  ))}
                </Box>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Key Features */}
      <Box sx={{ mb: 8 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 4, textAlign: 'center' }}>
          Key Features
        </Typography>
        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Paper sx={{ p: 4, height: '100%', textAlign: 'center' }}>
                <Box sx={{ mb: 2 }}>
                  {feature.icon}
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                  {feature.description}
                </Typography>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Author Section */}
      <Paper sx={{ p: 5, mb: 6, textAlign: 'center', bgcolor: '#f9fafb' }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 4 }}>
          About the Creator
        </Typography>
        <Avatar
          sx={{
            width: 100,
            height: 100,
            mx: 'auto',
            mb: 3,
            bgcolor: 'primary.main',
            fontSize: '2.5rem',
            fontWeight: 700,
          }}
        >
          NS
        </Avatar>
        <Typography variant="h5" sx={{ fontWeight: 600, mb: 2 }}>
          NextStepAI Team
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto', lineHeight: 1.8 }}>
          Built with passion by a team dedicated to helping professionals navigate their career paths using the power of artificial intelligence. Our mission is to democratize access to intelligent career guidance.
        </Typography>
      </Paper>

      {/* Mission Statement */}
      <Box sx={{ textAlign: 'center', py: 6 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
          Our Mission
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 800, mx: 'auto', lineHeight: 1.8, fontWeight: 400 }}>
          "To empower every professional with AI-driven insights that accelerate career growth and help them achieve their full potential."
        </Typography>
      </Box>
    </Container>
  );
};

export default About;
