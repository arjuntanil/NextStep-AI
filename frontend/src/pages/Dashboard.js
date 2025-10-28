import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Paper,
} from '@mui/material';
import {
  Description as CVIcon,
  Psychology as AdvisorIcon,
  Chat as CoachIcon,
  TrendingUp as GrowthIcon,
  Work as JobIcon,
  School as SkillIcon,
  ArrowForward as ArrowIcon,
} from '@mui/icons-material';

const Dashboard = () => {
  const navigate = useNavigate();

  const mainFeatures = [
    {
      title: 'CV Analyzer',
      description: 'Upload your resume and get AI-powered analysis with personalized job recommendations and skill insights.',
      icon: <CVIcon sx={{ fontSize: 50 }} />,
      path: '/cv-analyzer',
      color: '#1e3a8a',
    },
    {
      title: 'Career Advisor',
      description: 'Get expert career guidance from our fine-tuned AI model. Ask anything about your career path.',
      icon: <AdvisorIcon sx={{ fontSize: 50 }} />,
      path: '/career-advisor',
      color: '#3b82f6',
    },
    {
      title: 'RAG Coach',
      description: 'Upload documents and get contextual answers using advanced Retrieval-Augmented Generation technology.',
      icon: <CoachIcon sx={{ fontSize: 50 }} />,
      path: '/rag-coach',
      color: '#0ea5e9',
    },
  ];

  const benefits = [
    {
      icon: <GrowthIcon sx={{ fontSize: 40, color: '#1e3a8a' }} />,
      title: 'Accelerate Growth',
      description: 'Fast-track your career with data-driven insights',
    },
    {
      icon: <JobIcon sx={{ fontSize: 40, color: '#3b82f6' }} />,
      title: 'Find Perfect Jobs',
      description: 'AI-matched opportunities tailored to your skills',
    },
    {
      icon: <SkillIcon sx={{ fontSize: 40, color: '#0ea5e9' }} />,
      title: 'Skill Development',
      description: 'Identify and bridge skill gaps effectively',
    },
  ];

  return (
    <Container maxWidth="xl">
      {/* Hero Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%)',
          borderRadius: 3,
          p: { xs: 4, md: 8 },
          mb: 6,
          color: 'white',
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 30% 50%, rgba(59, 130, 246, 0.3), transparent 50%), radial-gradient(circle at 70% 50%, rgba(14, 165, 233, 0.3), transparent 50%)',
            animation: 'floatUpDown 10s ease-in-out infinite',
          },
        }}
      >
        <Grid container spacing={4} alignItems="center">
          <Grid item xs={12} md={7}>
            <Typography variant="h2" sx={{ fontWeight: 700, mb: 2 }}>
              Navigate Your Career with AI
            </Typography>
            <Typography variant="h6" sx={{ mb: 4, opacity: 0.95, lineHeight: 1.7 }}>
              Unlock your professional potential with intelligent career guidance, resume analysis, and personalized job recommendations powered by advanced AI.
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                size="large"
                onClick={() => navigate('/cv-analyzer')}
                sx={{
                  bgcolor: 'white',
                  color: '#ff6b35',
                  fontWeight: 700,
                  px: 4,
                  py: 1.5,
                  '&:hover': {
                    bgcolor: '#f9fafb',
                  },
                }}
              >
                Analyze Your CV
              </Button>
              <Button
                variant="outlined"
                size="large"
                onClick={() => navigate('/career-advisor')}
                sx={{
                  borderColor: 'white',
                  color: 'white',
                  fontWeight: 700,
                  px: 4,
                  py: 1.5,
                  borderWidth: 2,
                  '&:hover': {
                    borderColor: 'white',
                    bgcolor: 'rgba(255,255,255,0.1)',
                    borderWidth: 2,
                  },
                }}
              >
                Get Career Advice
              </Button>
            </Box>
          </Grid>
          <Grid item xs={12} md={5} sx={{ display: { xs: 'none', md: 'block' } }}>
            <Box
              sx={{
                fontSize: '200px',
                textAlign: 'center',
                filter: 'drop-shadow(0 10px 20px rgba(0,0,0,0.2))',
                animation: 'floatUpDown 6s ease-in-out infinite',
              }}
            >
              <svg width="200" height="200" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="100" cy="100" r="80" fill="white" opacity="0.2"/>
                <path d="M100 40 L120 80 L160 90 L130 120 L140 160 L100 140 L60 160 L70 120 L40 90 L80 80 Z" fill="white"/>
              </svg>
            </Box>
          </Grid>
        </Grid>
      </Box>

      {/* Benefits Section */}
      <Box sx={{ mb: 8 }}>
        <Grid container spacing={3}>
          {benefits.map((benefit, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Paper
                sx={{
                  p: 3,
                  textAlign: 'center',
                  height: '100%',
                  transition: 'transform 0.3s',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                  },
                }}
              >
                <Box sx={{ mb: 2 }}>{benefit.icon}</Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  {benefit.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {benefit.description}
                </Typography>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Our Core AI Features - Moved after Hero */}
      <Box sx={{ mb: 8 }}>
        <Typography variant="h3" sx={{ fontWeight: 700, mb: 1, textAlign: 'center', fontFamily: 'Space Grotesk' }}>
          Our Core AI Features
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 6, textAlign: 'center' }}>
          Everything you need to supercharge your career journey
        </Typography>
        <Grid container spacing={4}>
          {mainFeatures.map((feature, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'all 0.3s',
                  border: `2px solid ${feature.color}20`,
                  '&:hover': {
                    transform: 'translateY(-12px)',
                    boxShadow: `0 12px 24px ${feature.color}30`,
                    borderColor: feature.color,
                  },
                }}
              >
                <CardContent sx={{ flexGrow: 1, p: 4 }}>
                  <Box
                    sx={{
                      width: 100,
                      height: 100,
                      mb: 3,
                      mx: 'auto',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderRadius: '50%',
                      background: `${feature.color}15`,
                      animation: 'morphShape 8s ease-in-out infinite, floatUpDown 6s ease-in-out infinite',
                    }}
                  >
                    <Box sx={{ color: feature.color }}>
                      {feature.icon}
                    </Box>
                  </Box>
                  <Typography variant="h5" sx={{ fontWeight: 700, mb: 2, textAlign: 'center', fontFamily: 'Space Grotesk' }}>
                    {feature.title}
                  </Typography>
                  <Typography variant="body1" color="text.secondary" sx={{ mb: 3, textAlign: 'center', lineHeight: 1.7 }}>
                    {feature.description}
                  </Typography>
                  <Button
                    fullWidth
                    variant="contained"
                    onClick={() => navigate(feature.path)}
                    endIcon={<ArrowIcon />}
                    sx={{
                      bgcolor: feature.color,
                      fontWeight: 600,
                      py: 1.5,
                      fontFamily: 'Space Grotesk',
                      '&:hover': {
                        bgcolor: feature.color,
                        opacity: 0.9,
                      },
                    }}
                  >
                    Get Started
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* CTA Section */}
      <Paper
        sx={{
          p: 6,
          textAlign: 'center',
          bgcolor: '#eff6ff',
          border: '2px dashed #1e3a8a',
        }}
      >
        <Typography variant="h4" sx={{ fontWeight: 700, mb: 2 }}>
          Ready to Transform Your Career?
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 4, maxWidth: 600, mx: 'auto' }}>
          Join thousands of professionals who are already using NextStepAI to accelerate their career growth and land their dream jobs.
        </Typography>
        <Button
          variant="contained"
          size="large"
          onClick={() => navigate('/cv-analyzer')}
          sx={{
            bgcolor: '#1e3a8a',
            fontWeight: 700,
            px: 6,
            py: 2,
            fontSize: '1.1rem',
            fontFamily: 'Space Grotesk',
            '&:hover': {
              bgcolor: '#1e40af',
            },
          }}
        >
          Start Now - It's Free
        </Button>
      </Paper>
    </Container>
  );
};

export default Dashboard;
