import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Aurora from '../components/Aurora';
import {
  Container,
  Typography,
  Button,
  Box,
  Modal,
  Grid,
  Paper,
  Card,
  CardContent,
} from '@mui/material';
import {
  TrendingUp,
  Work,
  School,
  Speed,
  BarChart,
  VerifiedUser,
  Star,
} from '@mui/icons-material';

const Dashboard = () => {
  const [openModal, setOpenModal] = useState(null);
  const navigate = useNavigate();

  const handleOpen = (modal) => setOpenModal(modal);
  const handleClose = () => setOpenModal(null);

  const features = [
    {
      title: 'CV Analyzer',
      desc: 'Upload your CV to receive detailed analysis and improvement suggestions.',
      color: '#8b5cf6',
      path: '/cv-analyzer',
    },
    {
      title: 'AI Career Advisor',
      desc: 'Answer a few questions and receive personalized career insights powered by AI.',
      color: '#10b981',
      path: '/career-advisor',
    },
    {
      title: 'Resume Analyzer with JD',
      desc: 'Compare your resume with a job description and identify matching strengths.',
      color: '#3b82f6',
      path: '/cv-analyzer',
    },
  ];

  const handleGetStarted = (path) => {
    handleClose();
    navigate(path);
  };

  return (
    <>
      {/* Aurora Animated Background */}
      <Aurora
        colorStops={['#dc2626', '#f59e0b', '#000000']}
        blend={0.8}
        amplitude={1.5}
        speed={0.4}
      />
    
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        textAlign: 'center',
        px: 3,
        position: 'relative',
        zIndex: 1,
      }}
    >
      {/* Main Hero Section */}
      <Container maxWidth="md">
        <Typography
          variant="h2"
          sx={{
            fontWeight: 800,
            mb: 3,
            color: 'white',
            fontFamily: 'Space Grotesk, sans-serif',
            fontSize: { xs: '2.5rem', md: '3.5rem' },
          }}
        >
          NextStep AI
        </Typography>
        <Typography
          variant="h6"
          sx={{
            mb: 6,
            color: 'white',
            opacity: 0.9,
            lineHeight: 1.6,
            maxWidth: 800,
            mx: 'auto',
            fontSize: { xs: '1rem', md: '1.25rem' },
          }}
        >
          Unlock your professional potential with intelligent career guidance, resume analysis, and personalized job recommendations powered by advanced AI.
        </Typography>

        {/* Buttons Section */}
        <Grid container spacing={3} justifyContent="center" sx={{ mb: 4 }}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Button
                onClick={() => handleOpen(feature.title)}
                sx={{
                  width: '100%',
                  py: 2,
                  fontWeight: 700,
                  color: 'white',
                  borderRadius: '2rem',
                  border: `2px solid ${feature.color}`,
                  background: 'rgba(255, 255, 255, 0.05)',
                  backdropFilter: 'blur(10px)',
                  transition: 'all 0.3s',
                  fontSize: { xs: '0.9rem', md: '1rem' },
                  boxShadow: `0 0 20px ${feature.color}30`,
                  '&:hover': {
                    background: `${feature.color}20`,
                    borderColor: feature.color,
                    transform: 'translateY(-4px)',
                    boxShadow: `0 8px 25px ${feature.color}80`,
                  },
                }}
              >
                {feature.title}
              </Button>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* Benefits Section */}
      <Container maxWidth="lg" sx={{ mt: 8, mb: 8 }}>
        <Typography
          variant="h3"
          sx={{
            fontWeight: 800,
            textAlign: 'center',
            mb: 2,
            color: 'white',
            fontFamily: 'Space Grotesk, sans-serif',
          }}
        >
          Why Choose NextStep AI?
        </Typography>
        <Typography
          variant="body1"
          sx={{
            textAlign: 'center',
            mb: 6,
            color: 'rgba(255, 255, 255, 0.7)',
            maxWidth: 600,
            mx: 'auto',
          }}
        >
          Powered by cutting-edge AI to transform your career journey
        </Typography>
        <Grid container spacing={4}>
          {[
            {
              icon: <Work />,
              title: 'Smart Job Matching',
              desc: 'AI algorithms match you with the perfect opportunities',
              color: '#3b82f6',
            },
            {
              icon: <School />,
              title: 'Skill Development',
              desc: 'Identify gaps and get personalized learning paths',
              color: '#10b981',
            },
            {
              icon: <BarChart />,
              title: 'Data-Driven Insights',
              desc: 'Make informed career decisions with real-time analytics',
              color: '#6366f1',
            },
            {
              icon: <VerifiedUser />,
              title: 'Privacy First',
              desc: 'Your data is encrypted and secure at all times',
              color: '#8b5cf6',
            },
            {
              icon: <TrendingUp />,
              title: 'Career Growth',
              desc: 'Track your progress and accelerate your journey',
              color: '#10b981',
            },
          ].map((benefit, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card
                sx={{
                  height: '100%',
                  background: `linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%)`,
                  backdropFilter: 'blur(20px)',
                  border: `1px solid rgba(255, 255, 255, 0.2)`,
                  borderRadius: 3,
                  transition: 'all 0.3s',
                  position: 'relative',
                  overflow: 'hidden',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    background: `linear-gradient(135deg, ${benefit.color}20 0%, transparent 100%)`,
                    opacity: 0,
                    transition: 'opacity 0.3s',
                  },
                  '&:hover': {
                    transform: 'translateY(-8px) scale(1.02)',
                    boxShadow: `0 12px 40px ${benefit.color}50`,
                    borderColor: benefit.color,
                    '&::before': {
                      opacity: 1,
                    },
                  },
                }}
              >
                <CardContent sx={{ p: 3, position: 'relative', zIndex: 1 }}>
                  <Box
                    sx={{
                      color: benefit.color,
                      mb: 2,
                      fontSize: '2.5rem',
                      filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.2))',
                    }}
                  >
                    {benefit.icon}
                  </Box>
                  <Typography
                    variant="h6"
                    sx={{
                      fontWeight: 700,
                      mb: 1.5,
                      color: 'white',
                      fontFamily: 'Space Grotesk',
                      textShadow: '0 2px 8px rgba(0,0,0,0.8)',
                    }}
                  >
                    {benefit.title}
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{
                      color: 'rgba(255, 255, 255, 0.9)',
                      lineHeight: 1.6,
                      fontWeight: 500,
                      textShadow: '0 1px 4px rgba(0,0,0,0.6)',
                    }}
                  >
                    {benefit.desc}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* CTA Section */}
      <Container maxWidth="md" sx={{ mt: 8, mb: 8 }}>
        <Paper
          sx={{
            p: 6,
            textAlign: 'center',
            background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '2rem',
            boxShadow: '0 8px 32px rgba(139, 92, 246, 0.2)',
          }}
        >
          <Typography
            variant="h4"
            sx={{
              fontWeight: 800,
              mb: 2,
              color: 'white',
              fontFamily: 'Space Grotesk',
              textShadow: '0 2px 8px rgba(0,0,0,0.8)',
            }}
          >
            Ready to Transform Your Career?
          </Typography>
          <Typography
            variant="body1"
            sx={{
              mb: 4,
              color: 'rgba(255, 255, 255, 0.9)',
              maxWidth: 600,
              mx: 'auto',
              fontWeight: 500,
              textShadow: '0 1px 4px rgba(0,0,0,0.6)',
            }}
          >
            Join thousands of professionals who are already using NextStepAI to accelerate their career growth and land their dream jobs.
          </Typography>
          <Button
            variant="contained"
            size="large"
            onClick={() => navigate('/cv-analyzer')}
            sx={{
              background: 'linear-gradient(135deg, #8b5cf6 0%, #10b981 100%)',
              fontWeight: 700,
              px: 6,
              py: 1.5,
              fontSize: '1.1rem',
              borderRadius: '2rem',
              '&:hover': {
                background: 'linear-gradient(135deg, #7c3aed 0%, #059669 100%)',
                transform: 'translateY(-4px)',
                boxShadow: '0 8px 25px rgba(139, 92, 246, 0.5)',
              },
            }}
          >
            Get Started Now
          </Button>
        </Paper>
      </Container>

      {/* Feature Modals */}
      {features.map((feature, index) => (
        <Modal
          key={index}
          open={openModal === feature.title}
          onClose={handleClose}
          aria-labelledby={`${feature.title}-modal`}
          sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        >
          <Box
            sx={{
              bgcolor: 'rgba(255, 255, 255, 0.95)',
              borderRadius: 3,
              p: 4,
              maxWidth: 500,
              mx: 2,
              textAlign: 'center',
              color: '#1e293b',
              boxShadow: `0 8px 32px ${feature.color}40`,
              border: `2px solid rgba(255, 255, 255, 0.5)`,
              backdropFilter: 'blur(20px)',
            }}
          >
            <Typography variant="h4" sx={{ mb: 2, fontWeight: 700, color: '#1e293b' }}>
              {feature.title}
            </Typography>
            <Typography variant="body1" sx={{ mb: 3, color: '#475569', fontWeight: 500 }}>
              {feature.desc}
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Button
                onClick={() => handleGetStarted(feature.path)}
                variant="contained"
                sx={{
                  background: feature.color,
                  fontWeight: 700,
                  borderRadius: '1.5rem',
                  px: 4,
                  '&:hover': {
                    background: feature.color,
                    opacity: 0.9,
                    transform: 'scale(1.05)',
                  },
                }}
              >
                Get Started
              </Button>
              <Button
                onClick={handleClose}
                variant="outlined"
                sx={{
                  color: '#475569',
                  borderColor: feature.color,
                  fontWeight: 700,
                  borderRadius: '1.5rem',
                  px: 4,
                  '&:hover': {
                    borderColor: feature.color,
                    background: `${feature.color}20`,
                    color: feature.color,
                  },
                }}
              >
                Close
              </Button>
            </Box>
          </Box>
        </Modal>
      ))}
    </Box>
    </>
  );
};

export default Dashboard;
