import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Aurora from '../components/Aurora';
import {
  Box,
  Container,
  Typography,
  Button,
  Modal,
  Grid,
  AppBar,
  Toolbar,
} from '@mui/material';

const LandingPage = () => {
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
      color: '#3b82f6',
      path: '/career-advisor',
    },
    {
      title: 'Resume Analyzer with JD',
      desc: 'Compare your resume with a job description and identify matching strengths.',
      color: '#10b981',
      path: '/resume-analyzer',
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

      {/* Transparent Header */}
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          background: 'rgba(10, 14, 39, 0.6)',
          borderRadius: '2rem',
          backdropFilter: 'blur(20px)',
          width: { xs: '90%', sm: '80%', md: '70%' },
          left: '50%',
          transform: 'translateX(-50%)',
          mt: 2,
          border: '1px solid rgba(139, 92, 246, 0.2)',
        }}
      >
        <Toolbar sx={{ justifyContent: 'center', gap: { xs: 2, md: 4 } }}>
          <Button color="inherit" sx={{ color: 'white', fontWeight: 600 }}>
            Home
          </Button>
          <Button color="inherit" sx={{ color: 'white', fontWeight: 600 }}>
            About
          </Button>
          <Button color="inherit" sx={{ color: 'white', fontWeight: 600 }}>
            Services
          </Button>
          <Button color="inherit" sx={{ color: 'white', fontWeight: 600 }}>
            Contact
          </Button>
          <Button
            variant="outlined"
            onClick={() => navigate('/login')}
            sx={{
              color: 'white',
              borderColor: 'white',
              borderRadius: '1.5rem',
              px: 3,
              fontWeight: 600,
              '&:hover': {
                background: 'rgba(255, 255, 255, 0.1)',
                borderColor: '#10b981',
                color: '#10b981',
              },
            }}
          >
            Login
          </Button>
        </Toolbar>
      </AppBar>

      {/* Main Hero Section */}
      <Container
        maxWidth="md"
        sx={{
          mt: 20,
          textAlign: 'center',
          color: 'white',
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
        }}
      >
        <Typography
          variant="h2"
          sx={{
            fontWeight: 800,
            mb: 3,
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
            opacity: 0.9,
            lineHeight: 1.6,
            maxWidth: 800,
            mx: 'auto',
            fontSize: { xs: '1rem', md: '1.25rem' },
          }}
        >
          Unlock your professional potential with intelligent career guidance,
          resume analysis, and personalized job recommendations powered by
          advanced AI.
        </Typography>

        {/* Buttons Section */}
        <Grid container spacing={3} justifyContent="center">
          {features.map((feature, index) => (
            <Grid item xs={12} sm={4} key={index}>
              <Button
                onClick={() => handleOpen(feature.title)}
                sx={{
                  width: '100%',
                  py: 1.8,
                  fontWeight: 700,
                  color: 'white',
                  borderRadius: '2rem',
                  border: `2px solid ${feature.color}`,
                  background: 'rgba(255, 255, 255, 0.05)',
                  backdropFilter: 'blur(10px)',
                  transition: '0.3s',
                  fontSize: { xs: '0.9rem', md: '1rem' },
                  '&:hover': {
                    background: feature.color,
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
              bgcolor: 'rgba(10, 14, 39, 0.95)',
              borderRadius: 3,
              p: 4,
              maxWidth: 500,
              mx: 2,
              textAlign: 'center',
              color: 'white',
              boxShadow: `0 0 25px ${feature.color}`,
              border: `2px solid ${feature.color}`,
              backdropFilter: 'blur(20px)',
            }}
          >
            <Typography variant="h4" sx={{ mb: 2, fontWeight: 700 }}>
              {feature.title}
            </Typography>
            <Typography variant="body1" sx={{ mb: 3, opacity: 0.9 }}>
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
                  color: 'white',
                  borderColor: feature.color,
                  fontWeight: 700,
                  borderRadius: '1.5rem',
                  px: 4,
                  '&:hover': {
                    borderColor: feature.color,
                    background: `${feature.color}20`,
                  },
                }}
              >
                Close
              </Button>
            </Box>
          </Box>
        </Modal>
      ))}
    </>
  );
};

export default LandingPage;
