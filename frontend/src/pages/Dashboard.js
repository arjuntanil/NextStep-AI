import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Box,
  Paper,
} from '@mui/material';
import {
  Description as DescriptionIcon,
  Psychology as PsychologyIcon,
  Chat as ChatIcon,
  TrendingUp as TrendingUpIcon,
  WorkOutline as WorkIcon,
  School as SchoolIcon,
} from '@mui/icons-material';

const Dashboard = () => {
  const navigate = useNavigate();

  const features = [
    {
      title: 'CV Analyzer',
      description: 'Upload your resume and get AI-powered analysis with job recommendations tailored to your skills.',
      icon: <DescriptionIcon sx={{ fontSize: 60, color: 'primary.main' }} />,
      path: '/cv-analyzer',
      color: '#667eea',
    },
    {
      title: 'Career Advisor',
      description: 'Get personalized career guidance and advice from our fine-tuned AI model.',
      icon: <PsychologyIcon sx={{ fontSize: 60, color: 'secondary.main' }} />,
      path: '/career-advisor',
      color: '#764ba2',
    },
    {
      title: 'RAG Coach',
      description: 'Upload your documents and get contextual answers powered by Retrieval-Augmented Generation.',
      icon: <ChatIcon sx={{ fontSize: 60, color: 'success.main' }} />,
      path: '/rag-coach',
      color: '#43a047',
    },
  ];

  const stats = [
    {
      title: 'Job Matches',
      value: 'AI-Powered',
      icon: <WorkIcon sx={{ fontSize: 40 }} />,
      color: '#667eea',
    },
    {
      title: 'Career Growth',
      value: 'Personalized',
      icon: <TrendingUpIcon sx={{ fontSize: 40 }} />,
      color: '#764ba2',
    },
    {
      title: 'Skill Development',
      value: 'Guided',
      icon: <SchoolIcon sx={{ fontSize: 40 }} />,
      color: '#43a047',
    },
  ];

  return (
    <Container maxWidth="lg" className="fade-in">
      <Box sx={{ mb: 4 }}>
        <Typography 
          variant="h3" 
          gutterBottom 
          className="gradient-text"
          sx={{ fontWeight: 700 }}
        >
          Welcome to NextStepAI! 👋
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Your intelligent career navigation platform
        </Typography>
      </Box>

      {/* Stats Section */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {stats.map((stat, index) => (
          <Grid item xs={12} md={4} key={index}>
            <Paper
              elevation={0}
              className="glass-effect float-animation"
              sx={{
                p: 3,
                display: 'flex',
                alignItems: 'center',
                background: `linear-gradient(135deg, ${stat.color}33 0%, ${stat.color}11 100%)`,
                borderLeft: `4px solid ${stat.color}`,
                border: `1px solid ${stat.color}44`,
                borderRadius: 3,
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: `0 8px 24px ${stat.color}33`,
                },
              }}
            >
              <Box sx={{ color: stat.color, mr: 2 }}>
                {stat.icon}
              </Box>
              <Box>
                <Typography variant="h6" sx={{ fontWeight: 700 }}>
                  {stat.value}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {stat.title}
                </Typography>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>

      {/* Features Section */}
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 700 }}>
        Explore Our Features
      </Typography>
      <Grid container spacing={4}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={4} key={index}>
            <Card
              elevation={0}
              className="glass-effect glow"
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                borderRadius: 3,
                border: `1px solid ${feature.color}33`,
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-12px)',
                  boxShadow: `0 12px 40px ${feature.color}44`,
                  border: `1px solid ${feature.color}66`,
                },
              }}
            >
              <CardContent sx={{ flexGrow: 1, textAlign: 'center', pt: 4 }}>
                <Box sx={{ mb: 2 }}>
                  {feature.icon}
                </Box>
                <Typography gutterBottom variant="h5" component="div" sx={{ fontWeight: 700 }}>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
              <CardActions sx={{ justifyContent: 'center', pb: 3 }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={() => navigate(feature.path)}
                  sx={{
                    background: `linear-gradient(135deg, ${feature.color} 0%, ${feature.color}dd 100%)`,
                    px: 4,
                    py: 1.5,
                    fontWeight: 600,
                    borderRadius: 2,
                    boxShadow: `0 4px 16px ${feature.color}44`,
                    '&:hover': {
                      background: `linear-gradient(135deg, ${feature.color}dd 0%, ${feature.color}bb 100%)`,
                      boxShadow: `0 6px 24px ${feature.color}66`,
                      transform: 'translateY(-2px)',
                    },
                    transition: 'all 0.3s ease',
                  }}
                >
                  Get Started
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Info Section */}
      <Box sx={{ mt: 6, mb: 4 }}>
        <Paper 
          elevation={0} 
          className="glass-effect"
          sx={{ 
            p: 4, 
            borderRadius: 3,
            background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%)',
            border: '1px solid rgba(102, 126, 234, 0.3)',
          }}
        >
          <Typography variant="h5" gutterBottom sx={{ fontWeight: 700 }}>
            🚀 How It Works
          </Typography>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            <Grid item xs={12} md={4}>
              <Box>
                <Typography variant="h6" sx={{ fontWeight: 700, color: 'primary.main', mb: 1 }}>
                  1. Upload & Analyze
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Start by uploading your CV or resume. Our AI will analyze your skills and experience.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box>
                <Typography variant="h6" sx={{ fontWeight: 700, color: 'secondary.main', mb: 1 }}>
                  2. Get Insights
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Receive personalized job recommendations and career guidance based on your profile.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box>
                <Typography variant="h6" sx={{ fontWeight: 700, color: 'success.main', mb: 1 }}>
                  3. Take Action
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Follow AI-powered recommendations to advance your career and achieve your goals.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Box>
    </Container>
  );
};

export default Dashboard;
