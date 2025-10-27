import React, { useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Navigate } from 'react-router-dom';
import {
  Container,
  Box,
  Paper,
  Typography,
  Grid,
  Alert,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  Dashboard as DashboardIcon,
  People as PeopleIcon,
  Description as CVIcon,
  Psychology as CareerIcon,
} from '@mui/icons-material';

const AdminDashboard = () => {
  const { isAdmin } = useAuth();
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    // Generate sample data for the chart
    const generateData = () => {
      const data = [];
      const startDate = new Date('2024-09-01');
      
      for (let i = 0; i < 56; i++) {
        const currentDate = new Date(startDate);
        currentDate.setDate(startDate.getDate() + i);
        
        // Generate realistic usage patterns
        const cvBase = 15 + Math.random() * 10;
        const cvTrend = (i / 56) * 20;
        const cvNoise = Math.random() * 8 - 4;
        
        const careerBase = 10 + Math.random() * 8;
        const careerTrend = (i / 56) * 25;
        const careerNoise = Math.random() * 6 - 3;
        
        const resumeBase = 8 + Math.random() * 6;
        const resumeTrend = (i / 56) * 18;
        const resumeNoise = Math.random() * 5 - 2.5;
        
        data.push({
          date: currentDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          cvAnalyzer: Math.max(0, Math.round(cvBase + cvTrend + cvNoise)),
          careerAdvisor: Math.max(0, Math.round(careerBase + careerTrend + careerNoise)),
          resumeAnalyzer: Math.max(0, Math.round(resumeBase + resumeTrend + resumeNoise)),
        });
      }
      
      return data;
    };

    setChartData(generateData());
  }, []);

  if (!isAdmin()) {
    return <Navigate to="/" replace />;
  }

  const stats = [
    {
      title: 'Total Users',
      value: '1,234',
      icon: <PeopleIcon sx={{ fontSize: 40 }} />,
      color: '#667eea',
      change: '+12%',
    },
    {
      title: 'CV Analyses',
      value: '1,511',
      icon: <CVIcon sx={{ fontSize: 40 }} />,
      color: '#764ba2',
      change: '+158%',
    },
    {
      title: 'Career Queries',
      value: '1,111',
      icon: <CareerIcon sx={{ fontSize: 40 }} />,
      color: '#43a047',
      change: '+278%',
    },
  ];

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          📊 Admin Dashboard
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Monitor platform analytics and user engagement
        </Typography>
      </Box>

      {/* Stats Grid */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {stats.map((stat, index) => (
          <Grid item xs={12} md={4} key={index}>
            <Paper
              elevation={3}
              sx={{
                p: 3,
                display: 'flex',
                alignItems: 'center',
                background: `linear-gradient(135deg, ${stat.color}22 0%, ${stat.color}11 100%)`,
                borderLeft: `4px solid ${stat.color}`,
              }}
            >
              <Box sx={{ color: stat.color, mr: 2 }}>
                {stat.icon}
              </Box>
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  {stat.title}
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                  {stat.value}
                </Typography>
                <Typography variant="body2" color="success.main" sx={{ fontWeight: 'bold' }}>
                  {stat.change} this month
                </Typography>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>

      {/* User Engagement Chart */}
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold', mb: 3 }}>
          📈 User Engagement Over Time
        </Typography>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 12 }}
              interval={Math.floor(chartData.length / 10)}
            />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="cvAnalyzer"
              stroke="#667eea"
              strokeWidth={2}
              name="CV Analyzer"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="careerAdvisor"
              stroke="#764ba2"
              strokeWidth={2}
              name="Career Advisor"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="resumeAnalyzer"
              stroke="#43a047"
              strokeWidth={2}
              name="Resume Analyzer"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Paper>

      {/* Additional Info */}
      <Paper elevation={3} sx={{ p: 3, background: 'linear-gradient(135deg, #667eea22 0%, #764ba222 100%)' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <DashboardIcon sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
            Platform Overview
          </Typography>
        </Box>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" gutterBottom>
              <strong>Peak Usage:</strong> October 26, 2024
            </Typography>
            <Typography variant="body2" gutterBottom>
              <strong>Most Popular Feature:</strong> CV Analyzer
            </Typography>
            <Typography variant="body2" gutterBottom>
              <strong>Fastest Growing:</strong> Career Advisor (+278%)
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="body2" gutterBottom>
              <strong>Average Daily Engagement:</strong> 63 uses
            </Typography>
            <Typography variant="body2" gutterBottom>
              <strong>Platform Status:</strong> 🟢 Healthy
            </Typography>
            <Typography variant="body2" gutterBottom>
              <strong>Feature Diversity:</strong> 58% (Good)
            </Typography>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
};

export default AdminDashboard;
