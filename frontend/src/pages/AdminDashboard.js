import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import {
  TrendingUp as TrendingUpIcon,
  People as PeopleIcon,
  Assessment as AssessmentIcon,
  Stars as StarsIcon,
} from '@mui/icons-material';
import { adminAPI } from '../services/api';

const AdminDashboard = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const response = await adminAPI.getStats();
      setStats(response.data);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch admin statistics');
      console.error('Error fetching admin stats:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  if (!stats) return null;

  // KPI Cards Data
  const kpiCards = [
    {
      title: 'Total Users',
      value: stats.total_users,
      subtitle: `${stats.new_users_7days} new this week`,
      icon: <PeopleIcon />,
      color: '#4F46E5',
      bgColor: 'rgba(79, 70, 229, 0.1)',
    },
    {
      title: 'Active Users (30d)',
      value: stats.active_users_30days,
      subtitle: `${stats.active_users_7days} active this week`,
      icon: <TrendingUpIcon />,
      color: '#06B6D4',
      bgColor: 'rgba(6, 182, 212, 0.1)',
    },
    {
      title: 'Total Analyses',
      value: stats.total_analyses,
      subtitle: `${stats.analyses_7days} this week`,
      icon: <AssessmentIcon />,
      color: '#8B5CF6',
      bgColor: 'rgba(139, 92, 246, 0.1)',
    },
    {
      title: 'Avg Match Score',
      value: `${stats.avg_match_percentage?.toFixed(1) || 0}%`,
      subtitle: 'ATS Score Average',
      icon: <StarsIcon />,
      color: '#10B981',
      bgColor: 'rgba(16, 185, 129, 0.1)',
    },
  ];

  // Chart Colors
  const COLORS = ['#4F46E5', '#06B6D4', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444'];

  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%)',
      py: 4,
    }}>
      <Container maxWidth="xl">
        {/* Header */}
        <Box mb={4}>
          <Typography 
            variant="h3" 
            sx={{ 
              fontWeight: 700,
              background: 'linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              mb: 1,
              fontFamily: "'Space Grotesk', 'Poppins', sans-serif",
            }}
          >
            Admin Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Platform insights and analytics
          </Typography>
        </Box>

        {/* KPI Cards */}
        <Grid container spacing={3} mb={4}>
          {kpiCards.map((card, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card 
                sx={{ 
                  height: '100%',
                  background: '#FFFFFF',
                  border: '1px solid rgba(0,0,0,0.06)',
                  borderRadius: '12px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0 12px 24px rgba(0,0,0,0.12)',
                  }
                }}
              >
                <CardContent>
                  <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                    <Box 
                      sx={{ 
                        p: 1.5, 
                        borderRadius: '10px', 
                        bgcolor: card.bgColor,
                        color: card.color,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      {card.icon}
                    </Box>
                  </Box>
                  <Typography 
                    variant="h4" 
                    sx={{ 
                      fontWeight: 700, 
                      color: '#1A1A1A',
                      mb: 0.5,
                      fontFamily: "'Poppins', sans-serif",
                    }}
                  >
                    {card.value}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600, mb: 0.5 }}>
                    {card.title}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {card.subtitle}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Retention Metrics */}
        <Paper 
          sx={{ 
            p: 3, 
            mb: 4,
            background: '#FFFFFF',
            border: '1px solid rgba(0,0,0,0.06)',
            borderRadius: '12px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
          }}
        >
          <Typography variant="h6" sx={{ fontWeight: 700, mb: 2, fontFamily: "'Poppins', sans-serif" }}>
            Retention Metrics
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Box textAlign="center">
                <Typography variant="h3" sx={{ fontWeight: 700, color: '#4F46E5' }}>
                  {(stats.retention_rate * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">Overall Retention</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box textAlign="center">
                <Typography variant="h3" sx={{ fontWeight: 700, color: '#06B6D4' }}>
                  {(stats.retention_7days * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">7-Day Retention</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box textAlign="center">
                <Typography variant="h3" sx={{ fontWeight: 700, color: '#8B5CF6' }}>
                  {(stats.retention_30days * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">30-Day Retention</Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Charts Grid */}
        <Grid container spacing={3} mb={4}>
          {/* User Growth Chart */}
          <Grid item xs={12} lg={8}>
            <Paper 
              sx={{ 
                p: 3,
                background: '#FFFFFF',
                border: '1px solid rgba(0,0,0,0.06)',
                borderRadius: '12px',
                boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
              }}
            >
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 3, fontFamily: "'Poppins', sans-serif" }}>
                User Growth (30 Days)
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={stats.user_growth}>
                  <defs>
                    <linearGradient id="colorUsers" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4F46E5" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#4F46E5" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#6B7280"
                    style={{ fontSize: '12px', fontFamily: "'Poppins', sans-serif" }}
                  />
                  <YAxis 
                    stroke="#6B7280"
                    style={{ fontSize: '12px', fontFamily: "'Poppins', sans-serif" }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#FFFFFF',
                      border: '1px solid rgba(0,0,0,0.1)',
                      borderRadius: '8px',
                      fontFamily: "'Poppins', sans-serif",
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="count" 
                    stroke="#4F46E5" 
                    strokeWidth={2}
                    fillOpacity={1} 
                    fill="url(#colorUsers)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* Match Score Distribution */}
          <Grid item xs={12} lg={4}>
            <Paper 
              sx={{ 
                p: 3,
                background: '#FFFFFF',
                border: '1px solid rgba(0,0,0,0.06)',
                borderRadius: '12px',
                boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
              }}
            >
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 3, fontFamily: "'Poppins', sans-serif" }}>
                Match Score Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={[
                      { name: 'Excellent (80-100%)', value: stats.match_distribution?.filter(s => s >= 80).length || 0 },
                      { name: 'Good (60-79%)', value: stats.match_distribution?.filter(s => s >= 60 && s < 80).length || 0 },
                      { name: 'Fair (40-59%)', value: stats.match_distribution?.filter(s => s >= 40 && s < 60).length || 0 },
                      { name: 'Poor (<40%)', value: stats.match_distribution?.filter(s => s < 40).length || 0 },
                    ]}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {COLORS.map((color, index) => (
                      <Cell key={`cell-${index}`} fill={color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#FFFFFF',
                      border: '1px solid rgba(0,0,0,0.1)',
                      borderRadius: '8px',
                      fontFamily: "'Poppins', sans-serif",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* Top Recommended Jobs */}
          <Grid item xs={12} md={6}>
            <Paper 
              sx={{ 
                p: 3,
                background: '#FFFFFF',
                border: '1px solid rgba(0,0,0,0.06)',
                borderRadius: '12px',
                boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
              }}
            >
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 3, fontFamily: "'Poppins', sans-serif" }}>
                Top Recommended Jobs
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={stats.top_jobs?.slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis 
                    dataKey="job" 
                    angle={-45}
                    textAnchor="end"
                    height={100}
                    stroke="#6B7280"
                    style={{ fontSize: '10px', fontFamily: "'Poppins', sans-serif" }}
                  />
                  <YAxis 
                    stroke="#6B7280"
                    style={{ fontSize: '12px', fontFamily: "'Poppins', sans-serif" }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#FFFFFF',
                      border: '1px solid rgba(0,0,0,0.1)',
                      borderRadius: '8px',
                      fontFamily: "'Poppins', sans-serif",
                    }}
                  />
                  <Bar dataKey="count" fill="#4F46E5" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>

          {/* Top Missing Skills */}
          <Grid item xs={12} md={6}>
            <Paper 
              sx={{ 
                p: 3,
                background: '#FFFFFF',
                border: '1px solid rgba(0,0,0,0.06)',
                borderRadius: '12px',
                boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
              }}
            >
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 3, fontFamily: "'Poppins', sans-serif" }}>
                Top Missing Skills
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={stats.top_missing_skills?.slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis 
                    dataKey="skill" 
                    angle={-45}
                    textAnchor="end"
                    height={100}
                    stroke="#6B7280"
                    style={{ fontSize: '10px', fontFamily: "'Poppins', sans-serif" }}
                  />
                  <YAxis 
                    stroke="#6B7280"
                    style={{ fontSize: '12px', fontFamily: "'Poppins', sans-serif" }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#FFFFFF',
                      border: '1px solid rgba(0,0,0,0.1)',
                      borderRadius: '8px',
                      fontFamily: "'Poppins', sans-serif",
                    }}
                  />
                  <Bar dataKey="count" fill="#EF4444" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        </Grid>

        {/* Recent Activity */}
        <Paper 
          sx={{ 
            p: 3,
            background: '#FFFFFF',
            border: '1px solid rgba(0,0,0,0.06)',
            borderRadius: '12px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
          }}
        >
          <Typography variant="h6" sx={{ fontWeight: 700, mb: 3, fontFamily: "'Poppins', sans-serif" }}>
            Recent Activity
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Type</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>User</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Action</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Time</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {stats.recent_activity?.slice(0, 10).map((activity, index) => (
                  <TableRow key={index} hover>
                    <TableCell>
                      <Chip 
                        label={activity.type}
                        size="small"
                        sx={{ 
                          bgcolor: activity.type === 'analysis' ? 'rgba(79, 70, 229, 0.1)' : 'rgba(6, 182, 212, 0.1)',
                          color: activity.type === 'analysis' ? '#4F46E5' : '#06B6D4',
                          fontWeight: 600,
                        }}
                      />
                    </TableCell>
                    <TableCell>{activity.user}</TableCell>
                    <TableCell>{activity.action}</TableCell>
                    <TableCell sx={{ color: 'text.secondary', fontSize: '0.875rem' }}>
                      {new Date(activity.timestamp).toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      </Container>
    </Box>
  );
};

export default AdminDashboard;
