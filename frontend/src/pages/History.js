import React, { useState, useEffect } from 'react';
import { historyAPI } from '../services/api';
import {
  Container,
  Box,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Description as FileIcon,
  WorkOutline as JobIcon,
  TrendingUp as SkillIcon,
  CalendarToday as DateIcon,
} from '@mui/icons-material';

const History = () => {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    setLoading(true);
    setError('');
    
    try {
      const response = await historyAPI.getAnalyses();
      setAnalyses(response.data || []);
    } catch (err) {
      console.error('Error fetching history:', err);
      setError(err.response?.data?.detail || 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // Helper to safely parse JSON fields that might be strings
  const parseJsonField = (field) => {
    if (!field) return [];
    if (Array.isArray(field)) return field;
    if (typeof field === 'string') {
      try {
        const parsed = JSON.parse(field);
        return Array.isArray(parsed) ? parsed : [];
      } catch (e) {
        console.warn('Failed to parse JSON field:', e);
        return [];
      }
    }
    return [];
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
          <CircularProgress size={60} />
        </Box>
      </Container>
    );
  }

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
          My History
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ fontWeight: 500 }}>
          View your past CV analyses and job recommendations
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {analyses.length === 0 ? (
        <Paper 
          elevation={0} 
          className="glass-card"
          sx={{ 
            p: 8, 
            textAlign: 'center',
            borderRadius: 4,
            border: '1px solid rgba(245, 158, 11, 0.2)',
            background: 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(20px)',
          }}
        >
          <Box sx={{
            display: 'inline-flex',
            p: 3,
            borderRadius: 4,
            background: 'linear-gradient(135deg, rgba(245,158,11,0.1) 0%, rgba(239,68,68,0.1) 100%)',
            mb: 3,
            animation: 'float 3s ease-in-out infinite',
          }}>
            <FileIcon sx={{ fontSize: 100, color: '#f59e0b' }} />
          </Box>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 800, color: '#78350f', mb: 2 }}>
            No History Yet
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ fontSize: '1.1rem' }}>
            Your CV analyses will appear here once you start using the CV Analyzer
          </Typography>
        </Paper>
      ) : (
        <Box>
          <Paper 
            elevation={0} 
            className="glass-card"
            sx={{ 
              p: 3, 
              mb: 4,
              borderRadius: 3,
              border: '1px solid rgba(245, 158, 11, 0.2)',
              background: 'linear-gradient(135deg, rgba(245,158,11,0.05) 0%, rgba(239,68,68,0.05) 100%)',
              backdropFilter: 'blur(20px)',
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 800 }}>
              📄 Past CV Analyses ({analyses.length})
            </Typography>
          </Paper>

          {analyses.map((analysis, index) => (
            <Accordion 
              key={analysis.id || index} 
              sx={{ 
                mb: 2,
                borderRadius: 3,
                border: '1px solid rgba(245, 158, 11, 0.2)',
                background: 'rgba(255, 255, 255, 0.8)',
                backdropFilter: 'blur(20px)',
                boxShadow: 'none',
                animation: `slideInUp 0.3s ease-out ${index * 0.1}s both`,
                '&:before': {
                  display: 'none',
                },
                '&:hover': {
                  boxShadow: '0 8px 24px rgba(245, 158, 11, 0.15)',
                  borderColor: '#f59e0b',
                },
              }}
            >
              <AccordionSummary 
                expandIcon={<ExpandMoreIcon />}
                sx={{
                  '&:hover': {
                    background: 'rgba(245, 158, 11, 0.05)',
                  },
                }}
              >
                <Box sx={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', pr: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Box sx={{
                      p: 1.5,
                      borderRadius: 2,
                      background: 'linear-gradient(135deg, rgba(245,158,11,0.1) 0%, rgba(239,68,68,0.1) 100%)',
                    }}>
                      <FileIcon sx={{ color: '#f59e0b', fontSize: 28 }} />
                    </Box>
                    <Box>
                      <Typography variant="h6" sx={{ fontWeight: 800 }}>
                        {analysis.resume_filename || `Analysis #${index + 1}`}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                        <DateIcon sx={{ fontSize: 16 }} color="action" />
                        <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>
                          {formatDate(analysis.created_at)}
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
                  <Chip
                    label={`${parseJsonField(analysis.skills).length || analysis.total_skills_count || 0} Skills`}
                    sx={{
                      background: 'linear-gradient(135deg, #f59e0b 0%, #ef4444 100%)',
                      color: 'white',
                      fontWeight: 700,
                    }}
                  />
                </Box>
              </AccordionSummary>
              <AccordionDetails sx={{ p: 4 }}>
                <Divider sx={{ mb: 4, borderColor: 'rgba(245, 158, 11, 0.2)' }} />
                
                {/* Skills */}
                <Box sx={{ mb: 4 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                    <Box sx={{
                      p: 1.5,
                      borderRadius: 2,
                      background: 'linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(139,92,246,0.1) 100%)',
                      mr: 2,
                    }}>
                      <SkillIcon sx={{ color: 'primary.main', fontSize: 28 }} />
                    </Box>
                    <Typography variant="h6" sx={{ fontWeight: 800 }}>
                      Extracted Skills
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1.5 }}>
                    {(() => {
                      const skills = parseJsonField(analysis.skills);
                      return skills.length > 0 ? (
                        skills.map((skill, idx) => (
                          <Chip 
                            key={idx} 
                            label={skill} 
                            sx={{
                              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                              color: 'white',
                              fontWeight: 600,
                              py: 2.5,
                              borderRadius: 2,
                              animation: `scaleIn 0.3s ease-out ${idx * 0.05}s both`,
                              '&:hover': {
                                transform: 'translateY(-2px) scale(1.05)',
                                boxShadow: '0 4px 12px rgba(99, 102, 241, 0.4)',
                              },
                            }}
                          />
                        ))
                      ) : (
                        <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                          No skills recorded
                        </Typography>
                      );
                    })()}
                  </Box>
                </Box>

                {/* Job Recommendations */}
                {(() => {
                  const jobs = parseJsonField(analysis.recommended_jobs);
                  return jobs.length > 0 && (
                    <Box sx={{ mb: 3 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <JobIcon sx={{ mr: 1, color: 'secondary.main' }} />
                        <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                          Job Recommendations
                        </Typography>
                      </Box>
                      <Grid container spacing={2}>
                        {jobs.slice(0, 4).map((job, idx) => (
                          <Grid item xs={12} sm={6} key={idx}>
                            <Card variant="outlined">
                              <CardContent>
                                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }} gutterBottom>
                                  {job.job_title || job.title || 'Job Title'}
                                </Typography>
                                <Box sx={{ mb: 1 }}>
                                  <LinearProgress
                                    variant="determinate"
                                    value={job.match_score || job.similarity_score || 0}
                                    sx={{ height: 6, borderRadius: 3 }}
                                  />
                                  <Typography variant="caption" color="text.secondary">
                                    Match: {(job.match_score || job.similarity_score || 0).toFixed(1)}%
                                  </Typography>
                                </Box>
                                {job.company && (
                                  <Typography variant="body2" color="text.secondary">
                                    {job.company}
                                  </Typography>
                                )}
                              </CardContent>
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    </Box>
                  );
                })()}

                {/* Skills to Add */}
                {(() => {
                  const skillsToAdd = parseJsonField(analysis.skills_to_add);
                  return skillsToAdd.length > 0 && (
                    <Box>
                      <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                        💡 Skills to Develop
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {skillsToAdd.map((skill, idx) => (
                          <Chip key={idx} label={skill} color="success" variant="outlined" size="small" />
                        ))}
                      </Box>
                    </Box>
                  );
                })()}
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      )}
    </Container>
  );
};

export default History;
