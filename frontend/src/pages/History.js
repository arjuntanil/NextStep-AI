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
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          📜 My History
        </Typography>
        <Typography variant="h6" color="text.secondary">
          View your past CV analyses and job recommendations
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {analyses.length === 0 ? (
        <Paper elevation={3} sx={{ p: 6, textAlign: 'center' }}>
          <FileIcon sx={{ fontSize: 80, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h5" gutterBottom color="text.secondary">
            No History Yet
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Your CV analyses will appear here once you start using the CV Analyzer
          </Typography>
        </Paper>
      ) : (
        <Box>
          <Paper elevation={2} sx={{ p: 2, mb: 3, background: '#f5f5f5' }}>
            <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
              📄 Past CV Analyses ({analyses.length})
            </Typography>
          </Paper>

          {analyses.map((analysis, index) => (
            <Accordion key={analysis.id || index} sx={{ mb: 2 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', pr: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <FileIcon color="primary" />
                    <Box>
                      <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                        {analysis.resume_filename || `Analysis #${index + 1}`}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                        <DateIcon sx={{ fontSize: 16 }} color="action" />
                        <Typography variant="body2" color="text.secondary">
                          {formatDate(analysis.created_at)}
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
                  <Chip
                    label={`${analysis.total_skills_count || analysis.skills?.length || 0} Skills`}
                    color="primary"
                    size="small"
                  />
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Divider sx={{ mb: 3 }} />
                
                {/* Skills */}
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <SkillIcon sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                      Extracted Skills
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {analysis.skills && analysis.skills.length > 0 ? (
                      analysis.skills.map((skill, idx) => (
                        <Chip key={idx} label={skill} color="primary" variant="outlined" size="small" />
                      ))
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        No skills recorded
                      </Typography>
                    )}
                  </Box>
                </Box>

                {/* Job Recommendations */}
                {analysis.recommended_jobs && analysis.recommended_jobs.length > 0 && (
                  <Box sx={{ mb: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <JobIcon sx={{ mr: 1, color: 'secondary.main' }} />
                      <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                        Job Recommendations
                      </Typography>
                    </Box>
                    <Grid container spacing={2}>
                      {analysis.recommended_jobs.slice(0, 4).map((job, idx) => (
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
                )}

                {/* Skills to Add */}
                {analysis.skills_to_add && analysis.skills_to_add.length > 0 && (
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                      💡 Skills to Develop
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {analysis.skills_to_add.map((skill, idx) => (
                        <Chip key={idx} label={skill} color="success" variant="outlined" size="small" />
                      ))}
                    </Box>
                  </Box>
                )}
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      )}
    </Container>
  );
};

export default History;
