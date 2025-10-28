import React, { useState } from 'react';
import { cvAPI } from '../services/api';
import {
  Container,
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress,
  Link,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Description as FileIcon,
  WorkOutline as JobIcon,
  TrendingUp as SkillIcon,
  YouTube as YouTubeIcon,
  Description as DescriptionIcon,
} from '@mui/icons-material';

const CVAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await cvAPI.analyzeResume(formData);
      
      // Check if response contains error information
      if (response.data && typeof response.data === 'object' && response.data.detail) {
        throw new Error(response.data.detail);
      }
      
      setResult(response.data);
    } catch (err) {
      console.error('Error analyzing resume:', err);
      // Better error handling
      const errorMessage = err.response?.data?.detail || 
                          (err.response?.data ? JSON.stringify(err.response.data) : 'Failed to analyze resume. Please try again.');
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setFileName('');
    setResult(null);
    setError('');
  };

  // Helper function to safely render job titles
  const renderJobTitle = (job) => {
    if (typeof job === 'string') {
      return job;
    }
    if (job && typeof job === 'object') {
      return job.title || job.job_title || 'Job Title';
    }
    return 'Job Title';
  };

  // Helper function to safely render job companies
  const renderJobCompany = (job) => {
    if (typeof job === 'string') {
      return '';
    }
    if (job && typeof job === 'object') {
      return job.company || '';
    }
    return '';
  };

  // Helper function to safely render job links
  const renderJobLink = (job) => {
    if (typeof job === 'string') {
      return '#';
    }
    if (job && typeof job === 'object') {
      return job.link || '#';
    }
    return '#';
  };

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
          CV Analyzer
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ fontWeight: 500 }}>
          Upload your resume for AI-powered analysis and job recommendations
        </Typography>
      </Box>

      {/* Main Split Layout - Upload on Left, Results on Right */}
      <Grid container spacing={3}>
        {/* LEFT SIDE - Upload Section */}
        <Grid item xs={12} lg={4}>
          <Paper 
            elevation={0} 
            sx={{ 
              p: 3,
              position: 'sticky',
              top: 80,
              borderRadius: 3,
              border: '2px solid #1e3a8a',
              background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
              position: 'relative',
              overflow: 'hidden',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                zIndex: 0,
                opacity: 0.03,
                backgroundImage: 'radial-gradient(circle at 30% 50%, rgba(30, 58, 138, 0.3) 0%, transparent 40%)',
                animation: 'floatUpDown 15s ease-in-out infinite',
              },
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 3, fontFamily: 'Space Grotesk', color: '#1e3a8a', position: 'relative', zIndex: 1 }}>
              Upload Resume
            </Typography>
            <Box component="form" onSubmit={handleSubmit} sx={{ position: 'relative', zIndex: 1 }}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <input
                    accept=".pdf,.doc,.docx,.txt"
                    style={{ display: 'none' }}
                    id="resume-file"
                    type="file"
                    onChange={handleFileChange}
                  />
                  <label htmlFor="resume-file">
                    <Button
                      variant="outlined"
                      component="span"
                      startIcon={<UploadIcon />}
                      fullWidth
                      sx={{ 
                        py: 2.5, 
                        mb: 1.5,
                        borderRadius: 2,
                        borderWidth: 2,
                        borderStyle: 'dashed',
                        borderColor: fileName ? 'success.main' : '#1e3a8a',
                        background: fileName ? 'rgba(76, 175, 80, 0.05)' : 'rgba(30, 58, 138, 0.05)',
                        fontSize: '0.95rem',
                        fontWeight: 600,
                        fontFamily: 'Space Grotesk',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          borderWidth: 2,
                          transform: 'translateY(-2px)',
                        },
                      }}
                    >
                      Choose File
                    </Button>
                  </label>
                  {fileName && (
                    <Box 
                      sx={{ 
                        p: 2,
                        borderRadius: 2,
                        background: 'rgba(76, 175, 80, 0.1)',
                        border: '1px solid rgba(76, 175, 80, 0.3)',
                        mb: 2,
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <FileIcon sx={{ mr: 1.5, color: 'success.main', fontSize: 24 }} />
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>{fileName}</Typography>
                          <Typography variant="caption" color="text.secondary">Ready to analyze</Typography>
                        </Box>
                      </Box>
                    </Box>
                  )}
                </Grid>

                {error && (
                  <Grid item xs={12}>
                    <Alert severity="error" sx={{ fontSize: '0.85rem' }}>{error}</Alert>
                  </Grid>
                )}

                <Grid item xs={12}>
                  <Button
                    type="submit"
                    variant="contained"
                    fullWidth
                    disabled={loading || !file}
                    sx={{
                      py: 1.5,
                      fontSize: '1rem',
                      fontWeight: 700,
                      fontFamily: 'Space Grotesk',
                      borderRadius: 2,
                      bgcolor: '#1e3a8a',
                      '&:hover': {
                        bgcolor: '#1e40af',
                      },
                    }}
                  >
                    {loading ? <CircularProgress size={24} sx={{ color: 'white' }} /> : 'Analyze Resume'}
                  </Button>
                  {(result || fileName) && (
                    <Button
                      variant="outlined"
                      fullWidth
                      onClick={handleReset}
                      disabled={loading}
                      sx={{
                        mt: 1.5,
                        py: 1.5,
                        fontWeight: 600,
                        fontFamily: 'Space Grotesk',
                        borderWidth: 2,
                        '&:hover': {
                          borderWidth: 2,
                        },
                      }}
                    >
                      Reset
                    </Button>
                  )}
                </Grid>
              </Grid>
            </Box>

            {/* Quick Info */}
            <Box sx={{ mt: 3, p: 2, bgcolor: 'rgba(30, 58, 138, 0.05)', borderRadius: 2 }}>
              <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 1, color: '#1e3a8a' }}>
                📋 Supported Formats
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                PDF, DOC, DOCX, TXT
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* RIGHT SIDE - Results Section */}
        <Grid item xs={12} lg={8}>
          {loading && (
            <Paper 
              elevation={0}
              sx={{ 
                p: 6, 
                textAlign: 'center',
                borderRadius: 3,
                border: '2px solid #0ea5e9',
                background: 'rgba(255, 255, 255, 0.95)',
              }}
            >
              <Box sx={{ mb: 3, animation: 'pulse 2s ease-in-out infinite' }}>
                <CircularProgress size={60} thickness={4} sx={{ color: '#1e3a8a' }} />
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 1, fontFamily: 'Space Grotesk' }}>
                Analyzing your resume...
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Our AI is processing your information
              </Typography>
            </Paper>
          )}

          {!result && !loading && (
            <Paper
              elevation={0}
              sx={{
                p: 8,
                textAlign: 'center',
                borderRadius: 3,
                border: '2px dashed #cbd5e1',
                background: 'rgba(248, 250, 252, 0.5)',
              }}
            >
              <Box sx={{ opacity: 0.4, mb: 3 }}>
                <DescriptionIcon sx={{ fontSize: 80 }} />
              </Box>
              <Typography variant="h5" sx={{ fontWeight: 700, mb: 2, fontFamily: 'Space Grotesk', color: 'text.secondary' }}>
                Upload your resume to get started
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Get AI-powered insights, job matches, and skill recommendations
              </Typography>
            </Paper>
          )}

          {result && !loading && (
            <Box>
          {/* Match Percentage */}
          {result.match_percentage !== undefined && (
            <Paper 
              elevation={0} 
              className="glass-card"
              sx={{ 
                p: 5, 
                mb: 4,
                borderRadius: 4,
                border: '1px solid rgba(99, 102, 241, 0.2)',
                background: 'rgba(255, 255, 255, 0.8)',
                backdropFilter: 'blur(20px)',
                animation: 'scaleIn 0.4s ease-out',
              }}
            >
              <Typography 
                variant="h4" 
                gutterBottom 
                sx={{ 
                  fontWeight: 800,
                  mb: 3,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                }}
              >
                📊 Match Score
              </Typography>
              <Box sx={{ mb: 3 }}>
                <LinearProgress
                  variant="determinate"
                  value={result.match_percentage}
                  sx={{ 
                    height: 16, 
                    borderRadius: 8, 
                    mb: 2,
                    background: 'rgba(99, 102, 241, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 8,
                      background: 'linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%)',
                    },
                  }}
                />
                <Typography variant="h3" sx={{ fontWeight: 800, color: 'primary.main', mb: 1 }}>
                  {result.match_percentage.toFixed(1)}%
                </Typography>
                <Typography variant="h6" color="text.secondary" sx={{ fontWeight: 500 }}>
                  Match with Recommended Role
                </Typography>
              </Box>
              <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                This score represents how well your current skills match the requirements for the recommended job role.
              </Typography>
            </Paper>
          )}

          {/* Skills Section */}
          <Paper 
            elevation={0} 
            className="glass-card"
            sx={{ 
              p: 5, 
              mb: 4,
              borderRadius: 4,
              border: '1px solid rgba(99, 102, 241, 0.2)',
              background: 'rgba(255, 255, 255, 0.8)',
              backdropFilter: 'blur(20px)',
              animation: 'slideInLeft 0.4s ease-out 0.1s both',
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
              <Box sx={{
                p: 1.5,
                borderRadius: 2,
                background: 'linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(139,92,246,0.1) 100%)',
                mr: 2,
              }}>
                <SkillIcon sx={{ fontSize: 36, color: 'primary.main' }} />
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 800 }}>
                Extracted Skills
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1.5 }}>
              {result.skills && Array.isArray(result.skills) && result.skills.length > 0 ? (
                result.skills.map((skill, index) => (
                  <Chip
                    key={index}
                    label={typeof skill === 'string' ? skill : JSON.stringify(skill)}
                    sx={{
                      background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                      color: 'white',
                      fontWeight: 600,
                      fontSize: '0.9rem',
                      py: 2.5,
                      px: 1,
                      borderRadius: 2,
                      boxShadow: '0 2px 8px rgba(99, 102, 241, 0.3)',
                      animation: `scaleIn 0.3s ease-out ${index * 0.05}s both`,
                      '&:hover': {
                        transform: 'translateY(-2px) scale(1.05)',
                        boxShadow: '0 4px 12px rgba(99, 102, 241, 0.4)',
                      },
                    }}
                  />
                ))
              ) : (
                <Typography color="text.secondary" sx={{ fontStyle: 'italic' }}>No skills extracted</Typography>
              )}
            </Box>
          </Paper>

          {/* Two-Column Split Layout: Missing Skills | Job Recommendations */}
          <Grid container spacing={4} sx={{ mb: 4 }}>
            {/* LEFT COLUMN - Missing Skills with YouTube Links */}
            <Grid item xs={12} md={6}>
              <Paper
                elevation={0}
                sx={{
                  p: 4,
                  height: '100%',
                  borderRadius: 4,
                  border: '2px solid #1e3a8a',
                  background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
                  position: 'relative',
                  overflow: 'hidden',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '4px',
                    background: 'linear-gradient(90deg, #1e3a8a 0%, #0ea5e9 100%)',
                  },
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Box
                    sx={{
                      width: 48,
                      height: 48,
                      borderRadius: 2,
                      background: 'linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mr: 2,
                      animation: 'rotateIn 0.6s ease-out',
                    }}
                  >
                    <SkillIcon sx={{ color: 'white', fontSize: 28 }} />
                  </Box>
                  <Typography variant="h5" sx={{ fontWeight: 800, fontFamily: 'Space Grotesk', color: '#1e3a8a' }}>
                    Skills to Develop
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3, lineHeight: 1.7 }}>
                  Master these skills to improve your job match score. Each skill includes curated YouTube tutorials.
                </Typography>
                <Box sx={{ maxHeight: '600px', overflowY: 'auto', pr: 1 }}>
                  {result.missing_skills_with_links && Array.isArray(result.missing_skills_with_links) && result.missing_skills_with_links.length > 0 ? (
                    result.missing_skills_with_links.map((skillObj, index) => (
                      <Card
                        key={index}
                        sx={{
                          mb: 2,
                          border: '1px solid #3b82f6',
                          transition: 'all 0.3s ease',
                          animation: `slideInLeft 0.4s ease-out ${index * 0.1}s both`,
                          '&:hover': {
                            transform: 'translateX(8px)',
                            boxShadow: '0 8px 16px rgba(30, 58, 138, 0.2)',
                            borderColor: '#1e3a8a',
                          },
                        }}
                      >
                        <CardContent sx={{ p: 3 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                            <Typography variant="h6" sx={{ fontWeight: 700, fontFamily: 'Space Grotesk', flex: 1 }}>
                              {skillObj.skill_name || (typeof skillObj === 'string' ? skillObj : 'Skill')}
                            </Typography>
                            <Chip
                              label={`#${index + 1}`}
                              size="small"
                              sx={{
                                bgcolor: '#1e3a8a',
                                color: 'white',
                                fontWeight: 700,
                              }}
                            />
                          </Box>
                          {skillObj.youtube_link && skillObj.youtube_link !== '#' ? (
                            <Link href={skillObj.youtube_link} target="_blank" rel="noopener" underline="none">
                              <Button
                                variant="contained"
                                startIcon={<YouTubeIcon />}
                                fullWidth
                                sx={{
                                  bgcolor: '#1e3a8a',
                                  fontWeight: 600,
                                  fontFamily: 'Space Grotesk',
                                  '&:hover': {
                                    bgcolor: '#1e40af',
                                  },
                                }}
                              >
                                Watch Tutorial
                              </Button>
                            </Link>
                          ) : (
                            <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic', textAlign: 'center' }}>
                              Tutorial coming soon
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                    ))
                  ) : (
                    <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', py: 4, fontStyle: 'italic' }}>
                      No missing skills detected. Great job!
                    </Typography>
                  )}
                </Box>
              </Paper>
            </Grid>

            {/* RIGHT COLUMN - Job Recommendations */}
            <Grid item xs={12} md={6}>
              <Paper
                elevation={0}
                sx={{
                  p: 4,
                  height: '100%',
                  borderRadius: 4,
                  border: '2px solid #0ea5e9',
                  background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                  position: 'relative',
                  overflow: 'hidden',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '4px',
                    background: 'linear-gradient(90deg, #0ea5e9 0%, #06b6d4 100%)',
                  },
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Box
                    sx={{
                      width: 48,
                      height: 48,
                      borderRadius: 2,
                      background: 'linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mr: 2,
                      animation: 'rotateIn 0.6s ease-out 0.2s both',
                    }}
                  >
                    <JobIcon sx={{ color: 'white', fontSize: 28 }} />
                  </Box>
                  <Typography variant="h5" sx={{ fontWeight: 800, fontFamily: 'Space Grotesk', color: '#0284c7' }}>
                    Job Opportunities
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3, lineHeight: 1.7 }}>
                  Live job postings matched to your skills and experience. Click to apply directly.
                </Typography>
                <Box sx={{ maxHeight: '600px', overflowY: 'auto', pr: 1 }}>
                  {result.live_jobs && Array.isArray(result.live_jobs) && result.live_jobs.length > 0 ? (
                    result.live_jobs.slice(0, 10).map((job, index) => (
                      <Card
                        key={index}
                        sx={{
                          mb: 2,
                          border: '1px solid #0ea5e9',
                          transition: 'all 0.3s ease',
                          animation: `slideInRight 0.4s ease-out ${index * 0.1}s both`,
                          cursor: 'pointer',
                          '&:hover': {
                            transform: 'translateX(-8px)',
                            boxShadow: '0 8px 16px rgba(14, 165, 233, 0.2)',
                            borderColor: '#0284c7',
                          },
                        }}
                      >
                        <CardContent sx={{ p: 3 }}>
                          <Link href={renderJobLink(job)} target="_blank" rel="noopener" underline="none">
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                              <Typography variant="h6" sx={{ fontWeight: 700, fontFamily: 'Space Grotesk', color: '#0284c7', flex: 1, pr: 2 }}>
                                {renderJobTitle(job)}
                              </Typography>
                              <Chip
                                label="Apply"
                                size="small"
                                sx={{
                                  bgcolor: '#0ea5e9',
                                  color: 'white',
                                  fontWeight: 700,
                                }}
                              />
                            </Box>
                            {renderJobCompany(job) && (
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 1, fontWeight: 600 }}>
                                {renderJobCompany(job)}
                              </Typography>
                            )}
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                              Click to view full job details →
                            </Typography>
                          </Link>
                        </CardContent>
                      </Card>
                    ))
                  ) : (
                    <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', py: 4, fontStyle: 'italic' }}>
                      No job opportunities available at the moment.
                    </Typography>
                  )}
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </Box>
      )}
        </Grid>
      </Grid>
    </Container>
  );
};

export default CVAnalyzer;