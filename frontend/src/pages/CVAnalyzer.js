import React, { useState } from 'react';
import { cvAPI } from '../services/api';
import Aurora from '../components/Aurora';
import {
  Container,
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  CircularProgress,
  Grid,
  Chip,
  LinearProgress,
  Link,
  Divider,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Description as FileIcon,
  WorkOutline as JobIcon,
  TrendingUp as SkillIcon,
  YouTube as YouTubeIcon,
  Description as DescriptionIcon,
  Insights as InsightsIcon,
  CheckCircle as CheckIcon,
  ArrowForward as ArrowIcon,
  School as LearnIcon,
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
      
      if (response.data && typeof response.data === 'object' && response.data.detail) {
        throw new Error(response.data.detail);
      }
      
      setResult(response.data);
    } catch (err) {
      console.error('Error analyzing resume:', err);
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

  const renderJobTitle = (job) => {
    if (typeof job === 'string') {
      return job;
    }
    if (job && typeof job === 'object') {
      return job.title || job.job_title || 'Job Title';
    }
    return 'Job Title';
  };

  const renderJobCompany = (job) => {
    if (typeof job === 'string') {
      return '';
    }
    if (job && typeof job === 'object') {
      return job.company || '';
    }
    return '';
  };

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
    <>
      {/* Aurora Animated Background */}
      <Aurora
        colorStops={['#dc2626', '#f59e0b', '#000000']}
        blend={0.8}
        amplitude={1.5}
        speed={0.4}
      />
    
    <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1, pb: 8 }}>
      <Box sx={{ mb: 5 }}>
        <Typography 
          variant="h2" 
          gutterBottom 
          sx={{ 
            fontWeight: 700, 
            mb: 2,
            fontFamily: 'Space Grotesk',
            color: 'white',
            textShadow: '0 2px 8px rgba(0,0,0,0.8)',
          }}
        >
          CV Analyzer
        </Typography>
        <Typography variant="h6" sx={{ fontWeight: 500, color: 'rgba(255, 255, 255, 0.8)', textShadow: '0 2px 4px rgba(0,0,0,0.8)' }}>
          Upload your resume for AI-powered analysis and job recommendations
        </Typography>
      </Box>

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
              border: '1px solid rgba(255, 255, 255, 0.1)',
              background: 'rgba(26, 24, 24, 0.95)',
              backdropFilter: 'blur(10px)',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 3, fontFamily: 'Space Grotesk', color: 'white' }}>
              Upload Resume
            </Typography>
            <Box component="form" onSubmit={handleSubmit}>
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
                      startIcon={<UploadIcon sx={{ color: 'rgba(255,255,255,0.9)' }} />}
                      fullWidth
                      sx={{ 
                        py: 2.5, 
                        mb: 1.5,
                        borderRadius: 2,
                        borderWidth: 2,
                        borderStyle: 'dashed',
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        color: 'rgba(255,255,255,0.95)',
                        background: 'rgba(17, 24, 39, 0.5)',
                        fontSize: '0.95rem',
                        fontWeight: 600,
                        fontFamily: 'Space Grotesk',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          borderWidth: 2,
                          borderColor: 'rgba(255,255,255,0.4)',
                          background: 'rgba(17, 24, 39, 0.7)',
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
                        background: 'rgba(31, 41, 55, 0.6)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        mb: 2,
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <FileIcon sx={{ mr: 1.5, color: 'rgba(255,255,255,0.9)', fontSize: 24 }} />
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 600, color: 'rgba(255,255,255,0.95)' }}>{fileName}</Typography>
                          <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.7)' }}>Ready to analyze</Typography>
                        </Box>
                      </Box>
                    </Box>
                  )}
                </Grid>

                {error && (
                  <Grid item xs={12}>
                    <Alert severity="error" sx={{ fontSize: '0.85rem', background: 'rgba(31, 41, 55, 0.9)', color: 'white' }}>{error}</Alert>
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
                      background: '#6B46C1',
                      color: 'white',
                      '&:hover': {
                        background: '#7C3AED',
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
                        color: 'rgba(255,255,255,0.9)',
                        borderColor: 'rgba(255,255,255,0.2)',
                        '&:hover': {
                          borderWidth: 2,
                          borderColor: 'rgba(255,255,255,0.4)',
                        },
                      }}
                    >
                      Reset
                    </Button>
                  )}
                </Grid>
              </Grid>
            </Box>

            <Box sx={{ mt: 3, p: 2, background: 'rgba(31, 41, 55, 0.4)', borderRadius: 2, border: '1px solid rgba(255,255,255,0.05)' }}>
              <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 1, color: 'rgba(255,255,255,0.8)' }}>
                SUPPORTED FORMATS
              </Typography>
              <Typography variant="caption" sx={{ display: 'block', color: 'rgba(255,255,255,0.6)' }}>
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
                border: '1px solid rgba(255,255,255,0.1)',
                background: 'rgba(26, 24, 24, 0.95)',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
              }}
            >
              <Box sx={{ mb: 3, animation: 'pulse 2s ease-in-out infinite' }}>
                <CircularProgress size={60} thickness={4} sx={{ color: '#f59e0b' }} />
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 1, fontFamily: 'Space Grotesk', color: 'white' }}>
                Analyzing your resume...
              </Typography>
              <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)' }}>
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
                border: '1px dashed rgba(255, 255, 255, 0.15)',
                background: 'rgba(26, 24, 24, 0.95)',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
              }}
            >
              <Box sx={{ opacity: 0.6, mb: 3 }}>
                <DescriptionIcon sx={{ fontSize: 80, color: 'rgba(255,255,255,0.9)' }} />
              </Box>
              <Typography variant="h5" sx={{ fontWeight: 700, mb: 2, fontFamily: 'Space Grotesk', color: 'white' }}>
                Upload your resume to get started
              </Typography>
              <Typography variant="body1" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
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
                  sx={{ 
                    p: 5, 
                    mb: 4,
                    borderRadius: 3,
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    background: 'rgba(26, 24, 24, 0.95)',
                    backdropFilter: 'blur(10px)',
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
                    animation: 'scaleIn 0.4s ease-out',
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                    <Box sx={{
                      p: 1.5,
                      borderRadius: 2,
                      background: 'linear-gradient(135deg, #dc2626 0%, #f59e0b 100%)',
                    }}>
                      <InsightsIcon sx={{ fontSize: 36, color: 'white' }} />
                    </Box>
                    <Typography variant="h4" sx={{ fontWeight: 800, color: 'white', fontFamily: 'Space Grotesk' }}>
                      Match Score
                    </Typography>
                  </Box>
                  <Box sx={{ mb: 3 }}>
                    <LinearProgress
                      variant="determinate"
                      value={result.match_percentage}
                      sx={{ 
                        height: 12, 
                        borderRadius: 6, 
                        mb: 2,
                        background: 'rgba(255, 255, 255, 0.1)',
                        '& .MuiLinearProgress-bar': {
                          borderRadius: 6,
                          background: 'linear-gradient(90deg, #dc2626 0%, #f59e0b 100%)',
                        },
                      }}
                    />
                    <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
                      <Typography variant="h2" sx={{ fontWeight: 800, color: '#f59e0b' }}>
                        {result.match_percentage.toFixed(1)}
                      </Typography>
                      <Typography variant="h4" sx={{ fontWeight: 600, color: 'rgba(255,255,255,0.7)' }}>
                        %
                      </Typography>
                    </Box>
                    <Typography variant="h6" sx={{ fontWeight: 500, color: 'rgba(255,255,255,0.8)', mt: 1 }}>
                      Match with Recommended Role
                    </Typography>
                  </Box>
                  <Divider sx={{ background: 'rgba(255,255,255,0.1)', my: 2 }} />
                  <Typography variant="body2" sx={{ lineHeight: 1.8, color: 'rgba(255,255,255,0.7)' }}>
                    This score represents how well your current skills match the requirements for the recommended job role.
                  </Typography>
                </Paper>
              )}

              {/* Skills Section */}
              <Paper 
                elevation={0} 
                sx={{ 
                  p: 5, 
                  mb: 4,
                  borderRadius: 3,
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  background: 'rgba(26, 24, 24, 0.95)',
                  backdropFilter: 'blur(10px)',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
                  animation: 'slideInLeft 0.4s ease-out 0.1s both',
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 4, gap: 2 }}>
                  <Box sx={{
                    p: 1.5,
                    borderRadius: 2,
                    background: 'rgba(220, 38, 38, 0.2)',
                  }}>
                    <SkillIcon sx={{ fontSize: 36, color: '#dc2626' }} />
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 800, color: 'white', fontFamily: 'Space Grotesk' }}>
                    Extracted Skills
                  </Typography>
                </Box>
                <Divider sx={{ background: 'rgba(255,255,255,0.1)', mb: 3 }} />
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1.5 }}>
                  {result.skills && Array.isArray(result.skills) && result.skills.length > 0 ? (
                    result.skills.map((skill, index) => (
                      <Chip
                        key={index}
                        label={typeof skill === 'string' ? skill : JSON.stringify(skill)}
                        sx={{
                          background: 'rgba(31, 41, 55, 0.8)',
                          color: 'rgba(255,255,255,0.9)',
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                          fontWeight: 600,
                          fontSize: '0.85rem',
                          py: 1,
                          px: 1.5,
                          borderRadius: 2,
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            transform: 'translateY(-2px)',
                            borderColor: 'rgba(255, 255, 255, 0.2)',
                            background: 'rgba(31, 41, 55, 1)',
                          },
                        }}
                      />
                    ))
                  ) : (
                    <Typography sx={{ fontStyle: 'italic', color: 'rgba(255,255,255,0.6)' }}>No skills extracted</Typography>
                  )}
                </Box>
              </Paper>

              {/* Two-Column Split Layout: Missing Skills | Job Recommendations */}
              <Grid container spacing={3}>
                {/* LEFT COLUMN - Missing Skills with YouTube Links */}
                <Grid item xs={12} md={6}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 4,
                      height: '100%',
                      borderRadius: 3,
                      border: '2px solid rgba(107, 70, 193, 0.3)',
                      background: 'rgba(26, 24, 24, 0.95)',
                      backdropFilter: 'blur(10px)',
                      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
                      position: 'relative',
                      overflow: 'hidden',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        height: '4px',
                        background: 'linear-gradient(90deg, #6B46C1 0%, #EC4899 100%)',
                      },
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, gap: 2 }}>
                      <Box
                        sx={{
                          width: 44,
                          height: 44,
                          borderRadius: '12px',
                          background: 'linear-gradient(135deg, #6B46C1 0%, #EC4899 100%)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        <LearnIcon sx={{ color: 'white', fontSize: 24 }} />
                      </Box>
                      <Typography variant="h5" sx={{ fontWeight: 800, fontFamily: 'Space Grotesk', color: 'white' }}>
                        Skills to Develop
                      </Typography>
                    </Box>
                    <Divider sx={{ background: 'rgba(107, 70, 193, 0.2)', mb: 3 }} />
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, maxHeight: '600px', overflowY: 'auto', pr: 1 }}>
                      {result.missing_skills_with_links && Array.isArray(result.missing_skills_with_links) && result.missing_skills_with_links.length > 0 ? (
                        result.missing_skills_with_links.map((skillObj, index) => (
                          <Box
                            key={index}
                            sx={{
                              p: 2.5,
                              borderLeft: '4px solid #6B46C1',
                              background: 'rgba(38, 34, 34, 0.6)',
                              borderRadius: 2,
                              transition: 'all 0.3s ease',
                              '&:hover': {
                                background: 'rgba(38, 34, 34, 1)',
                                borderLeftColor: '#EC4899',
                                transform: 'translateX(4px)',
                              },
                            }}
                          >
                            <Typography variant="body1" sx={{ fontWeight: 600, fontFamily: 'Space Grotesk', color: 'white', mb: 1.5 }}>
                              {skillObj.skill_name || (typeof skillObj === 'string' ? skillObj : 'Skill')}
                            </Typography>
                            {skillObj.youtube_link && skillObj.youtube_link !== '#' ? (
                              <Link href={skillObj.youtube_link} target="_blank" rel="noopener" underline="none">
                                <Button
                                  size="small"
                                  variant="outlined"
                                  startIcon={<YouTubeIcon />}
                                  fullWidth
                                  sx={{
                                    borderColor: 'rgba(107, 70, 193, 0.5)',
                                    color: '#6B46C1',
                                    fontWeight: 600,
                                    fontFamily: 'Space Grotesk',
                                    fontSize: '0.75rem',
                                    '&:hover': {
                                      borderColor: '#6B46C1',
                                      background: 'rgba(107, 70, 193, 0.1)',
                                    },
                                  }}
                                >
                                  Watch
                                </Button>
                              </Link>
                            ) : (
                              <Typography variant="body2" sx={{ fontStyle: 'italic', textAlign: 'center', color: 'rgba(255,255,255,0.4)' }}>
                                Coming soon
                              </Typography>
                           )}
                          </Box>
                        ))
                      ) : (
                        <Typography variant="body1" sx={{ textAlign: 'center', py: 4, fontStyle: 'italic', color: 'rgba(255,255,255,0.6)' }}>
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
                      borderRadius: 3,
                      border: '2px solid rgba(245, 158, 11, 0.3)',
                      background: 'rgba(26, 24, 24, 0.95)',
                      backdropFilter: 'blur(10px)',
                      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
                      position: 'relative',
                      overflow: 'hidden',
                      '&::after': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        right: 0,
                        bottom: 0,
                        width: '4px',
                        background: 'linear-gradient(180deg, #F59E0B 0%, #EF4444 100%)',
                      },
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, gap: 2 }}>
                      <Box
                        sx={{
                          width: 44,
                          height: 44,
                          borderRadius: '12px',
                          background: 'linear-gradient(135deg, #F59E0B 0%, #EF4444 100%)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        <JobIcon sx={{ color: 'white', fontSize: 24 }} />
                      </Box>
                      <Typography variant="h5" sx={{ fontWeight: 800, fontFamily: 'Space Grotesk', color: 'white' }}>
                        Job Opportunities
                      </Typography>
                    </Box>
                    <Divider sx={{ background: 'rgba(245, 158, 11, 0.2)', mb: 3 }} />
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, maxHeight: '600px', overflowY: 'auto', pr: 1 }}>
                      {result.live_jobs && Array.isArray(result.live_jobs) && result.live_jobs.length > 0 ? (
                        result.live_jobs.slice(0, 10).map((job, index) => (
                          <Box
                            key={index}
                            sx={{
                              p: 2.5,
                              borderRight: '4px solid #F59E0B',
                              background: 'rgba(42, 34, 30, 0.6)',
                              borderRadius: 2,
                              transition: 'all 0.3s ease',
                              cursor: 'pointer',
                              '&:hover': {
                                background: 'rgba(42, 34, 30, 1)',
                                borderRightColor: '#EF4444',
                                transform: 'translateX(-4px)',
                              },
                            }}
                          >
                            <Link href={renderJobLink(job)} target="_blank" rel="noopener" underline="none">
                              <Typography variant="body1" sx={{ fontWeight: 700, fontFamily: 'Space Grotesk', color: 'white', mb: 1.5, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                                {renderJobTitle(job)}
                              </Typography>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                {renderJobCompany(job) && (
                                  <Typography variant="body2" sx={{ fontWeight: 600, color: 'rgba(245, 158, 11, 0.8)' }}>
                                    {renderJobCompany(job)}
                                  </Typography>
                                )}
                                <Chip
                                  label="Apply"
                                  size="small"
                                  icon={<ArrowIcon sx={{ color: 'white' }} />}
                                  sx={{
                                    background: 'linear-gradient(135deg, #F59E0B 0%, #EF4444 100%)',
                                    color: 'white',
                                    fontWeight: 700,
                                    fontSize: '0.7rem',
                                  }}
                                />
                              </Box>
                            </Link>
                          </Box>
                        ))
                      ) : (
                        <Typography variant="body1" sx={{ textAlign: 'center', py: 4, fontStyle: 'italic', color: 'rgba(255,255,255,0.6)' }}>
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
    </>
  );
};

export default CVAnalyzer;
