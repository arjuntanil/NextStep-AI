import React, { useState } from 'react';
import { cvAPI, ragAPI } from '../services/api';
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
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Description as FileIcon,
  Insights as InsightsIcon,
} from '@mui/icons-material';

const ResumeAnalyzer = () => {
  const [resumeFile, setResumeFile] = useState(null);
  const [jdFile, setJdFile] = useState(null);
  const [resumeName, setResumeName] = useState('');
  const [jdName, setJdName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const handleFileChange = (setterFile, setterName) => (event) => {
    const f = event.target.files[0];
    if (f) {
      setterFile(f);
      setterName(f.name);
      setError('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!resumeFile || !jdFile) {
      setError('Please select both Resume and Job Description files');
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);
    try {
      // Preferred endpoint if available
      const fd = new FormData();
      fd.append('resume', resumeFile);
      fd.append('job_description', jdFile);
      try {
        const resp = await cvAPI.analyzeResumeWithJD(fd);
        setResult(resp.data);
      } catch (primaryErr) {
        // Fallback to rag coach multi-upload flow
        const fd2 = new FormData();
        fd2.append('resume', resumeFile);
        fd2.append('job_description', jdFile);
        await ragAPI.uploadDocuments(fd2);
        const processed = await ragAPI.getProcessedResult();
        setResult(processed.data);
      }
    } catch (err) {
      const detail = err.response?.data?.detail;
      const errorMessage = typeof detail === 'string' ? detail : (detail ? JSON.stringify(detail) : 'Failed to analyze. Please try again.');
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const ats = (() => {
    if (!result) return null;
    const v = result.ats_score ?? result.match_percentage ?? result.score ?? null;
    if (v == null) return null;
    const n = Math.max(0, Math.min(100, Number(v)));
    return n;
  })();

  return (
    <>
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
            Resume Analyzer (With JD)
          </Typography>
          <Typography variant="h6" sx={{ fontWeight: 500, color: 'rgba(255, 255, 255, 0.8)', textShadow: '0 2px 4px rgba(0,0,0,0.8)' }}>
            Upload resume and job description to get ATS insights
          </Typography>
        </Box>

        <Grid container spacing={3}>
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
                Upload Files
              </Typography>
              <Box component="form" onSubmit={handleSubmit}>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <input accept=".pdf,.doc,.docx,.txt" style={{ display: 'none' }} id="resume-file" type="file" onChange={handleFileChange(setResumeFile, setResumeName)} />
                    <label htmlFor="resume-file">
                      <Button variant="outlined" component="span" startIcon={<UploadIcon sx={{ color: 'rgba(255,255,255,0.9)' }} />} fullWidth sx={{ py: 2.5, mb: 1.5, borderRadius: 2, borderWidth: 2, borderStyle: 'dashed', borderColor: 'rgba(255, 255, 255, 0.2)', color: 'rgba(255,255,255,0.95)', background: 'rgba(17, 24, 39, 0.5)', fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Space Grotesk', '&:hover': { borderWidth: 2, borderColor: 'rgba(255,255,255,0.4)', background: 'rgba(17, 24, 39, 0.7)' } }}>
                        Choose Resume
                      </Button>
                    </label>
                    {resumeName && (
                      <Box sx={{ p: 2, borderRadius: 2, background: 'rgba(31, 41, 55, 0.6)', border: '1px solid rgba(255,255,255,0.1)', mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <FileIcon sx={{ mr: 1.5, color: 'rgba(255,255,255,0.9)', fontSize: 24 }} />
                          <Typography variant="body2" sx={{ fontWeight: 600, color: 'rgba(255,255,255,0.95)' }}>{resumeName}</Typography>
                        </Box>
                      </Box>
                    )}
                  </Grid>

                  <Grid item xs={12}>
                    <input accept=".pdf,.doc,.docx,.txt" style={{ display: 'none' }} id="jd-file" type="file" onChange={handleFileChange(setJdFile, setJdName)} />
                    <label htmlFor="jd-file">
                      <Button variant="outlined" component="span" startIcon={<UploadIcon sx={{ color: 'rgba(255,255,255,0.9)' }} />} fullWidth sx={{ py: 2.5, mb: 1.5, borderRadius: 2, borderWidth: 2, borderStyle: 'dashed', borderColor: 'rgba(255, 255, 255, 0.2)', color: 'rgba(255,255,255,0.95)', background: 'rgba(17, 24, 39, 0.5)', fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Space Grotesk', '&:hover': { borderWidth: 2, borderColor: 'rgba(255,255,255,0.4)', background: 'rgba(17, 24, 39, 0.7)' } }}>
                        Choose Job Description
                      </Button>
                    </label>
                    {jdName && (
                      <Box sx={{ p: 2, borderRadius: 2, background: 'rgba(31, 41, 55, 0.6)', border: '1px solid rgba(255,255,255,0.1)', mb: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <FileIcon sx={{ mr: 1.5, color: 'rgba(255,255,255,0.9)', fontSize: 24 }} />
                          <Typography variant="body2" sx={{ fontWeight: 600, color: 'rgba(255,255,255,0.95)' }}>{jdName}</Typography>
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
                    <Button type="submit" variant="contained" fullWidth disabled={loading || !resumeFile || !jdFile} sx={{ py: 1.5, fontSize: '1rem', fontWeight: 700, fontFamily: 'Space Grotesk', borderRadius: 2, background: '#6B46C1', color: 'white', '&:hover': { background: '#7C3AED' } }}>
                      {loading ? <CircularProgress size={24} sx={{ color: 'white' }} /> : 'Analyze'}
                    </Button>
                  </Grid>
                </Grid>
              </Box>
            </Paper>
          </Grid>

          <Grid item xs={12} lg={8}>
            {loading && (
              <Paper elevation={0} sx={{ p: 6, textAlign: 'center', borderRadius: 3, border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(26, 24, 24, 0.95)', backdropFilter: 'blur(10px)', boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)' }}>
                <Box sx={{ mb: 3, animation: 'pulse 2s ease-in-out infinite' }}>
                  <CircularProgress size={60} thickness={4} sx={{ color: '#f59e0b' }} />
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 700, mb: 1, fontFamily: 'Space Grotesk', color: 'white' }}>
                  Analyzing...
                </Typography>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                  Processing your resume and job description
                </Typography>
              </Paper>
            )}

            {!result && !loading && (
              <Paper elevation={0} sx={{ p: 8, textAlign: 'center', borderRadius: 3, border: '1px dashed rgba(255, 255, 255, 0.15)', background: 'rgba(26, 24, 24, 0.95)', backdropFilter: 'blur(10px)', boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)' }}>
                <Box sx={{ opacity: 0.6, mb: 3 }}>
                  <FileIcon sx={{ fontSize: 80, color: 'rgba(255,255,255,0.9)' }} />
                </Box>
                <Typography variant="h5" sx={{ fontWeight: 700, mb: 2, fontFamily: 'Space Grotesk', color: 'white' }}>
                  Upload your files to get started
                </Typography>
                <Typography variant="body1" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Get ATS score and detailed comparisons
                </Typography>
              </Paper>
            )}

            {result && !loading && (
              <Box>
                {/* ATS Score Card */}
                {ats != null && (
                  <Paper 
                    elevation={0} 
                    sx={{ 
                      p: 4, 
                      mb: 4, 
                      borderRadius: 3, 
                      border: '1px solid rgba(245, 158, 11, 0.3)', 
                      background: 'linear-gradient(135deg, rgba(220, 38, 38, 0.05) 0%, rgba(245, 158, 11, 0.05) 100%)',
                      backdropFilter: 'blur(10px)',
                    }}
                  >
                    <Box sx={{ textAlign: 'center' }}>
                      {/* Icon and Title */}
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1.5, mb: 3 }}>
                        <InsightsIcon sx={{ fontSize: 28, color: '#f59e0b' }} />
                        <Typography 
                          variant="h5" 
                          sx={{ 
                            fontWeight: 700, 
                            color: 'white', 
                            fontFamily: 'Space Grotesk' 
                          }}
                        >
                          ATS Score
                        </Typography>
                      </Box>
                      
                      {/* Score Display */}
                      <Box sx={{ mb: 2 }}>
                        <Typography 
                          variant="h1" 
                          sx={{ 
                            fontWeight: 800, 
                            fontSize: '4rem',
                            background: 'linear-gradient(135deg, #dc2626 0%, #f59e0b 100%)',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                            lineHeight: 1,
                            mb: 0.5
                          }}
                        >
                          {ats.toFixed(0)}%
                        </Typography>
                      </Box>

                      {/* Description */}
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          color: 'rgba(255,255,255,0.7)',
                          maxWidth: 400,
                          mx: 'auto',
                          fontSize: '0.95rem'
                        }}
                      >
                        Resume alignment with job description
                      </Typography>

                      {/* Progress Bar */}
                      <Box 
                        sx={{ 
                          mt: 3, 
                          height: 8, 
                          borderRadius: 4,
                          background: 'rgba(255,255,255,0.1)',
                          overflow: 'hidden',
                          position: 'relative'
                        }}
                      >
                        <Box 
                          sx={{ 
                            height: '100%', 
                            width: `${ats}%`,
                            background: 'linear-gradient(90deg, #dc2626 0%, #f59e0b 100%)',
                            borderRadius: 4,
                            transition: 'width 1s ease-out'
                          }} 
                        />
                      </Box>
                    </Box>
                  </Paper>
                )}

                {/* Matching Skills */}
                {result.matching_skills && result.matching_skills.length > 0 && (
                  <Paper elevation={0} sx={{ p: 4, mb: 3, borderRadius: 3, border: '1px solid rgba(34, 197, 94, 0.3)', background: 'rgba(26, 24, 24, 0.95)', backdropFilter: 'blur(10px)' }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, mb: 2, color: '#22c55e', fontFamily: 'Space Grotesk', display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box component="span" sx={{ fontSize: '1.5rem' }}>✓</Box>
                      Matching Skills ({result.matching_skills.length})
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {result.matching_skills.map((skill, idx) => (
                        <Box 
                          key={idx} 
                          sx={{ 
                            px: 2, 
                            py: 0.75, 
                            borderRadius: 2, 
                            background: 'rgba(34, 197, 94, 0.1)', 
                            border: '1px solid rgba(34, 197, 94, 0.3)', 
                            color: '#22c55e',
                            fontSize: '0.875rem',
                            fontWeight: 600,
                            textTransform: 'capitalize'
                          }}
                        >
                          {skill}
                        </Box>
                      ))}
                    </Box>
                  </Paper>
                )}

                {/* Missing Skills */}
                {result.missing_skills && result.missing_skills.length > 0 && (
                  <Paper elevation={0} sx={{ p: 4, mb: 3, borderRadius: 3, border: '1px solid rgba(239, 68, 68, 0.3)', background: 'rgba(26, 24, 24, 0.95)', backdropFilter: 'blur(10px)' }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, mb: 2, color: '#ef4444', fontFamily: 'Space Grotesk', display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box component="span" sx={{ fontSize: '1.5rem' }}>!</Box>
                      Missing Skills ({result.missing_skills.length})
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {result.missing_skills.map((skill, idx) => {
                        const linkData = result.missing_skills_with_links?.find(item => item.skill_name === skill);
                        const hasLink = linkData && linkData.youtube_link && linkData.youtube_link !== '#';
                        
                        return (
                          <Box 
                            key={idx}
                            component={hasLink ? 'a' : 'div'}
                            href={hasLink ? linkData.youtube_link : undefined}
                            target={hasLink ? '_blank' : undefined}
                            rel={hasLink ? 'noopener noreferrer' : undefined}
                            sx={{ 
                              px: 2, 
                              py: 0.75, 
                              borderRadius: 2, 
                              background: 'rgba(239, 68, 68, 0.1)', 
                              border: '1px solid rgba(239, 68, 68, 0.3)', 
                              color: '#ef4444',
                              fontSize: '0.875rem',
                              fontWeight: 600,
                              textTransform: 'capitalize',
                              textDecoration: 'none',
                              cursor: hasLink ? 'pointer' : 'default',
                              transition: 'all 0.2s',
                              '&:hover': hasLink ? {
                                background: 'rgba(239, 68, 68, 0.2)',
                                borderColor: 'rgba(239, 68, 68, 0.5)',
                                transform: 'translateY(-2px)'
                              } : {}
                            }}
                          >
                            {skill}
                            {hasLink && <Box component="span" sx={{ ml: 0.5, fontSize: '0.75rem' }}>🎥</Box>}
                          </Box>
                        );
                      })}
                    </Box>
                  </Paper>
                )}

                {/* Layout Feedback */}
                {result.layout_feedback && (
                  <Paper elevation={0} sx={{ p: 4, mb: 3, borderRadius: 3, border: '1px solid rgba(59, 130, 246, 0.3)', background: 'rgba(26, 24, 24, 0.95)', backdropFilter: 'blur(10px)' }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, mb: 3, color: '#3b82f6', fontFamily: 'Space Grotesk', display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box component="span" sx={{ fontSize: '1.5rem' }}>💡</Box>
                      ATS Optimization Tips
                    </Typography>
                    <Box sx={{ 
                      color: 'rgba(255,255,255,0.85)', 
                      lineHeight: 1.8,
                      '& ul': { 
                        pl: 2, 
                        mb: 0,
                        listStyleType: 'none'
                      },
                      '& li': {
                        position: 'relative',
                        pl: 3,
                        mb: 2,
                        '&:before': {
                          content: '"▸"',
                          position: 'absolute',
                          left: 0,
                          color: '#3b82f6',
                          fontWeight: 'bold'
                        }
                      }
                    }}>
                      {result.layout_feedback.split('\n').map((line, idx) => {
                        // Convert markdown-style bullets to list items
                        if (line.trim().startsWith('*   ')) {
                          return (
                            <Box key={idx} component="li" sx={{ mb: 2 }}>
                              {line.trim().substring(4)}
                            </Box>
                          );
                        } else if (line.trim()) {
                          return (
                            <Typography key={idx} variant="body2" sx={{ mb: 2, color: 'rgba(255,255,255,0.7)' }}>
                              {line}
                            </Typography>
                          );
                        }
                        return null;
                      })}
                    </Box>
                  </Paper>
                )}

                {/* All Resume Skills */}
                {result.skills && result.skills.length > 0 && (
                  <Paper elevation={0} sx={{ p: 4, borderRadius: 3, border: '1px solid rgba(168, 85, 247, 0.3)', background: 'rgba(26, 24, 24, 0.95)', backdropFilter: 'blur(10px)' }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, mb: 2, color: '#a855f7', fontFamily: 'Space Grotesk' }}>
                      All Detected Skills ({result.skills.length})
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {result.skills.map((skill, idx) => (
                        <Box 
                          key={idx} 
                          sx={{ 
                            px: 2, 
                            py: 0.75, 
                            borderRadius: 2, 
                            background: 'rgba(168, 85, 247, 0.1)', 
                            border: '1px solid rgba(168, 85, 247, 0.2)', 
                            color: 'rgba(255,255,255,0.8)',
                            fontSize: '0.875rem',
                            fontWeight: 500,
                            textTransform: 'capitalize'
                          }}
                        >
                          {skill}
                        </Box>
                      ))}
                    </Box>
                  </Paper>
                )}
              </Box>
            )}
          </Grid>
        </Grid>
      </Container>
    </>
  );
};

export default ResumeAnalyzer;


