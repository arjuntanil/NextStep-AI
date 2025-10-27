import React, { useState } from 'react';
import { cvAPI } from '../services/api';
import {
  Container,
  Box,
  Paper,
  Typography,
  Button,
  TextField,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Grid,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  LinearProgress,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Description as FileIcon,
  WorkOutline as JobIcon,
  TrendingUp as SkillIcon,
} from '@mui/icons-material';

const CVAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [jobDescription, setJobDescription] = useState('');
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
      if (jobDescription.trim()) {
        formData.append('job_description', jobDescription);
      }

      const response = await cvAPI.analyzeResume(formData);
      setResult(response.data);
    } catch (err) {
      console.error('Error analyzing resume:', err);
      setError(err.response?.data?.detail || 'Failed to analyze resume. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setFileName('');
    setJobDescription('');
    setResult(null);
    setError('');
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
          📄 CV Analyzer
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Upload your resume for AI-powered analysis and job recommendations
        </Typography>
      </Box>

      {/* Upload Form */}
      <Paper elevation={3} sx={{ p: 4, mb: 4 }}>
        <Box component="form" onSubmit={handleSubmit}>
          <Grid container spacing={3}>
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
                  sx={{ py: 2, mb: 1 }}
                >
                  {fileName || 'Choose Resume File'}
                </Button>
              </label>
              {fileName && (
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <FileIcon sx={{ mr: 1, color: 'primary.main' }} />
                  <Typography variant="body2">{fileName}</Typography>
                </Box>
              )}
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={6}
                label="Job Description (Optional)"
                placeholder="Paste the job description here for better matching..."
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                variant="outlined"
              />
            </Grid>

            {error && (
              <Grid item xs={12}>
                <Alert severity="error">{error}</Alert>
              </Grid>
            )}

            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button
                  type="submit"
                  variant="contained"
                  size="large"
                  disabled={loading || !file}
                  fullWidth
                >
                  {loading ? <CircularProgress size={24} /> : 'Analyze Resume'}
                </Button>
                <Button
                  variant="outlined"
                  size="large"
                  onClick={handleReset}
                  disabled={loading}
                >
                  Reset
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Box>
      </Paper>

      {/* Results */}
      {loading && (
        <Paper elevation={3} sx={{ p: 4, textAlign: 'center' }}>
          <CircularProgress size={60} sx={{ mb: 2 }} />
          <Typography variant="h6">Analyzing your resume...</Typography>
          <Typography variant="body2" color="text.secondary">
            This may take a few moments
          </Typography>
        </Paper>
      )}

      {result && !loading && (
        <Box>
          {/* Skills Section */}
          <Paper elevation={3} sx={{ p: 4, mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <SkillIcon sx={{ fontSize: 32, mr: 1, color: 'primary.main' }} />
              <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                Extracted Skills
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {result.skills && result.skills.length > 0 ? (
                result.skills.map((skill, index) => (
                  <Chip
                    key={index}
                    label={skill}
                    color="primary"
                    variant="outlined"
                  />
                ))
              ) : (
                <Typography color="text.secondary">No skills extracted</Typography>
              )}
            </Box>
          </Paper>

          {/* Job Recommendations */}
          <Paper elevation={3} sx={{ p: 4, mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <JobIcon sx={{ fontSize: 32, mr: 1, color: 'secondary.main' }} />
              <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                Job Recommendations
              </Typography>
            </Box>
            {result.recommended_jobs && result.recommended_jobs.length > 0 ? (
              <Grid container spacing={2}>
                {result.recommended_jobs.slice(0, 6).map((job, index) => (
                  <Grid item xs={12} md={6} key={index}>
                    <Card variant="outlined" sx={{ height: '100%' }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                          {job.job_title || job.title || 'Job Title'}
                        </Typography>
                        <Box sx={{ mb: 2 }}>
                          <LinearProgress
                            variant="determinate"
                            value={job.match_score || job.similarity_score || 0}
                            sx={{ height: 8, borderRadius: 4, mb: 1 }}
                          />
                          <Typography variant="body2" color="text.secondary">
                            Match: {(job.match_score || job.similarity_score || 0).toFixed(1)}%
                          </Typography>
                        </Box>
                        {job.company && (
                          <Typography variant="body2" color="text.secondary">
                            Company: {job.company}
                          </Typography>
                        )}
                        {job.location && (
                          <Typography variant="body2" color="text.secondary">
                            Location: {job.location}
                          </Typography>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Typography color="text.secondary">No job recommendations available</Typography>
            )}
          </Paper>

          {/* Skills to Add */}
          {result.skills_to_add && result.skills_to_add.length > 0 && (
            <Paper elevation={3} sx={{ p: 4 }}>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold' }}>
                💡 Skills to Develop
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
                {result.skills_to_add.map((skill, index) => (
                  <Chip
                    key={index}
                    label={skill}
                    color="success"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Paper>
          )}
        </Box>
      )}
    </Container>
  );
};

export default CVAnalyzer;
