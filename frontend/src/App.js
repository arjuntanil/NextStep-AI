import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import theme from './theme';
import { AuthProvider } from './contexts/AuthContext';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import LandingPage from './pages/LandingPage';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import CVAnalyzer from './pages/CVAnalyzer';
import ResumeAnalyzer from './pages/ResumeAnalyzer';
import CareerAdvisor from './pages/CareerAdvisor';
import RAGCoach from './pages/RAGCoach';
import History from './pages/History';
import About from './pages/About';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <Router>
          <Routes>
            <Route path="/welcome" element={<LandingPage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <Layout />
                </ProtectedRoute>
              }
            >
              <Route index element={<Dashboard />} />
              <Route path="cv-analyzer" element={<CVAnalyzer />} />
              <Route path="resume-analyzer" element={<ResumeAnalyzer />} />
              <Route path="career-advisor" element={<CareerAdvisor />} />
              <Route path="rag-coach" element={<RAGCoach />} />
              <Route path="history" element={<History />} />
              <Route path="about" element={<About />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
