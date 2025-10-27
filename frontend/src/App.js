import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AuthProvider } from './contexts/AuthContext';
import Layout from './components/Layout';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import CVAnalyzer from './pages/CVAnalyzer';
import CareerAdvisor from './pages/CareerAdvisor';
import RAGCoach from './pages/RAGCoach';
import History from './pages/History';
import AdminDashboard from './pages/AdminDashboard';
import ProtectedRoute from './components/ProtectedRoute';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#667eea',
      light: '#8b9fee',
      dark: '#4a5fc1',
    },
    secondary: {
      main: '#764ba2',
      light: '#9d6ec8',
      dark: '#533477',
    },
    background: {
      default: '#0a0e27',
      paper: '#131842',
    },
    text: {
      primary: '#e4e6eb',
      secondary: '#b0b3b8',
    },
    success: {
      main: '#00d9ff',
    },
    error: {
      main: '#ff4757',
    },
    warning: {
      main: '#ffa502',
    },
  },
  typography: {
    fontFamily: [
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontWeight: 700,
      letterSpacing: '-0.02em',
    },
    h2: {
      fontWeight: 700,
      letterSpacing: '-0.01em',
    },
    h3: {
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(0, 0, 0, 0.4)',
    '0px 4px 8px rgba(0, 0, 0, 0.4)',
    '0px 8px 16px rgba(0, 0, 0, 0.4)',
    '0px 12px 24px rgba(0, 0, 0, 0.5)',
    '0px 16px 32px rgba(0, 0, 0, 0.5)',
    '0px 20px 40px rgba(0, 0, 0, 0.5)',
    '0px 24px 48px rgba(0, 0, 0, 0.6)',
    '0px 2px 4px rgba(102, 126, 234, 0.3)',
    '0px 4px 8px rgba(102, 126, 234, 0.3)',
    '0px 8px 16px rgba(102, 126, 234, 0.3)',
    '0px 12px 24px rgba(102, 126, 234, 0.4)',
    '0px 16px 32px rgba(102, 126, 234, 0.4)',
    '0px 20px 40px rgba(102, 126, 234, 0.4)',
    '0px 24px 48px rgba(102, 126, 234, 0.5)',
    '0px 2px 10px rgba(102, 126, 234, 0.5)',
    '0px 4px 20px rgba(102, 126, 234, 0.5)',
    '0px 8px 30px rgba(102, 126, 234, 0.5)',
    '0px 12px 40px rgba(102, 126, 234, 0.6)',
    '0px 16px 50px rgba(102, 126, 234, 0.6)',
    '0px 20px 60px rgba(102, 126, 234, 0.6)',
    '0px 24px 70px rgba(102, 126, 234, 0.7)',
    '0px 28px 80px rgba(102, 126, 234, 0.7)',
    '0px 32px 90px rgba(102, 126, 234, 0.7)',
    '0px 36px 100px rgba(102, 126, 234, 0.8)',
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
          padding: '10px 24px',
        },
        contained: {
          boxShadow: '0 4px 14px 0 rgba(102, 126, 234, 0.39)',
          '&:hover': {
            boxShadow: '0 6px 20px rgba(102, 126, 234, 0.5)',
            transform: 'translateY(-1px)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderRadius: 16,
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <Router>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
              <Route index element={<Dashboard />} />
              <Route path="cv-analyzer" element={<CVAnalyzer />} />
              <Route path="career-advisor" element={<CareerAdvisor />} />
              <Route path="rag-coach" element={<RAGCoach />} />
              <Route path="history" element={<History />} />
              <Route path="admin" element={<AdminDashboard />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
