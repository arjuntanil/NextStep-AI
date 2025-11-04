import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import Aurora from './Aurora';
import {
  AppBar,
  Box,
  Toolbar,
  Typography,
  Button,
  Container,
  Menu,
  MenuItem,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Person as PersonIcon,
  Logout as LogoutIcon,
} from '@mui/icons-material';

const Layout = () => {
  const [featuresAnchor, setFeaturesAnchor] = useState(null);
  const [userAnchor, setUserAnchor] = useState(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const { user, logout, isAdmin } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleFeaturesClick = (event) => {
    setFeaturesAnchor(event.currentTarget);
  };

  const handleFeaturesClose = () => {
    setFeaturesAnchor(null);
  };

  const handleUserClick = (event) => {
    setUserAnchor(event.currentTarget);
  };

  const handleUserClose = () => {
    setUserAnchor(null);
  };

  const handleLogout = () => {
    handleUserClose();
    logout();
    navigate('/login');
  };

  const handleMobileToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const navigateToPage = (path) => {
    navigate(path);
    handleFeaturesClose();
    setMobileOpen(false);
  };

  const features = [
    { name: 'CV Analyzer', path: '/cv-analyzer' },
    { name: 'Resume Analyzer with JD', path: '/resume-analyzer' },
    { name: 'Career Advisor', path: '/career-advisor' },
    { name: 'RAG Coach', path: '/rag-coach' },
    { name: 'History', path: '/history' },
  ];

  if (isAdmin()) {
    features.push({ name: 'Admin Panel', path: '/admin' });
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative' }}>
      {/* Aurora Background */}
      <Box sx={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: -1, bgcolor: '#0a0e27' }}>
        <Aurora
          colorStops={["#8b5cf6", "#3b82f6", "#10b981"]}
          blend={0.8}
          amplitude={1.5}
          speed={0.4}
        />
      </Box>
      
      {/* Header */}
      <AppBar 
        position="sticky" 
        elevation={0}
        sx={{ 
          bgcolor: 'transparent',
          border: 'none',
          boxShadow: 'none',
          py: 2,
        }}
      >
        <Container maxWidth="lg">
          <Toolbar disableGutters sx={{ justifyContent: 'space-between', p: 1.25, borderRadius: 3, backdropFilter: 'blur(20px)', bgcolor: 'rgba(10, 14, 39, 0.8)', border: '1px solid rgba(139, 92, 246, 0.2)' }}>
            {/* Logo */}
            <Box 
              onClick={() => navigate('/')}
              sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                cursor: 'pointer',
                '&:hover': { opacity: 0.8 },
              }}
            >
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 800,
                  color: '#FFFFFF',
                  letterSpacing: '-0.5px',
                  fontFamily: 'Space Grotesk',
                }}
              >
                NextStepAI
              </Typography>
            </Box>

            {/* Desktop Navigation */}
            <Box sx={{ display: { xs: 'none', md: 'flex' }, alignItems: 'center', gap: 1 }}>
              <Button
                onClick={() => navigate('/')}
                sx={{
                  color: location.pathname === '/' ? '#8b5cf6' : 'rgba(255, 255, 255, 0.9)',
                  fontWeight: 600,
                  px: 2,
                }}
              >
                Home
              </Button>
              <Button
                onClick={handleFeaturesClick}
                sx={{
                  color: 'rgba(255, 255, 255, 0.9)',
                  fontWeight: 600,
                  px: 2,
                }}
              >
                Features
              </Button>
              <Menu
                anchorEl={featuresAnchor}
                open={Boolean(featuresAnchor)}
                onClose={handleFeaturesClose}
                PaperProps={{
                  sx: {
                    mt: 1,
                    minWidth: 200,
                    boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                  },
                }}
              >
                {features.map((feature) => (
                  <MenuItem
                    key={feature.path}
                    onClick={() => navigateToPage(feature.path)}
                    selected={location.pathname === feature.path}
                    sx={{
                      py: 1.5,
                      fontWeight: 500,
                      '&.Mui-selected': {
                        bgcolor: 'primary.light',
                        color: 'white',
                        '&:hover': {
                          bgcolor: 'primary.main',
                        },
                      },
                    }}
                  >
                    {feature.name}
                  </MenuItem>
                ))}
              </Menu>
              <Button
                onClick={() => navigate('/about')}
                sx={{
                  color: location.pathname === '/about' ? '#10b981' : 'rgba(255, 255, 255, 0.9)',
                  fontWeight: 600,
                  px: 2,
                }}
              >
                About
              </Button>

              {/* User Menu */}
              <Button
                onClick={handleUserClick}
                startIcon={<PersonIcon />}
                variant="outlined"
                sx={{
                  ml: 2,
                  borderWidth: 2,
                  borderColor: 'rgba(255, 255, 255, 0.3)',
                  color: '#FFFFFF',
                  '&:hover': {
                    borderWidth: 2,
                    borderColor: 'rgba(255, 255, 255, 0.6)',
                    bgcolor: 'rgba(255, 255, 255, 0.1)',
                  },
                }}
              >
                {user?.email?.split('@')[0] || 'User'}
              </Button>
              <Menu
                anchorEl={userAnchor}
                open={Boolean(userAnchor)}
                onClose={handleUserClose}
                PaperProps={{
                  sx: {
                    mt: 1,
                    minWidth: 180,
                    boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                  },
                }}
              >
                <MenuItem disabled sx={{ opacity: 1, fontWeight: 600 }}>
                  {user?.email}
                </MenuItem>
                <MenuItem disabled sx={{ opacity: 1 }}>
                  Role: {user?.role || 'user'}
                </MenuItem>
                <MenuItem onClick={handleLogout} sx={{ color: 'error.main', fontWeight: 600 }}>
                  <LogoutIcon sx={{ mr: 1, fontSize: 20 }} />
                  Logout
                </MenuItem>
              </Menu>
            </Box>

            {/* Mobile Menu Icon */}
            <IconButton
              sx={{ display: { xs: 'flex', md: 'none' } }}
              onClick={handleMobileToggle}
            >
              <MenuIcon />
            </IconButton>
          </Toolbar>
        </Container>
      </AppBar>

      {/* Mobile Drawer */}
      <Drawer
        anchor="right"
        open={mobileOpen}
        onClose={handleMobileToggle}
        PaperProps={{
          sx: { width: 280 },
        }}
      >
        <Box sx={{ p: 3 }}>
          <Typography variant="h6" sx={{ fontWeight: 700, mb: 3 }}>
            Menu
          </Typography>
          <List>
            <ListItem disablePadding>
              <ListItemButton onClick={() => navigateToPage('/')}>
                <ListItemText primary="Home" primaryTypographyProps={{ fontWeight: 600 }} />
              </ListItemButton>
            </ListItem>
            {features.map((feature) => (
              <ListItem key={feature.path} disablePadding>
                <ListItemButton onClick={() => navigateToPage(feature.path)}>
                  <ListItemText primary={feature.name} primaryTypographyProps={{ fontWeight: 600 }} />
                </ListItemButton>
              </ListItem>
            ))}
            <ListItem disablePadding>
              <ListItemButton onClick={() => navigateToPage('/about')}>
                <ListItemText primary="About" primaryTypographyProps={{ fontWeight: 600 }} />
              </ListItemButton>
            </ListItem>
            <ListItem disablePadding sx={{ mt: 2 }}>
              <ListItemButton onClick={handleLogout} sx={{ color: 'error.main' }}>
                <LogoutIcon sx={{ mr: 1 }} />
                <ListItemText primary="Logout" primaryTypographyProps={{ fontWeight: 600 }} />
              </ListItemButton>
            </ListItem>
          </List>
        </Box>
      </Drawer>

      {/* Main Content */}
      <Box 
        component="main" 
        sx={{ 
          flexGrow: 1, 
          py: 4,
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(10, 14, 39, 0.7)',
            backdropFilter: 'blur(10px)',
            zIndex: -1,
          },
        }}
      >
        <Container maxWidth="xl">
          <Outlet />
        </Container>
      </Box>

      {/* Footer */}
      <Box
        component="footer"
        sx={{
          py: 3,
          px: 2,
          mt: 'auto',
          bgcolor: 'rgba(10, 14, 39, 0.9)',
          backdropFilter: 'blur(10px)',
          borderTop: '1px solid',
          borderColor: 'rgba(139, 92, 246, 0.2)',
        }}
      >
        <Container maxWidth="xl">
          <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }} align="center">
            © {new Date().getFullYear()} NextStepAI. All rights reserved.
          </Typography>
        </Container>
      </Box>
    </Box>
  );
};

export default Layout;
