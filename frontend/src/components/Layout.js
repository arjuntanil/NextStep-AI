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
      <Box sx={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: -1, bgcolor: '#000000' }}>
        <Aurora
          colorStops={["#8b5cf6", "#6366f1", "#3b82f6", "#10b981"]}
          blend={0.8}
          amplitude={1.8}
          speed={0.5}
        />
      </Box>
      
      {/* Header */}
      <AppBar 
        position="sticky" 
        elevation={0}
        sx={{ 
          background: 'rgba(10, 14, 39, 0.6)',
          borderRadius: { xs: 0, md: '2rem' },
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(139, 92, 246, 0.2)',
          boxShadow: 'none',
          width: { xs: '100%', md: '90%' },
          mx: 'auto',
          mt: { xs: 0, md: 2 },
          left: '50%',
          transform: { xs: 'none', md: 'translateX(-50%)' },
        }}
      >
        <Container maxWidth="xl">
          <Toolbar disableGutters sx={{ justifyContent: 'space-between', px: { xs: 2, md: 4 } }}>
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
                  color: 'white',
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
                  color: location.pathname === '/' ? '#10b981' : 'rgba(255, 255, 255, 0.9)',
                  fontWeight: 600,
                  px: 2,
                  '&:hover': { color: '#10b981' },
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
                  '&:hover': { color: '#3b82f6' },
                }}
              >
                Services
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
                    bgcolor: 'rgba(10, 14, 39, 0.95)',
                    backdropFilter: 'blur(20px)',
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
                      color: 'white',
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
                  '&:hover': { color: '#10b981' },
                }}
              >
                About
              </Button>
              <Button
                sx={{
                  color: 'rgba(255, 255, 255, 0.9)',
                  fontWeight: 600,
                  px: 2,
                  '&:hover': { color: '#10b981' },
                }}
              >
                Contact
              </Button>

              {/* User Menu - Login Button on Right */}
              <Button
                onClick={handleUserClick}
                startIcon={<PersonIcon />}
                variant="outlined"
                sx={{
                  ml: 3,
                  borderWidth: 2,
                  borderColor: 'rgba(255, 255, 255, 0.3)',
                  color: 'rgba(255, 255, 255, 0.9)',
                  borderRadius: '12px',
                  '&:hover': {
                    borderWidth: 2,
                    borderColor: '#10b981',
                    bgcolor: 'rgba(16, 185, 129, 0.1)',
                    color: '#10b981',
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
                <MenuItem disabled sx={{ opacity: 1, fontWeight: 600, color: 'white' }}>
                  {user?.email}
                </MenuItem>
                <MenuItem disabled sx={{ opacity: 1, color: 'rgba(255, 255, 255, 0.7)' }}>
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
          bgcolor: 'rgba(10, 14, 39, 0.6)',
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
