import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
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
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Header */}
      <AppBar 
        position="sticky" 
        elevation={1}
        sx={{ 
          bgcolor: 'white', 
          borderBottom: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Container maxWidth="xl">
          <Toolbar disableGutters sx={{ justifyContent: 'space-between' }}>
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
                  background: 'linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
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
                  color: location.pathname === '/' ? 'primary.main' : 'text.primary',
                  fontWeight: 600,
                  px: 2,
                }}
              >
                Home
              </Button>
              <Button
                onClick={handleFeaturesClick}
                sx={{
                  color: 'text.primary',
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
                  color: location.pathname === '/about' ? 'primary.main' : 'text.primary',
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
                  '&:hover': {
                    borderWidth: 2,
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
      <Box component="main" sx={{ flexGrow: 1, bgcolor: 'background.default', py: 4 }}>
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
          bgcolor: '#f9fafb',
          borderTop: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Container maxWidth="xl">
          <Typography variant="body2" color="text.secondary" align="center">
            © {new Date().getFullYear()} NextStepAI. All rights reserved.
          </Typography>
        </Container>
      </Box>
    </Box>
  );
};

export default Layout;
