import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Handle 401 responses
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const authAPI = {
  register: (email, password, fullName) => 
    api.post('/auth/register', { email, password, full_name: fullName }),
  
  login: (email, password) => 
    api.post('/auth/manual-login', { email, password }),
  
  getMe: () => 
    api.get('/users/me'),
};

export const cvAPI = {
  analyzeResume: (formData) => 
    api.post('/analyze_resume/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
};

export const careerAPI = {
  queryCareerPath: (question) => 
    api.post('/query-career-path/', { question }),
  
  getCareerAdvice: (question) => 
    api.post('/career-advice-ai', { question }),
  
  getModelStatus: () => 
    api.get('/model-status'),
};

export const ragAPI = {
  uploadDocument: (formData) => 
    api.post('/rag-coach/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  
  query: (question, sessionId) => 
    api.post('/rag-coach/query', { question, session_id: sessionId }),
  
  getStatus: () => 
    api.get('/rag-coach/status'),
};

export const historyAPI = {
  getAnalyses: () => 
    api.get('/history/analyses'),
  
  getQueries: () => 
    api.get('/history/queries'),
  
  getRAGQueries: () => 
    api.get('/history/rag-queries'),
};

export default api;
