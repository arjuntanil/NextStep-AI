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
    
    // For multipart/form-data, delete Content-Type to let browser set it
    if (config.headers['Content-Type'] === undefined) {
      delete config.headers['Content-Type'];
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Handle responses and errors
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
      headers: { 'Content-Type': undefined },  // Let browser set boundary
    }),
  analyzeResumeWithJD: (formData) =>
    api.post('/analyze_resume_with_jd/', formData, {
      headers: { 'Content-Type': undefined },  // Let browser set boundary
    }),
};

export const careerAPI = {
  queryCareerPath: (question) => 
    api.post('/query-career-path/', { text: question }),
  
  getCareerAdvice: (question, maxLength = 200, temperature = 0.7) => 
    api.post('/career-advice-ai', { 
      text: question,
      max_length: maxLength,
      temperature: temperature
    }),
  
  getModelStatus: () => 
    api.get('/model-status'),
};

export const ragAPI = {
  uploadDocument: (formData) => 
    api.post('/rag-coach/upload', formData, {
      headers: { 'Content-Type': undefined },  // Let browser set boundary
    }),
  
  uploadDocuments: (formData) => {
    // Direct axios call to bypass potential interceptor issues
    const token = localStorage.getItem('token');
    return axios.post(`${API_BASE_URL}/rag-coach/upload`, formData, {
      headers: {
        ...(token && { 'Authorization': `Bearer ${token}` }),
        // Don't set Content-Type - let browser handle it
      },
    });
  },
  
  getProcessedResult: () => 
    api.get('/rag-coach/processed-result'),
  
  query: (question) => 
    api.post('/rag-coach/query', { question }),
  
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

export const adminAPI = {
  login: (email, password) =>
    api.post('/admin/login', { email, password }),
  
  getStats: () =>
    api.get('/admin/stats'),
  
  getUsers: (params) =>
    api.get('/admin/users', { params }),
  
  getUser: (userId) =>
    api.get(`/admin/user/${userId}`),
  
  suspendUser: (userId) =>
    api.put(`/admin/user/${userId}/suspend`),
  
  activateUser: (userId) =>
    api.put(`/admin/user/${userId}/activate`),
  
  deleteUser: (userId) =>
    api.delete(`/admin/user/${userId}`),
  
  createUser: (userData) =>
    api.post('/admin/user/create', userData),
};

export default api;