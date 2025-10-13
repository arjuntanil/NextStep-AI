# skills_db.py
# This is the master list of skills, expanded to be multi-domain.
SKILLS_DB = sorted(list(set([

    # --- IT: Programming & Scripting ---
    'python', 'javascript', 'java', 'c#', 'c++', 'typescript', 'php', 'swift', 'kotlin', 'go', 
    'rust', 'ruby', 'scala', 'r', 'sql', 'bash', 'powershell', 'perl', 'lua',

    # --- IT: Web Development ---
    'html', 'css', 'react', 'react.js', 'angular', 'vue.js', 'svelte', 'jquery', 'bootstrap', 'tailwind css', 
    'node.js', 'express.js', 'django', 'flask', 'fastapi', 'ruby on rails', 'asp.net', 'laravel', 'spring boot',
    'graphql', 'restful apis', 'rest', 'sass', 'less', 'webpack', 'babel',

    # --- IT: Databases & Data Stores ---
    'mysql', 'postgresql', 'sqlite', 'mongodb', 'redis', 'microsoft sql server', 'oracle', 'firebase', 
    'dynamodb', 'elasticsearch', 'cassandra', 'couchdb', 'neo4j',

    # --- IT: Mobile Development ---
    'ios development', 'android development', 'react native', 'flutter', 'swiftui', 'jetpack compose', 'xamarin', 'dart',

    # --- IT: Data Science, ML & AI ---
    'machine learning', 'deep learning', 'data science', 'natural language processing', 'nlp', 
    'computer vision', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
    'matplotlib', 'seaborn', 'plotly', 'jupyter', 'apache spark', 'spark', 'hadoop', 'hive', 'airflow', 
    'data mining', 'data analysis', 'statistical analysis', 'data modeling', 'etl',

    # --- IT: DevOps, Cloud & Infrastructure ---
    'aws', 'azure', 'google cloud platform', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 
    'gitlab', 'ansible', 'terraform', 'ci/cd', 'linux', 'ubuntu', 'centos', 'serverless', 'aws lambda',
    'network administration', 'system administration',

    # --- Business, Management & Operations ---
    'project management', 'product management', 'agile', 'scrum', 'kanban', 'pmp', 'lean manufacturing', 'six sigma',
    'business strategy', 'operations management', 'supply chain management', 'logistics', 'procurement', 'inventory management',
    'business development', 'market research', 'competitive analysis', 'stakeholder management', 'business analysis',
    'erp', 'crm', 'salesforce', 'sap',

    # --- Finance & Accounting ---
    'financial analysis', 'risk management', 'financial reporting', 'compliance', 'kyc', 'aml', 'anti-money laundering', 
    'credit analysis', 'portfolio management', 'investment banking', 'wealth management', 'underwriting', 
    'financial modeling', 'trade finance', 'treasury', 'auditing', 'taxation', 'bookkeeping', 'gaap', 'ifrs',
    'accounts payable', 'accounts receivable', 'general ledger', 'quickbooks', 'tally', 'sap fico',

    # --- Marketing & Sales ---
    'digital marketing', 'seo', 'sem', 'ppc', 'content marketing', 'email marketing', 'social media marketing',
    'google analytics', 'google ads', 'hubspot', 'market research', 'brand management', 'public relations', 'pr',
    'sales', 'lead generation', 'negotiation', 'client relationship management',

    # --- Design & Creative ---
    'ui design', 'ux design', 'ui/ux', 'user research', 'wireframing', 'prototyping', 'figma', 'sketch', 
    'adobe xd', 'invision', 'graphic design', 'illustration', 'typography', 'adobe creative suite', 
    'photoshop', 'illustrator', 'indesign', 'after effects', 'video editing', 'motion graphics',

    # --- Healthcare & Life Sciences ---
    'patient care', 'nursing', 'medical billing', 'medical coding', 'hipaa', 'electronic health records', 'ehr',
    'clinical trials', 'pharmacology', 'bioinformatics', 'genomics', 'lab techniques',

    # --- General Engineering (Non-Software) ---
    'cad', 'autocad', 'solidworks', 'catia', 'fea', 'mechanical engineering', 'electrical engineering',
    'civil engineering', 'circuit design', 'manufacturing',

    # --- Human Resources ---
    'talent acquisition', 'recruiting', 'sourcing', 'interviewing', 'onboarding', 'employee relations',
    'performance management', 'hris', 'compensation and benefits',

    # --- General Soft Skills ---
    'communication', 'teamwork', 'problem-solving', 'leadership', 'time management', 'critical thinking', 
    'adaptability', 'collaboration', 'mentoring', 'public speaking', 'technical writing'
])))