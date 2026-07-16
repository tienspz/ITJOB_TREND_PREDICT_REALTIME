/**
 * IT Job Market Intelligence - Frontend Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = window.API_BASE || '';
    const KNOWN_SKILL_GROUPS = {
        prog: ['Python', 'Java', 'JavaScript', 'C++', 'SQL'],
        cloud: ['AWS', 'Azure', 'Docker', 'Kubernetes', 'CI/CD'],
        ai: ['Machine Learning', 'PyTorch', 'TensorFlow', 'AI', 'Data Science'],
        db: ['PostgreSQL', 'MongoDB', 'MySQL', 'Oracle', 'Redis'],
        devops: ['Terraform', 'Ansible', 'CI/CD', 'GitHub Actions', 'Jenkins'],
        framework: ['React', 'Node.js', 'Spring', 'Vue.js', 'Angular'],
        dataeng: ['Spark', 'Airflow', 'DBT', 'Hadoop', 'Kafka'],
        sec: ['Cybersecurity', 'Penetration Testing', 'Cloud Security', 'IAM', 'Compliance'],
    };

    const modelMetrics = {
        r2: document.getElementById('model-r2'),
        mae: document.getElementById('model-mae'),
        cvR2: document.getElementById('cv-r2'),
        cvMae: document.getElementById('cv-mae'),
    };

    const statCards = {
        salary: document.getElementById('stat-salary'),
        jobs: document.getElementById('stat-jobs'),
        domain: document.getElementById('stat-domain'),
        skill: document.getElementById('stat-skill'),
    };

    async function fetchMeta() {
        try {
            const res = await fetch(`${API_BASE}/api/meta`);
            if (!res.ok) {
                console.warn('Failed to load metadata:', res.status);
                return null;
            }
            return await res.json();
        } catch (err) {
            console.error('Meta fetch error:', err);
            return null;
        }
    }

    function populateSelect(selectId, options, defaultValue) {
        const select = document.getElementById(selectId);
        if (!select || !Array.isArray(options)) {
            return;
        }
        select.innerHTML = options.map(value => `\n            <option value="${value}"${value === defaultValue ? ' selected' : ''}>${value}</option>`).join('');
    }

    function setText(id, value) {
        const el = document.getElementById(id);
        if (el) {
            el.innerText = value != null ? value : '—';
        }
    }

    function buildSkillBadges() {
        const badge = (category, skill) => `<span class="badge rounded-pill skill-badge" data-category="${category}">${skill}</span>`;

        // The salary form has 4 visual groups; each badge keeps the real
        // model category (prog/cloud/ai/db/devops/framework/dataeng/sec)
        // that maps onto the hidden pred-* inputs.
        const groupLayout = {
            'group-prog': [['prog', ['Python', 'Java', 'JavaScript', 'C++', 'SQL']]],
            'group-cloud': [['cloud', ['AWS', 'Azure']], ['devops', ['Docker', 'Kubernetes', 'CI/CD']]],
            'group-data': [['ai', ['Machine Learning', 'PyTorch']], ['db', ['MongoDB', 'PostgreSQL']], ['dataeng', ['Spark']]],
            'group-other': [['framework', ['React', 'Node.js', 'Spring']], ['sec', ['Cybersecurity']]],
        };
        Object.entries(groupLayout).forEach(([groupId, categories]) => {
            const group = document.getElementById(groupId);
            if (!group) return;
            group.innerHTML = categories.map(([cat, skills]) => skills.map(s => badge(cat, s)).join('')).join('');
        });

        const clSkills = document.getElementById('cl-skills');
        if (clSkills) {
            const clusterSkills = [
                ['prog', ['Python', 'Java', 'SQL']],
                ['cloud', ['AWS', 'Azure']],
                ['devops', ['Docker', 'Kubernetes']],
                ['ai', ['Machine Learning', 'PyTorch']],
                ['db', ['MongoDB']],
                ['framework', ['React']],
                ['sec', ['Cybersecurity']],
            ];
            clSkills.innerHTML = clusterSkills.map(([cat, skills]) => skills.map(s => badge(cat, s)).join('')).join('');
        }
    }

    function setModelMetaValues(meta) {
        if (!meta) return;
        const salaryMeta = meta.salary_meta || {};
        // data_rows comes from /api/meta; fall back to train+test split sizes
        const dataRows = salaryMeta.data_rows
            || ((salaryMeta.train_size || 0) + (salaryMeta.test_size || 0));
        setText('stat-salary', salaryMeta.mean_salary ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(salaryMeta.mean_salary) : '$0');
        setText('stat-jobs', dataRows ? dataRows.toLocaleString() : '—');
        setText('stat-domain', salaryMeta.top_domain || salaryMeta.it_domain?.[0] || '—');
        setText('stat-skill', salaryMeta.top_skill || '—');

        if (modelMetrics.r2) {
            modelMetrics.r2.innerText = `R² = ${salaryMeta.r2_score != null ? salaryMeta.r2_score.toFixed(3) : '-'}`;
        }
        if (modelMetrics.mae) {
            modelMetrics.mae.innerText = `MAE = ${salaryMeta.mae != null ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(salaryMeta.mae) : '$0'}`;
        }
        if (modelMetrics.cvR2) {
            modelMetrics.cvR2.innerText = modelMetrics.r2 ? modelMetrics.r2.innerText : `R² = ${salaryMeta.r2_score != null ? salaryMeta.r2_score.toFixed(3) : '-'}`;
        }
        if (modelMetrics.cvMae) {
            modelMetrics.cvMae.innerText = modelMetrics.mae ? modelMetrics.mae.innerText : `MAE = ${salaryMeta.mae != null ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(salaryMeta.mae) : '$0'}`;
        }
    }

    const FALLBACK_DOMAINS = ['Software Engineering', 'Data Science', 'DevOps/SRE', 'Cybersecurity', 'QA/Testing'];
    const FALLBACK_STATES = ['CA', 'TX', 'NY', 'WA', 'FL', 'Remote'];

    async function initMeta() {
        const meta = await fetchMeta();
        if (!meta) {
            buildSkillBadges();
            populateSelect('hist-domain', FALLBACK_DOMAINS, FALLBACK_DOMAINS[0]);
            populateSelect('hist-state', FALLBACK_STATES, FALLBACK_STATES[0]);
            return;
        }
        window._meta = meta;

        const salaryMeta = meta.salary_meta || {};
        const demandMeta = meta.demand_meta || {};
        const clusterMeta = meta.cluster_meta || {};

        populateSelect('pred-domain', salaryMeta.it_domain || [], salaryMeta.it_domain?.[0]);
        populateSelect('pred-seniority', salaryMeta.seniority_level || [], 'Mid');
        populateSelect('pred-job-type', salaryMeta.job_type || [], salaryMeta.job_type?.[0]);
        populateSelect('pred-state', salaryMeta.state || [], 'CA');

        populateSelect('dem-domain', demandMeta.it_domain || salaryMeta.it_domain || [], demandMeta.it_domain?.[0] || salaryMeta.it_domain?.[0]);
        populateSelect('dem-state', demandMeta.state || salaryMeta.state || [], demandMeta.state?.[0] || salaryMeta.state?.[0]);
        populateSelect('dem-seniority', demandMeta.seniority_level || salaryMeta.seniority_level || [], 'Mid');
        populateSelect('dem-job-type', demandMeta.job_type || salaryMeta.job_type || [], demandMeta.job_type?.[0] || salaryMeta.job_type?.[0]);

        populateSelect('cl-domain', clusterMeta.it_domain || salaryMeta.it_domain || [], clusterMeta.it_domain?.[0] || salaryMeta.it_domain?.[0]);
        populateSelect('cl-seniority', clusterMeta.seniority_level || salaryMeta.seniority_level || [], 'Mid');
        populateSelect('cl-job-type', clusterMeta.job_type || salaryMeta.job_type || [], clusterMeta.job_type?.[0] || salaryMeta.job_type?.[0]);
        populateSelect('cl-state', clusterMeta.state || salaryMeta.state || [], 'CA');

        populateSelect('hist-domain', salaryMeta.it_domain || FALLBACK_DOMAINS, salaryMeta.it_domain?.[0] || FALLBACK_DOMAINS[0]);
        populateSelect('hist-state', salaryMeta.state || FALLBACK_STATES, 'CA');

        setModelMetaValues(meta);
        buildSkillBadges();
    }

    // -----------------------------------------------------------------------
    // Trend Forecasting Logic (Kaggle + Realtime API)
    // -----------------------------------------------------------------------
    const btnFetchTrend = document.getElementById('btn-fetch-trend');
    let trendChartInstance = null;

    initMeta();

    if (btnFetchTrend) {
        btnFetchTrend.addEventListener('click', async () => {
            const loadingDiv = document.getElementById('trend-loading');
            const canvas = document.getElementById('trendChart');
            const statusBadge = document.getElementById('trend-status');
            
            btnFetchTrend.disabled = true;
            canvas.style.display = 'none';
            loadingDiv.classList.remove('d-none');
            
            try {
                const response = await fetch(`${API_BASE}/api/realtime-trends`);
                const data = await response.json();
                
                if (data.status === 'success') {
                    // Update status badge
                    if (data.source === 'rapidapi' || data.source === 'kaggle+realtime') {
                        statusBadge.className = 'badge bg-success';
                        statusBadge.innerText = 'Nguồn: Kaggle + API Realtime (Live)';
                    } else if (data.source === 'backup_csv') {
                        statusBadge.className = 'badge bg-warning text-dark';
                        statusBadge.innerText = 'Nguồn: Fallback CSV';
                    } else {
                        statusBadge.className = 'badge bg-warning text-dark';
                        statusBadge.innerText = 'Nguồn: Fallback CSV';
                    }
                    
                    const labels = [];
                    const actualData = [];
                    const forecastData = [];
                    
                    // Process historical data
                    data.historical.forEach(item => {
                        labels.push(item.month);
                        actualData.push(item.job_count);
                        forecastData.push(null); // No forecast for historical
                    });
                    
                    // Connect the lines
                    if (actualData.length > 0) {
                        forecastData[actualData.length - 1] = actualData[actualData.length - 1];
                    }
                    
                    // Process forecast data
                    data.forecast.forEach(item => {
                        labels.push(item.month);
                        actualData.push(null); // No actual data for future
                        forecastData.push(item.job_count);
                    });
                    
                    // Render Chart
                    if (trendChartInstance) {
                        trendChartInstance.destroy();
                    }
                    
                    const ctx = canvas.getContext('2d');
                    trendChartInstance = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [
                                {
                                    label: 'Số lượng Jobs (Thực tế)',
                                    data: actualData,
                                    borderColor: 'rgba(54, 162, 235, 1)',
                                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                    borderWidth: 2,
                                    fill: true,
                                    tension: 0.3
                                },
                                {
                                    label: 'Dự báo (Linear Regression)',
                                    data: forecastData,
                                    borderColor: 'rgba(255, 206, 86, 1)',
                                    borderWidth: 2,
                                    borderDash: [5, 5],
                                    fill: false,
                                    tension: 0.3
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'top',
                                    labels: { color: '#e0e0e0' }
                                },
                                tooltip: {
                                    mode: 'index',
                                    intersect: false,
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                    ticks: { color: '#e0e0e0' }
                                },
                                x: {
                                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                    ticks: { color: '#e0e0e0' }
                                }
                            }
                        }
                    });
                    
                    canvas.style.display = 'block';
                } else {
                    alert('Lỗi: ' + data.error);
                }
            } catch (err) {
                console.error(err);
                alert('Không thể tải dữ liệu xu hướng.');
            } finally {
                loadingDiv.classList.add('d-none');
                btnFetchTrend.disabled = false;
            }
        });
    }

    // -----------------------------------------------------------------------
    // Smart Skill Selector Logic
    // -----------------------------------------------------------------------

    // Category mapping counts. Badges are (re)built dynamically by
    // buildSkillBadges(), so use event delegation on the form.
    const categoryCounts = {
        prog: 0, cloud: 0, ai: 0, db: 0, devops: 0, framework: 0, dataeng: 0, sec: 0
    };

    const salaryFormEl = document.getElementById('salaryForm');
    if (salaryFormEl) {
        salaryFormEl.addEventListener('click', (e) => {
            const badge = e.target.closest('.skill-badge');
            if (!badge) return;
            badge.classList.toggle('active');
            const category = badge.getAttribute('data-category');
            if (!(category in categoryCounts)) return;

            categoryCounts[category] += badge.classList.contains('active') ? 1 : -1;
            document.getElementById('pred-' + category).value = categoryCounts[category];

            const totalSkills = Object.values(categoryCounts).reduce((a, b) => a + b, 0);
            document.getElementById('pred-num-skills').value = totalSkills;
        });
    }

    // -----------------------------------------------------------------------
    // Salary Predictor
    // -----------------------------------------------------------------------
    const salaryForm = document.getElementById('salaryForm');
    if (salaryForm) {
        salaryForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btn = document.getElementById('btn-predict');
            btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Đang dự đoán...';
            btn.disabled = true;
            
            // Collect features
            const features = {
                it_domain: document.getElementById('pred-domain').value,
                seniority_level: document.getElementById('pred-seniority').value,
                job_type: document.getElementById('pred-job-type').value,
                state: document.getElementById('pred-state').value,
                num_skills: parseInt(document.getElementById('pred-num-skills').value) || 0,
                
                skill_programming: parseInt(document.getElementById('pred-prog').value) || 0,
                skill_cloud: parseInt(document.getElementById('pred-cloud').value) || 0,
                skill_ai_ml: parseInt(document.getElementById('pred-ai').value) || 0,
                skill_database: parseInt(document.getElementById('pred-db').value) || 0,
                skill_devops: parseInt(document.getElementById('pred-devops').value) || 0,
                skill_framework: parseInt(document.getElementById('pred-framework').value) || 0,
                skill_data_engineering: parseInt(document.getElementById('pred-dataeng').value) || 0,
                skill_security: parseInt(document.getElementById('pred-sec').value) || 0,
            };
            
            // Calculate skill diversity (count of non-zero categories)
            let divCount = 0;
            const catKeys = ['pred-prog', 'pred-cloud', 'pred-ai', 'pred-db', 'pred-devops', 'pred-framework', 'pred-dataeng', 'pred-sec'];
            catKeys.forEach(id => {
                if (parseInt(document.getElementById(id).value) > 0) divCount++;
            });
            features.skill_diversity = divCount;
            features.skill_soft_skills = 1; // Default
            
            try {
                const response = await fetch(`${API_BASE}/api/predict_salary`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(features)
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    const resultDiv = document.getElementById('salary-result');
                    const valueEl = document.getElementById('salary-value');
                    
                    // Format number with commas
                    const formattedSalary = new Intl.NumberFormat('en-US', {
                        style: 'currency',
                        currency: 'USD',
                        maximumFractionDigits: 0
                    }).format(data.predicted_salary);
                    
                    valueEl.innerText = formattedSalary;
                    resultDiv.classList.remove('d-none');
                    resultDiv.classList.add('animate-fade-in');
                    
                    // Animate number counting up (optional polish)
                    animateValue(valueEl, 0, data.predicted_salary, 1000);
                } else {
                    alert('Lỗi: ' + data.error);
                }
            } catch (err) {
                console.error(err);
                alert('Không thể kết nối với API dự đoán.');
            } finally {
                btn.innerHTML = '<span>Dự Đoán Lương</span>';
                btn.disabled = false;
            }
        });
    }

    // -----------------------------------------------------------------------
    // CV Parser
    // -----------------------------------------------------------------------
    const cvInput = document.getElementById('cvFileInput');
    const uploadArea = document.getElementById('uploadArea');
    
    if (cvInput && uploadArea) {
        
        // Drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('dragover');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('dragover');
            }, false);
        });
        
        uploadArea.addEventListener('drop', (e) => {
            let dt = e.dataTransfer;
            let files = dt.files;
            if (files.length > 0) {
                cvInput.files = files;
                handleCVUpload(files[0]);
            }
        });
        
        cvInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                handleCVUpload(this.files[0]);
            }
        });
    }
    
    async function handleCVUpload(file) {
        const loadingDiv = document.getElementById('cv-loading');
        const resultDiv = document.getElementById('cv-result');
        const area = document.getElementById('uploadArea');
        
        area.classList.add('d-none');
        loadingDiv.classList.remove('d-none');
        resultDiv.classList.add('d-none');
        
        const formData = new FormData();
        formData.append('cv', file);
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout
        
        try {
            const response = await fetch(`${API_BASE}/api/upload_cv`, {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            const data = await response.json();
            
            if (response.ok) {
                // Populate results
                document.getElementById('cv-seniority').innerText = data.inferred_seniority;
                
                const badgesContainer = document.getElementById('cv-skills-badges');
                badgesContainer.innerHTML = '';
                
                if (data.skills_found.length > 0) {
                    // unique skills only
                    const uniqueSkills = [...new Set(data.skills_found)];
                    uniqueSkills.forEach(skill => {
                        const span = document.createElement('span');
                        span.className = 'skill-badge';
                        span.innerText = skill;
                        badgesContainer.appendChild(span);
                    });
                } else {
                    badgesContainer.innerHTML = '<span class="text-muted small">Không phát hiện kỹ năng IT cụ thể.</span>';
                }
                
                const formattedSalary = new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD',
                    maximumFractionDigits: 0
                }).format(data.predicted_salary);
                
                document.getElementById('cv-salary').innerText = formattedSalary;
                
                loadingDiv.classList.add('d-none');
                resultDiv.classList.remove('d-none');
                resultDiv.classList.add('animate-fade-in');
            } else {
                alert('Lỗi: ' + (data.error || 'Đã có lỗi xảy ra'));
                loadingDiv.classList.add('d-none');
                area.classList.remove('d-none');
            }
        } catch (err) {
            clearTimeout(timeoutId);
            console.error(err);
            if (err.name === 'AbortError') {
                alert('Hết thời gian xử lý (Timeout). Vui lòng thử lại.');
            } else {
                alert('Không thể xử lý CV.');
            }
            loadingDiv.classList.add('d-none');
            area.classList.remove('d-none');
        }
    }
    
    window.resetCVUpload = function() {
        document.getElementById('cv-result').classList.add('d-none');
        document.getElementById('uploadArea').classList.remove('d-none');
        document.getElementById('cvFileInput').value = '';
    };

    // -----------------------------------------------------------------------
    // Demand Forecasting
    // -----------------------------------------------------------------------
    const demandForm = document.getElementById('demandForm');
    if (demandForm) {
        demandForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btn = document.getElementById('btn-demand');
            if (btn) {
                btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Đang tính...';
                btn.disabled = true;
            }
            
            const features = {
                it_domain: document.getElementById('dem-domain').value,
                state: document.getElementById('dem-state').value,
                seniority_level: document.getElementById('dem-seniority').value,
                job_type: document.getElementById('dem-job-type') ? document.getElementById('dem-job-type').value : "Remote"
            };
            
            try {
                const response = await fetch(`${API_BASE}/api/predict_demand`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(features)
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    const resultBox = document.getElementById('demand-result-box');
                    const scoreVal = document.getElementById('demand-score-val');
                    const interp = document.getElementById('demand-interp');
                    const bar = document.getElementById('demand-bar');
                    
                    scoreVal.innerText = data.demand_score;
                    interp.innerText = "Nhu cầu " + (data.interpretation === 'High' ? 'Cao' : (data.interpretation === 'Medium' ? 'Trung bình' : 'Thấp'));
                    
                    // Colors based on interpretation
                    bar.className = 'progress-bar';
                    if (data.interpretation === 'High') {
                        bar.classList.add('bg-success');
                        interp.className = 'text-uppercase tracking-wide text-success fw-bold';
                    } else if (data.interpretation === 'Medium') {
                        bar.classList.add('bg-warning');
                        interp.className = 'text-uppercase tracking-wide text-warning fw-bold';
                    } else {
                        bar.classList.add('bg-danger');
                        interp.className = 'text-uppercase tracking-wide text-danger fw-bold';
                    }
                    
                    bar.style.width = data.demand_score + '%';
                    
                    resultBox.classList.remove('d-none');
                    resultBox.classList.add('animate-fade-in');
                } else {
                    alert('Lỗi: ' + (data.error || 'Đã có lỗi xảy ra'));
                }
            } catch (err) {
                console.error(err);
                alert('Không thể kết nối với API dự báo nhu cầu.');
            } finally {
                if (btn) {
                    btn.innerHTML = 'Tính Điểm Nhu Cầu';
                    btn.disabled = false;
                }
            }
        });
    }

    // Number animation utility
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            // Ease out cubic
            const easeProgress = 1 - Math.pow(1 - progress, 3);
            const current = Math.floor(easeProgress * (end - start) + start);

            obj.innerHTML = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                maximumFractionDigits: 0
            }).format(current);

            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD',
                    maximumFractionDigits: 0
                }).format(end);
            }
        };
        window.requestAnimationFrame(step);
    }

    // -----------------------------------------------------------------------
    // REAL-TIME MARKET TAB
    // -----------------------------------------------------------------------
    let rtSkillsChart = null;
    let rtCompareChart = null;
    let rtVelocityChart = null;

    const chartColors = {
        blue: 'rgba(54,162,235,1)', blueBg: 'rgba(54,162,235,0.3)',
        green: 'rgba(75,192,192,1)', greenBg: 'rgba(75,192,192,0.3)',
        orange: 'rgba(255,159,64,1)', orangeBg: 'rgba(255,159,64,0.3)',
        purple: 'rgba(153,102,255,1)', purpleBg: 'rgba(153,102,255,0.3)',
    };
    const gridColor = 'rgba(255,255,255,0.1)';
    const textColor = '#e0e0e0';

    function destroyRtCharts() {
        if (rtSkillsChart) { rtSkillsChart.destroy(); rtSkillsChart = null; }
        if (rtCompareChart) { rtCompareChart.destroy(); rtCompareChart = null; }
        if (rtVelocityChart) { rtVelocityChart.destroy(); rtVelocityChart = null; }
    }

    function renderRtCharts(data) {
        destroyRtCharts();

        // 1. Top Skills Bar Chart
        const skills = data.top_skills_7d || data.top_skills || [];
        const skillsCtx = document.getElementById('rtSkillsChart').getContext('2d');
        rtSkillsChart = new Chart(skillsCtx, {
            type: 'bar',
            data: {
                labels: skills.map(s => s.skill),
                datasets: [{
                    label: 'Tần suất (%)',
                    data: skills.map(s => s.share),
                    backgroundColor: chartColors.blueBg,
                    borderColor: chartColors.blue,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, grid: { color: gridColor }, ticks: { color: textColor } }, x: { grid: { color: gridColor }, ticks: { color: textColor } } },
                plugins: { legend: { labels: { color: textColor } } }
            }
        });

        // 2. Historical vs Realtime Comparison (grouped bar)
        const compRows = data.comparison.rows || [];
        const labels = compRows.map(r => r.skill.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()));
        rtCompareChart = new Chart(document.getElementById('rtCompareChart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Historical (Kaggle)', data: compRows.map(r => r.historical_share), backgroundColor: chartColors.purpleBg, borderColor: chartColors.purple, borderWidth: 1 },
                    { label: 'Real-time', data: compRows.map(r => r.realtime_share), backgroundColor: chartColors.greenBg, borderColor: chartColors.green, borderWidth: 1 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                scales: { x: { beginAtZero: true, grid: { color: gridColor }, ticks: { color: textColor } }, y: { grid: { color: gridColor }, ticks: { color: textColor } } },
                plugins: { legend: { labels: { color: textColor } } }
            }
        });

        // 3. Hiring Velocity Line Chart
        const velocity = data.hiring_velocity || [];
        rtVelocityChart = new Chart(document.getElementById('rtVelocityChart').getContext('2d'), {
            type: 'line',
            data: {
                labels: velocity.map(v => v.date.slice(5)),
                datasets: [{
                    label: 'Việc làm / ngày',
                    data: velocity.map(v => v.count),
                    borderColor: chartColors.orange,
                    backgroundColor: chartColors.orangeBg,
                    fill: true, tension: 0.4, borderWidth: 2
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, grid: { color: gridColor }, ticks: { color: textColor } }, x: { grid: { color: gridColor }, ticks: { color: textColor } } },
                plugins: { legend: { labels: { color: textColor } } }
            }
        });
    }

    function populateRtStatCards(data) {
        document.getElementById('rt-total-jobs').innerText = data.total_jobs;

        const topSkill = (data.top_skills_7d || data.top_skills || [])[0];
        if (topSkill) {
            document.getElementById('rt-top-skill').innerText = topSkill.skill;
            document.getElementById('rt-top-skill-share').innerText = topSkill.share + '% tần suất';
        }

        const topLoc = (data.top_locations || [])[0];
        if (topLoc) {
            document.getElementById('rt-top-location').innerText = topLoc.state;
            document.getElementById('rt-top-location-share').innerText = topLoc.share + '% việc làm';
        }

        const topCo = (data.top_companies || [])[0];
        if (topCo) {
            document.getElementById('rt-top-company').innerText = topCo.company;
            document.getElementById('rt-top-company-share').innerText = topCo.share + '% việc làm';
        }

        // Source badge
        const badge = document.getElementById('rt-source-badge');
        const note = document.getElementById('rt-source-note');
        if (data.source === 'live') {
            badge.className = 'badge bg-success'; badge.innerText = 'Live';
        } else if (data.source === 'cache') {
            badge.className = 'badge bg-info text-dark'; badge.innerText = 'Cached';
        } else {
            badge.className = 'badge bg-warning text-dark'; badge.innerText = 'Snapshot (Demo)';
        }
        note.innerText = data.meta?.note || '';
    }

    function populateRtJobs(data) {
        const tbody = document.getElementById('rt-jobs-tbody');
        const jobs = data.jobs || [];
        document.getElementById('rt-jobs-count').innerText = jobs.length;

        if (!jobs.length) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">Không có dữ liệu</td></tr>';
            return;
        }
        tbody.innerHTML = jobs.slice(0, 20).map(j => `
            <tr>
                <td><strong>${j.title || '-'}</strong></td>
                <td>${j.company || '-'}</td>
                <td>${j.location || '-'}</td>
                <td>${(j.skills || []).map(s => '<span class="skill-badge me-1">' + s + '</span>').join('') || '<span class="text-muted">—</span>'}</td>
                <td><span class="badge bg-secondary">${j.experience || '-'}</span></td>
            </tr>
        `).join('');
    }

    function populateRtRoles(data) {
        const el = document.getElementById('rt-roles-list');
        const roles = data.emerging_roles || data.top_roles || [];
        if (!roles.length) { el.innerHTML = '<p class="text-muted">-</p>'; return; }
        el.innerHTML = roles.slice(0, 6).map(r => `
            <div class="d-flex justify-content-between align-items-center mb-2">
                <span>${r.title || '-'}</span>
                <span class="badge bg-primary">${r.count || 0}</span>
            </div>
        `).join('');
    }

    async function fetchRealtimeData() {
        const btn = document.getElementById('btn-fetch-realtime');
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Đang tải...';

        try {
            const resp = await fetch(`${API_BASE}/api/realtime/report`);
            const data = await resp.json();

            if (!resp.ok || data.error) { alert(data.error || 'Lỗi tải dữ liệu realtime'); return; }

            populateRtStatCards(data);
            renderRtCharts(data);
            populateRtJobs(data);
            populateRtRoles(data);

            if (data.comparison?.conclusion) {
                document.getElementById('rt-conclusion').innerText = data.comparison.conclusion;
            }
        } catch (err) {
            console.error(err);
            alert('Không thể tải dữ liệu realtime.');
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-arrow-clockwise"></i> Làm mới dữ liệu';
        }
    }

    // Wire the refresh button
    const btnFetchRt = document.getElementById('btn-fetch-realtime');
    if (btnFetchRt) btnFetchRt.addEventListener('click', fetchRealtimeData);

    // Auto-load when user switches to the realtime tab
    const realtimeTab = document.getElementById('realtime-tab');
    if (realtimeTab) {
        realtimeTab.addEventListener('shown.bs.tab', function () {
            if (!document.getElementById('rt-total-jobs').innerText || document.getElementById('rt-total-jobs').innerText === '—') {
                fetchRealtimeData();
            }
        });
    }

    // -----------------------------------------------------------------------
    // Cluster Skill Selector (separate counter from salary form)
    // -----------------------------------------------------------------------
    // Delegated like the salary form: badges are rebuilt by buildSkillBadges()
    const clCatCounts = { prog: 0, cloud: 0, ai: 0, db: 0, devops: 0, framework: 0, dataeng: 0, sec: 0 };
    const clSkillsContainer = document.getElementById('cl-skills');
    if (clSkillsContainer) {
        clSkillsContainer.addEventListener('click', (e) => {
            const badge = e.target.closest('.skill-badge');
            if (!badge) return;
            badge.classList.toggle('active');
            const cat = badge.getAttribute('data-category');
            if (!(cat in clCatCounts)) return;
            clCatCounts[cat] += badge.classList.contains('active') ? 1 : -1;
            const el = document.getElementById('cl-' + cat);
            if (el) el.value = clCatCounts[cat];
        });
    }

    // -----------------------------------------------------------------------
    // Cluster Classification
    // -----------------------------------------------------------------------
    const clusterForm = document.getElementById('clusterForm');
    if (clusterForm) {
        clusterForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('btn-cluster');
            btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Đang phân loại...';
            btn.disabled = true;

            const catIds = ['cl-prog', 'cl-cloud', 'cl-ai', 'cl-db', 'cl-devops', 'cl-framework', 'cl-dataeng', 'cl-sec'];
            const features = {
                it_domain: document.getElementById('cl-domain').value,
                seniority_level: document.getElementById('cl-seniority').value,
                job_type: document.getElementById('cl-job-type').value,
                state: document.getElementById('cl-state').value,
                num_skills: 0,
                skill_programming: parseInt(document.getElementById('cl-prog').value) || 0,
                skill_cloud: parseInt(document.getElementById('cl-cloud').value) || 0,
                skill_ai_ml: parseInt(document.getElementById('cl-ai').value) || 0,
                skill_database: parseInt(document.getElementById('cl-db').value) || 0,
                skill_devops: parseInt(document.getElementById('cl-devops').value) || 0,
                skill_framework: parseInt(document.getElementById('cl-framework').value) || 0,
                skill_data_engineering: parseInt(document.getElementById('cl-dataeng').value) || 0,
                skill_security: parseInt(document.getElementById('cl-sec').value) || 0,
            };
            features.num_skills = catIds.reduce((sum, id) => sum + (parseInt(document.getElementById(id).value) || 0), 0);

            try {
                const resp = await fetch(`${API_BASE}/api/cluster`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(features)
                });
                const data = await resp.json();
                if (resp.ok) {
                    const box = document.getElementById('cluster-result-box');
                    document.getElementById('cluster-id-val').innerText = 'Cụm ' + data.cluster_id;
                    document.getElementById('cluster-desc').innerText = data.description;
                    box.classList.remove('d-none');
                    box.classList.add('animate-fade-in');
                } else {
                    alert('Lỗi: ' + (data.error || 'Đã có lỗi xảy ra'));
                }
            } catch (err) {
                console.error(err);
                alert('Không thể kết nối với API phân cụm.');
            } finally {
                btn.innerHTML = 'Phân loại';
                btn.disabled = false;
            }
        });
    }

    // -----------------------------------------------------------------------
    // Skills Comparison
    // -----------------------------------------------------------------------
    const skillsForm = document.getElementById('skillsForm');
    if (skillsForm) {
        skillsForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('btn-skills');
            btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Đang phân tích...';
            btn.disabled = true;
            const input = document.getElementById('skills-input').value;
            const skills = input.split(',').map(s => s.trim()).filter(Boolean);
            try {
                const resp = await fetch(`${API_BASE}/api/compare_skills`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ skills })
                });
                const data = await resp.json();
                if (resp.ok) {
                    document.getElementById('skills-market-value').innerText =
                        new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(data.market_value);
                    document.getElementById('skills-percentile').innerText = `Top ${100 - data.percentile}%`;
                    document.getElementById('skills-percentile').className = `badge ${data.percentile > 60 ? 'bg-success' : data.percentile > 30 ? 'bg-warning text-dark' : 'bg-secondary'}`;
                    document.getElementById('skills-in-demand').innerText = data.in_demand ? 'Đang có nhu cầu cao' : 'Nhu cầu trung bình';
                    document.getElementById('skills-in-demand').className = `badge ms-1 ${data.in_demand ? 'bg-success' : 'bg-secondary'}`;
                    document.getElementById('skills-related-roles').innerHTML = data.related_roles.map(r => `<span class="badge bg-info me-1">${r}</span>`).join('') || '<p class="text-muted">—</p>';
                    document.getElementById('skills-gaps').innerHTML = data.skill_gaps.map(g => `<p class="text-warning small mb-1"><i class="bi bi-lightbulb"></i> ${g}</p>`).join('') || '<p class="text-success small">Không thiếu kỹ năng!</p>';
                    document.getElementById('skills-result').classList.remove('d-none');
                    document.getElementById('skills-result').classList.add('animate-fade-in');
                } else {
                    alert('Lỗi: ' + (data.error || 'Đã có lỗi'));
                }
            } catch (err) {
                console.error(err);
                alert('Không thể kết nối API so sánh kỹ năng.');
            } finally {
                btn.innerHTML = '<span>Phân Tích Kỹ Năng</span>';
                btn.disabled = false;
            }
        });
    }

    // -----------------------------------------------------------------------
    // Historical Trends
    // -----------------------------------------------------------------------
    const historyForm = document.getElementById('historyForm');
    let historyChartInstance = null;
    if (historyForm) {
        historyForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('btn-history');
            btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Đang tải...';
            btn.disabled = true;
            try {
                const resp = await fetch(`${API_BASE}/api/historical_trends`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        domain: document.getElementById('hist-domain').value,
                        state: document.getElementById('hist-state').value,
                        months: parseInt(document.getElementById('hist-months').value),
                    })
                });
                const data = await resp.json();
                if (resp.ok) {
                    document.getElementById('history-chart-container').classList.remove('d-none');
                    document.getElementById('history-growth').innerText = `Tốc độ tăng trưởng: ${data.growth_percentage > 0 ? '+' : ''}${data.growth_percentage.toFixed(1)}%/tháng`;
                    if (historyChartInstance) historyChartInstance.destroy();
                    const ctx = document.getElementById('historyChart').getContext('2d');
                    historyChartInstance = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.labels,
                            datasets: [
                                {
                                    label: 'Lương trung bình ($)',
                                    data: data.salary_trend,
                                    borderColor: '#00d2ff',
                                    backgroundColor: 'rgba(0,210,255,0.1)',
                                    yAxisID: 'y',
                                    tension: 0.3,
                                    fill: true,
                                },
                                {
                                    label: 'Số lượng việc làm',
                                    data: data.demand_trend,
                                    borderColor: '#ffc107',
                                    backgroundColor: 'rgba(255,193,7,0.1)',
                                    yAxisID: 'y1',
                                    tension: 0.3,
                                    fill: true,
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            interaction: { mode: 'index', intersect: false },
                            plugins: {
                                legend: { labels: { color: '#e0e0e0' } }
                            },
                            scales: {
                                y: {
                                    type: 'linear',
                                    display: true,
                                    position: 'left',
                                    grid: { color: 'rgba(255,255,255,0.05)' },
                                    ticks: { color: '#e0e0e0', callback: v => '$' + v.toLocaleString() }
                                },
                                y1: {
                                    type: 'linear',
                                    display: true,
                                    position: 'right',
                                    grid: { drawOnChartArea: false },
                                    ticks: { color: '#e0e0e0' }
                                },
                                x: {
                                    grid: { color: 'rgba(255,255,255,0.05)' },
                                    ticks: { color: '#e0e0e0' }
                                }
                            }
                        }
                    });
                } else {
                    alert('Lỗi: ' + (data.error || 'Không có dữ liệu'));
                }
            } catch (err) {
                console.error(err);
                alert('Không thể tải xu hướng lịch sử.');
            } finally {
                btn.innerHTML = '<span>Xem Xu Hướng</span>';
                btn.disabled = false;
            }
        });
    }
});
