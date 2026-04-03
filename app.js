let rawData = null;
let aiModel = null;
let accChartInstance = null;
let rewChartInstance = null;

// Initialize charts globally
async function initDashboard() {
    try {
        const response = await fetch('results.json');
        if (!response.ok) throw new Error("File missing");
        rawData = await response.json();
        setupButtons();
        updateDashboard('easy');
        
        // Fetch AI Model for live demo
        const modelRes = await fetch('ai_model.json');
        if (modelRes.ok) {
            aiModel = await modelRes.json();
        }
    } catch (e) {
        console.error(e);
        document.getElementById('fetch-warning').style.display = 'block';
    }
}

function setupButtons() {
    const btns = document.querySelectorAll('.diff-btn');
    btns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            btns.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            updateDashboard(e.target.dataset.diff);
        });
    });
}

function updateDashboard(difficulty) {
    if (!rawData) return;
    const data = rawData[difficulty];
    
    const labels = data.map(d => d.agent);
    const accuracies = data.map(d => d.accuracy * 100);
    const rewards = data.map(d => d.avg_reward);

    const topAcc = Math.max(...accuracies).toFixed(1);
    const topRew = Math.max(...rewards).toFixed(2);
    document.getElementById('top-acc').innerText = topAcc + '%';
    document.getElementById('top-reward').innerText = topRew;

    const colors = [
        'rgba(148, 163, 184, 0.7)', // Random
        'rgba(244, 63, 94, 0.7)',   // Human Rule
        'rgba(56, 189, 248, 0.9)'   // Q-Learning (brighter)
    ];

    updateChart('accChart', accChartInstance, labels, accuracies, 'Accuracy (%)', colors, (newChart) => accChartInstance = newChart);
    updateChart('rewChart', rewChartInstance, labels, rewards, 'Avg Reward', colors, (newChart) => rewChartInstance = newChart);
}

function updateChart(canvasId, chartInstance, labels, dataPoints, title, colors, updateRef) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    if (chartInstance) {
        chartInstance.destroy();
    }

    const newChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: title,
                data: dataPoints,
                backgroundColor: colors,
                borderRadius: 8,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                title: { display: true, text: title, color: '#f8fafc', font: { size: 16, weight: 'bold' } }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { weight: 'bold' } }
                }
            },
            animation: { duration: 800, easing: 'easeOutQuart' }
        }
    });

    updateRef(newChart);
}

// LIVE AI EVALUATION
document.getElementById('btn-evaluate').addEventListener('click', () => {
    if (!aiModel) {
        alert("The AI model hasn't been generated yet! Run 'python main.py' locally first so it exports ai_model.json.");
        return;
    }
    
    let inc = parseInt(document.getElementById('demo-income').value);
    let cred = parseInt(document.getElementById('demo-credit').value);
    let debt = parseInt(document.getElementById('demo-debt').value);
    let loan = parseInt(document.getElementById('demo-loan').value);
    
    // Discretize state exactly like the python QLearningAgent
    let incBucket = inc < 50000 ? 0 : (inc < 80000 ? 1 : 2);
    let credBucket = cred < 600 ? 0 : (cred < 700 ? 1 : 2);
    let dti = debt / Math.max(1, inc);
    let debtBucket = dti < 0.3 ? 0 : 1;
    let lti = loan / Math.max(1, inc);
    let loanBucket = lti < 0.3 ? 0 : 1;
    
    let stateKey = `${incBucket}_${credBucket}_${debtBucket}_${loanBucket}`;
    
    let resultDiv = document.getElementById('demo-result');
    let statusH3 = document.getElementById('result-status');
    let reasonP = document.getElementById('result-reason');
    
    resultDiv.classList.remove('hidden', 'approved', 'rejected');
    
    let qValues = aiModel[stateKey];
    
    if (!qValues) {
        // Tie / Unexplored
        resultDiv.classList.add('rejected'); // Default safe visually
        statusH3.innerText = "⚠️ UNKNOWN PROFILE";
        statusH3.style.color = "#f59e0b";
        reasonP.innerText = "The AI hasn't seen enough data structurally identical to this profile during its 10,000 game run. Try a more common scenario!";
    } else {
        let rejectPts = qValues[0].toFixed(2);
        let approvePts = qValues[1].toFixed(2);
        
        if (qValues[1] > qValues[0]) {
            resultDiv.classList.add('approved');
            statusH3.innerText = "✅ APPROVED";
            statusH3.style.color = "#10b981";
            reasonP.innerText = `The AI algorithm expects to secure a positive ${approvePts} return points by backing you, vs ${rejectPts} points by rejecting.`;
        } else {
            resultDiv.classList.add('rejected');
            statusH3.innerText = "❌ REJECTED";
            statusH3.style.color = "#f43f5e";
            reasonP.innerText = `The AI flagged this. It avoided an expected ${approvePts} loss and safely nets ${rejectPts} points by rejecting.`;
        }
    }
});

document.addEventListener('DOMContentLoaded', initDashboard);
