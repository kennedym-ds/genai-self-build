const HomePage = {
    render() {
        return `
            <div class="page-container">
                <div class="main-header">
                    <h1>🛸 GenAI Self-Build</h1>
                    <h3>The Complete Journey: From Text to Intelligence</h3>
                    <p><em>Demystifying how AI really works, one concept at a time</em></p>
                </div>
                
                <h2>🌟 Welcome to the GenAI Self-Build Workshop Series!</h2>
                <p>This interactive demo brings together <strong>6 workshops</strong> that demystify how modern AI systems like ChatGPT actually work.</p>
                
                <div class="alert alert-info">
                    <h3>🎯 The Core Insight</h3>
                    <p><strong>AI doesn't understand language the way you do.</strong><br>
                    It transforms text through a series of mathematical operations.<br>
                    Each workshop reveals one piece of this puzzle.</p>
                </div>
                
                <hr>
                
                <h3>🛤️ Choose Your Path</h3>
                <p>To begin the workshop, select whether you want to learn using familiar words or visual shapes.</p>
                
                <div class="card-grid">
                    <button class="mega-button" onclick="AppState.setLearningPath('Text')">
                        📝 Start Text Journey<br><small>(Words & Sentences)</small>
                    </button>
                    <button class="mega-button" onclick="AppState.setLearningPath('Visual')">
                        🛸 Start Visual Journey<br><small>(Zara's Shapes)</small>
                    </button>
                </div>
                
                <hr>
                
                <h3>🔄 The Complete Pipeline</h3>
                <div class="pipeline-flow">
                    <div class="pipeline-step">📝 Text</div>
                    <span class="pipeline-arrow">→</span>
                    <div class="pipeline-step">🔢 Tokens</div>
                    <span class="pipeline-arrow">→</span>
                    <div class="pipeline-step">📊 Embeddings</div>
                    <span class="pipeline-arrow">→</span>
                    <div class="pipeline-step">🔍 Search</div>
                    <span class="pipeline-arrow">→</span>
                    <div class="pipeline-step">👀 Attention</div>
                    <span class="pipeline-arrow">→</span>
                    <div class="pipeline-step">🧠 Transform</div>
                    <span class="pipeline-arrow">→</span>
                    <div class="pipeline-step">💬 Output</div>
                </div>
                
                <hr>
                
                <div class="alert alert-tip">
                    👈 <strong>Select a workshop from the sidebar</strong> to explore each concept interactively!
                </div>
            </div>
        `;
    }
};
