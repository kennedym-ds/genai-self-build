const HomePage = {
    render() {
        return `
            <div class="page-container">
                <div class="main-header">
                    <h1>🛸 GenAI Self-Build</h1>
                    <h3>Zara's Dual Mission: Language + Vision</h3>
                    <p><em>An interactive story of how one scientist builds an AI system from first principles</em></p>
                </div>
                
                <h2>🎬 The Hook</h2>
                <div class="alert alert-info">
                    <p><strong>Imagine you've never read a word. Never seen a sentence.</strong><br>
                    You come from a world of pure mathematics — numbers and equations are your native tongue.<br>
                    Now someone drops you on Earth and says: <strong>"Read this."</strong></p>
                </div>
                
                <h2>🛸 Meet Zara</h2>
                <p>Zara is a Zorathian scientist who crash-lands on Earth with brilliant mathematical instincts—but no intuition for human words or visual scenes. Every symbol feels foreign. Every image feels noisy. She has to build understanding from zero, step by step.</p>

                <h2>🎯 Her Mission</h2>
                <p>Across six chapters, Zara assembles the same architecture that powers modern AI systems. Each chapter is one crucial component; together they become a complete intelligence pipeline.</p>

                <h2>🗺️ The Chapter Map</h2>
                <div class="card-grid">
                    <div class="workshop-card">
                        <h3>📕 Chapter 1: The Codebook</h3>
                        <p><strong>Curiosity + Frustration</strong> → Zara breaks overwhelming input into learnable pieces and builds her first codebook.</p>
                    </div>
                    <div class="workshop-card">
                        <h3>📗 Chapter 2: The Map</h3>
                        <p><strong>Wonder</strong> → She creates a meaning map where similar ideas live close together.</p>
                    </div>
                    <div class="workshop-card">
                        <h3>📘 Chapter 3: The Library</h3>
                        <p><strong>Confidence</strong> → Zara organizes memory so she can retrieve the most relevant patterns quickly.</p>
                    </div>
                    <div class="workshop-card">
                        <h3>📙 Chapter 4: The Spotlight</h3>
                        <p><strong>"Aha!" Moment</strong> → She learns to focus on what matters most in context.</p>
                    </div>
                    <div class="workshop-card">
                        <h3>📓 Chapter 5: The Brain</h3>
                        <p><strong>Pride</strong> → Zara assembles the full transformer brain from her earlier breakthroughs.</p>
                    </div>
                    <div class="workshop-card">
                        <h3>📔 Chapter 6: The Search Engine</h3>
                        <p><strong>Joy</strong> → She connects to real knowledge and grounds generation in evidence.</p>
                    </div>
                </div>
                
                <hr>
                
                <h2>🛤️ Path Selection</h2>
                <p>Zara's journey can be told two ways. Choose how you want to follow along:</p>
                
                <div class="card-grid">
                    <button class="mega-button" onclick="AppState.setLearningPath('Text')">
                        📝 Through Words<br><small>See how AI processes human language</small>
                    </button>
                    <button class="mega-button" onclick="AppState.setLearningPath('Visual')">
                        🎨 Through Shapes<br><small>See how AI processes visual scenes</small>
                    </button>
                </div>
                
                <hr>
                
                <div class="alert alert-tip">
                    <h3>🚀 Ready to begin?</h3>
                    <p>This is not just a tutorial. It's Zara's survival story—and your guided tour of modern AI architecture.</p>
                    <button class="mega-button" style="margin-top: 10px;" onclick="window.location.hash='#tokenization'">
                        Begin Chapter 1 →
                    </button>
                </div>
            </div>
        `;
    }
};
