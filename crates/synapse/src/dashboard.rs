/// Embedded web dashboard — a single HTML page served from Rust.
/// No React. No build tools. No npm. Just HTML + JS + Tailwind CDN.
/// Normal people can open a browser and chat with their AI.

pub const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TITAN Synapse</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  body { font-family: 'Inter', sans-serif; background: #0a0a0f; color: #e0e0e0; }
  .glow { box-shadow: 0 0 20px rgba(139, 92, 246, 0.15); }
  .pulse { animation: pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
  .msg-user { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); }
  .msg-ai { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
  #chat-input:focus { box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.5); }
  .specialist-tag { font-size: 0.65rem; padding: 1px 6px; border-radius: 9999px; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #4c1d95; border-radius: 3px; }
</style>
</head>
<body class="min-h-screen flex flex-col">

<!-- Header -->
<header class="border-b border-purple-900/30 px-6 py-3 flex items-center justify-between">
  <div class="flex items-center gap-3">
    <div class="text-2xl font-bold bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
      ⚡ SYNAPSE
    </div>
    <span class="text-xs text-gray-500">v0.1.0</span>
    <span id="status-dot" class="w-2 h-2 rounded-full bg-green-500 pulse"></span>
  </div>
  <div class="flex items-center gap-4 text-xs text-gray-400">
    <span id="model-info">Loading...</span>
    <button onclick="showPanel('stats')" class="hover:text-purple-400 transition">📊 Stats</button>
    <button onclick="showPanel('confidence')" class="hover:text-purple-400 transition">🧠 Brain</button>
  </div>
</header>

<!-- Main content -->
<div class="flex flex-1 overflow-hidden">
  <!-- Chat area -->
  <main class="flex-1 flex flex-col">
    <!-- Messages -->
    <div id="messages" class="flex-1 overflow-y-auto px-6 py-4 space-y-4">
      <div class="text-center text-gray-500 mt-20">
        <div class="text-4xl mb-3">🧠</div>
        <div class="text-lg font-semibold">TITAN Synapse</div>
        <div class="text-sm mt-1">Small models that think together. And learn.</div>
        <div class="text-xs mt-2 text-gray-600">Every conversation makes me smarter.</div>
      </div>
    </div>

    <!-- Input -->
    <div class="border-t border-purple-900/30 px-6 py-4">
      <form onsubmit="sendMessage(event)" class="flex gap-3">
        <input id="chat-input" type="text" placeholder="Ask me anything..."
          class="flex-1 bg-gray-900/50 border border-purple-900/30 rounded-xl px-4 py-3 text-sm
                 outline-none transition placeholder-gray-600" autocomplete="off">
        <button type="submit" id="send-btn"
          class="bg-purple-600 hover:bg-purple-500 px-6 py-3 rounded-xl text-sm font-medium transition">
          Send
        </button>
      </form>
      <div class="flex items-center gap-4 mt-2 text-xs text-gray-500">
        <span id="typing-indicator" class="hidden text-purple-400">⚡ Thinking...</span>
        <span id="tok-speed" class="ml-auto"></span>
      </div>
    </div>
  </main>

  <!-- Side panel -->
  <aside id="side-panel" class="hidden w-80 border-l border-purple-900/30 overflow-y-auto p-4">
    <div id="panel-content"></div>
  </aside>
</div>

<script>
const API = window.location.origin;
let messageHistory = [];

async function sendMessage(e) {
  e.preventDefault();
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  addMessage('user', msg);
  messageHistory.push({ role: 'user', content: msg });

  const typing = document.getElementById('typing-indicator');
  const speedEl = document.getElementById('tok-speed');
  typing.classList.remove('hidden');
  document.getElementById('send-btn').disabled = true;

  const start = performance.now();

  try {
    const resp = await fetch(API + '/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'synapse',
        messages: messageHistory.slice(-10), // Last 10 messages for context
        max_tokens: 2048,
        temperature: 0.7,
      }),
    });

    const data = await resp.json();
    const elapsed = ((performance.now() - start) / 1000).toFixed(1);
    const content = data.choices?.[0]?.message?.content || 'Error: No response';
    const usage = data.usage || {};

    addMessage('assistant', content, usage);
    messageHistory.push({ role: 'assistant', content });

    const tokPerSec = usage.completion_tokens
      ? (usage.completion_tokens / parseFloat(elapsed)).toFixed(0)
      : '?';
    speedEl.textContent = `${usage.completion_tokens || '?'} tokens · ${elapsed}s · ${tokPerSec} tok/s`;
  } catch (err) {
    addMessage('assistant', `Connection error: ${err.message}. Is the server running?`);
  }

  typing.classList.add('hidden');
  document.getElementById('send-btn').disabled = false;
}

function addMessage(role, content, usage) {
  const container = document.getElementById('messages');
  // Remove welcome message on first real message
  if (messageHistory.length <= 1 && role === 'user') {
    container.innerHTML = '';
  }

  const div = document.createElement('div');
  div.className = `rounded-xl px-4 py-3 max-w-3xl ${role === 'user' ? 'msg-user ml-auto' : 'msg-ai'}`;

  const label = document.createElement('div');
  label.className = 'text-xs font-semibold mb-1 ' + (role === 'user' ? 'text-purple-300' : 'text-cyan-300');
  label.textContent = role === 'user' ? 'You' : 'Synapse';
  if (usage && usage.prompt_tokens) {
    const badge = document.createElement('span');
    badge.className = 'specialist-tag bg-purple-900/50 text-purple-300 ml-2';
    badge.textContent = `${usage.prompt_tokens}→${usage.completion_tokens} tokens`;
    label.appendChild(badge);
  }

  const text = document.createElement('div');
  text.className = 'text-sm leading-relaxed whitespace-pre-wrap';
  // Simple markdown-like formatting
  text.innerHTML = formatContent(content);

  div.appendChild(label);
  div.appendChild(text);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function formatContent(text) {
  // Code blocks
  text = text.replace(/```(\w*)\n([\s\S]*?)```/g,
    '<pre class="bg-black/40 rounded-lg p-3 my-2 overflow-x-auto text-xs"><code>$2</code></pre>');
  // Inline code
  text = text.replace(/`([^`]+)`/g, '<code class="bg-black/30 px-1 rounded text-purple-300">$1</code>');
  // Bold
  text = text.replace(/\*\*([^*]+)\*\*/g, '<strong class="text-white">$1</strong>');
  // Lists
  text = text.replace(/^(\d+)\. /gm, '<span class="text-purple-400">$1.</span> ');
  text = text.replace(/^- /gm, '<span class="text-cyan-400">•</span> ');
  return text;
}

async function showPanel(type) {
  const panel = document.getElementById('side-panel');
  const content = document.getElementById('panel-content');
  panel.classList.toggle('hidden');

  if (type === 'stats') {
    content.innerHTML = '<div class="text-center text-gray-500">Loading...</div>';
    try {
      const resp = await fetch(API + '/api/status');
      const data = await resp.json();
      content.innerHTML = `
        <h3 class="text-sm font-bold text-purple-300 mb-3">📊 System Status</h3>
        <div class="space-y-2 text-xs">
          <div class="flex justify-between"><span class="text-gray-400">Version</span><span>${data.version}</span></div>
          <div class="flex justify-between"><span class="text-gray-400">Models</span><span>${data.models_loaded?.length || 0}</span></div>
          <div class="flex justify-between"><span class="text-gray-400">Specialists</span><span>${data.specialists?.length || 0}</span></div>
          <div class="flex justify-between"><span class="text-gray-400">Adapters</span><span>${data.adapters?.length || 0}</span></div>
          <hr class="border-purple-900/30 my-2">
          <h4 class="font-semibold text-cyan-300">Knowledge Graph</h4>
          <div class="flex justify-between"><span class="text-gray-400">Facts known</span><span>${data.knowledge?.facts || 0}</span></div>
          <div class="flex justify-between"><span class="text-gray-400">Conversations</span><span>${data.knowledge?.conversations || 0}</span></div>
          <div class="flex justify-between"><span class="text-gray-400">Preference pairs</span><span>${data.knowledge?.preference_pairs || 0}</span></div>
          <hr class="border-purple-900/30 my-2">
          <h4 class="font-semibold text-cyan-300">Loaded Models</h4>
          ${(data.models_loaded || []).map(m => `<div class="text-gray-300">• ${m}</div>`).join('')}
          <hr class="border-purple-900/30 my-2">
          <h4 class="font-semibold text-cyan-300">Hebbian Pathways</h4>
          ${(data.hebbian_routing?.top_pathways || []).map(p =>
            `<div class="flex justify-between"><span class="text-gray-400">${p.pathway}</span><span>×${p.strength}</span></div>`
          ).join('')}
        </div>
      `;
    } catch (e) {
      content.innerHTML = '<div class="text-red-400 text-xs">Failed to load status</div>';
    }
  } else if (type === 'confidence') {
    content.innerHTML = '<div class="text-center text-gray-500">Loading...</div>';
    try {
      const resp = await fetch(API + '/api/confidence');
      const data = await resp.json();
      const m = data.metacognition;
      content.innerHTML = `
        <h3 class="text-sm font-bold text-purple-300 mb-3">🧠 Metacognition</h3>
        <p class="text-xs text-gray-400 mb-3">What the system knows about itself</p>
        <div class="space-y-3">
          <h4 class="font-semibold text-cyan-300 text-xs">Specialist Confidence</h4>
          ${(m.specialists || []).map(s => `
            <div class="bg-gray-900/50 rounded-lg p-2 text-xs">
              <div class="flex justify-between font-medium">
                <span>${s.specialist}</span>
                <span class="text-purple-300">${s.avg_tok_per_sec?.toFixed(0)} tok/s</span>
              </div>
              <div class="flex justify-between text-gray-500 mt-1">
                <span>${s.requests} requests</span>
                <span>score: ${s.avg_score?.toFixed(1)}/5</span>
              </div>
              <div class="w-full bg-gray-800 rounded-full h-1 mt-1">
                <div class="bg-purple-500 h-1 rounded-full" style="width: ${(s.avg_score/5*100)}%"></div>
              </div>
            </div>
          `).join('')}
          <hr class="border-purple-900/30">
          <h4 class="font-semibold text-cyan-300 text-xs">Learning Status</h4>
          <div class="text-xs space-y-1">
            <div class="flex justify-between"><span class="text-gray-400">Facts learned</span><span>${m.learning_status?.facts_known || 0}</span></div>
            <div class="flex justify-between"><span class="text-gray-400">Conversations</span><span>${m.learning_status?.conversations_logged || 0}</span></div>
            <div class="flex justify-between"><span class="text-gray-400">Preferences</span><span>${m.learning_status?.preferences_collected || 0}</span></div>
          </div>
        </div>
      `;
    } catch (e) {
      content.innerHTML = '<div class="text-red-400 text-xs">Failed to load confidence data</div>';
    }
  }
}

// Load initial status
async function init() {
  try {
    const resp = await fetch(API + '/api/status');
    const data = await resp.json();
    document.getElementById('model-info').textContent =
      `${data.models_loaded?.length || 0} models · ${data.specialists?.length || 0} specialists · ${data.knowledge?.facts || 0} facts`;
  } catch (e) {
    document.getElementById('model-info').textContent = 'Disconnected';
    document.getElementById('status-dot').className = 'w-2 h-2 rounded-full bg-red-500';
  }
}

// Allow Enter to send, Shift+Enter for newline
document.getElementById('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    document.querySelector('form').dispatchEvent(new Event('submit'));
  }
});

init();
</script>
</body>
</html>"#;
