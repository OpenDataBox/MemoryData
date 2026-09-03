"""Visualize trajectories and memory from MemoryAgent outputs in ATIF format."""

import json
import logging
from html import escape
from pathlib import Path

logger = logging.getLogger(__name__)


def _generate_memory_html(memory_data: dict, standalone: bool = False) -> str:
    """Generate HTML content for memory visualization.
    
    Args:
        memory_data: Memory data dict with status, knowledge, procedural
        standalone: If True, include full HTML structure; if False, return fragment
    
    Returns:
        HTML string
    """
    status = memory_data.get('status', '')
    knowledge = memory_data.get('knowledge', [])
    procedural = memory_data.get('procedural', [])
    
    parts = []
    
    if standalone:
        parts.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Memory Visualization</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #1a1a2e;
            color: #eee;
        }
        .header {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 28px;
        }
        .header p {
            margin: 5px 0;
            opacity: 0.9;
        }
        .memory-stats {
            background: #16213e;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }
        .stat-item {
            display: flex;
            flex-direction: column;
        }
        .stat-label {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 18px;
            font-weight: bold;
            color: #f5576c;
        }
        .memory-section {
            background: #16213e;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .section-header {
            padding: 15px 20px;
            font-weight: bold;
            font-size: 16px;
            border-bottom: 2px solid;
        }
        .section-header.status {
            background-color: #2d2d5e;
            border-bottom-color: #9c27b0;
            color: #ce93d8;
        }
        .section-header.knowledge {
            background-color: #1e3a5f;
            border-bottom-color: #2196f3;
            color: #64b5f6;
        }
        .section-header.procedural {
            background-color: #1e4d2b;
            border-bottom-color: #4caf50;
            color: #81c784;
        }
        .section-content {
            padding: 20px;
        }
        .memory-entry {
            background: #0f0f1a;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            border-left: 3px solid #444;
        }
        .memory-entry.env { border-left-color: #ff9800; }
        .memory-entry.path { border-left-color: #2196f3; }
        .memory-entry.fact { border-left-color: #9c27b0; }
        .memory-entry.bug { border-left-color: #f44336; }
        .memory-entry.perf { border-left-color: #4caf50; }
        .entry-id {
            font-size: 11px;
            color: #666;
            margin-bottom: 8px;
        }
        .entry-content {
            color: #ddd;
        }
        .entry-meta {
            font-size: 11px;
            color: #555;
            margin-top: 8px;
        }
        .tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            margin-right: 8px;
        }
        .tag-env { background: #ff9800; color: #000; }
        .tag-path { background: #2196f3; color: #fff; }
        .tag-fact { background: #9c27b0; color: #fff; }
        .tag-bug { background: #f44336; color: #fff; }
        .tag-perf { background: #4caf50; color: #fff; }
        .status-box {
            background: #0f0f1a;
            border-radius: 6px;
            padding: 20px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.8;
            white-space: pre-wrap;
            border-left: 3px solid #9c27b0;
        }
        .empty-state {
            color: #666;
            font-style: italic;
            padding: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
""")
    
    # Memory header (standalone only)
    if standalone:
        parts.append(f"""
    <div class="header">
        <h1>🧠 Memory Visualization</h1>
        <p><strong>Total Entries:</strong> {len(knowledge) + len(procedural)}</p>
    </div>
    <div class="memory-stats">
        <div class="stat-item">
            <span class="stat-label">Status Length</span>
            <span class="stat-value">{len(status)} chars</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Knowledge Entries</span>
            <span class="stat-value">{len(knowledge)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Procedural Entries</span>
            <span class="stat-value">{len(procedural)}</span>
        </div>
    </div>
""")
    else:
        # Embedded header
        parts.append(f"""
    <div class="memory-section">
        <div class="section-header status">
            <span>🧠 MEMORY STATE ({len(knowledge)} knowledge, {len(procedural)} procedural)</span>
        </div>
    </div>
""")
    
    # Status section
    parts.append("""
    <div class="memory-section">
        <div class="section-header status">
            📊 Working Status
        </div>
        <div class="section-content">
""")
    
    if status:
        parts.append(f"""            <div class="status-box">{escape(status)}</div>
""")
    else:
        parts.append("""            <div class="empty-state">No status set</div>
""")
    
    parts.append("""        </div>
    </div>
""")
    
    # Knowledge section
    parts.append("""
    <div class="memory-section">
        <div class="section-header knowledge">
            📚 Knowledge Memory
        </div>
        <div class="section-content">
""")
    
    if knowledge:
        for entry in knowledge:
            entry_id = entry.get('id', '?')
            content = entry.get('content', '')
            created_at = entry.get('created_at', '')
            access_count = entry.get('access_count', 0)
            
            # Detect tag type
            entry_class = ""
            tag_html = ""
            if content.startswith('[ENV]'):
                entry_class = "env"
                tag_html = '<span class="tag tag-env">ENV</span>'
            elif content.startswith('[PATH]'):
                entry_class = "path"
                tag_html = '<span class="tag tag-path">PATH</span>'
            elif content.startswith('[FACT]'):
                entry_class = "fact"
                tag_html = '<span class="tag tag-fact">FACT</span>'
            
            parts.append(f"""            <div class="memory-entry {entry_class}">
                <div class="entry-id">ID: {escape(entry_id)}</div>
                <div class="entry-content">{tag_html}{escape(content)}</div>
                <div class="entry-meta">Created: {escape(created_at[:19] if created_at else 'unknown')} | Accessed: {access_count} times</div>
            </div>
""")
    else:
        parts.append("""            <div class="empty-state">No knowledge entries</div>
""")
    
    parts.append("""        </div>
    </div>
""")
    
    # Procedural section
    parts.append("""
    <div class="memory-section">
        <div class="section-header procedural">
            🔧 Procedural Memory
        </div>
        <div class="section-content">
""")
    
    if procedural:
        for entry in procedural:
            entry_id = entry.get('id', '?')
            content = entry.get('content', '')
            created_at = entry.get('created_at', '')
            access_count = entry.get('access_count', 0)
            
            # Detect tag type
            entry_class = ""
            tag_html = ""
            if content.startswith('[BUG]'):
                entry_class = "bug"
                tag_html = '<span class="tag tag-bug">BUG</span>'
            elif content.startswith('[PERF]'):
                entry_class = "perf"
                tag_html = '<span class="tag tag-perf">PERF</span>'
            
            parts.append(f"""            <div class="memory-entry {entry_class}">
                <div class="entry-id">ID: {escape(entry_id)}</div>
                <div class="entry-content">{tag_html}{escape(content)}</div>
                <div class="entry-meta">Created: {escape(created_at[:19] if created_at else 'unknown')} | Accessed: {access_count} times</div>
            </div>
""")
    else:
        parts.append("""            <div class="empty-state">No procedural entries</div>
""")
    
    parts.append("""        </div>
    </div>
""")
    
    if standalone:
        parts.append("""
</body>
</html>
""")
    
    return ''.join(parts)


def visualize_memory(memory_file: Path, save_html: bool = True) -> Path | None:
    """Read and display memory from memory.json file.

    Args:
        memory_file: Path to the memory.json file
        save_html: If True, save visualization as HTML file

    Returns:
        Path to the generated HTML file, or None if save_html is False
    """
    memory_file = Path(memory_file)
    
    with open(memory_file, 'r') as f:
        data = json.load(f)
    
    html_content = _generate_memory_html(data, standalone=True)
    
    if save_html:
        html_file = memory_file.parent / "memory.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.debug(f"Generated memory visualization: {html_file}")
        return html_file
    
    return None


def visualize_trajectory(trajectory_file: Path, save_html: bool = True) -> Path | None:
    """Read and display a trajectory from ATIF-v1.5 format JSON.

    Also loads trajectory_memory.json (if present) to show injected memory
    context inline at the corresponding action agent steps.

    Args:
        trajectory_file: Path to the trajectory.json file
        save_html: If True, save visualization as HTML file

    Returns:
        Path to the generated HTML file, or None if save_html is False
    """
    trajectory_file = Path(trajectory_file)
    
    with open(trajectory_file, 'r') as f:
        data = json.load(f)

    # Extract metadata
    schema_version = data.get('schema_version', 'unknown')
    session_id = data.get('session_id', 'unknown')
    agent_info = data.get('agent', {})
    agent_name = agent_info.get('name', 'unknown')
    agent_version = agent_info.get('version', 'unknown')
    model_name = agent_info.get('model_name', 'unknown')
    steps = data.get('steps', [])

    # Load memory trajectory for injection data (if exists)
    # The trajectory_memory.json is in the same directory as trajectory.json
    memory_traj_path = trajectory_file.parent / "trajectory_memory.json"
    injection_by_step: dict[int, str] = {}
    memory_ops_by_step: dict[int, list] = {}
    memory_snapshot_by_step: dict[int, dict] = {}
    memory_summary: dict = {}
    if memory_traj_path.exists():
        try:
            with open(memory_traj_path, 'r') as f:
                mem_data = json.load(f)
            memory_summary = mem_data.get('summary', {})
            for ms in mem_data.get('steps', []):
                # round_num is 0-indexed; agent_step_counter is 1-indexed
                # Memory at round_num=N injects into the next agent turn (agent_step N+1)
                agent_step = ms.get('round_num', 0) + 1
                if ms.get('should_inject') and ms.get('context_for_action'):
                    injection_by_step[agent_step] = ms['context_for_action']
                # Always store operations and snapshot for every trigger
                if ms.get('operations'):
                    memory_ops_by_step[agent_step] = ms['operations']
                if ms.get('memory_snapshot'):
                    memory_snapshot_by_step[agent_step] = ms['memory_snapshot']
        except Exception as e:
            logger.warning(f"Failed to load memory trajectory: {e}")

    # Prepare output path
    output_dir = trajectory_file.parent
    task_name = output_dir.parent.name  # e.g., draft_dp_2e599c56
    role_name = output_dir.name  # e.g., agent or verifier
    html_file = output_dir / f"trajectory_{role_name}.html"

    # Generate HTML
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Trajectory Visualization</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #1a1a2e;
            color: #eee;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 28px;
        }
        .header p {
            margin: 5px 0;
            opacity: 0.9;
        }
        .stats {
            background: #16213e;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }
        .stat-item {
            display: flex;
            flex-direction: column;
        }
        .stat-label {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 18px;
            font-weight: bold;
            color: #4caf50;
        }
        .step {
            background: #16213e;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .step-header {
            padding: 15px 20px;
            font-weight: bold;
            font-size: 16px;
            border-bottom: 2px solid;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .step-header.user {
            background-color: #1e3a5f;
            border-bottom-color: #2196f3;
            color: #64b5f6;
        }
        .step-header.agent {
            background-color: #1e4d2b;
            border-bottom-color: #4caf50;
            color: #81c784;
        }
        .step-header.system {
            background-color: #4a1a1a;
            border-bottom-color: #f44336;
            color: #ef9a9a;
        }
        .step-content {
            padding: 20px;
        }
        .message-box {
            background: #0f0f1a;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
            white-space: pre-wrap;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            max-height: 500px;
            overflow-y: auto;
            border-left: 3px solid #444;
        }
        .tool-call {
            background: #1a1a3e;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
            border-left: 3px solid #9c27b0;
        }
        .tool-call-header {
            color: #ce93d8;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .tool-args {
            background: #0f0f1a;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
        .memory-context-box {
            background: #1a2a2a;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
            border-left: 3px solid #00bcd4;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
            color: #80deea;
        }
        .observation {
            background: #1a2a1a;
            border-radius: 6px;
            padding: 15px;
            margin: 10px 0;
            border-left: 3px solid #4caf50;
        }
        .observation-header {
            color: #81c784;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .observation-content {
            background: #0f0f1a;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }
        .metrics {
            background: #2a2a3e;
            border-radius: 6px;
            padding: 10px 15px;
            margin: 10px 0;
            font-size: 12px;
            color: #aaa;
            display: flex;
            gap: 20px;
        }
        .timestamp {
            font-size: 12px;
            color: #888;
        }
        .emoji {
            font-size: 20px;
            margin-right: 8px;
        }
        .section-label {
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .collapsible {
            cursor: pointer;
            user-select: none;
        }
        .collapsible:hover {
            opacity: 0.8;
        }
        .collapsed .message-box,
        .collapsed .memory-context-box,
        .collapsed .tool-call,
        .collapsed .observation,
        .collapsed .metrics {
            display: none;
        }
    </style>
    <script>
        function toggleStep(stepId) {
            const step = document.getElementById(stepId);
            step.classList.toggle('collapsed');
        }
    </script>
</head>
<body>
""")

    # Calculate totals and detect summarization
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    max_prompt_tokens = 0
    max_prompt_step = 0
    summarization_steps: list[int] = []  # step_ids where summarization occurred
    truncation_steps: list[int] = []  # step_ids where truncation occurred
    for step in steps:
        if step.get('metrics'):
            pt = step['metrics'].get('prompt_tokens', 0)
            total_prompt_tokens += pt
            total_completion_tokens += step['metrics'].get('completion_tokens', 0)
            total_cost += step['metrics'].get('cost_usd', 0)
            if pt > max_prompt_tokens:
                max_prompt_tokens = pt
                max_prompt_step = step.get('step_id', 0)
        # Detect summarization steps
        if (step.get('source') == 'system'
                and 'summarization' in (step.get('message', '') or '').lower()):
            summarization_steps.append(step.get('step_id', 0))
        # Detect truncation steps
        extra = step.get('extra', {}) or {}
        if extra.get('context_truncated'):
            truncation_steps.append(step.get('step_id', 0))

    html_parts.append(f"""
    <div class="header">
        <h1>🔍 Trajectory Visualization</h1>
        <p><strong>Task:</strong> {escape(task_name)}</p>
        <p><strong>Role:</strong> {escape(role_name)}</p>
        <p><strong>Session:</strong> {escape(session_id[:16])}...</p>
    </div>
    <div class="stats">
        <div class="stat-item">
            <span class="stat-label">Schema</span>
            <span class="stat-value">{escape(schema_version)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Agent</span>
            <span class="stat-value">{escape(agent_name)} v{escape(agent_version)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Model</span>
            <span class="stat-value">{escape(model_name)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Total Steps</span>
            <span class="stat-value">{len(steps)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Prompt Tokens</span>
            <span class="stat-value">{total_prompt_tokens:,}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Completion Tokens</span>
            <span class="stat-value">{total_completion_tokens:,}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Total Cost</span>
            <span class="stat-value">${total_cost:.4f}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Max Context Length</span>
            <span class="stat-value">{max_prompt_tokens:,} tokens</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Max Context Step</span>
            <span class="stat-value">Step {max_prompt_step}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Summarizations</span>
            <span class="stat-value" style="color: {'#f44336' if summarization_steps else '#4caf50'};">{len(summarization_steps)}{' (steps: ' + ', '.join(map(str, summarization_steps)) + ')' if summarization_steps else ''}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Truncations</span>
            <span class="stat-value" style="color: {'#ff9800' if truncation_steps else '#4caf50'};">{len(truncation_steps)}{' (steps: ' + ', '.join(map(str, truncation_steps)) + ')' if truncation_steps else ''}</span>
        </div>
""")

    # Add memory stats if available
    if memory_summary:
        n_inj = memory_summary.get('total_injections', 0)
        n_noop = memory_summary.get('total_noops', 0)
        n_trig = memory_summary.get('total_triggers', 0)
        html_parts.append(f"""
        <div class="stat-item">
            <span class="stat-label">Memory Injections</span>
            <span class="stat-value" style="color: #00bcd4;">{n_inj} / {n_trig}</span>
        </div>
""")

    html_parts.append("    </div>\n")

    # Build a mapping from step_id to action step count.
    # The action agent loop numbers steps as: step 1 = initial prompt,
    # step 2 = episode 0 + 2, step 3 = episode 1 + 2, etc.
    # In the trajectory, agent steps are interleaved with user steps.
    # The memory trigger fires at step_count = episode + 2 for episode >= 0,
    # and step_count = 1 for the initial prompt.
    # We match by counting agent steps: the Nth agent step corresponds to
    # step_count = N (since memory triggers at every step starting from 1).
    agent_step_counter = 0

    # Process each step
    for step in steps:
        step_id = step.get('step_id', '?')
        source = step.get('source', 'unknown')
        timestamp = step.get('timestamp', '')
        message = step.get('message', '')
        tool_calls = step.get('tool_calls', [])
        observation = step.get('observation') or {}
        metrics = step.get('metrics') or {}

        # Track agent steps to match with memory injection step_count
        if source == 'agent':
            agent_step_counter += 1
        # Look up injection for this step (user step after agent step N
        # corresponds to step_count = N+1 in memory trajectory)
        memory_context = injection_by_step.get(agent_step_counter, '')

        if source == 'user':
            emoji = "👤"
            role_label = "USER"
        elif source == 'agent':
            emoji = "🤖"
            role_label = "AGENT"
        elif source == 'system':
            extra = step.get('extra', {}) or {}
            if extra.get('context_length_exceeded'):
                emoji = "🚫"
                role_label = "CONTEXT LENGTH EXCEEDED"
            elif extra.get('context_truncated'):
                emoji = "✂️"
                role_label = f"CONTEXT TRUNCATED (#{extra.get('truncation_count', '?')}: {extra.get('messages_before', '?')} → {extra.get('messages_after', '?')} msgs)"
            else:
                emoji = "⚙️"
                role_label = "SYSTEM"
        else:
            emoji = "❓"
            role_label = source.upper()

        html_parts.append(f"""
    <div class="step" id="step-{step_id}">
        <div class="step-header {source} collapsible" onclick="toggleStep('step-{step_id}')">
            <span><span class="emoji">{emoji}</span>STEP {step_id}: {escape(role_label)}</span>
            <span class="timestamp">{escape(timestamp[:19] if timestamp else '')}</span>
        </div>
        <div class="step-content">
""")

        # Memory operations and context for this agent step
        if source == 'agent' and (agent_step_counter in memory_ops_by_step or memory_context):
            ops = memory_ops_by_step.get(agent_step_counter, [])
            snapshot = memory_snapshot_by_step.get(agent_step_counter, {})

            html_parts.append(f"""
            <div style="background:#0d2626; border:1px solid #00bcd4; border-radius:8px; padding:15px; margin:10px 0;">
                <div style="color:#00bcd4; font-weight:bold; font-size:14px; margin-bottom:10px;">
                    🧠 Memory Agent (Round {agent_step_counter - 1})
                    {'<span style="background:#00bcd4;color:#000;padding:2px 8px;border-radius:4px;font-size:11px;margin-left:8px;">INJECT</span>' if memory_context else '<span style="background:#666;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;margin-left:8px;">NO-OP</span>'}
                </div>
""")
            # Show operations
            if ops:
                html_parts.append('                <div style="margin-bottom:10px;">')
                for op in ops:
                    action = op.get('action', '?')
                    content = op.get('content', '') or ''
                    mem_id = op.get('memory_id', '')
                    if action == 'delete':
                        color = '#f44336'
                    elif action == 'update_status':
                        color = '#ff9800'
                    else:
                        color = '#4caf50'
                    content_display = content
                    html_parts.append(f"""
                    <div style="background:#0f1a1a; border-left:3px solid {color}; padding:8px 12px; margin:5px 0; border-radius:4px;">
                        <span style="color:{color}; font-weight:bold; font-size:12px;">{escape(action.upper())}</span>
                        {f'<span style="color:#666; font-size:11px;"> [{mem_id}]</span>' if mem_id else ''}
                        <div style="font-family:monospace; font-size:11px; color:#aaa; margin-top:4px; white-space:pre-wrap;">{escape(content_display)}</div>
                    </div>
""")
                html_parts.append('                </div>')

            # Show injected context
            if memory_context:
                html_parts.append(f"""
                <div style="margin-top:10px;">
                    <div style="color:#80deea; font-size:12px; text-transform:uppercase; margin-bottom:5px;">Injected Context → Action Agent</div>
                    <div class="memory-context-box">{escape(memory_context)}</div>
                </div>
""")
            # Show snapshot summary
            if snapshot:
                html_parts.append(f"""
                <div style="color:#555; font-size:11px; margin-top:8px;">
                    Memory: {snapshot.get('knowledge_count', 0)} knowledge, {snapshot.get('procedural_count', 0)} procedural |
                    Status: {escape(snapshot.get('status', ''))}
                </div>
""")
            html_parts.append("            </div>\n")

        # Message
        if message:
            # Truncate very long messages for display
            display_message = message
            html_parts.append(f"""
            <div class="section-label">Message</div>
            <div class="message-box">{escape(display_message)}</div>
""")

        # Tool calls
        if tool_calls:
            for tc in tool_calls:
                tc_id = tc.get('tool_call_id', '')
                func_name = tc.get('function_name', '')
                args = tc.get('arguments', {})
                args_str = json.dumps(args, indent=2)
                html_parts.append(f"""
            <div class="tool-call">
                <div class="tool-call-header">🔧 Tool Call: {escape(func_name)} ({escape(tc_id)})</div>
                <div class="tool-args">{escape(args_str)}</div>
            </div>
""")

        # Observation (skip for user steps — message already contains the tool response)
        if observation and source != 'user':
            results = observation.get('results', [])
            for i, result in enumerate(results):
                content = result.get('content', '')
                # Truncate very long observations
                display_content = content
                html_parts.append(f"""
            <div class="observation">
                <div class="observation-header">📋 Observation {i+1}</div>
                <div class="observation-content">{escape(display_content)}</div>
            </div>
""")

        # Metrics
        if metrics:
            prompt_tokens = metrics.get('prompt_tokens', 0)
            completion_tokens = metrics.get('completion_tokens', 0)
            cost = metrics.get('cost_usd', 0)
            html_parts.append(f"""
            <div class="metrics">
                <span>Prompt: {prompt_tokens:,} tokens</span>
                <span>Completion: {completion_tokens:,} tokens</span>
                <span>Cost: ${cost:.6f}</span>
            </div>
""")

        html_parts.append("""
        </div>
    </div>
""")

    html_parts.append("""
</body>
</html>
""")

    # Save HTML file
    if save_html:
        html_content = ''.join(html_parts)
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.debug(f"Generated trajectory visualization: {html_file}")
        return html_file
    
    return None


def process_run_directory(run_dir: Path) -> dict[str, list[Path]]:
    """Process all trajectories and memory files in a run directory.

    Args:
        run_dir: Path to the run directory (e.g., run_20260213_072245)

    Returns:
        Dict with 'trajectories' and 'memories' lists of generated HTML paths
    """
    run_dir = Path(run_dir)
    result = {'trajectories': [], 'memories': []}
    
    # Find all trajectory.json files, separating agent vs memory trajectories
    all_traj_files = list(run_dir.rglob("trajectory*.json"))
    agent_traj_files = [f for f in all_traj_files if "_memory" not in f.name]
    memory_traj_files = [f for f in all_traj_files if "_memory" in f.name]

    if agent_traj_files:
        logger.info(f"Found {len(agent_traj_files)} agent trajectory files in {run_dir.name}/")
        for traj_file in sorted(agent_traj_files):
            try:
                html_path = visualize_trajectory(traj_file, save_html=True)
                if html_path:
                    result['trajectories'].append(html_path)
            except Exception as e:
                logger.error(f"Failed to visualize {traj_file}: {e}")
    else:
        logger.warning(f"No trajectory.json files found in: {run_dir}")

    if memory_traj_files:
        logger.info(f"Found {len(memory_traj_files)} memory trajectory files in {run_dir.name}/")
        for traj_file in sorted(memory_traj_files):
            try:
                html_path = visualize_memory_trajectory(traj_file, save_html=True)
                if html_path:
                    result['trajectories'].append(html_path)
            except Exception as e:
                logger.error(f"Failed to visualize memory trajectory {traj_file}: {e}")

    # Find all memory.json files
    memory_files = list(run_dir.rglob("memory.json"))
    
    if memory_files:
        logger.info(f"Found {len(memory_files)} memory files in {run_dir.name}/")
        for mem_file in sorted(memory_files):
            try:
                html_path = visualize_memory(mem_file, save_html=True)
                if html_path:
                    result['memories'].append(html_path)
            except Exception as e:
                logger.error(f"Failed to visualize {mem_file}: {e}")

    total = len(result['trajectories']) + len(result['memories'])
    logger.info(f"Generated {total} HTML visualizations ({len(result['trajectories'])} trajectories, {len(result['memories'])} memories)")
    return result


def visualize_task_with_memory(task_dir: Path, save_html: bool = True) -> Path | None:
    """Visualize a task's trajectory with embedded memory view.
    
    Creates a combined HTML file showing both the trajectory and memory state.
    
    Args:
        task_dir: Path to the task directory (e.g., outputs/run_xxx/task_name/)
        save_html: If True, save visualization as HTML file
    
    Returns:
        Path to the generated HTML file, or None if not saved
    """
    task_dir = Path(task_dir)
    
    # Find trajectory file
    agent_dir = task_dir / "agent"
    trajectory_files = list(agent_dir.glob("trajectory*.json"))
    if not trajectory_files:
        logger.warning(f"No trajectory file found in {agent_dir}")
        return None
    
    trajectory_file = trajectory_files[0]
    
    # Check for memory file
    memory_file = task_dir / "memory.json"
    has_memory = memory_file.exists()
    
    # Read trajectory
    with open(trajectory_file, 'r') as f:
        traj_data = json.load(f)
    
    # Read memory if exists
    memory_data = None
    if has_memory:
        with open(memory_file, 'r') as f:
            memory_data = json.load(f)
    
    raise NotImplementedError("visualize_task_with_memory is not yet implemented")


def visualize_memory_trajectory(trajectory_file: Path, save_html: bool = True) -> Path | None:
    """Visualize the Memory Agent trajectory from trajectory_memory.json.
    
    Creates an HTML file showing all Memory Agent calls, their operations,
    and the context generated for the Action Agent.
    
    Args:
        trajectory_file: Path to trajectory_memory.json
        save_html: If True, save visualization as HTML file
    
    Returns:
        Path to the generated HTML file, or None if save_html is False
    """
    trajectory_file = Path(trajectory_file)
    
    with open(trajectory_file, 'r') as f:
        data = json.load(f)
    
    # Extract info
    schema_version = data.get('schema_version', 'unknown')
    memory_model = data.get('memory_model_name', 'unknown')
    trigger_config = data.get('trigger_config', {})
    summary = data.get('summary', {})
    steps = data.get('steps', [])
    
    html_file = trajectory_file.parent / "trajectory_memory.html"
    
    # Generate HTML
    html_parts = []
    html_parts.append('''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Memory Agent Trajectory</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #1a1a2e;
            color: #eee;
        }
        .header {
            background: linear-gradient(135deg, #00bcd4 0%, #00897b 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .header h1 { margin: 0 0 10px 0; font-size: 28px; }
        .header p { margin: 5px 0; opacity: 0.9; }
        .stats {
            background: #16213e;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }
        .stat-item { display: flex; flex-direction: column; }
        .stat-label { font-size: 12px; color: #888; text-transform: uppercase; }
        .stat-value { font-size: 20px; font-weight: bold; color: #00bcd4; }
        .step {
            background: #16213e;
            border-radius: 10px;
            margin-bottom: 15px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .step-header {
            background: #1f3a5f;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }
        .step-header:hover { background: #254670; }
        .step-content { padding: 20px; }
        .section-label {
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
            margin: 15px 0 5px 0;
        }
        .observation-box {
            background: #0f0f1a;
            padding: 15px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
            color: #aaa;
        }
        .context-box {
            background: #1a2a2a;
            padding: 15px;
            border-radius: 6px;
            border-left: 3px solid #00bcd4;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
            color: #80deea;
        }
        .operation {
            background: #1a2a1a;
            border-radius: 6px;
            padding: 12px;
            margin: 8px 0;
            border-left: 3px solid #4caf50;
        }
        .operation.delete { border-left-color: #f44336; }
        .operation.update_status { border-left-color: #ff9800; }
        .op-action {
            font-weight: bold;
            color: #4caf50;
            margin-bottom: 5px;
        }
        .op-content {
            font-family: monospace;
            font-size: 12px;
            color: #ccc;
        }
        .memory-snapshot {
            background: #252540;
            padding: 10px 15px;
            border-radius: 6px;
            margin-top: 10px;
            font-size: 12px;
            color: #888;
        }
        .error {
            background: #2a1a1a;
            border-left: 3px solid #f44336;
            padding: 15px;
            border-radius: 6px;
            color: #f88;
        }
        .timestamp { color: #666; font-size: 12px; }
        .collapsed .step-content { display: none; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            margin-left: 8px;
            vertical-align: middle;
        }
        .badge-llm { background: #4caf50; color: #fff; }
        .badge-bm25 { background: #ff9800; color: #000; }
        .badge-error { background: #f44336; color: #fff; }
        .badge-unknown { background: #666; color: #fff; }
    </style>
    <script>
        function toggleStep(stepId) {
            document.getElementById(stepId).classList.toggle('collapsed');
        }
    </script>
</head>
<body>
''')

    context_source_counts = summary.get('context_source_counts', {})

    html_parts.append(f'''
    <div class="header">
        <h1>🧠 Memory Agent Trajectory</h1>
        <p><strong>Model:</strong> {escape(memory_model)}</p>
        <p><strong>Schema:</strong> {escape(schema_version)}</p>
    </div>
    <div class="stats">
        <div class="stat-item">
            <span class="stat-label">Total Triggers</span>
            <span class="stat-value">{summary.get('total_triggers', 0)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Total Operations</span>
            <span class="stat-value">{summary.get('total_operations', 0)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Injections</span>
            <span class="stat-value">{summary.get('total_injections', 0)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">No-ops</span>
            <span class="stat-value">{summary.get('total_noops', 0)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Final Memory Size</span>
            <span class="stat-value">{summary.get('final_memory_size', 0)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Trigger Interval</span>
            <span class="stat-value">{trigger_config.get('interval', '?')}</span>
        </div>
    </div>
''')
    
    for step in steps:
        step_id = step.get('step_id', '?')
        trigger_step = step.get('trigger_step', '?')
        timestamp = (step.get('timestamp') or '')[:19]
        user_prompt = step.get('user_prompt_phase1') or step.get('user_prompt') or ''
        observation = step.get('observation') or step.get('observation_preview') or ''
        operations = step.get('operations', [])
        context = step.get('context_for_action', '')
        error = step.get('error')
        snapshot = step.get('memory_snapshot', {})
        should_inject = step.get('should_inject', False)

        # Badge for step header
        if error:
            header_badge = '<span class="badge badge-error">ERR</span>'
        elif should_inject and context:
            header_badge = '<span class="badge badge-llm">INJECT</span>'
        else:
            header_badge = '<span class="badge badge-unknown">NO-OP</span>'

        # Show full user prompt if available, otherwise fall back to observation
        input_label = "Full User Prompt" if user_prompt else "Observation Input"
        input_content = user_prompt or observation

        html_parts.append(f'''
    <div class="step" id="step-{step_id}">
        <div class="step-header" onclick="toggleStep('step-{step_id}')">
            <span>🧠 Memory Trigger #{step_id} (at Action Step {trigger_step}) {header_badge}</span>
            <span class="timestamp">{escape(timestamp)}</span>
        </div>
        <div class="step-content">
            <div class="section-label">{input_label}</div>
            <div class="observation-box">{escape(input_content)}</div>
''')
        
        if error:
            html_parts.append(f'''
            <div class="section-label">Error</div>
            <div class="error">{escape(error)}</div>
''')
        else:
            if operations:
                html_parts.append(f'            <div class="section-label">Operations ({len(operations)})</div>')
                for op in operations:
                    action = op.get('action', '?')
                    content = op.get('content', '') or ''
                    mem_id = op.get('memory_id', '')
                    op_class = action if action in ['delete', 'update_status'] else ''
                    html_parts.append(f'''
            <div class="operation {op_class}">
                <div class="op-action">{escape(action.upper())}{f' [{mem_id}]' if mem_id else ''}</div>
                <div class="op-content">{escape(content)}</div>
            </div>
''')
            else:
                html_parts.append('            <div class="section-label">Operations (0)</div>')
            
            if context:
                html_parts.append(f'''
            <div class="section-label">Context for Action Agent</div>
            <div class="context-box">{escape(context)}</div>
''')
            
            if snapshot:
                html_parts.append(f'''
            <div class="memory-snapshot">
                <strong>Memory Snapshot:</strong> Status: "{escape(snapshot.get('status', '')[:100])}" | 
                Knowledge: {snapshot.get('knowledge_count', 0)} | 
                Procedural: {snapshot.get('procedural_count', 0)}
            </div>
''')
        
        html_parts.append('''
        </div>
    </div>
''')
    
    html_parts.append('''
</body>
</html>
''')
    
    if save_html:
        html_content = ''.join(html_parts)
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.debug(f"Generated memory trajectory visualization: {html_file}")
        return html_file
    
    return None
