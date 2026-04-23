/**
 * GPUDB Real-Time Benchmark Dashboard
 * ────────────────────────────────────
 * D3.js visualization engine with built-in simulation mode
 * for demonstrating all features without a live C++ backend.
 */

// ── Global State ────────────────────────────────────────────────────────────
const state = {
    benchmarks: [],        // Array of benchmark results
    gpuStats: [],          // Array of GPU utilization snapshots
    maxSpeedup: 0,
    mode: 'idle',          // 'idle' | 'simulating' | 'connected' | 'file'
    simTimer: null,
    simIndex: 0,
    wsConnection: null,
};

// ── Color Palette (Light Blue & Cyan Theme) ───────────────────────────────
const COLORS = {
    cyan:       '#0891b2',
    cyanDim:    'rgba(8, 145, 178, 0.12)',
    orange:     '#ea580c',
    orangeDim:  'rgba(234, 88, 12, 0.12)',
    violet:     '#6d28d9',
    green:      '#059669',
    red:        '#dc2626',
    yellow:     '#d97706',
    gridLine:   'rgba(8, 145, 178, 0.06)',
    axisTick:   'rgba(8, 145, 178, 0.12)',
    textMuted:  '#4a85b0',
    textSec:    '#1b5a8a',
};

// ── Simulated Benchmark Data ────────────────────────────────────────────────
const SIMULATED_BENCHMARKS = [
    { operation: 'Filter-1M',   rows: 1000000,   cpu_ms: 18.5,   gpu_ms: 0.9,   bandwidth_gbps: 8.9  },
    { operation: 'Sum-1M',      rows: 1000000,   cpu_ms: 12.3,   gpu_ms: 0.4,   bandwidth_gbps: 20.0 },
    { operation: 'Filter-10M',  rows: 10000000,  cpu_ms: 185.2,  gpu_ms: 3.1,   bandwidth_gbps: 25.8 },
    { operation: 'Sum-10M',     rows: 10000000,  cpu_ms: 128.4,  gpu_ms: 1.8,   bandwidth_gbps: 44.4 },
    { operation: 'GroupBy-10M', rows: 10000000,  cpu_ms: 410.0,  gpu_ms: 8.5,   bandwidth_gbps: 14.1 },
    { operation: 'Join-1Mx10M', rows: 11000000,  cpu_ms: 1250.0, gpu_ms: 12.3,  bandwidth_gbps: 10.7 },
    { operation: 'Filter-100M', rows: 100000000, cpu_ms: 1840.5, gpu_ms: 15.2,  bandwidth_gbps: 52.6 },
    { operation: 'Sum-100M',    rows: 100000000, cpu_ms: 1280.0, gpu_ms: 10.6,  bandwidth_gbps: 75.5 },
];

// ── Tooltip singleton ───────────────────────────────────────────────────────
let tooltip;

// ═══════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    tooltip = d3.select('#tooltip');
    initCharts();
    bindControls();
    updateKPIs();
    showToast('Dashboard ready — click ▶ Simulate to demo, or upload a JSON file.');
});

function bindControls() {
    document.getElementById('btn-simulate').addEventListener('click', toggleSimulation);
    document.getElementById('file-input').addEventListener('change', handleFileUpload);
    document.getElementById('btn-clear').addEventListener('click', clearData);
}

// ═══════════════════════════════════════════════════════════════════════════
// SIMULATION ENGINE
// ═══════════════════════════════════════════════════════════════════════════

function toggleSimulation() {
    const btn = document.getElementById('btn-simulate');

    if (state.mode === 'simulating') {
        stopSimulation();
        btn.textContent = '▶ Simulate';
        btn.classList.remove('active');
        return;
    }

    clearData();
    state.mode = 'simulating';
    state.simIndex = 0;
    btn.textContent = '⏹ Stop';
    btn.classList.add('active');
    setConnectionStatus('simulating', 'Simulating');

    // Stream one benchmark result every 1.5 seconds
    pushNextSimResult();
    state.simTimer = setInterval(() => {
        pushNextSimResult();
    }, 1500);

    // Also simulate GPU stats at 200ms intervals
    state.gpuStatsTimer = setInterval(() => {
        pushSimGpuStats();
    }, 200);
}

function stopSimulation() {
    state.mode = 'idle';
    clearInterval(state.simTimer);
    clearInterval(state.gpuStatsTimer);
    state.simTimer = null;
    state.gpuStatsTimer = null;
    setConnectionStatus('disconnected', 'Idle');
}

function pushNextSimResult() {
    if (state.simIndex >= SIMULATED_BENCHMARKS.length) {
        stopSimulation();
        const btn = document.getElementById('btn-simulate');
        btn.textContent = '▶ Simulate';
        btn.classList.remove('active');
        showToast('Simulation complete — all benchmarks rendered.');
        return;
    }

    const raw = SIMULATED_BENCHMARKS[state.simIndex];
    const jitter = () => 0.9 + Math.random() * 0.2; // ±10% jitter
    const entry = {
        ...raw,
        type: 'benchmark',
        cpu_ms: +(raw.cpu_ms * jitter()).toFixed(2),
        gpu_ms: +(raw.gpu_ms * jitter()).toFixed(2),
        bandwidth_gbps: +(raw.bandwidth_gbps * jitter()).toFixed(2),
        timestamp: Date.now(),
    };
    entry.speedup = +(entry.cpu_ms / entry.gpu_ms).toFixed(2);
    entry.mean_ms = entry.gpu_ms;
    entry.median_ms = entry.gpu_ms;
    entry.p95_ms = +(entry.gpu_ms * 1.2).toFixed(2);

    ingestBenchmark(entry);
    state.simIndex++;
}

function pushSimGpuStats() {
    const progress = state.simIndex / SIMULATED_BENCHMARKS.length;
    const baseUtil = state.simIndex > 0 ? 45 + progress * 50 : 5;
    const baseVram = 350 + progress * 2800;

    const entry = {
        type: 'gpu_stats',
        gpu_util_pct: Math.min(100, Math.round(baseUtil + (Math.random() - 0.5) * 20)),
        vram_used_mb: Math.round(baseVram + (Math.random() - 0.5) * 200),
        vram_total_mb: 4096,
        temperature_c: Math.round(55 + progress * 25 + (Math.random() - 0.5) * 5),
        timestamp: Date.now(),
    };

    ingestGpuStats(entry);
}

// ═══════════════════════════════════════════════════════════════════════════
// FILE UPLOAD
// ═══════════════════════════════════════════════════════════════════════════

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
        try {
            const text = ev.target.result;
            let data;

            if (file.name.endsWith('.csv')) {
                // Parse benchmark CSV: operator,rows,mean_ms,median_ms,p95_ms,cpu_ms,speedup,bw_gbps,...
                const lines = text.trim().split('\n');
                const headers = lines[0].trim().split(',');
                data = lines.slice(1).map(line => {
                    const cols = line.trim().split(',');
                    const row = {};
                    headers.forEach((h, i) => row[h.trim()] = cols[i]?.trim());
                    return {
                        type: 'benchmark',
                        operation: row['operator'] || row['operation'] || 'Unknown',
                        rows: parseInt(row['rows']) || 0,
                        cpu_ms: parseFloat(row['cpu_ms']) || 0,
                        gpu_ms: parseFloat(row['median_ms']) || parseFloat(row['gpu_ms']) || 0,
                        speedup: parseFloat(row['speedup']) || 0,
                        bandwidth_gbps: parseFloat(row['bw_gbps']) || parseFloat(row['bandwidth_gbps']) || 0,
                        mean_ms: parseFloat(row['mean_ms']) || 0,
                        median_ms: parseFloat(row['median_ms']) || 0,
                        p95_ms: parseFloat(row['p95_ms']) || 0,
                    };
                }).filter(r => r.gpu_ms > 0);

                // Recalculate speedup if it was 0 or missing
                data.forEach(d => {
                    if ((!d.speedup || d.speedup < 0.001) && d.cpu_ms > 0 && d.gpu_ms > 0) {
                        d.speedup = +(d.cpu_ms / d.gpu_ms).toFixed(2);
                    }
                });
            } else {
                // JSON or JSONL
                try {
                    data = JSON.parse(text);
                } catch {
                    data = text.trim().split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
                }
                if (!Array.isArray(data)) data = [data];
            }

            clearData();
            state.mode = 'file';
            setConnectionStatus('connected', 'File Loaded');

            let i = 0;
            const interval = setInterval(() => {
                if (i >= data.length) {
                    clearInterval(interval);
                    showToast(`Loaded ${data.length} results from ${file.name}`);
                    return;
                }
                const entry = data[i];
                if (!entry.speedup && entry.cpu_ms && entry.gpu_ms) {
                    entry.speedup = +(entry.cpu_ms / entry.gpu_ms).toFixed(2);
                }
                entry.type = entry.type || 'benchmark';
                ingestBenchmark(entry);
                i++;
            }, 300);

        } catch (err) {
            showToast('Error parsing file: ' + err.message);
        }
    };
    reader.readAsText(file);
    e.target.value = '';
}

// ═══════════════════════════════════════════════════════════════════════════
// DATA INGESTION
// ═══════════════════════════════════════════════════════════════════════════

function ingestBenchmark(entry) {
    state.benchmarks.push(entry);
    if (entry.speedup > state.maxSpeedup) {
        state.maxSpeedup = entry.speedup;
    }
    updateKPIs(entry);
    updateBarChart();
    updateBandwidthChart();
    updateSpeedupGauge(entry);
    updateResultsTable(entry);
}

function ingestGpuStats(entry) {
    state.gpuStats.push(entry);
    // Keep last 300 samples
    if (state.gpuStats.length > 300) state.gpuStats.shift();
    updateGpuHealthChart();
    updateGpuKPIs(entry);
}

function clearData() {
    state.benchmarks = [];
    state.gpuStats = [];
    state.maxSpeedup = 0;
    state.simIndex = 0;
    stopSimulation();
    state.mode = 'idle';

    document.getElementById('btn-simulate').textContent = '▶ Simulate';
    document.getElementById('btn-simulate').classList.remove('active');
    setConnectionStatus('disconnected', 'Idle');

    updateKPIs();
    clearChart('bar-chart');
    clearChart('bandwidth-chart');
    clearChart('gpu-health-chart');
    resetSpeedupGauge();
    clearResultsTable();
}

// ═══════════════════════════════════════════════════════════════════════════
// KPI CARDS
// ═══════════════════════════════════════════════════════════════════════════

function updateKPIs(latest) {
    const n = state.benchmarks.length;

    document.getElementById('kpi-peak-speedup').textContent =
        n ? state.maxSpeedup.toFixed(1) + '×' : '—';
    document.getElementById('kpi-benchmarks').textContent =
        n ? n.toString() : '—';

    if (latest) {
        document.getElementById('kpi-last-gpu').textContent =
            latest.gpu_ms.toFixed(1) + ' ms';
        document.getElementById('kpi-peak-bw').textContent =
            Math.max(...state.benchmarks.map(b => b.bandwidth_gbps)).toFixed(1) + ' GB/s';

        // Flash effect
        document.querySelectorAll('.kpi-card').forEach(card => {
            card.classList.remove('flash');
            void card.offsetWidth; // Reflow
            card.classList.add('flash');
        });
    } else {
        document.getElementById('kpi-last-gpu').textContent = '—';
        document.getElementById('kpi-peak-bw').textContent = '—';
    }
}

function updateGpuKPIs(stats) {
    document.getElementById('kpi-gpu-util').textContent = stats.gpu_util_pct + '%';
    document.getElementById('kpi-vram').textContent =
        (stats.vram_used_mb / 1024).toFixed(1) + ' GB';
}

// ═══════════════════════════════════════════════════════════════════════════
// CHART 1: GROUPED BAR CHART (CPU vs GPU)
// ═══════════════════════════════════════════════════════════════════════════

let barChartInitialized = false;
let barSvg, barX0, barX1, barY, barWidth, barHeight, barMargin;

function initBarChart() {
    const container = document.getElementById('bar-chart');
    const rect = container.getBoundingClientRect();
    barMargin = { top: 20, right: 24, bottom: 60, left: 60 };
    barWidth = rect.width - barMargin.left - barMargin.right;
    barHeight = rect.height - barMargin.top - barMargin.bottom;

    barSvg = d3.select('#bar-chart')
        .append('svg')
        .attr('viewBox', `0 0 ${rect.width} ${rect.height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .append('g')
        .attr('transform', `translate(${barMargin.left},${barMargin.top})`);

    barX0 = d3.scaleBand().rangeRound([0, barWidth]).paddingInner(0.25).paddingOuter(0.1);
    barX1 = d3.scaleBand().padding(0.12);
    barY = d3.scaleLog().range([barHeight, 0]).clamp(true);

    // Axes groups
    barSvg.append('g').attr('class', 'axis x-axis').attr('transform', `translate(0,${barHeight})`);
    barSvg.append('g').attr('class', 'axis y-axis');

    // Y-axis label
    barSvg.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('y', -45)
        .attr('x', -barHeight / 2)
        .attr('text-anchor', 'middle')
        .style('fill', COLORS.textMuted)
        .style('font-size', '10px')
        .style('font-family', "'JetBrains Mono', monospace")
        .text('Time (ms) — log scale');

    // Grid lines group
    barSvg.append('g').attr('class', 'grid-lines');

    barChartInitialized = true;
}

function updateBarChart() {
    if (!barChartInitialized) return;

    const data = state.benchmarks;
    const operations = data.map(d => d.operation);
    const metrics = ['cpu_ms', 'gpu_ms'];

    barX0.domain(operations);
    barX1.domain(metrics).rangeRound([0, barX0.bandwidth()]);

    const maxVal = d3.max(data, d => Math.max(d.cpu_ms, d.gpu_ms)) || 100;
    const minVal = d3.min(data, d => Math.min(d.cpu_ms, d.gpu_ms)) || 0.1;
    barY.domain([Math.max(0.05, minVal * 0.5), maxVal * 1.5]);

    const t = d3.transition().duration(600).ease(d3.easeCubicOut);

    // Update axes
    barSvg.select('.x-axis')
        .transition(t)
        .call(d3.axisBottom(barX0))
        .selectAll('text')
        .attr('transform', 'rotate(-25)')
        .style('text-anchor', 'end')
        .style('fill', COLORS.textMuted)
        .style('font-size', '9px');

    barSvg.select('.y-axis')
        .transition(t)
        .call(d3.axisLeft(barY).ticks(6, '.1f').tickFormat(d => {
            if (d >= 1000) return (d/1000).toFixed(0) + 'k';
            if (d >= 1) return d.toFixed(0);
            return d.toFixed(1);
        }))
        .selectAll('text')
        .style('fill', COLORS.textMuted);

    // Grid lines
    const gridData = barY.ticks(6);
    const gridLines = barSvg.select('.grid-lines').selectAll('.grid-line').data(gridData);
    gridLines.enter()
        .append('line').attr('class', 'grid-line')
        .merge(gridLines)
        .transition(t)
        .attr('x1', 0).attr('x2', barWidth)
        .attr('y1', d => barY(d)).attr('y2', d => barY(d));
    gridLines.exit().remove();

    // Bar groups
    const groups = barSvg.selectAll('.bar-group').data(data, d => d.operation);

    const groupsEnter = groups.enter()
        .append('g')
        .attr('class', 'bar-group')
        .attr('transform', d => `translate(${barX0(d.operation)},0)`);

    const groupsMerge = groupsEnter.merge(groups)
        .transition(t)
        .attr('transform', d => `translate(${barX0(d.operation)},0)`);

    groups.exit().transition(t).style('opacity', 0).remove();

    // Individual bars within each group
    const allGroups = barSvg.selectAll('.bar-group');

    allGroups.each(function(d) {
        const group = d3.select(this);
        const bars = group.selectAll('.bar').data(metrics.map(m => ({
            key: m,
            value: d[m],
            op: d.operation,
            data: d
        })), dd => dd.key);

        bars.enter()
            .append('rect')
            .attr('class', 'bar')
            .attr('x', dd => barX1(dd.key))
            .attr('width', barX1.bandwidth())
            .attr('y', barHeight)
            .attr('height', 0)
            .attr('rx', 3)
            .attr('fill', dd => dd.key === 'gpu_ms' ? COLORS.cyan : COLORS.orange)
            .attr('opacity', 0.85)
            .on('mouseover', function(event, dd) {
                d3.select(this).attr('opacity', 1).attr('filter', 'brightness(1.2)');
                showTooltip(event, dd);
            })
            .on('mousemove', (event) => moveTooltip(event))
            .on('mouseout', function() {
                d3.select(this).attr('opacity', 0.85).attr('filter', null);
                hideTooltip();
            })
            .transition(t)
            .attr('y', dd => barY(Math.max(0.05, dd.value)))
            .attr('height', dd => barHeight - barY(Math.max(0.05, dd.value)));

        bars.transition(t)
            .attr('x', dd => barX1(dd.key))
            .attr('width', barX1.bandwidth())
            .attr('y', dd => barY(Math.max(0.05, dd.value)))
            .attr('height', dd => barHeight - barY(Math.max(0.05, dd.value)));
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// CHART 2: BANDWIDTH AREA CHART
// ═══════════════════════════════════════════════════════════════════════════

let bwChartInitialized = false;
let bwSvg, bwX, bwY, bwWidth, bwHeight, bwMargin, bwArea, bwLine;

function initBandwidthChart() {
    const container = document.getElementById('bandwidth-chart');
    const rect = container.getBoundingClientRect();
    bwMargin = { top: 20, right: 24, bottom: 40, left: 55 };
    bwWidth = rect.width - bwMargin.left - bwMargin.right;
    bwHeight = rect.height - bwMargin.top - bwMargin.bottom;

    const svg = d3.select('#bandwidth-chart')
        .append('svg')
        .attr('viewBox', `0 0 ${rect.width} ${rect.height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

    // Gradient definition
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
        .attr('id', 'bw-gradient')
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '0%').attr('y2', '100%');
    gradient.append('stop').attr('offset', '0%').attr('stop-color', COLORS.violet).attr('stop-opacity', 0.25);
    gradient.append('stop').attr('offset', '100%').attr('stop-color', COLORS.violet).attr('stop-opacity', 0.03);

    bwSvg = svg.append('g')
        .attr('transform', `translate(${bwMargin.left},${bwMargin.top})`);

    bwX = d3.scaleLinear().range([0, bwWidth]);
    bwY = d3.scaleLinear().range([bwHeight, 0]);

    bwArea = d3.area()
        .x((d, i) => bwX(i))
        .y0(bwHeight)
        .y1(d => bwY(d.bandwidth_gbps))
        .curve(d3.curveMonotoneX);

    bwLine = d3.line()
        .x((d, i) => bwX(i))
        .y(d => bwY(d.bandwidth_gbps))
        .curve(d3.curveMonotoneX);

    bwSvg.append('g').attr('class', 'axis x-axis').attr('transform', `translate(0,${bwHeight})`);
    bwSvg.append('g').attr('class', 'axis y-axis');

    bwSvg.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('y', -40)
        .attr('x', -bwHeight / 2)
        .attr('text-anchor', 'middle')
        .style('fill', COLORS.textMuted)
        .style('font-size', '10px')
        .style('font-family', "'JetBrains Mono', monospace")
        .text('Bandwidth (GB/s)');

    bwSvg.append('path').attr('class', 'bw-area');
    bwSvg.append('path').attr('class', 'bw-line');
    bwSvg.append('g').attr('class', 'bw-dots');

    bwChartInitialized = true;
}

function updateBandwidthChart() {
    if (!bwChartInitialized) return;

    const data = state.benchmarks;
    if (data.length === 0) return;

    const t = d3.transition().duration(600).ease(d3.easeCubicOut);

    bwX.domain([0, Math.max(1, data.length - 1)]);
    bwY.domain([0, d3.max(data, d => d.bandwidth_gbps) * 1.2]);

    bwSvg.select('.x-axis').transition(t)
        .call(d3.axisBottom(bwX).ticks(data.length).tickFormat(i => {
            return data[i] ? data[i].operation : '';
        }))
        .selectAll('text')
        .attr('transform', 'rotate(-25)')
        .style('text-anchor', 'end')
        .style('fill', COLORS.textMuted)
        .style('font-size', '9px');

    bwSvg.select('.y-axis').transition(t)
        .call(d3.axisLeft(bwY).ticks(5).tickFormat(d => d.toFixed(0)))
        .selectAll('text')
        .style('fill', COLORS.textMuted);

    // Area
    bwSvg.select('.bw-area')
        .datum(data)
        .transition(t)
        .attr('d', bwArea)
        .attr('fill', 'url(#bw-gradient)');

    // Line
    bwSvg.select('.bw-line')
        .datum(data)
        .transition(t)
        .attr('d', bwLine)
        .attr('fill', 'none')
        .attr('stroke', COLORS.violet)
        .attr('stroke-width', 2.5)
        .attr('filter', 'drop-shadow(0 0 4px rgba(124, 58, 237, 0.25))');

    // Dots
    const dots = bwSvg.select('.bw-dots').selectAll('.bw-dot').data(data, (d, i) => i);

    dots.enter()
        .append('circle')
        .attr('class', 'bw-dot')
        .attr('r', 0)
        .attr('fill', COLORS.violet)
        .attr('stroke', '#dceefb')
        .attr('stroke-width', 2)
        .on('mouseover', function(event, d) {
            d3.select(this).transition().duration(150).attr('r', 7);
            showBwTooltip(event, d);
        })
        .on('mousemove', (event) => moveTooltip(event))
        .on('mouseout', function() {
            d3.select(this).transition().duration(150).attr('r', 4.5);
            hideTooltip();
        })
        .transition(t)
        .attr('cx', (d, i) => bwX(i))
        .attr('cy', d => bwY(d.bandwidth_gbps))
        .attr('r', 4.5);

    dots.transition(t)
        .attr('cx', (d, i) => bwX(i))
        .attr('cy', d => bwY(d.bandwidth_gbps));
}

// ═══════════════════════════════════════════════════════════════════════════
// CHART 3: SPEEDUP GAUGE
// ═══════════════════════════════════════════════════════════════════════════

let gaugeInitialized = false;
let gaugeSvg, gaugeArc, gaugeArcBg;

function initSpeedupGauge() {
    const container = document.getElementById('speedup-gauge');
    const rect = container.getBoundingClientRect();
    const size = Math.min(rect.width, 220);
    const radius = size / 2 - 10;

    gaugeSvg = d3.select('#speedup-gauge')
        .append('svg')
        .attr('viewBox', `0 0 ${size} ${size * 0.65}`)
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .append('g')
        .attr('transform', `translate(${size/2},${size * 0.58})`);

    const arcGen = d3.arc()
        .innerRadius(radius - 12)
        .outerRadius(radius)
        .startAngle(-Math.PI * 0.75);

    // Background arc
    gaugeSvg.append('path')
        .attr('class', 'gauge-bg')
        .attr('d', arcGen.endAngle(Math.PI * 0.75)())
        .attr('fill', 'rgba(8, 145, 178, 0.08)');

    // Foreground arc
    gaugeArc = gaugeSvg.append('path')
        .attr('class', 'gauge-fg')
        .attr('d', arcGen.endAngle(-Math.PI * 0.75)())
        .attr('fill', COLORS.cyan);

    // Gradient for arc
    const defs = d3.select('#speedup-gauge svg').append('defs');
    const arcGradient = defs.append('linearGradient')
        .attr('id', 'gauge-gradient')
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '100%').attr('y2', '0%');
    arcGradient.append('stop').attr('offset', '0%').attr('stop-color', COLORS.yellow);
    arcGradient.append('stop').attr('offset', '50%').attr('stop-color', COLORS.green);
    arcGradient.append('stop').attr('offset', '100%').attr('stop-color', COLORS.cyan);

    gaugeArc.attr('fill', 'url(#gauge-gradient)');

    gaugeInitialized = true;
}

function updateSpeedupGauge(latest) {
    if (!gaugeInitialized) return;

    const speedup = latest.speedup;
    const maxGauge = Math.max(150, state.maxSpeedup * 1.2);
    const fraction = Math.min(speedup / maxGauge, 1);
    const endAngle = -Math.PI * 0.75 + fraction * Math.PI * 1.5;

    const arcGen = d3.arc()
        .innerRadius(90)
        .outerRadius(102)
        .startAngle(-Math.PI * 0.75);

    gaugeArc.transition()
        .duration(800)
        .ease(d3.easeCubicOut)
        .attrTween('d', function() {
            const current = d3.select(this).attr('data-end') || (-Math.PI * 0.75);
            const interp = d3.interpolate(+current, endAngle);
            d3.select(this).attr('data-end', endAngle);
            return t => arcGen.endAngle(interp(t))();
        });

    // Update gauge text
    const valueEl = document.getElementById('gauge-speedup-value');
    const currentVal = parseFloat(valueEl.textContent) || 0;
    animateValue(valueEl, currentVal, speedup, 600, 1, '×');

    valueEl.classList.remove('flash-update');
    void valueEl.offsetWidth;
    valueEl.classList.add('flash-update');

    document.getElementById('gauge-op-name').textContent = latest.operation;
}

function resetSpeedupGauge() {
    document.getElementById('gauge-speedup-value').textContent = '—';
    document.getElementById('gauge-op-name').textContent = 'Waiting...';

    if (gaugeInitialized && gaugeArc) {
        const arcGen = d3.arc()
            .innerRadius(90)
            .outerRadius(102)
            .startAngle(-Math.PI * 0.75);

        gaugeArc.transition()
            .duration(400)
            .attr('d', arcGen.endAngle(-Math.PI * 0.75)())
            .attr('data-end', -Math.PI * 0.75);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CHART 4: GPU HEALTH TIMELINE
// ═══════════════════════════════════════════════════════════════════════════

let gpuChartInitialized = false;
let gpuSvg, gpuX, gpuYUtil, gpuYVram, gpuWidth, gpuHeight, gpuMargin;

function initGpuHealthChart() {
    const container = document.getElementById('gpu-health-chart');
    const rect = container.getBoundingClientRect();
    gpuMargin = { top: 20, right: 55, bottom: 30, left: 55 };
    gpuWidth = rect.width - gpuMargin.left - gpuMargin.right;
    gpuHeight = rect.height - gpuMargin.top - gpuMargin.bottom;

    const svg = d3.select('#gpu-health-chart')
        .append('svg')
        .attr('viewBox', `0 0 ${rect.width} ${rect.height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

    // Gradient for VRAM area
    const defs = svg.append('defs');
    const vramGrad = defs.append('linearGradient')
        .attr('id', 'vram-gradient')
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '0%').attr('y2', '100%');
    vramGrad.append('stop').attr('offset', '0%').attr('stop-color', COLORS.green).attr('stop-opacity', 0.2);
    vramGrad.append('stop').attr('offset', '100%').attr('stop-color', COLORS.green).attr('stop-opacity', 0.03);

    gpuSvg = svg.append('g')
        .attr('transform', `translate(${gpuMargin.left},${gpuMargin.top})`);

    gpuX = d3.scaleLinear().range([0, gpuWidth]);
    gpuYUtil = d3.scaleLinear().domain([0, 100]).range([gpuHeight, 0]);
    gpuYVram = d3.scaleLinear().range([gpuHeight, 0]);

    gpuSvg.append('g').attr('class', 'axis x-axis').attr('transform', `translate(0,${gpuHeight})`);
    gpuSvg.append('g').attr('class', 'axis y-axis-left');
    gpuSvg.append('g').attr('class', 'axis y-axis-right').attr('transform', `translate(${gpuWidth},0)`);

    // Left Y label
    gpuSvg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', -42).attr('x', -gpuHeight/2)
        .attr('text-anchor', 'middle')
        .style('fill', COLORS.green).style('font-size', '9px')
        .style('font-family', "'JetBrains Mono', monospace")
        .text('VRAM (MB)');

    // Right Y label
    gpuSvg.append('text')
        .attr('transform', 'rotate(90)')
        .attr('y', -gpuWidth - 42).attr('x', gpuHeight/2)
        .attr('text-anchor', 'middle')
        .style('fill', COLORS.cyan).style('font-size', '9px')
        .style('font-family', "'JetBrains Mono', monospace")
        .text('Utilization (%)');

    gpuSvg.append('path').attr('class', 'vram-area');
    gpuSvg.append('path').attr('class', 'vram-line');
    gpuSvg.append('path').attr('class', 'util-line');

    gpuChartInitialized = true;
}

function updateGpuHealthChart() {
    if (!gpuChartInitialized) return;

    const data = state.gpuStats;
    if (data.length < 2) return;

    const t = d3.transition().duration(200).ease(d3.easeLinear);

    gpuX.domain([0, Math.max(50, data.length - 1)]);
    gpuYVram.domain([0, (data[0]?.vram_total_mb || 4096) * 1.05]);

    gpuSvg.select('.x-axis').transition(t)
        .call(d3.axisBottom(gpuX).ticks(5).tickFormat(d => ''))
        .selectAll('text').style('fill', COLORS.textMuted);

    gpuSvg.select('.y-axis-left').transition(t)
        .call(d3.axisLeft(gpuYVram).ticks(5).tickFormat(d => (d/1024).toFixed(1)+'G'))
        .selectAll('text').style('fill', COLORS.green);

    gpuSvg.select('.y-axis-right').transition(t)
        .call(d3.axisRight(gpuYUtil).ticks(5).tickFormat(d => d+'%'))
        .selectAll('text').style('fill', COLORS.cyan);

    // VRAM area
    const vramArea = d3.area()
        .x((d, i) => gpuX(i))
        .y0(gpuHeight)
        .y1(d => gpuYVram(d.vram_used_mb))
        .curve(d3.curveMonotoneX);

    gpuSvg.select('.vram-area')
        .datum(data)
        .transition(t)
        .attr('d', vramArea)
        .attr('fill', 'url(#vram-gradient)');

    // VRAM line
    const vramLine = d3.line()
        .x((d, i) => gpuX(i))
        .y(d => gpuYVram(d.vram_used_mb))
        .curve(d3.curveMonotoneX);

    gpuSvg.select('.vram-line')
        .datum(data)
        .transition(t)
        .attr('d', vramLine)
        .attr('fill', 'none')
        .attr('stroke', COLORS.green)
        .attr('stroke-width', 2);

    // Utilization line
    const utilLine = d3.line()
        .x((d, i) => gpuX(i))
        .y(d => gpuYUtil(d.gpu_util_pct))
        .curve(d3.curveMonotoneX);

    gpuSvg.select('.util-line')
        .datum(data)
        .transition(t)
        .attr('d', utilLine)
        .attr('fill', 'none')
        .attr('stroke', COLORS.cyan)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '4 2');
}

// ═══════════════════════════════════════════════════════════════════════════
// RESULTS TABLE (in gauge panel)
// ═══════════════════════════════════════════════════════════════════════════

function updateResultsTable(entry) {
    const tbody = document.getElementById('results-tbody');
    const row = document.createElement('tr');
    row.innerHTML = `
        <td style="color:${COLORS.cyan}">${entry.operation}</td>
        <td>${entry.cpu_ms.toFixed(1)}</td>
        <td style="color:${COLORS.cyan}">${entry.gpu_ms.toFixed(1)}</td>
        <td style="color:${COLORS.green};font-weight:600">${entry.speedup.toFixed(1)}×</td>
    `;
    tbody.appendChild(row);

    // Auto-scroll
    const container = tbody.closest('.results-table-container');
    if (container) container.scrollTop = container.scrollHeight;
}

function clearResultsTable() {
    const tbody = document.getElementById('results-tbody');
    if (tbody) tbody.innerHTML = '';
}

// ═══════════════════════════════════════════════════════════════════════════
// TOOLTIPS
// ═══════════════════════════════════════════════════════════════════════════

function showTooltip(event, d) {
    const label = d.key === 'gpu_ms' ? 'GPU' : 'CPU';
    const color = d.key === 'gpu_ms' ? COLORS.cyan : COLORS.orange;
    tooltip.html(`
        <div class="tt-title" style="color:${color}">${d.op} — ${label}</div>
        <div class="tt-row"><span class="tt-label">Time</span><span class="tt-value">${d.value.toFixed(2)} ms</span></div>
        <div class="tt-row"><span class="tt-label">Speedup</span><span class="tt-value">${d.data.speedup.toFixed(1)}×</span></div>
        <div class="tt-row"><span class="tt-label">Bandwidth</span><span class="tt-value">${d.data.bandwidth_gbps.toFixed(1)} GB/s</span></div>
        <div class="tt-row"><span class="tt-label">Rows</span><span class="tt-value">${formatNumber(d.data.rows)}</span></div>
    `).classed('visible', true);
    moveTooltip(event);
}

function showBwTooltip(event, d) {
    tooltip.html(`
        <div class="tt-title" style="color:${COLORS.violet}">${d.operation}</div>
        <div class="tt-row"><span class="tt-label">PCIe BW</span><span class="tt-value">${d.bandwidth_gbps.toFixed(1)} GB/s</span></div>
        <div class="tt-row"><span class="tt-label">GPU Time</span><span class="tt-value">${d.gpu_ms.toFixed(2)} ms</span></div>
        <div class="tt-row"><span class="tt-label">Data Size</span><span class="tt-value">${formatBytes(d.rows * 8)}</span></div>
    `).classed('visible', true);
    moveTooltip(event);
}

function moveTooltip(event) {
    const x = event.pageX + 14;
    const y = event.pageY - 14;
    tooltip.style('left', x + 'px').style('top', y + 'px');
}

function hideTooltip() {
    tooltip.classed('visible', false);
}

// ═══════════════════════════════════════════════════════════════════════════
// UI HELPERS
// ═══════════════════════════════════════════════════════════════════════════

function setConnectionStatus(status, label) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    dot.className = 'status-dot ' + status;
    text.textContent = label;
}

function showToast(message) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3500);
}

function animateValue(el, from, to, duration, decimals, suffix) {
    const start = performance.now();
    const update = (now) => {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
        const current = from + (to - from) * eased;
        el.textContent = current.toFixed(decimals) + (suffix || '');
        if (progress < 1) requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
}

function formatNumber(n) {
    return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function formatBytes(bytes) {
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + ' GB';
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    return (bytes / 1e3).toFixed(0) + ' KB';
}

function clearChart(id) {
    const el = document.getElementById(id);
    const svg = el?.querySelector('svg');
    if (svg) svg.remove();

    // Re-initialize
    if (id === 'bar-chart') { barChartInitialized = false; initBarChart(); }
    if (id === 'bandwidth-chart') { bwChartInitialized = false; initBandwidthChart(); }
    if (id === 'gpu-health-chart') { gpuChartInitialized = false; initGpuHealthChart(); }
}

function initCharts() {
    // Defer initialization to ensure containers have dimensions
    requestAnimationFrame(() => {
        initBarChart();
        initBandwidthChart();
        initSpeedupGauge();
        initGpuHealthChart();
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// OPTIONAL: WebSocket / SSE connection (for live C++ backend)
// ═══════════════════════════════════════════════════════════════════════════

function connectWebSocket(url) {
    if (state.wsConnection) state.wsConnection.close();

    try {
        const ws = new WebSocket(url || 'ws://localhost:9002');
        state.wsConnection = ws;

        ws.onopen = () => {
            state.mode = 'connected';
            setConnectionStatus('connected', 'Live');
            showToast('Connected to benchmark server');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'benchmark') {
                    if (!data.speedup && data.cpu_ms && data.gpu_ms) {
                        data.speedup = data.cpu_ms / data.gpu_ms;
                    }
                    ingestBenchmark(data);
                } else if (data.type === 'gpu_stats') {
                    ingestGpuStats(data);
                }
            } catch (e) { console.warn('WS parse error:', e); }
        };

        ws.onclose = () => {
            state.mode = 'idle';
            setConnectionStatus('disconnected', 'Disconnected');
        };

        ws.onerror = () => {
            setConnectionStatus('disconnected', 'Error');
        };
    } catch (e) {
        console.warn('WebSocket not available:', e);
    }
}

// SSE fallback for the Python bridge
function connectSSE(url) {
    const evtSource = new EventSource(url || 'http://localhost:8080/stream');

    evtSource.onopen = () => {
        state.mode = 'connected';
        setConnectionStatus('connected', 'Live (SSE)');
    };

    evtSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'benchmark') {
                if (!data.speedup && data.cpu_ms && data.gpu_ms) {
                    data.speedup = data.cpu_ms / data.gpu_ms;
                }
                ingestBenchmark(data);
            } else if (data.type === 'gpu_stats') {
                ingestGpuStats(data);
            }
        } catch (e) { console.warn('SSE parse error:', e); }
    };

    evtSource.onerror = () => {
        setConnectionStatus('disconnected', 'SSE Error');
    };
}

// Expose for console usage
window.connectWebSocket = connectWebSocket;
window.connectSSE = connectSSE;


