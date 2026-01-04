/**
 * CEWP Dashboard - Main JavaScript
 * =================================
 * Interactive visualization using Deck.gl + MapLibre GL JS
 * Connects to FastAPI backend for prediction data
 */

// =============================================================================
// 1. CONFIGURATION & STATE
// =============================================================================

const CONFIG = {
    API_BASE: window.location.origin,  // Same origin as served page
    MAP_CENTER: [20.9, 6.6],           // CAR center [lng, lat]
    MAP_ZOOM: 5.5,
    H3_RESOLUTION: 5,
    TILE_SOURCES: {
        // Free raster tile sources (no API key required)
        cartodb_positron: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
        cartodb_dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
        osm: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        stamen_terrain: 'https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.png'
    }
};

// Application state
const STATE = {
    currentHorizon: '14d',
    currentDate: null,
    availableDates: [],
    predictions: null,
    rivers: null,
    roads: null,
    events: null,
    hexgrid: null,
    
    // Layer visibility
    layers: {
        hexagons: true,
        rivers: false,
        roads: false,
        events: false
    },
    
    // Map instances
    map: null,
    deckOverlay: null,
    
    // Loading state
    isLoading: true
};

// =============================================================================
// 2. DATA SOURCES (Static Content)
// =============================================================================

const dataSources = {
    'Environmental': ['CHIRPS (Precipitation)', 'ERA5 (Climate)', 'MODIS (Vegetation)', 'VIIRS (Nighttime Lights)', 'JRC (Surface Water)'],
    'Conflict': ['ACLED (Events/Fatalities)', 'GDELT (Media/Tone)', 'IODA (Internet Outages)'],
    'Socio-Political': ['EPR (Ethnic Power Relations)', 'IOM DTM (Displacement)', 'FEWS NET (IPC Phases)'],
    'Economic': ['Yahoo Finance (Gold/Oil)', 'Local Market Prices'],
    'Infrastructure': ['GRIP4 (Roads)', 'HydroRIVERS', 'IPIS (Mines)', 'Settlements'],
    'Demographics': ['WorldPop (Population Density)']
};

const modelPerformance = {
    '14d': { xgb_auc: 0.88, lgb_auc: 0.87, xgb_recall: 0.65, lgb_recall: 0.64 },
    '1m':  { xgb_auc: 0.85, lgb_auc: 0.84, xgb_recall: 0.58, lgb_recall: 0.56 },
    '3m':  { xgb_auc: 0.79, lgb_auc: 0.78, xgb_recall: 0.45, lgb_recall: 0.44 }
};

const phaseDetails = {
    'static': { 
        title: 'Static Ingestion', 
        desc: 'Processes invariant geography. Generates the H3 Grid (3,407 cells), computes distances to roads/rivers, and generates terrain derivatives (Slope/TRI) from Copernicus DEM.' 
    },
    'dynamic': { 
        title: 'Dynamic Ingestion', 
        desc: 'Fetches time-series data. Handles API logic for ACLED, GDELT (BigQuery), and GEE. Performs spatial disaggregation to map administrative data (like IPC phases) onto the grid.' 
    },
    'engineering': { 
        title: 'Feature Engineering', 
        desc: 'The core transformation engine. Computes 14-day temporal lags, exponential decays (30d/90d), and environmental anomalies relative to a rolling baseline. Integrates EPR status entropy.' 
    },
    'modeling': { 
        title: 'Modeling Strategy', 
        desc: 'Trains a Two-Stage Hurdle Ensemble. Stage 1 trains thematic sub-models (Geography, Socio-Political, etc.). Stage 2 stacks them using Meta-Learners (Logistic & Ridge) to calibrate probability and intensity.' 
    }
};

// =============================================================================
// 3. API FUNCTIONS
// =============================================================================

async function fetchJSON(endpoint) {
    try {
        const response = await fetch(`${CONFIG.API_BASE}${endpoint}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        return null;
    }
}

async function checkAPIHealth() {
    const health = await fetchJSON('/api/health');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    
    if (health && health.status === 'healthy') {
        statusDot.classList.remove('bg-yellow-400', 'bg-red-500', 'animate-pulse');
        statusDot.classList.add('bg-green-400');
        statusText.textContent = 'Connected';
        return true;
    } else {
        statusDot.classList.remove('bg-yellow-400', 'bg-green-400', 'animate-pulse');
        statusDot.classList.add('bg-red-500');
        statusText.textContent = 'Disconnected';
        return false;
    }
}

async function fetchAvailableDates() {
    const data = await fetchJSON('/api/dates');
    if (data && data.dates) {
        STATE.availableDates = data.dates;
        STATE.currentDate = data.latest;
        initializeDateControls();
    }
}

async function fetchPredictions() {
    updateLoadingStatus('Fetching predictions...');
    const params = new URLSearchParams({
        horizon: STATE.currentHorizon
    });
    if (STATE.currentDate) {
        params.append('date', STATE.currentDate);
    }
    
    const data = await fetchJSON(`/api/predictions?${params}`);
    if (data) {
        STATE.predictions = data;
        console.log(`Loaded ${data.features?.length || 0} prediction hexagons`);
    }
    return data;
}

async function fetchStaticFeatures() {
    updateLoadingStatus('Loading rivers and roads...');
    const data = await fetchJSON('/api/features/static');
    if (data) {
        STATE.rivers = data.rivers || null;
        STATE.roads = data.roads || null;
    }
    return data;
}

async function fetchConflictEvents() {
    const data = await fetchJSON('/api/events?limit=500');
    if (data) {
        STATE.events = data;
    }
    return data;
}

async function fetchStats() {
    const stats = await fetchJSON('/api/stats');
    if (stats) {
        if (stats.hexagon_count) {
            document.getElementById('metric-hexagons').textContent = stats.hexagon_count.toLocaleString();
        }
        if (stats.event_count) {
            document.getElementById('metric-events').textContent = stats.event_count.toLocaleString();
        }
    }
}

// =============================================================================
// 4. COLOR UTILITIES
// =============================================================================

function probabilityToColor(prob) {
    // Color ramp: Green -> Yellow -> Orange -> Red -> Dark Red
    const colors = [
        [16, 185, 129],    // Green (0.0)
        [251, 191, 36],    // Yellow (0.25)
        [249, 115, 22],    // Orange (0.5)
        [239, 68, 68],     // Red (0.75)
        [127, 29, 29]      // Dark Red (1.0)
    ];
    
    const p = Math.max(0, Math.min(1, prob));
    const idx = p * (colors.length - 1);
    const lower = Math.floor(idx);
    const upper = Math.min(lower + 1, colors.length - 1);
    const t = idx - lower;
    
    return [
        Math.round(colors[lower][0] + t * (colors[upper][0] - colors[lower][0])),
        Math.round(colors[lower][1] + t * (colors[upper][1] - colors[lower][1])),
        Math.round(colors[lower][2] + t * (colors[upper][2] - colors[lower][2])),
        180  // Alpha
    ];
}

// =============================================================================
// 5. DECK.GL LAYERS
// =============================================================================

function createHexagonLayer() {
    if (!STATE.predictions || !STATE.predictions.features) {
        return null;
    }
    
    return new deck.GeoJsonLayer({
        id: 'hexagons',
        data: STATE.predictions,
        visible: STATE.layers.hexagons,
        
        // Polygon styling
        filled: true,
        stroked: true,
        extruded: false,
        
        getFillColor: f => {
            const prob = f.properties.pred_proba || 0;
            return probabilityToColor(prob);
        },
        getLineColor: [100, 100, 100, 100],
        getLineWidth: 1,
        lineWidthMinPixels: 0.5,
        
        // Picking/Interaction
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 100],
        
        // Events
        onHover: info => handleHexHover(info),
        onClick: info => handleHexClick(info),
        
        // Update triggers
        updateTriggers: {
            getFillColor: [STATE.currentHorizon, STATE.currentDate]
        }
    });
}

function createRiversLayer() {
    if (!STATE.rivers) return null;
    
    return new deck.GeoJsonLayer({
        id: 'rivers',
        data: STATE.rivers,
        visible: STATE.layers.rivers,
        
        stroked: true,
        filled: false,
        getLineColor: [30, 144, 255, 200],  // Dodger blue
        getLineWidth: 2,
        lineWidthMinPixels: 1,
        lineWidthMaxPixels: 4
    });
}

function createRoadsLayer() {
    if (!STATE.roads) return null;
    
    return new deck.GeoJsonLayer({
        id: 'roads',
        data: STATE.roads,
        visible: STATE.layers.roads,
        
        stroked: true,
        filled: false,
        getLineColor: [139, 69, 19, 180],  // Saddle brown
        getLineWidth: 1,
        lineWidthMinPixels: 0.5,
        lineWidthMaxPixels: 3
    });
}

function createEventsLayer() {
    if (!STATE.events || !STATE.events.features) return null;
    
    return new deck.GeoJsonLayer({
        id: 'events',
        data: STATE.events,
        visible: STATE.layers.events,
        
        // Point styling
        pointType: 'circle',
        filled: true,
        stroked: true,
        
        getPointRadius: f => {
            const fatalities = f.properties.fatalities || 0;
            return Math.max(4, Math.min(20, 4 + fatalities * 0.5));
        },
        getFillColor: f => {
            const fatalities = f.properties.fatalities || 0;
            if (fatalities === 0) return [255, 165, 0, 180];  // Orange
            if (fatalities < 5) return [255, 69, 0, 200];      // Red-orange
            return [139, 0, 0, 220];                           // Dark red
        },
        getLineColor: [255, 255, 255, 200],
        getLineWidth: 1,
        
        pointRadiusMinPixels: 3,
        pointRadiusMaxPixels: 20,
        
        pickable: true,
        onHover: info => handleEventHover(info)
    });
}

function buildDeckLayers() {
    const layers = [];
    
    // Order matters: bottom to top
    const roadsLayer = createRoadsLayer();
    if (roadsLayer) layers.push(roadsLayer);
    
    const riversLayer = createRiversLayer();
    if (riversLayer) layers.push(riversLayer);
    
    const hexLayer = createHexagonLayer();
    if (hexLayer) layers.push(hexLayer);
    
    const eventsLayer = createEventsLayer();
    if (eventsLayer) layers.push(eventsLayer);
    
    return layers;
}

// =============================================================================
// 6. MAP INITIALIZATION
// =============================================================================

function initializeMap() {
    updateLoadingStatus('Initializing map...');
    
    // Create MapLibre GL map with raster tiles
    STATE.map = new maplibregl.Map({
        container: 'map-container',
        style: {
            version: 8,
            sources: {
                'carto-light': {
                    type: 'raster',
                    tiles: [
                        'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
                        'https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
                        'https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'
                    ],
                    tileSize: 256,
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                }
            },
            layers: [
                {
                    id: 'carto-light-layer',
                    type: 'raster',
                    source: 'carto-light',
                    minzoom: 0,
                    maxzoom: 19
                }
            ]
        },
        center: CONFIG.MAP_CENTER,
        zoom: CONFIG.MAP_ZOOM,
        minZoom: 3,
        maxZoom: 12
    });
    
    // Add navigation controls
    STATE.map.addControl(new maplibregl.NavigationControl(), 'top-right');
    STATE.map.addControl(new maplibregl.ScaleControl({ maxWidth: 200, unit: 'metric' }), 'bottom-left');
    
    // Track mouse position
    STATE.map.on('mousemove', (e) => {
        const coords = e.lngLat;
        document.getElementById('cursor-coords').textContent = 
            `Lat: ${coords.lat.toFixed(4)}, Lng: ${coords.lng.toFixed(4)}`;
    });
    
    // Initialize Deck.gl overlay once map loads
    STATE.map.on('load', () => {
        initializeDeckOverlay();
    });
}

function initializeDeckOverlay() {
    STATE.deckOverlay = new deck.MapboxOverlay({
        interleaved: false,
        layers: buildDeckLayers()
    });
    
    STATE.map.addControl(STATE.deckOverlay);
    
    // Hide loading overlay
    hideLoadingOverlay();
}

function updateDeckLayers() {
    if (STATE.deckOverlay) {
        STATE.deckOverlay.setProps({
            layers: buildDeckLayers()
        });
    }
}

// =============================================================================
// 7. EVENT HANDLERS
// =============================================================================

function handleHexHover(info) {
    const hexInfo = document.getElementById('hex-info');
    
    if (!info.object) {
        hexInfo.innerHTML = '<p class="text-slate-500 italic">Hover over a hexagon to see details.</p>';
        return;
    }
    
    const props = info.object.properties;
    const prob = props.pred_proba || 0;
    const fatalities = props.pred_fatalities || 0;
    const h3Index = props.h3_index;
    
    // Determine risk level
    let riskLevel, riskColor;
    if (prob < 0.2) {
        riskLevel = 'Low';
        riskColor = 'text-green-600';
    } else if (prob < 0.5) {
        riskLevel = 'Moderate';
        riskColor = 'text-yellow-600';
    } else if (prob < 0.75) {
        riskLevel = 'High';
        riskColor = 'text-orange-600';
    } else {
        riskLevel = 'Critical';
        riskColor = 'text-red-600';
    }
    
    hexInfo.innerHTML = `
        <div class="space-y-2">
            <div class="flex justify-between">
                <span class="text-slate-500">Risk Level:</span>
                <span class="font-bold ${riskColor}">${riskLevel}</span>
            </div>
            <div class="flex justify-between">
                <span class="text-slate-500">Probability:</span>
                <span class="font-semibold">${(prob * 100).toFixed(1)}%</span>
            </div>
            <div class="flex justify-between">
                <span class="text-slate-500">Exp. Fatalities:</span>
                <span class="font-semibold">${fatalities.toFixed(1)}</span>
            </div>
            <div class="pt-2 border-t border-blue-200">
                <span class="text-xs text-slate-400 font-mono">H3: ${h3Index}</span>
            </div>
        </div>
    `;
}

function handleHexClick(info) {
    if (!info.object) return;
    
    // Could expand this to show detailed analysis panel
    console.log('Clicked hex:', info.object.properties);
}

function handleEventHover(info) {
    // Could show event tooltip
    if (info.object) {
        console.log('Event:', info.object.properties);
    }
}

// =============================================================================
// 8. UI CONTROLS
// =============================================================================

function initializeDateControls() {
    const datePicker = document.getElementById('date-picker');
    const dateSlider = document.getElementById('date-slider');
    
    if (STATE.availableDates.length > 0) {
        // Set date picker value
        datePicker.value = STATE.currentDate;
        
        // Configure slider
        dateSlider.max = STATE.availableDates.length - 1;
        dateSlider.value = 0;  // Latest date
        
        // Update labels
        document.getElementById('slider-start').textContent = STATE.availableDates[STATE.availableDates.length - 1];
        document.getElementById('slider-end').textContent = STATE.availableDates[0];
    }
    
    // Event listeners
    datePicker.addEventListener('change', async (e) => {
        STATE.currentDate = e.target.value;
        await refreshPredictions();
    });
    
    dateSlider.addEventListener('input', async (e) => {
        const idx = parseInt(e.target.value);
        STATE.currentDate = STATE.availableDates[idx];
        datePicker.value = STATE.currentDate;
        await refreshPredictions();
    });
}

function initializeLayerToggles() {
    const toggles = {
        'layer-hexagons': 'hexagons',
        'layer-rivers': 'rivers',
        'layer-roads': 'roads',
        'layer-events': 'events'
    };
    
    Object.entries(toggles).forEach(([elementId, layerKey]) => {
        const checkbox = document.getElementById(elementId);
        const toggle = checkbox.nextElementSibling;
        
        // Set initial state
        if (STATE.layers[layerKey]) {
            checkbox.checked = true;
            toggle.classList.add('bg-blue-500');
            toggle.classList.remove('bg-slate-300');
            toggle.querySelector('.toggle-dot').classList.add('right-0.5');
            toggle.querySelector('.toggle-dot').classList.remove('left-0.5');
        }
        
        checkbox.addEventListener('change', () => {
            STATE.layers[layerKey] = checkbox.checked;
            
            // Update toggle visual
            if (checkbox.checked) {
                toggle.classList.add('bg-blue-500');
                toggle.classList.remove('bg-slate-300');
                toggle.querySelector('.toggle-dot').classList.add('right-0.5');
                toggle.querySelector('.toggle-dot').classList.remove('left-0.5');
            } else {
                toggle.classList.remove('bg-blue-500');
                toggle.classList.add('bg-slate-300');
                toggle.querySelector('.toggle-dot').classList.remove('right-0.5');
                toggle.querySelector('.toggle-dot').classList.add('left-0.5');
            }
            
            // Update map layers
            updateDeckLayers();
        });
    });
}

// Global function for horizon buttons
window.setHorizon = async function(horizon) {
    STATE.currentHorizon = horizon;
    
    // Update button styles
    document.querySelectorAll('.horizon-btn').forEach(btn => {
        if (btn.dataset.hz === horizon) {
            btn.classList.add('bg-blue-500', 'text-white');
            btn.classList.remove('bg-slate-100', 'text-slate-600', 'hover:bg-slate-200');
        } else {
            btn.classList.remove('bg-blue-500', 'text-white');
            btn.classList.add('bg-slate-100', 'text-slate-600', 'hover:bg-slate-200');
        }
    });
    
    await refreshPredictions();
};

// Global refresh function
window.refreshPredictions = async function() {
    showLoadingOverlay();
    await fetchPredictions();
    updateDeckLayers();
    hideLoadingOverlay();
};

// =============================================================================
// 9. LOADING STATE
// =============================================================================

function updateLoadingStatus(message) {
    const statusEl = document.getElementById('loading-status');
    if (statusEl) {
        statusEl.textContent = message;
    }
}

function showLoadingOverlay() {
    const overlay = document.getElementById('map-loading');
    if (overlay) {
        overlay.classList.remove('hidden');
    }
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('map-loading');
    if (overlay) {
        overlay.classList.add('hidden');
    }
    STATE.isLoading = false;
}

// =============================================================================
// 10. CHARTS (Performance & Data Sources)
// =============================================================================

function initializeCharts() {
    // Data Sources Doughnut Chart
    const ctxData = document.getElementById('dataChart');
    if (ctxData) {
        window.dataChart = new Chart(ctxData.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: Object.keys(dataSources),
                datasets: [{
                    data: Object.values(dataSources).map(arr => arr.length),
                    backgroundColor: ['#10b981', '#ef4444', '#f59e0b', '#3b82f6', '#6366f1', '#8b5cf6'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 10 } } }
                },
                onClick: (e, elements) => {
                    if (elements.length > 0) {
                        const label = window.dataChart.data.labels[elements[0].index];
                        updateSourceList(label);
                    }
                }
            }
        });
        
        updateSourceList('Environmental');
    }
    
    // PR-AUC Chart
    const ctxAuc = document.getElementById('aucChart');
    if (ctxAuc) {
        window.aucChart = new Chart(ctxAuc.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['XGBoost', 'LightGBM'],
                datasets: [{
                    label: 'PR-AUC Score',
                    data: [modelPerformance['14d'].xgb_auc, modelPerformance['14d'].lgb_auc],
                    backgroundColor: ['#3b82f6', '#94a3b8'],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: false, min: 0.5, max: 1.0 } },
                plugins: { legend: { display: false } }
            }
        });
    }
    
    // Recall Chart
    const ctxRecall = document.getElementById('recallChart');
    if (ctxRecall) {
        window.recallChart = new Chart(ctxRecall.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['XGBoost', 'LightGBM'],
                datasets: [{
                    label: 'Top-10% Recall',
                    data: [modelPerformance['14d'].xgb_recall, modelPerformance['14d'].lgb_recall],
                    backgroundColor: ['#10b981', '#cbd5e1'],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, max: 1.0 } },
                plugins: { legend: { display: false } }
            }
        });
    }
}

function updateSourceList(category) {
    const listContainer = document.getElementById('source-list');
    if (!listContainer) return;
    
    const sources = dataSources[category] || [];
    let html = `<h5 class="font-bold text-slate-700 text-sm mb-2">${category} Sources</h5><ul class="text-sm space-y-2">`;
    sources.forEach(s => {
        html += `<li class="flex items-center"><i class="fa-solid fa-database text-slate-400 mr-2"></i> ${s}</li>`;
    });
    html += '</ul>';
    listContainer.innerHTML = html;
}

// Global function for performance horizon buttons
window.updatePerfHorizon = function(hz) {
    document.querySelectorAll('.perf-horizon-btn').forEach(btn => {
        if (btn.dataset.hz === hz) {
            btn.classList.add('bg-blue-500', 'text-white');
            btn.classList.remove('text-slate-600', 'hover:bg-slate-100');
        } else {
            btn.classList.remove('bg-blue-500', 'text-white');
            btn.classList.add('text-slate-600', 'hover:bg-slate-100');
        }
    });
    
    if (window.aucChart) {
        window.aucChart.data.datasets[0].data = [modelPerformance[hz].xgb_auc, modelPerformance[hz].lgb_auc];
        window.aucChart.update();
    }
    if (window.recallChart) {
        window.recallChart.data.datasets[0].data = [modelPerformance[hz].xgb_recall, modelPerformance[hz].lgb_recall];
        window.recallChart.update();
    }
};

// =============================================================================
// 11. PHASE DETAILS
// =============================================================================

window.showPhaseDetails = function(phase) {
    const detailBox = document.getElementById('phase-detail');
    const title = document.getElementById('phase-title');
    const desc = document.getElementById('phase-desc');
    
    if (phaseDetails[phase]) {
        title.textContent = phaseDetails[phase].title;
        desc.textContent = phaseDetails[phase].desc;
        detailBox.classList.remove('hidden');
        detailBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
};

// =============================================================================
// 12. MAIN INITIALIZATION
// =============================================================================

async function initializeApp() {
    console.log('🚀 CEWP Dashboard Initializing...');
    
    // Check API health
    const apiHealthy = await checkAPIHealth();
    if (!apiHealthy) {
        console.warn('API not available - some features may be limited');
    }
    
    // Fetch initial data
    await fetchAvailableDates();
    await fetchStats();
    
    // Initialize map
    initializeMap();
    
    // Load data in parallel
    await Promise.all([
        fetchPredictions(),
        fetchStaticFeatures(),
        fetchConflictEvents()
    ]);
    
    // Initialize UI
    initializeLayerToggles();
    initializeCharts();
    
    // Update layers once data is loaded
    if (STATE.deckOverlay) {
        updateDeckLayers();
    }
    
    console.log('✅ Dashboard initialization complete');
}

// Run on DOM ready
document.addEventListener('DOMContentLoaded', initializeApp);
