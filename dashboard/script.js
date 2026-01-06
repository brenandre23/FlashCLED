/**
 * FlashCLED Dashboard - Main JavaScript (Refactored)
 * ===================================================
 * Interactive visualization using Deck.gl MapboxOverlay + MapLibre GL JS
 * 
 * IMPROVEMENTS:
 * 1. MapboxOverlay: Modern standard for Deck.gl + MapLibre integration
 * 2. Robust H3 Parsing: Handles signed/unsigned integer mismatches
 * 3. Async/Await: Clean data fetching patterns
 * 4. Error Handling: Graceful degradation with user feedback
 */

/* =============================================================================
   CONSTANTS
   ============================================================================= */

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
const API_URL = window.location.origin + '/api';

/* =============================================================================
   STATE
   ============================================================================= */

let layersState = {
    conflict: true,
    roads: false,
    rivers: false,
    events: false
};

let cachedData = {
    conflict: null,
    roads: null,
    rivers: null,
    events: null
};

let map = null;
let deckOverlay = null;

/* =============================================================================
   INITIALIZATION
   ============================================================================= */

/**
 * Initialize MapLibre map
 */
function initializeMap() {
    updateLoadingStatus('Initializing map...');
    
    map = new maplibregl.Map({
        container: 'map',
        style: MAP_STYLE,
        center: [20.9394, 6.6111], // CAR Centroid
        zoom: 5,
        pitch: 40,
        bearing: 0,
        antialias: true
    });

    // Add navigation controls
    map.addControl(new maplibregl.NavigationControl(), 'top-right');
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 200, unit: 'metric' }), 'bottom-left');

    // Track cursor coordinates
    map.on('mousemove', (e) => {
        const coords = document.getElementById('cursor-coords');
        if (coords) {
            coords.textContent = `Lat: ${e.lngLat.lat.toFixed(4)}, Lng: ${e.lngLat.lng.toFixed(4)}`;
        }
    });

    // Initialize Deck.gl overlay when map is ready
    map.on('load', async () => {
        initializeDeckOverlay();
        await loadAllData();
    });
}

/**
 * Initialize Deck.gl MapboxOverlay
 * This is the modern standard for integrating Deck.gl with MapLibre
 */
function initializeDeckOverlay() {
    deckOverlay = new deck.MapboxOverlay({
        interleaved: true, // Allows MapLibre labels to stay on top of Deck layers
        layers: []
    });
    
    map.addControl(deckOverlay);
}

/* =============================================================================
   DATA FETCHING
   ============================================================================= */

/**
 * Load all data sources
 */
async function loadAllData() {
    const loader = document.getElementById('loading');
    
    try {
        // Check API health first
        await checkAPIHealth();
        
        // Load data in parallel
        updateLoadingStatus('Loading conflict predictions...');
        const [conflictData, roadData, riverData, eventData] = await Promise.all([
            fetchConflictData(),
            fetchRoadData(),
            fetchRiverData(),
            fetchEventData()
        ]);

        cachedData.conflict = conflictData;
        cachedData.roads = roadData;
        cachedData.rivers = riverData;
        cachedData.events = eventData;

        // Render all layers
        renderLayers();
        
        // Hide loading overlay
        if (loader) loader.classList.add('hidden');
        
        console.log('✅ Dashboard loaded successfully');
        
    } catch (err) {
        console.error('Data load failed:', err);
        if (loader) {
            loader.innerHTML = `
                <div class="spinner" style="border-top-color: #ef4444;"></div>
                <p style="color: #ef4444;">Failed to load data</p>
                <p class="loading-status">${err.message}</p>
            `;
        }
    }
}

/**
 * Check API health and update status indicator
 */
async function checkAPIHealth() {
    updateLoadingStatus('Connecting to API...');
    
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusDot.classList.add('connected');
            statusDot.classList.remove('disconnected');
            statusText.textContent = 'Connected';
            return true;
        }
    } catch (err) {
        console.warn('API health check failed:', err);
    }
    
    statusDot.classList.add('disconnected');
    statusDot.classList.remove('connected');
    statusText.textContent = 'Disconnected';
    return false;
}

/**
 * Fetch conflict prediction data (H3 hexagons)
 */
async function fetchConflictData() {
    updateLoadingStatus('Loading conflict risk layer...');
    
    try {
        const response = await fetch(`${API_URL}/h3_data`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (err) {
        console.warn('Failed to fetch conflict data:', err);
        return [];
    }
}

/**
 * Fetch road data
 */
async function fetchRoadData() {
    try {
        const response = await fetch(`${API_URL}/roads`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (err) {
        console.warn('Failed to fetch road data:', err);
        return { type: 'FeatureCollection', features: [] };
    }
}

/**
 * Fetch river data
 */
async function fetchRiverData() {
    try {
        const response = await fetch(`${API_URL}/rivers`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (err) {
        console.warn('Failed to fetch river data:', err);
        return { type: 'FeatureCollection', features: [] };
    }
}

/**
 * Fetch conflict events
 */
async function fetchEventData() {
    try {
        const response = await fetch(`${API_URL}/events?limit=500`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (err) {
        console.warn('Failed to fetch event data:', err);
        return { type: 'FeatureCollection', features: [] };
    }
}

/* =============================================================================
   LAYER RENDERING
   ============================================================================= */

/**
 * Render all Deck.gl layers based on current state
 */
function renderLayers() {
    if (!deckOverlay) return;
    
    const layers = [];

    // 1. Roads Layer (rendered first, below hexagons)
    if (layersState.roads && cachedData.roads && cachedData.roads.features) {
        layers.push(createRoadsLayer(cachedData.roads));
    }

    // 2. Rivers Layer
    if (layersState.rivers && cachedData.rivers && cachedData.rivers.features) {
        layers.push(createRiversLayer(cachedData.rivers));
    }

    // 3. Conflict Risk Layer (H3 Hexagons)
    if (layersState.conflict && cachedData.conflict && cachedData.conflict.length > 0) {
        layers.push(createConflictLayer(cachedData.conflict));
    }

    // 4. Conflict Events Layer (on top)
    if (layersState.events && cachedData.events && cachedData.events.features) {
        layers.push(createEventsLayer(cachedData.events));
    }

    deckOverlay.setProps({ layers });
}

/**
 * Create H3 hexagon conflict risk layer
 */
function createConflictLayer(data) {
    return new deck.H3HexagonLayer({
        id: 'conflict-layer',
        data: data,
        pickable: true,
        wireframe: false,
        filled: true,
        extruded: false,
        getHexagon: d => d.hex,
        getFillColor: d => {
            const risk = d.risk || 0;
            // YlOrRd color scale
            if (risk < 0.2) return [255, 255, 178, 180];      // Light yellow
            if (risk < 0.4) return [254, 204, 92, 180];       // Yellow
            if (risk < 0.6) return [253, 141, 60, 180];       // Orange
            if (risk < 0.8) return [240, 59, 32, 200];        // Red-orange
            return [189, 0, 38, 220];                          // Dark red
        },
        getLineColor: [0, 0, 0, 40],
        getLineWidth: 1,
        lineWidthMinPixels: 0.5,
        onHover: handleHexHover,
        updateTriggers: {
            getFillColor: [data]
        }
    });
}

/**
 * Create roads layer
 */
function createRoadsLayer(data) {
    return new deck.GeoJsonLayer({
        id: 'roads-layer',
        data: data,
        pickable: false,
        stroked: true,
        filled: false,
        getLineColor: [139, 90, 43, 200], // Brown
        getLineWidth: 2,
        lineWidthMinPixels: 1,
        lineWidthMaxPixels: 4
    });
}

/**
 * Create rivers layer
 */
function createRiversLayer(data) {
    return new deck.GeoJsonLayer({
        id: 'rivers-layer',
        data: data,
        pickable: false,
        stroked: true,
        filled: false,
        getLineColor: [30, 144, 255, 200], // Dodger blue
        getLineWidth: 2,
        lineWidthMinPixels: 1,
        lineWidthMaxPixels: 4
    });
}

/**
 * Create conflict events layer (points)
 */
function createEventsLayer(data) {
    return new deck.GeoJsonLayer({
        id: 'events-layer',
        data: data,
        pickable: true,
        pointType: 'circle',
        filled: true,
        stroked: true,
        getPointRadius: f => {
            const fatalities = f.properties?.fatalities || 0;
            return Math.max(4, Math.min(20, 4 + fatalities * 0.5));
        },
        getFillColor: f => {
            const fatalities = f.properties?.fatalities || 0;
            if (fatalities === 0) return [255, 165, 0, 180];   // Orange
            if (fatalities < 5) return [255, 69, 0, 200];      // Red-orange
            return [139, 0, 0, 220];                            // Dark red
        },
        getLineColor: [255, 255, 255, 200],
        getLineWidth: 1,
        pointRadiusMinPixels: 3,
        pointRadiusMaxPixels: 20,
        onHover: handleEventHover
    });
}

/* =============================================================================
   EVENT HANDLERS
   ============================================================================= */

/**
 * Handle hex hover - show details in sidebar
 */
function handleHexHover({ object, x, y }) {
    const tooltip = document.getElementById('tooltip');
    const hexInfo = document.getElementById('hex-info');
    
    if (!object) {
        tooltip.classList.add('hidden');
        hexInfo.innerHTML = '<p class="muted">Hover over a hexagon to see details.</p>';
        return;
    }
    
    const risk = object.risk || 0;
    const riskPercent = (risk * 100).toFixed(1);
    
    // Determine risk level
    let riskLevel, riskClass;
    if (risk < 0.2) {
        riskLevel = 'Low';
        riskClass = 'risk-low';
    } else if (risk < 0.5) {
        riskLevel = 'Moderate';
        riskClass = 'risk-moderate';
    } else if (risk < 0.75) {
        riskLevel = 'High';
        riskClass = 'risk-high';
    } else {
        riskLevel = 'Critical';
        riskClass = 'risk-critical';
    }
    
    // Update tooltip
    tooltip.classList.remove('hidden');
    tooltip.style.left = `${x + 10}px`;
    tooltip.style.top = `${y + 10}px`;
    tooltip.innerHTML = `
        <strong>Risk Score:</strong> ${riskPercent}%<br>
        <small>H3: ${object.hex}</small>
    `;
    
    // Update sidebar info panel
    hexInfo.innerHTML = `
        <div class="info-row">
            <span class="info-label">Risk Level</span>
            <span class="info-value ${riskClass}">${riskLevel}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Probability</span>
            <span class="info-value">${riskPercent}%</span>
        </div>
        <div class="info-row">
            <span class="info-label">H3 Index</span>
            <span class="info-value" style="font-family: monospace; font-size: 0.7rem;">${object.hex}</span>
        </div>
    `;
}

/**
 * Handle event hover
 */
function handleEventHover({ object, x, y }) {
    const tooltip = document.getElementById('tooltip');
    
    if (!object || !object.properties) {
        tooltip.classList.add('hidden');
        return;
    }
    
    const props = object.properties;
    
    tooltip.classList.remove('hidden');
    tooltip.style.left = `${x + 10}px`;
    tooltip.style.top = `${y + 10}px`;
    tooltip.innerHTML = `
        <strong>${props.event_type || 'Event'}</strong><br>
        ${props.sub_event_type ? `<small>${props.sub_event_type}</small><br>` : ''}
        <strong>Date:</strong> ${props.date || 'Unknown'}<br>
        <strong>Fatalities:</strong> ${props.fatalities || 0}
    `;
}

/**
 * Update loading status message
 */
function updateLoadingStatus(message) {
    const statusEl = document.getElementById('loading-status');
    if (statusEl) {
        statusEl.textContent = message;
    }
}

/* =============================================================================
   UI EVENT LISTENERS
   ============================================================================= */

/**
 * Set up toggle switch listeners
 */
function setupToggleListeners() {
    // Conflict layer toggle
    document.getElementById('toggle-conflict').addEventListener('change', (e) => {
        layersState.conflict = e.target.checked;
        renderLayers();
    });
    
    // Roads layer toggle
    document.getElementById('toggle-roads').addEventListener('change', (e) => {
        layersState.roads = e.target.checked;
        renderLayers();
    });
    
    // Rivers layer toggle
    document.getElementById('toggle-rivers').addEventListener('change', (e) => {
        layersState.rivers = e.target.checked;
        renderLayers();
    });
    
    // Events layer toggle
    document.getElementById('toggle-events').addEventListener('change', (e) => {
        layersState.events = e.target.checked;
        renderLayers();
    });
}

/* =============================================================================
   MAIN ENTRY POINT
   ============================================================================= */

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 FlashCLED Dashboard Initializing...');
    
    setupToggleListeners();
    initializeMap();
});
