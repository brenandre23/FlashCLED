/**
 * FlashCLED Three-Map Data Explorer
 * =============================================================================
 * Architecture:
 * - Map 1 (Predictions): Uses /api/predictions + /api/dates/predictions ONLY
 * - Map 2 (Temporal):    Uses /api/temporal_feature + /api/dates/temporal ONLY
 * - Map 3 (Static):      Uses /api/static_feature ONLY
 * 
 * No data mixing between maps. Slider direction fixed: right = later date.
 * =============================================================================
 */

const CONFIG = {
    API_URL: window.location.origin + '/api',
    STYLE_URL: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    INITIAL_VIEW: {
        longitude: 20.9394,
        latitude: 6.6111,
        zoom: 5.5,
        pitch: 30,
        bearing: 0
    }
};

const PREDICTION_HORIZONS = ['14d', '1m', '3m'];
const PREDICTION_LEARNERS = ['xgboost', 'lightgbm'];
// legacy stub to avoid ReferenceError from prior zoom code paths
const zoomPluginRegistered = false;

// =============================================================================
// COLOR RAMPS (Dataset-specific, no external libs)
// =============================================================================

const COLOR_RAMPS = {
    // Map 1: Predictions (light rose → dark red)
    predictions: [
        [255, 245, 240],
        [254, 224, 210],
        [252, 187, 161],
        [252, 146, 114],
        [251, 106, 74],
        [239, 59, 44],
        [203, 24, 29],
        [165, 15, 21],
        [103, 0, 13]
    ],
    
    // Map 2: Temporal Features (feature-dependent)
    temporal: {
        precip: [[240, 249, 255], [189, 215, 231], [107, 174, 214], [49, 130, 189], [8, 81, 156]],
        ndvi: [[247, 252, 245], [199, 233, 192], [127, 205, 145], [65, 171, 93], [0, 109, 44]],
        temp: [[255, 247, 236], [254, 232, 200], [253, 187, 132], [252, 141, 60], [204, 76, 2]],
        soil: [[240, 255, 255], [178, 226, 226], [102, 194, 164], [44, 162, 95], [0, 109, 68]],
        nightlights: [[255, 255, 229], [255, 247, 188], [254, 227, 145], [254, 196, 79], [217, 144, 32]],
        conflict: [[247, 247, 247], [204, 204, 204], [150, 150, 150], [99, 99, 99], [37, 37, 37]],
        market: [[255, 248, 240], [253, 220, 186], [250, 183, 132], [236, 137, 81], [180, 90, 40]],
        gdelt: [[103, 169, 207], [166, 189, 219], [224, 224, 224], [253, 174, 107], [230, 85, 13]],
        default: [[247, 251, 255], [198, 219, 239], [107, 174, 214], [33, 113, 181], [8, 48, 107]]
    },
    
    // Map 3: Static Features (light lavender → dark purple)
    static: [
        [252, 251, 253],
        [239, 237, 245],
        [218, 218, 235],
        [188, 189, 220],
        [158, 154, 200],
        [128, 125, 186],
        [106, 81, 163],
        [84, 39, 143],
        [63, 0, 125]
    ],
    static_distance: [
        [255, 247, 236],
        [254, 232, 200],
        [253, 212, 158],
        [253, 187, 132],
        [252, 141, 89],
        [239, 101, 72],
        [215, 48, 31],
        [179, 0, 0],
        [127, 0, 0]
    ]
};

// Map feature names to color ramp categories
const TEMPORAL_FEATURE_COLOR_MAP = {
    chirps_precip_anomaly: 'precip',
    era5_temp_anomaly: 'temp',
    era5_soil_moisture_anomaly: 'soil',
    ndvi_anomaly: 'ndvi',
    nightlights_intensity: 'nightlights',
    fatalities_14d_sum: 'conflict',
    fatalities_1m_lag: 'conflict',
    protest_count_lag1: 'conflict',
    riot_count_lag1: 'conflict',
    regional_risk_score_lag1: 'conflict',
    gdelt_event_count: 'default',
    gdelt_avg_tone: 'gdelt',
    gdelt_goldstein_mean: 'gdelt',
    gdelt_mentions_total: 'default',
    price_maize: 'market',
    price_rice: 'market',
    price_oil: 'market',
    price_sorghum: 'market',
    price_cassava: 'market',
    price_groundnuts: 'market'
};

// Static feature allowlist (must match backend allowlist)
const STATIC_FEATURE_OPTIONS = {
    Distance: [
        "dist_to_capital",
        "dist_to_border",
        "dist_to_city",
        "dist_to_road",
        "dist_to_river",
        "dist_to_market_km",
        "dist_to_diamond_mine",
        "dist_to_gold_mine",
        "dist_to_large_mine",
        "dist_to_controlled_mine",
        "dist_to_large_gold_mine"
    ],
    Geographic: ["elevation_mean", "slope_mean", "terrain_ruggedness_index"],
    Administrative: ["admin1", "admin2", "admin3"]
};
// Non-spatial temporal features (do not render on map)
const NON_SPATIAL_TEMPORAL = [
    "gold_price_usd_lag1",
    "oil_price_usd_lag1",
    "sp500_index_lag1",
    "eur_usd_rate_lag1"
];
const TEMPORAL_EXTRUSION = {
    pop_log: 3000, // halved from prior 6000
    iom_displacement_count_lag1: 4000
};
const STATIC_DISTANCE_FEATURES = new Set([
    "dist_to_capital",
    "dist_to_border",
    "dist_to_city",
    "dist_to_road",
    "dist_to_river",
    "dist_to_market_km",
    "dist_to_diamond_mine",
    "dist_to_gold_mine",
    "dist_to_large_mine",
    "dist_to_controlled_mine",
    "dist_to_large_gold_mine"
]);
let lastNonSpatialNoticeFeature = null;

// Chart cache for temporal trend
const SERIES_CACHE = {};
const SUMMARY_CACHE = {};
let predTrendChart = null;
let trendChart = null;
let trendMarkerIndex = null;
let trendLabels = [];
const TREND_WINDOW_STEPS = 6; // update chart every 6 timesteps (~12 weeks)
let predExplainChart = null;
const COMPARE_FEATURES = [
    // Environmental
    "chirps_precip_anomaly",
    "era5_temp_anomaly",
    "era5_soil_moisture_anomaly",
    "ndvi_anomaly",
    "nightlights_intensity",
    // Macroeconomics
    "gold_price_usd_lag1",
    "oil_price_usd_lag1",
    "sp500_index_lag1",
    "eur_usd_rate_lag1",
    // Market prices
    "price_maize",
    "price_rice",
    "price_oil",
    "price_sorghum",
    "price_cassava",
    "price_groundnuts"
];
let activeCompareFeatures = new Set();

const PRED_FORWARD_ONLY = { enabled: false };
const PRED_HISTORY_WINDOW = { steps: null }; // null = all

// =============================================================================
// STATE MANAGEMENT (Isolated per map)
// =============================================================================

const state = {
    activeTab: 'predictions',
    
    predictions: {
        dates: [],
        selectedDateIndex: 0,
        data: null,
        eventsData: null,
        showLayer: true,
        showEvents: false,
        opacity: 0.8,
        cube: {},
        cubeLoaded: false,
        horizon: '3m',
        learner: 'xgboost'
    },
    
    temporal: {
        dates: [],
        selectedDateIndex: 0,
        feature: 'chirps_precip_anomaly',
        data: null,
        showLayer: true,
        opacity: 0.8,
        cache: {},
        stats: { min: 0, max: 1 },
        normalize: false,
        showMax: false,
        showMin: false
    },
    
    static: {
        feature: 'dist_to_capital',
        data: null,
        showLayer: true,
        opacity: 0.8,
        cache: {},
        stats: { min: 0, max: 1 },
        breaks: null
    },
    
    selection: {
        predictions: null,
        temporal: null,
        static: null,
        charts: {}
    }
};

// Map instances
const maps = {};
const deckOverlays = {};

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 FlashCLED Three-Map Explorer Initializing...');
    
    if (window.feather) feather.replace();
    
    await checkSystemHealth();
    
    setupTabNavigation();
    setupEventListeners();

    // Sync selectors with initial state
    const horizonSlider = document.getElementById('pred-horizon-slider');
    if (horizonSlider) horizonSlider.value = PREDICTION_HORIZONS.indexOf(state.predictions.horizon);
    const horizonDisplay = document.getElementById('pred-horizon-display');
    if (horizonDisplay) horizonDisplay.textContent = state.predictions.horizon;
    const learnerSelect = document.getElementById('pred-learner-select');
    if (learnerSelect) learnerSelect.value = state.predictions.learner;
    
    // Initialize all three maps
    initializeMap('predictions');
    initializeMap('temporal');
    initializeMap('static');
    
    // Load initial data for predictions (default tab)
    await loadPredictionsDates();
    await loadPredictionsData();
    
    // Pre-load temporal data and trend chart
    await loadTemporalDates();
    await loadTemporalData();
    await refreshTrendChart();
});

// =============================================================================
// MAP INITIALIZATION
// =============================================================================

function initializeMap(mapId) {
    const container = document.getElementById(`map-${mapId}`);
    if (!container) return;
    
    maps[mapId] = new maplibregl.Map({
        container: container,
        style: CONFIG.STYLE_URL,
        center: [CONFIG.INITIAL_VIEW.longitude, CONFIG.INITIAL_VIEW.latitude],
        zoom: CONFIG.INITIAL_VIEW.zoom,
        pitch: CONFIG.INITIAL_VIEW.pitch,
        bearing: CONFIG.INITIAL_VIEW.bearing,
        antialias: true
    });
    
    maps[mapId].addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'bottom-right');
    
    maps[mapId].on('load', () => {
        deckOverlays[mapId] = new deck.MapboxOverlay({
            interleaved: false,
            layers: []
        });
        maps[mapId].addControl(deckOverlays[mapId]);

        // Force render once overlay exists (critical fix)
        if (mapId === 'predictions') renderPredictionsLayer();
        if (mapId === 'temporal') renderTemporalLayer();
        if (mapId === 'static') renderStaticLayer();
        
        document.getElementById(`loading-${mapId}`).classList.add('hidden');
    });
    
    maps[mapId].on('mousemove', (e) => {
        document.getElementById(`coords-${mapId}`).textContent = 
            `${e.lngLat.lat.toFixed(4)}, ${e.lngLat.lng.toFixed(4)}`;
    });
}

// =============================================================================
// TAB NAVIGATION
// =============================================================================

function setupTabNavigation() {
    const tabs = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.map-panel');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', async () => {
            const targetTab = tab.dataset.tab;
            
            // Update active states
            tabs.forEach(t => {
                t.classList.remove('active');
                t.setAttribute('aria-selected', 'false');
            });
            tab.classList.add('active');
            tab.setAttribute('aria-selected', 'true');
            
            panels.forEach(p => p.classList.remove('active'));
            document.getElementById(`panel-${targetTab}`).classList.add('active');
            
            state.activeTab = targetTab;
            
            // Resize map on tab switch
            setTimeout(() => {
                if (maps[targetTab]) {
                    maps[targetTab].resize();
                }

                // Force render after resize (fixes hidden-tab race)
                if (targetTab === 'predictions') renderPredictionsLayer();
                if (targetTab === 'temporal') renderTemporalLayer();
                if (targetTab === 'static') renderStaticLayer();
            }, 100);
            
            // Load data if not already loaded
            if (targetTab === 'temporal' && state.temporal.dates.length === 0) {
                await loadTemporalDates();
            }
            if (targetTab === 'temporal' && !state.temporal.data) {
                await loadTemporalData();
            }
            if (targetTab === 'static' && !state.static.data) {
                await loadStaticData();
            }
        });
    });
}

// =============================================================================
// DATA LOADING: PREDICTIONS (Map 1)
// =============================================================================

function buildPredictionQuery(basePath, params = {}) {
    const search = new URLSearchParams();
    const horizon = params.horizon || state.predictions.horizon;
    const learner = params.hasOwnProperty('learner') ? params.learner : state.predictions.learner;

    if (horizon) search.append('horizon', horizon);
    if (learner) search.append('learner', learner);
    if (params.date) search.append('date', params.date);

    return `${CONFIG.API_URL}${basePath}?${search.toString()}`;
}

async function loadPredictionsDates() {
    try {
        const res = await fetch(buildPredictionQuery('/dates/predictions'));
        const data = await res.json();
        
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }
        
        // Dates should already be sorted ascending from server
        // But verify and fix if not
        let dates = data.dates || [];
        if (dates.length > 1 && dates[0] > dates[1]) {
            console.warn('Dates were descending, reversing to ascending');
            dates = dates.reverse();
        }
        
        state.predictions.dates = dates;
        if (dates.length > 0) {
            state.predictions.selectedDateIndex = dates.length - 1;
        }
    } catch (e) {
        console.error('Failed to load prediction dates:', e);
        showToast('Failed to load prediction dates', 'error');
    }
}

async function loadPredictionsData(date = null) {
    const selectedDate = date || state.predictions.dates[state.predictions.selectedDateIndex];
    if (!selectedDate) return;
    
    // Check cube cache first
    if (state.predictions.cubeLoaded && state.predictions.cube[selectedDate]) {
        state.predictions.data = state.predictions.cube[selectedDate];
        renderPredictionsLayer();
        await refreshPredictionSelection();
        return;
    }
    
    document.getElementById('loading-predictions').classList.remove('hidden');
    
    try {
        const res = await fetch(buildPredictionQuery('/predictions', { date: selectedDate }));
        const data = await res.json();
        
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }
        
        state.predictions.data = data;
        renderPredictionsLayer();
        await refreshPredictionSelection();
    } catch (e) {
        console.error('Failed to load predictions:', e);
        showToast('Failed to load predictions', 'error');
    } finally {
        document.getElementById('loading-predictions').classList.add('hidden');
    }
}

async function loadPredictionsCube() {
    if (state.predictions.cubeLoaded) return;
    
    showToast('Loading animation data...', 'info');
    
    try {
        const res = await fetch(buildPredictionQuery('/predictions/cube'));
        const data = await res.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Pivot to cube structure
        data.forEach(row => {
            if (!state.predictions.cube[row.date]) {
                state.predictions.cube[row.date] = [];
            }
            state.predictions.cube[row.date].push({
                hex: row.h3_index,
                risk: row.risk,
                fatalities: row.fatal,
                expected_fatalities: row.expected_fatal
            });
        });
        
        state.predictions.cubeLoaded = true;
        await refreshPredictionSelection();
        showToast('Animation ready', 'success');
    } catch (e) {
        console.error('Cube load failed:', e);
        showToast('Animation data unavailable', 'error');
    }
}

async function loadEventsData() {
    if (state.predictions.eventsData) return;
    
    try {
        const res = await fetch(`${CONFIG.API_URL}/events?limit=1000`);
        state.predictions.eventsData = await res.json();
        renderPredictionsLayer();
    } catch (e) {
        showToast('Failed to load events', 'error');
    }
}

// =============================================================================
// DATA LOADING: TEMPORAL FEATURES (Map 2)
// =============================================================================

async function loadTemporalDates() {
    try {
        const res = await fetch(`${CONFIG.API_URL}/dates/temporal`);
        const data = await res.json();
        
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }
        
        let dates = data.dates || [];
        if (dates.length > 1 && dates[0] > dates[1]) {
            console.warn('Temporal dates were descending, reversing');
            dates = dates.reverse();
        }
        
        state.temporal.dates = dates;
        
        const slider = document.getElementById('temporal-date-slider');
        const display = document.getElementById('temporal-date-display');
        
        if (dates.length > 0) {
            slider.disabled = false;
            slider.min = 0;
            slider.max = dates.length - 1;
            slider.value = dates.length - 1;
            state.temporal.selectedDateIndex = dates.length - 1;
            display.textContent = dates[dates.length - 1];
        } else {
            display.textContent = 'No data';
        }
    } catch (e) {
        console.error('Failed to load temporal dates:', e);
    }
}

async function loadTemporalData() {
    const feature = state.temporal.feature;
    const date = state.temporal.dates[state.temporal.selectedDateIndex];
    if (!date) return;
    const isNonSpatial = NON_SPATIAL_TEMPORAL.includes(feature);
    if (!isNonSpatial && !state.selection.temporal) {
        const inspector = document.getElementById('temporal-inspector-content');
        if (inspector) inspector.innerHTML = '<p class="muted">Click a hex to inspect.</p>';
    }
    
    // Check cache
    const cacheKey = `${feature}_${date}`;
    if (state.temporal.cache[cacheKey]) {
        state.temporal.data = state.temporal.cache[cacheKey].data;
        state.temporal.stats = state.temporal.cache[cacheKey].stats;
        renderTemporalLayer();
        updateTemporalLegend();
        return;
    }
    
    try {
        const res = await fetch(`${CONFIG.API_URL}/temporal_feature?feature=${feature}&date=${date}`);
        const data = await res.json();
        
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }
        
        // Calculate stats for normalization
        const values = data.filter(d => d.value !== null).map(d => d.value);
        // For elevation, keep full min/max so extrusion is continuous across full range
        const stats = feature === 'elevation_mean'
            ? { min: Math.min(...values), max: Math.max(...values) }
            : calculatePercentileStats(values);
        
        state.temporal.data = data;
        state.temporal.stats = stats;
        state.temporal.cache[cacheKey] = { data, stats };
        
        if (isNonSpatial && lastNonSpatialNoticeFeature !== feature) {
            showToast(`${formatFeatureName(feature)} is non-spatial. Map layer hidden; use the trend chart.`, 'info');
            lastNonSpatialNoticeFeature = feature;
            const inspector = document.getElementById('temporal-inspector-content');
            if (inspector) {
                inspector.innerHTML = '<p class="muted">This macro series has no spatial surface. Use the trend chart below.</p>';
            }
        }

        renderTemporalLayer();
        updateTemporalLegend();
        // Update trend marker every step; refresh data only when moving to a new 6-step window
        await updateTrendMarker(date);
    } catch (e) {
        console.error('Failed to load temporal feature:', e);
        showToast('Failed to load temporal feature', 'error');
    }
}

function getComparisonHex() {
    if (state.temporal.data && state.temporal.data.length) return state.temporal.data[0].hex;
    if (state.static.data && state.static.data.length) return state.static.data[0].hex;
    return '855a5a1bfffffff';
}

async function fetchTemporalSeries(feature, hexOverride = null) {
    const hex = hexOverride || getComparisonHex();
    const cacheKey = `${feature}|${hex}`;
    if (SERIES_CACHE[cacheKey]) return SERIES_CACHE[cacheKey];
    try {
        const res = await fetch(`${CONFIG.API_URL}/analytics/temporal/hex/${hex}?feature=${feature}`);
        const data = await res.json();
        if (data.error || !data.history) return null;
        SERIES_CACHE[cacheKey] = data.history;
        return data.history;
    } catch (e) {
        console.error('Failed to fetch temporal series', feature, e);
        return null;
    }
}

async function fetchTemporalSummary(feature) {
    if (SUMMARY_CACHE[feature]) return SUMMARY_CACHE[feature];
    try {
        const res = await fetch(`${CONFIG.API_URL}/analytics/temporal/summary?feature=${feature}`);
        const data = await res.json();
        if (data.error) return null;
        SUMMARY_CACHE[feature] = data;
        return data;
    } catch (e) {
        console.error('Failed to fetch temporal summary', feature, e);
        return null;
    }
}

async function fetchStaticSnapshot(hex) {
    try {
        const res = await fetch(`${CONFIG.API_URL}/analytics/static/hex/${hex}`);
        const data = await res.json();
        if (data.error) return null;
        return data.values || null;
    } catch (e) {
        console.error('Failed to fetch static snapshot', e);
        return null;
    }
}

function buildChunkSeries(history, labels) {
    if (!history || !history.dates) return { values: [] };
    const map = {};
    for (let i = 0; i < history.dates.length; i++) {
        map[history.dates[i]] = history.values[i];
    }
    return {
        values: labels.map(d => map[d] ?? null)
    };
}

function normalizeSeries(values) {
    const vals = values.slice();
    const firstIdx = vals.findIndex(v => v !== null && v !== undefined);
    if (firstIdx === -1) return vals;
    const base = vals[firstIdx];
    if (base === 0) return vals;
    return vals.map(v => (v === null || v === undefined) ? null : (v - base) / Math.abs(base));
}

function buildLineChart(ctx, datasets, labels) {
    const trendMarker = {
        id: 'trendMarker',
        afterDraw: chart => {
            if (trendMarkerIndex === null) return;
            const xScale = chart.scales.x;
            const yScale = chart.scales.y;
            if (!xScale || !yScale) return;
            const xPos = xScale.getPixelForValue(trendMarkerIndex);
            const ctx = chart.ctx;
            ctx.save();
            ctx.strokeStyle = '#ffcc00';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(xPos, yScale.top);
            ctx.lineTo(xPos, yScale.bottom);
            ctx.stroke();
            ctx.restore();
        }
    };

    return new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: true, position: 'bottom' } },
            scales: {
                x: { ticks: { color: '#666', font: { size: 9 } }, grid: { display: false } },
                y: { ticks: { color: '#666', font: { size: 9 } }, grid: { display: false } }
            }
        },
        plugins: [trendMarker]
    });
}

async function refreshTrendChart() {
    const trendCanvas = document.getElementById('trend-chart');
    if (!trendCanvas) return;
    const compareSelect = document.getElementById('trend-compare-select');
    const mainFeature = state.temporal.feature;
    const date = state.temporal.dates[state.temporal.selectedDateIndex];
    if (!date) return;
    const normalize = state.temporal.normalize;
    const showMax = state.temporal.showMax;
    const showMin = state.temporal.showMin;

    // Determine 6-step window chunk for the current date
    const idx = state.temporal.selectedDateIndex;
    const chunkStart = Math.floor(idx / TREND_WINDOW_STEPS) * TREND_WINDOW_STEPS;
    const datesWindow = state.temporal.dates.slice(chunkStart, chunkStart + TREND_WINDOW_STEPS);
    if (!datesWindow.length) return;

    const colors = ['#f39c12', '#27ae60', '#2980b9', '#8e44ad', '#e74c3c', '#16a085'];
    const datasets = [];
    let labels = datesWindow;

    const mainHist = await fetchTemporalSeries(mainFeature);
    const mainSeries = buildChunkSeries(mainHist, labels);
    trendMarkerIndex = null;
    trendLabels = labels;
    if (mainSeries.values.length) {
        const markerIdx = labels.indexOf(date);
        trendMarkerIndex = markerIdx >= 0 ? markerIdx : labels.length - 1;
        datasets.push({
            label: formatFeatureName(mainFeature),
            data: normalize ? normalizeSeries(mainSeries.values) : mainSeries.values,
            borderColor: colors[0],
            backgroundColor: 'transparent',
            tension: 0.3,
            pointRadius: 0
        });
    }

    // Compare overlays (toggleable)
    let colorIdx = 1;
    for (const compareFeature of activeCompareFeatures) {
        const compHist = await fetchTemporalSeries(compareFeature);
        const compSeries = buildChunkSeries(compHist, labels);
        const nonNull = compSeries.values.filter(v => v !== null);
        if (nonNull.length >= 2) {
            datasets.push({
                label: formatFeatureName(compareFeature),
                data: normalize ? normalizeSeries(compSeries.values) : compSeries.values,
                borderColor: colors[colorIdx % colors.length],
                backgroundColor: 'transparent',
                tension: 0.3,
                pointRadius: 0,
                borderDash: [4, 2]
            });
            colorIdx += 1;
        }
    }

    // Selected cell overlay
    if (state.selection.temporal) {
        const selHex = state.selection.temporal;
        const selHist = await fetchTemporalSeries(mainFeature, selHex);
        const selSeries = buildChunkSeries(selHist, labels);
        const nonNull = selSeries.values.filter(v => v !== null);
        if (nonNull.length >= 2) {
            datasets.push({
                label: `Selected Hex (${selHex.slice(0, 6)}…)`,
                data: normalize ? normalizeSeries(selSeries.values) : selSeries.values,
                borderColor: '#ff4b4b',
                backgroundColor: 'transparent',
                tension: 0.3,
                pointRadius: 0,
                borderDash: [2, 2]
            });
        }
    }

    if (labels.length && datasets.length) {
        // Add band lines for max/min (global per-date)
        if (showMax || showMin) {
            const summary = await fetchTemporalSummary(mainFeature);
            if (summary && summary.dates) {
                const summaryMap = {};
                summary.dates.forEach((d, idx) => {
                    summaryMap[d] = { max: summary.max[idx], min: summary.min[idx] };
                });
                const maxVals = labels.map(d => summaryMap[d]?.max ?? null);
                const minVals = labels.map(d => summaryMap[d]?.min ?? null);
                const addBand = (vals, label) => {
                    const valid = vals.filter(v => v !== null);
                    if (!valid.length) return;
                    datasets.push({
                        label,
                        data: normalize ? normalizeSeries(vals) : vals,
                        borderColor: 'rgba(255, 255, 255, 0.25)',
                        borderDash: [2, 2],
                        pointRadius: 0,
                        tension: 0,
                        fill: false
                    });
                };
                if (showMax) addBand(maxVals, 'Window Max');
                if (showMin) addBand(minVals, 'Window Min');
            }
        }

        if (trendChart) trendChart.destroy();
        trendChart = buildLineChart(trendCanvas, datasets, labels.map(d => d.slice(0, 10)));
    }
}

function updateTrendMarker(date) {
    // If current date is outside the active window, refresh the chart for the new chunk
    if (!trendLabels.length || trendLabels.indexOf(date) === -1) {
        return refreshTrendChart();
    }
    const idx = trendLabels.indexOf(date);
    trendMarkerIndex = idx >= 0 ? idx : trendLabels.length - 1;
    if (trendChart) trendChart.update('none');
}

// =============================================================================
// DATA LOADING: STATIC FEATURES (Map 3)
// =============================================================================

async function loadStaticData() {
    const feature = state.static.feature;
    
    // Check cache
    if (state.static.cache[feature]) {
        state.static.data = state.static.cache[feature].data;
        state.static.stats = state.static.cache[feature].stats;
        state.static.breaks = state.static.cache[feature].breaks || null;
        renderStaticLayer();
        updateStaticLegend();
        // Refresh inspector if a selection exists
        if (state.selection.static) {
            const obj = state.static.data.find(d => d.hex === state.selection.static);
            if (obj) await renderStaticInspector(obj);
        }
        return;
    }
    
    document.getElementById('loading-static').classList.remove('hidden');
    
    try {
        const res = await fetch(`${CONFIG.API_URL}/static_feature?feature=${feature}`);
        const data = await res.json();
        
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }
        
        const values = data.filter(d => d.value !== null).map(d => d.value);
        const stats = calculatePercentileStats(values);
        
        state.static.data = data;
        state.static.stats = stats;

        // Compute Jenks breaks for distance features
        let breaks = null;
        if (STATIC_DISTANCE_FEATURES.has(feature) && values.length >= 2) {
            const k = Math.min(COLOR_RAMPS.static_distance.length, Math.max(2, Math.floor(values.length / 10)));
            breaks = computeJenks(values, k);
        }
        state.static.breaks = breaks;
        state.static.cache[feature] = { data, stats, breaks };
        
        renderStaticLayer();
        updateStaticLegend();
        if (state.selection.static) {
            const obj = data.find(d => d.hex === state.selection.static);
            if (obj) await renderStaticInspector(obj);
        }
    } catch (e) {
        console.error('Failed to load static feature:', e);
        showToast('Failed to load static feature', 'error');
    } finally {
        document.getElementById('loading-static').classList.add('hidden');
    }
}

// =============================================================================
// RENDERING: PREDICTIONS (Map 1)
// =============================================================================

function renderPredictionsLayer() {
    if (!deckOverlays.predictions) return;
    
    const layers = [];
    
    if (state.predictions.showLayer && state.predictions.data) {
        layers.push(new deck.H3HexagonLayer({
            id: 'predictions-layer',
            data: state.predictions.data,
            pickable: true,
            filled: true,
            extruded: true,
            getHexagon: d => d.hex,
            getFillColor: d => getPredictionColor(d.risk),
            getElevation: d => (d.fatalities || 0) * 500,
            elevationScale: 1,
            opacity: state.predictions.opacity,
            material: { ambient: 0.6, diffuse: 0.6, shininess: 32, specularColor: [51, 51, 51] },
            transitions: { getFillColor: { duration: 300 }, getElevation: { duration: 300 } },
            onClick: info => handlePredictionClick(info),
            onHover: info => updateTooltip('predictions', info)
        }));
    }
    
    if (state.predictions.showEvents && state.predictions.eventsData?.features) {
        layers.push(new deck.GeoJsonLayer({
            id: 'events-layer',
            data: state.predictions.eventsData,
            pickable: true,
            stroked: true,
            filled: true,
            pointType: 'circle',
            getPointRadius: d => Math.max(3, (d.properties.fatalities || 0) * 0.5),
            getFillColor: [255, 75, 75, 180],
            getLineColor: [255, 255, 255, 100],
            getLineWidth: 1,
            pointRadiusUnits: 'pixels',
            onHover: info => updateEventTooltip('predictions', info)
        }));
    }
    
    deckOverlays.predictions.setProps({ layers });
}

function getPredictionColor(risk) {
    if (risk === null || risk === undefined) return [0, 0, 0, 0];
    
    const ramp = COLOR_RAMPS.predictions;
    // Stretch low values to improve visual contrast for small risks
    const stretched = Math.pow(Math.max(0, Math.min(1, risk)), 0.35);
    const t = Math.max(0, Math.min(1, stretched));
    const idx = t * (ramp.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, ramp.length - 1);
    const frac = idx - lo;
    
    return [
        Math.round(ramp[lo][0] + frac * (ramp[hi][0] - ramp[lo][0])),
        Math.round(ramp[lo][1] + frac * (ramp[hi][1] - ramp[lo][1])),
        Math.round(ramp[lo][2] + frac * (ramp[hi][2] - ramp[lo][2])),
        200
    ];
}

function getTemporalElevation(value, feature) {
    if (value === null || value === undefined) return 0;
    const heightScale = TEMPORAL_EXTRUSION[feature];
    if (!heightScale) return 0;
    const { min, max } = state.temporal.stats;
    const range = max - min || 1;
    const t = Math.max(0, Math.min(1, (value - min) / range));
    return t * heightScale;
}


// =============================================================================
// RENDERING: TEMPORAL (Map 2)
// =============================================================================

function renderTemporalLayer() {
    if (!deckOverlays.temporal) return;
    
    const layers = [];
    
    const feature = state.temporal.feature;
    const isExtruded = Boolean(TEMPORAL_EXTRUSION[feature]);
    const isNonSpatial = NON_SPATIAL_TEMPORAL.includes(feature);
    
    if (isNonSpatial) {
        deckOverlays.temporal.setProps({ layers });
        const inspector = document.getElementById('temporal-inspector-content');
        if (inspector) {
            inspector.innerHTML = '<p class="muted">This macro series has no spatial surface. Use the trend chart below.</p>';
        }
        return;
    }
    
    if (state.temporal.showLayer && state.temporal.data) {
        layers.push(new deck.H3HexagonLayer({
            id: 'temporal-layer',
            data: state.temporal.data,
            pickable: true,
            filled: true,
            extruded: isExtruded,
            getHexagon: d => d.hex,
            getFillColor: d => getTemporalColor(d.value, feature),
            getElevation: d => getTemporalElevation(d.value, feature),
            elevationScale: isExtruded ? 30 : 1,
            elevationRange: [0, 60000],
            opacity: state.temporal.opacity,
            transitions: { getFillColor: { duration: 300 } },
            onClick: info => handleTemporalClick(info),
            onHover: info => updateTooltip('temporal', info)
        }));
    }
    
    deckOverlays.temporal.setProps({ layers });
}

function getTemporalColor(value, feature) {
    if (value === null || value === undefined) return [0, 0, 0, 0];
    
    const { min, max } = state.temporal.stats;
    const range = max - min || 1;
    const t = Math.max(0, Math.min(1, (value - min) / range));
    
    const rampKey = TEMPORAL_FEATURE_COLOR_MAP[feature] || 'default';
    const ramp = COLOR_RAMPS.temporal[rampKey];
    
    const idx = t * (ramp.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, ramp.length - 1);
    const frac = idx - lo;
    
    return [
        Math.round(ramp[lo][0] + frac * (ramp[hi][0] - ramp[lo][0])),
        Math.round(ramp[lo][1] + frac * (ramp[hi][1] - ramp[lo][1])),
        Math.round(ramp[lo][2] + frac * (ramp[hi][2] - ramp[lo][2])),
        255
    ];
}

function updateTemporalLegend() {
    const feature = state.temporal.feature;
    const { min, max } = state.temporal.stats;
    
    document.getElementById('temporal-legend-title').textContent = formatFeatureName(feature);
    document.getElementById('temporal-legend-min').textContent = formatValue(min);
    document.getElementById('temporal-legend-max').textContent = formatValue(max);
    
    // Update gradient
    const rampKey = TEMPORAL_FEATURE_COLOR_MAP[feature] || 'default';
    const ramp = COLOR_RAMPS.temporal[rampKey];
    const gradient = ramp.map((c, i) => `rgb(${c.join(',')}) ${(i / (ramp.length - 1) * 100).toFixed(0)}%`).join(', ');
    document.querySelector('.temporal-gradient').style.background = `linear-gradient(to right, ${gradient})`;
}

// =============================================================================
// RENDERING: STATIC (Map 3)
// =============================================================================

function renderStaticLayer() {
    if (!deckOverlays.static) return;
    
    const layers = [];
    
    if (state.static.showLayer && state.static.data) {
        const feature = state.static.feature;
        const extrude = feature === 'elevation_mean';
        layers.push(new deck.H3HexagonLayer({
            id: 'static-layer',
            data: state.static.data,
            pickable: true,
            filled: true,
            extruded: extrude,
            getHexagon: d => d.hex,
            getFillColor: d => getStaticColor(d.value, feature),
            getElevation: d => extrude ? getStaticElevation(d.value) : 0,
            elevationScale: extrude ? 50 : 1,
            elevationRange: [0, 30000],
            opacity: state.static.opacity,
            transitions: { getFillColor: { duration: 300 } },
            onClick: info => handleStaticClick(info),
            onHover: info => updateTooltip('static', info)
        }));
    }
    
    deckOverlays.static.setProps({ layers });
}

function getStaticColor(value, feature) {
    if (value === null || value === undefined) return [0, 0, 0, 0];
    
    const ramp = STATIC_DISTANCE_FEATURES.has(feature) ? COLOR_RAMPS.static_distance : COLOR_RAMPS.static;
    const breaks = STATIC_DISTANCE_FEATURES.has(feature) ? state.static.breaks : null;

    if (breaks && breaks.length > 1) {
        // Jenks/natural breaks classification
        let idx = breaks.findIndex(b => value <= b);
        if (idx === -1) idx = breaks.length - 1;
        const t = Math.max(0, Math.min(1, idx / (ramp.length - 1)));
        const pos = t * (ramp.length - 1);
        const lo = Math.floor(pos);
        const hi = Math.min(lo + 1, ramp.length - 1);
        const frac = pos - lo;
        return [
            Math.round(ramp[lo][0] + frac * (ramp[hi][0] - ramp[lo][0])),
            Math.round(ramp[lo][1] + frac * (ramp[hi][1] - ramp[lo][1])),
            Math.round(ramp[lo][2] + frac * (ramp[hi][2] - ramp[lo][2])),
            255
        ];
    }

    const { min, max } = state.static.stats;
    const range = max - min || 1;
    const t = Math.max(0, Math.min(1, (value - min) / range));
    
    const idx = t * (ramp.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, ramp.length - 1);
    const frac = idx - lo;
    
    return [
        Math.round(ramp[lo][0] + frac * (ramp[hi][0] - ramp[lo][0])),
        Math.round(ramp[lo][1] + frac * (ramp[hi][1] - ramp[lo][1])),
        Math.round(ramp[lo][2] + frac * (ramp[hi][2] - ramp[lo][2])),
        255
    ];
}

function getStaticElevation(value) {
    if (value === null || value === undefined) return 0;
    const { min, max } = state.static.stats;
    const range = max - min || 1;
    const t = Math.max(0, Math.min(1, (value - min) / range));
    const heightScale = 500; // modest extrusion for elevation
    return t * heightScale;
}

// Jenks / natural breaks helper for static distance features
function computeJenks(values, k) {
    if (!values.length || k < 2) return null;
    const sorted = values.slice().sort((a, b) => a - b);
    const n = sorted.length;
    k = Math.min(k, n);

    const lower = Array.from({ length: n + 1 }, () => Array(k + 1).fill(0));
    const variance = Array.from({ length: n + 1 }, () => Array(k + 1).fill(0));

    for (let i = 1; i <= k; i++) {
        lower[1][i] = 1;
        variance[1][i] = 0;
        for (let j = 2; j <= n; j++) {
            variance[j][i] = Infinity;
        }
    }

    let prefixSum = Array(n + 1).fill(0);
    let prefixSq = Array(n + 1).fill(0);
    for (let i = 1; i <= n; i++) {
        prefixSum[i] = prefixSum[i - 1] + sorted[i - 1];
        prefixSq[i] = prefixSq[i - 1] + sorted[i - 1] * sorted[i - 1];
    }

    function varianceBetween(i, j) {
        const count = j - i + 1;
        const sum = prefixSum[j] - prefixSum[i - 1];
        const sq = prefixSq[j] - prefixSq[i - 1];
        const mean = sum / count;
        return sq - count * mean * mean;
    }

    for (let l = 2; l <= n; l++) {
        for (let m = 2; m <= k; m++) {
            let best = Infinity;
            let bestIdx = -1;
            for (let i = 1; i <= l - 1; i++) {
                const cost = variance[i][m - 1] + varianceBetween(i + 1, l);
                if (cost < best) {
                    best = cost;
                    bestIdx = i;
                }
            }
            lower[l][m] = bestIdx;
            variance[l][m] = best;
        }
    }

    const breaks = [];
    let kClass = k;
    let l = n;
    while (kClass > 0) {
        const idx = lower[l][kClass];
        breaks[kClass - 1] = sorted[l - 1];
        l = idx;
        kClass -= 1;
    }
    return breaks;
}

function updateStaticLegend() {
    const feature = state.static.feature;
    const { min, max } = state.static.stats;
    
    document.getElementById('static-legend-title').textContent = formatFeatureName(feature);
    document.getElementById('static-legend-min').textContent = formatValue(min);
    document.getElementById('static-legend-max').textContent = formatValue(max);
}

// =============================================================================
// CLICK HANDLERS & INSPECTORS
// =============================================================================

async function handlePredictionClick({ object }) {
    if (!object) return;
    
    state.selection.predictions = object.hex;
    const content = document.getElementById('pred-inspector-content');
    
    content.innerHTML = '<div class="skeleton-loader"><div class="skeleton-line"></div></div>';
    
    try {
        const params = new URLSearchParams({ horizon: state.predictions.horizon });
        if (state.predictions.learner) params.append('learner', state.predictions.learner);
        const [historyRes, explainRes] = await Promise.all([
            fetch(`${CONFIG.API_URL}/analytics/prediction/hex/${object.hex}?${params.toString()}`),
            fetch(`${CONFIG.API_URL}/analytics/prediction/hex/${object.hex}/explanations?${params.toString()}`)
        ]);
        if (!historyRes.ok || !explainRes.ok) throw new Error('Prediction fetch failed');
        const historyData = await historyRes.json();
        const explainData = await explainRes.json();
        const actualHist = await fetchActualFatalHistory(object.hex, historyData.history?.dates || []);
        if (historyData.history) {
            historyData.history.actual_fatalities = actualHist;
        }
        renderPredictionInspector(historyData, object, explainData);
    } catch (e) {
        content.innerHTML = '<p class="error">Failed to load data</p>';
        console.error('Prediction inspector error', e);
    }
}

async function refreshPredictionSelection() {
    if (!state.selection.predictions) return;
    const hex = state.selection.predictions;
    // Try to find current map object for values
    const obj = (state.predictions.data || []).find(d => d.hex === hex);
    const params = new URLSearchParams({ horizon: state.predictions.horizon });
    if (state.predictions.learner) params.append('learner', state.predictions.learner);
    if (PRED_FORWARD_ONLY.enabled) params.append('forward_only', 'true');
    try {
        const [historyRes, explainRes] = await Promise.all([
            fetch(`${CONFIG.API_URL}/analytics/prediction/hex/${hex}?${params.toString()}`),
            fetch(`${CONFIG.API_URL}/analytics/prediction/hex/${hex}/explanations?${params.toString()}`)
        ]);
        if (!historyRes.ok || !explainRes.ok) throw new Error('Prediction fetch failed');
        const historyData = await historyRes.json();
        const explainData = await explainRes.json();
        // Fetch actual fatalities (ACLED) history for this hex to overlay
        const actualHist = await fetchActualFatalHistory(hex, historyData.history?.dates || []);
        if (historyData.history) {
            historyData.history.actual_fatalities = actualHist;
        }
        renderPredictionInspector(historyData, obj || { hex, risk: null, fatalities: null, expected_fatalities: null }, explainData);
    } catch (e) {
        console.error('Failed to refresh selection analytics', e);
    }
}

function renderPredictionInspector(data, object, explainData) {
    const content = document.getElementById('pred-inspector-content');
    
    content.innerHTML = `
        <div class="inspector-meta">
            <span class="label">H3:</span>
            <span class="mono">${object.hex.toString().substring(0, 12)}...</span>
        </div>
        <div class="inspector-meta">
            <span class="label">Horizon:</span>
            <span class="mono">${state.predictions.horizon}</span>
        </div>
        <div class="inspector-meta">
            <span class="label">Learner:</span>
            <span class="mono">${state.predictions.learner || 'all'}</span>
        </div>
        <div class="inspector-stats">
            <div class="stat-card">
                <span class="stat-label">Current Risk</span>
                <span class="stat-value risk">${((object.risk || 0) * 100).toFixed(1)}%</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Fatalities</span>
                <span class="stat-value">${(object.fatalities || 0).toFixed(1)}</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Expected Fatalities</span>
                <span class="stat-value">${(function() {
                    const exp = (object.expected_fatalities !== undefined && object.expected_fatalities !== null)
                        ? object.expected_fatalities
                        : (object.risk || 0) * (object.fatalities || 0);
                    return (exp || 0).toFixed(2);
                })()}</span>
            </div>
        </div>
    `;
    
    if (data.history && data.history.dates.length > 0) {
        renderPredictionTrend(data.history);
    }

    renderPredictionExplanations(explainData);
}

async function fetchActualFatalHistory(hex, dateLabels) {
    if (!hex || !dateLabels || !dateLabels.length) return [];
    try {
        const res = await fetch(`${CONFIG.API_URL}/analytics/temporal/hex/${hex}?feature=fatalities_14d_sum`);
        const data = await res.json();
        if (data.error || !data.history) return [];
        const map = {};
        data.history.dates.forEach((d, idx) => { map[d] = data.history.values[idx]; });
        return dateLabels.map(d => map[d] ?? null);
    } catch (e) {
        console.error('Failed to fetch actual fatalities history', e);
        return [];
    }
}

function renderPredictionTrend(history) {
    if (!history || !history.dates || !history.dates.length) return;
    const ctx = document.getElementById('pred-trend-chart');
    if (!ctx) return;
    if (predTrendChart) predTrendChart.destroy();

    const selectedDate = state.predictions.dates[state.predictions.selectedDateIndex];
    let dates = history.dates.map(d => d.slice(0, 10));
    let risk = history.risk || [];
    let predFatal = history.fatalities || [];
    let expectedFatal = history.expected_fatalities || [];
    let actualFatal = history.actual_fatalities || [];
    const forwardOnly = PRED_FORWARD_ONLY.enabled;

    if (forwardOnly) {
        // Future-only view: keep next steps from today, limit to 6, drop actuals
        const today = new Date().toISOString().slice(0, 10);
        const idx = dates.findIndex(d => d >= today);
        if (idx !== -1) {
            dates = dates.slice(idx, idx + TREND_WINDOW_STEPS);
            risk = risk.slice(idx, idx + TREND_WINDOW_STEPS);
            predFatal = predFatal.slice(idx, idx + TREND_WINDOW_STEPS);
            expectedFatal = expectedFatal.slice(idx, idx + TREND_WINDOW_STEPS);
        } else {
            dates = [];
            risk = [];
            predFatal = [];
            expectedFatal = [];
        }
        actualFatal = [];
        trendMarkerIndex = null;
    } else {
        // Full history view (no windowing), include actuals
        const markerIdx = selectedDate ? dates.indexOf(selectedDate) : dates.length - 1;
        trendMarkerIndex = markerIdx >= 0 ? markerIdx : null;
        // Optional history window: keep last N steps if set
        if (PRED_HISTORY_WINDOW.steps && dates.length > PRED_HISTORY_WINDOW.steps) {
            const start = Math.max(0, dates.length - PRED_HISTORY_WINDOW.steps);
            dates = dates.slice(start);
            risk = risk.slice(start);
            predFatal = predFatal.slice(start);
            expectedFatal = expectedFatal.slice(start);
            actualFatal = actualFatal.slice(start);
            if (trendMarkerIndex !== null) {
                trendMarkerIndex = Math.max(0, trendMarkerIndex - start);
            }
        }
    }

    const datasets = [
        {
            label: 'Predicted Risk',
            data: risk,
            borderColor: '#ffb347',
            backgroundColor: 'transparent',
            tension: 0.3,
            pointRadius: 0
        },
        {
            label: 'Predicted Fatalities',
            data: predFatal,
            borderColor: '#7fb3ff',
            backgroundColor: 'transparent',
            tension: 0.3,
            pointRadius: 0,
            yAxisID: 'y1'
        },
        {
            label: 'Expected Fatalities',
            data: expectedFatal,
            borderColor: '#ffa0bf',
            backgroundColor: 'transparent',
            tension: 0.3,
            pointRadius: 0,
            yAxisID: 'y1'
        }
    ];
    if (actualFatal.length) {
        datasets.push({
            label: 'Actual Fatalities (ACLED)',
            data: actualFatal,
            borderColor: '#00e5ff',
            backgroundColor: 'transparent',
            tension: 0.3,
            pointRadius: 0,
            yAxisID: 'y1'
        });
    }

    predTrendChart = new Chart(ctx, {
        type: 'line',
        data: { labels: dates, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: true, position: 'bottom' } },
            scales: {
                x: { ticks: { color: '#666', font: { size: 9 } }, grid: { display: false } },
                y: { position: 'left', ticks: { color: '#ffb347', font: { size: 9 } }, grid: { display: false }, min: 0, max: 1 },
                y1: { position: 'right', ticks: { color: '#7fb3ff', font: { size: 9 } }, grid: { display: false } }
            }
        }
    });
}

function renderPredictionExplanations(explainData) {
    const container = document.getElementById('pred-explain');
    if (!container) return;
    if (!explainData || explainData.error || !explainData.explanations) {
        container.innerHTML = '<p class="muted">No explanations available</p>';
        if (predExplainChart) predExplainChart.destroy();
        predExplainChart = null;
        return;
    }

    const groups = explainData.explanations;
    if (!groups.length) {
        container.innerHTML = '<p class="muted">No explanations available</p>';
        if (predExplainChart) predExplainChart.destroy();
        predExplainChart = null;
        return;
    }

    const labels = groups.map(g => g.group);
    const values = groups.map(g => g.contribution || 0);

    container.innerHTML = '<canvas id="pred-explain-chart" height="220"></canvas>';
    const ctx = document.getElementById('pred-explain-chart');
    if (predExplainChart) predExplainChart.destroy();

    predExplainChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Contribution',
                data: values,
                backgroundColor: values.map(v => v >= 0 ? '#4caf50' : '#e53935')
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#666', font: { size: 10 } }, grid: { display: false } },
                y: { ticks: { color: '#666', font: { size: 9 } }, grid: { display: true } }
            }
        }
    });
}

async function handleTemporalClick({ object }) {
    if (!object) return;
    
    state.selection.temporal = object.hex;
    const content = document.getElementById('temporal-inspector-content');
    
    content.innerHTML = `
        <div class="inspector-meta">
            <span class="label">H3:</span>
            <span class="mono">${object.hex.toString().substring(0, 12)}...</span>
        </div>
        <div class="inspector-stats">
            <div class="stat-card">
                <span class="stat-label">${formatFeatureName(state.temporal.feature)}</span>
                <span class="stat-value">${formatValue(object.value)}</span>
            </div>
        </div>
    `;
    
    // Add selected cell signature to trend chart
    await refreshTrendChart();
}

async function handleStaticClick({ object }) {
    if (!object) return;
    
    state.selection.static = object.hex;
    await renderStaticInspector(object);
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function setupEventListeners() {
    // --- Predictions Controls ---
    const horizonSlider = document.getElementById('pred-horizon-slider');
    if (horizonSlider) {
        horizonSlider.addEventListener('input', async e => {
            const idx = parseInt(e.target.value, 10);
            const horizon = PREDICTION_HORIZONS[idx] || '3m';
            state.predictions.horizon = horizon;
            const horizonDisplay = document.getElementById('pred-horizon-display');
            if (horizonDisplay) horizonDisplay.textContent = horizon;
            state.predictions.cubeLoaded = false;
            state.predictions.cube = {};
            await loadPredictionsDates();
            await loadPredictionsData();
        });
    }

    const learnerSelect = document.getElementById('pred-learner-select');
    if (learnerSelect) {
        learnerSelect.addEventListener('change', async e => {
            state.predictions.learner = e.target.value;
            state.predictions.cubeLoaded = false;
            state.predictions.cube = {};
            await loadPredictionsDates();
            await loadPredictionsData();
        });
    }
    
    document.getElementById('pred-show-layer').addEventListener('change', e => {
        state.predictions.showLayer = e.target.checked;
        renderPredictionsLayer();
    });
    
    document.getElementById('pred-show-events').addEventListener('change', async e => {
        state.predictions.showEvents = e.target.checked;
        if (state.predictions.showEvents) await loadEventsData();
        renderPredictionsLayer();
    });
    
    document.getElementById('pred-opacity').addEventListener('input', e => {
        state.predictions.opacity = parseFloat(e.target.value);
        renderPredictionsLayer();
    });
    
    document.getElementById('pred-clear-selection').addEventListener('click', () => {
        state.selection.predictions = null;
        document.getElementById('pred-inspector-content').innerHTML = '<p class="muted">Click a hex to inspect.</p>';
        if (predTrendChart) {
            predTrendChart.destroy();
            predTrendChart = null;
        }
    });
    const forwardToggle = document.getElementById('pred-forward-toggle');
    if (forwardToggle) {
        forwardToggle.addEventListener('change', () => {
            PRED_FORWARD_ONLY.enabled = forwardToggle.checked;
            refreshPredictionSelection();
        });
    }
    const historyWindowSelect = document.getElementById('pred-history-window');
    if (historyWindowSelect) {
        historyWindowSelect.addEventListener('change', () => {
            const val = historyWindowSelect.value;
            if (val === 'all') {
                PRED_HISTORY_WINDOW.steps = null;
            } else {
                const n = parseInt(val, 10);
                PRED_HISTORY_WINDOW.steps = Number.isFinite(n) ? n : null;
            }
            refreshPredictionSelection();
        });
    }
    
    // --- Temporal Controls ---
    document.getElementById('temporal-feature-select').addEventListener('change', async e => {
        state.temporal.feature = e.target.value;
        stopTemporalPlayback();
        await loadTemporalData();
        await refreshTrendChart();
    });
    
    document.getElementById('temporal-date-slider').addEventListener('input', e => {
        const idx = parseInt(e.target.value);
        state.temporal.selectedDateIndex = idx;
        const date = state.temporal.dates[idx];
        document.getElementById('temporal-date-display').textContent = date || 'No data';
        // Load on input so the map updates immediately as the slider moves
        loadTemporalData();
        refreshTrendChart();
    });
    
    document.getElementById('temporal-date-slider').addEventListener('change', () => {
        if (!isTemporalAutoPlaying) {
            stopTemporalPlayback();
        }
    });
    
    document.getElementById('temporal-show-layer').addEventListener('change', e => {
        state.temporal.showLayer = e.target.checked;
        renderTemporalLayer();
    });
    
    document.getElementById('temporal-opacity').addEventListener('input', e => {
        state.temporal.opacity = parseFloat(e.target.value);
        renderTemporalLayer();
    });

    const temporalPlayBtn = document.getElementById('temporal-play-btn');
    if (temporalPlayBtn) {
        temporalPlayBtn.addEventListener('click', toggleTemporalPlayback);
    }
    
    const compareSelect = document.getElementById('trend-compare-select');
    if (compareSelect) {
        // Populate compare options from curated list
        compareSelect.innerHTML = '<option value=\"\">Compare with...</option>';
        COMPARE_FEATURES.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f;
            opt.textContent = formatFeatureName(f);
            compareSelect.appendChild(opt);
        });
        compareSelect.addEventListener('change', () => {
            const val = compareSelect.value;
            if (!val) {
                activeCompareFeatures.clear();
            } else if (activeCompareFeatures.has(val)) {
                activeCompareFeatures.delete(val);
            } else {
                activeCompareFeatures.add(val);
            }
            // Reset select so same option can be toggled again
            compareSelect.value = '';
            refreshTrendChart();
        });
    }
    
    const normToggle = document.getElementById('trend-normalize-toggle');
    if (normToggle) {
        normToggle.checked = state.temporal.normalize;
        normToggle.addEventListener('change', () => {
            state.temporal.normalize = normToggle.checked;
            refreshTrendChart();
        });
    }

    const maxToggle = document.getElementById('trend-max-toggle');
    if (maxToggle) {
        maxToggle.checked = state.temporal.showMax;
        maxToggle.addEventListener('change', () => {
            state.temporal.showMax = maxToggle.checked;
            refreshTrendChart();
        });
    }

    const minToggle = document.getElementById('trend-min-toggle');
    if (minToggle) {
        minToggle.checked = state.temporal.showMin;
        minToggle.addEventListener('change', () => {
            state.temporal.showMin = minToggle.checked;
            refreshTrendChart();
        });
    }
    
    document.getElementById('temporal-clear-selection').addEventListener('click', () => {
        state.selection.temporal = null;
        document.getElementById('temporal-inspector-content').innerHTML = '<p class="muted">Click a hex to inspect.</p>';
        refreshTrendChart();
    });
    
    // --- Static Controls ---
    document.getElementById('static-feature-select').addEventListener('change', async e => {
        state.static.feature = e.target.value;
        await loadStaticData();
    });
    
    document.getElementById('static-show-layer').addEventListener('change', e => {
        state.static.showLayer = e.target.checked;
        renderStaticLayer();
    });
    
    document.getElementById('static-opacity').addEventListener('input', e => {
        state.static.opacity = parseFloat(e.target.value);
        renderStaticLayer();
    });
    
    document.getElementById('static-clear-selection').addEventListener('click', () => {
        state.selection.static = null;
        document.getElementById('static-inspector-content').innerHTML = '<p class="muted">Click a hex to inspect.</p>';
    });
}

// =============================================================================
// ANIMATION PLAYBACK
// =============================================================================

let temporalPlaybackInterval = null;
let isTemporalAutoPlaying = false;

function stopTemporalPlayback() {
    const btn = document.getElementById('temporal-play-btn');
    isTemporalAutoPlaying = false;
    if (temporalPlaybackInterval) {
        clearInterval(temporalPlaybackInterval);
        temporalPlaybackInterval = null;
    }
    if (btn) {
        btn.innerHTML = '<i data-feather="play"></i><span>Animate</span>';
        if (window.feather) feather.replace();
    }
}

function getNextTemporalIndex(currentIdx) {
    const dates = state.temporal.dates;
    if (!dates.length) return null;
    const step = 1; // advance one timestep (≈2 weeks) per tick
    const nextIdx = currentIdx + step;
    if (nextIdx >= dates.length) return null; // stop at latest
    return nextIdx;
}

function toggleTemporalPlayback() {
    const btn = document.getElementById('temporal-play-btn');
    if (!btn) return;
    const icon = btn.querySelector('i');

    if (temporalPlaybackInterval) {
        stopTemporalPlayback();
        return;
    }

    if (!state.temporal.dates.length) return;

    isTemporalAutoPlaying = true;
    if (btn) {
        btn.innerHTML = '<i data-feather="pause"></i><span>Pause</span>';
    }
    if (window.feather) feather.replace();

    temporalPlaybackInterval = setInterval(async () => {
        const slider = document.getElementById('temporal-date-slider');
        const nextIdx = getNextTemporalIndex(state.temporal.selectedDateIndex);

        if (nextIdx === null) {
            // Reached latest; stop at last index
            stopTemporalPlayback();
            return;
        }

        state.temporal.selectedDateIndex = nextIdx;
        if (slider) slider.value = nextIdx;
        const date = state.temporal.dates[nextIdx];
        document.getElementById('temporal-date-display').textContent = date;

        await loadTemporalData();
        await refreshTrendChart();
    }, 800);
}

// =============================================================================
// UTILITIES
// =============================================================================

async function checkSystemHealth() {
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    
    try {
        const res = await fetch(`${CONFIG.API_URL}/health`);
        const data = await res.json();
        
        if (data.status === 'healthy') {
            statusDot.className = 'status-dot connected';
            statusText.textContent = 'Connected';
        } else {
            throw new Error('Unhealthy');
        }
    } catch (e) {
        statusDot.className = 'status-dot disconnected';
        statusText.textContent = 'Offline';
        showToast('Backend connection failed', 'error');
    }
}

function calculatePercentileStats(values) {
    if (!values.length) return { min: 0, max: 1 };
    
    const sorted = [...values].sort((a, b) => a - b);
    const p5Idx = Math.floor(sorted.length * 0.05);
    const p95Idx = Math.floor(sorted.length * 0.95);
    
    return {
        min: sorted[p5Idx] || sorted[0],
        max: sorted[p95Idx] || sorted[sorted.length - 1]
    };
}

function formatFeatureName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatValue(val) {
    if (val === null || val === undefined) return 'N/A';
    if (Math.abs(val) >= 1000) return val.toFixed(0);
    if (Math.abs(val) >= 1) return val.toFixed(2);
    return val.toFixed(4);
}

async function renderStaticInspector(object) {
    const content = document.getElementById('static-inspector-content');
    if (!content) return;

    const hexStr = object.hex.toString();
    const snapshot = await fetchStaticSnapshot(hexStr);

    let extraStats = '';
    if (snapshot) {
        const entries = Object.entries(snapshot)
            .filter(([k]) => k !== state.static.feature)
            .map(([k, v]) => `
                <div class="stat-card">
                    <span class="stat-label">${formatFeatureName(k)}</span>
                    <span class="stat-value">${formatValue(v)}</span>
                </div>
            `).join('');
        extraStats = entries ? `<div class="inspector-stats">${entries}</div>` : '';
    }

    content.innerHTML = `
        <div class="inspector-meta">
            <span class="label">H3:</span>
            <span class="mono">${hexStr.substring(0, 12)}...</span>
        </div>
        <div class="inspector-stats">
            <div class="stat-card">
                <span class="stat-label">${formatFeatureName(state.static.feature)}</span>
                <span class="stat-value">${formatValue(object.value)}</span>
            </div>
        </div>
        ${extraStats}
    `;
}

function updateTooltip(mapId, { object, x, y }) {
    const tooltip = document.getElementById(`tooltip-${mapId}`);
    if (!tooltip) return;
    
    if (object) {
        tooltip.classList.remove('hidden');
        tooltip.style.left = `${x + 10}px`;
        tooltip.style.top = `${y + 10}px`;
        
        let content = '';
        if (mapId === 'predictions') {
            const exp = (object.expected_fatalities !== undefined && object.expected_fatalities !== null)
                ? object.expected_fatalities
                : (object.risk || 0) * (object.fatalities || 0);
            content = `<strong>Risk:</strong> ${((object.risk || 0) * 100).toFixed(1)}%<br><strong>Fatalities:</strong> ${(object.fatalities || 0).toFixed(1)}<br><strong>Expected:</strong> ${(exp || 0).toFixed(2)}`;
        } else {
            content = `<strong>Value:</strong> ${formatValue(object.value)}`;
        }
        tooltip.innerHTML = content;
    } else {
        tooltip.classList.add('hidden');
    }
}

function updateEventTooltip(mapId, { object, x, y }) {
    const tooltip = document.getElementById(`tooltip-${mapId}`);
    if (!tooltip) return;
    
    if (object?.properties) {
        tooltip.classList.remove('hidden');
        tooltip.style.left = `${x + 10}px`;
        tooltip.style.top = `${y + 10}px`;
        tooltip.innerHTML = `
            <strong>${object.properties.type}</strong><br>
            Date: ${object.properties.date}<br>
            Fatalities: ${object.properties.fatalities}
        `;
    } else {
        tooltip.classList.add('hidden');
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <i data-feather="${type === 'success' ? 'check-circle' : type === 'error' ? 'alert-circle' : 'info'}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    if (window.feather) feather.replace();
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
