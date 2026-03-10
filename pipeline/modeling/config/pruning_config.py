# Pruning configuration for stability selection and rescue logic.
# Toggle behavior (e.g., enable_rescue) is controlled in pruning.py.

# CLASS A: Static context features (never pruned)
STATIC_CONTEXT = [
    # Temporal
    'month_sin', 'month_cos', 'is_dry_season',
    # Geographic / Infrastructure
    'dist_to_capital', 'dist_to_border',
    'dist_to_road', 'dist_to_river', 'dist_to_city',
    'dist_to_market_km',
    # Terrain
    'terrain_ruggedness_index', 'elevation_mean', 'slope_mean',
    # EPR (Ethnic Power Relations)
    'epr_excluded_groups_count', 'epr_status_mean',
    'epr_discriminated_groups_count', 'ethnic_group_count',

    # --- NEW: PROTECTED LATE-STARTERS ---
    # These features start too late to pass stability checks in early windows
    # but are critical for modern predictions.
    # IODA (starts 2022-02-01)
    'ioda_outage_score',
    'ioda_data_available',
    # Dynamic World (starts 2015-06-27)
    'landcover_grass',
    'landcover_crops',
    'landcover_trees',
    'landcover_bare',
    'landcover_built',
    'landcover_data_available',
    # IOM DTM (starts 2018-01-31)
    'iom_displacement_count_lag1',
    'iom_data_available',
]

# CLASS C: Family guards — if any child survives, keep the flag
FAMILY_GUARDS = {
    'food_data_available': [
        'price_maize', 'price_rice', 'price_oil',
        'price_sorghum', 'price_cassava', 'price_groundnuts',
        'food_price_index',
    ],
    'iom_data_available': ['iom_displacement_count_lag1'],
    'viirs_data_available': ['ntl_mean', 'ntl_peak', 'ntl_kinetic_delta'],
    'gdelt_data_available': [
        'gdelt_event_count', 'gdelt_predatory_action_decay_30d',
        'gdelt_shock_signal',
    ],
    'econ_data_available': [
        'gold_price_usd_lag1', 'oil_price_usd_lag1',
        'eur_usd_rate_lag1', 'sp500_index_lag1',
    ],
    'landcover_data_available': [
        'landcover_grass', 'landcover_crops', 'landcover_trees',
        'landcover_bare', 'landcover_built',
    ],
}

# CLASS B: Asymmetric rescue bundles — parent rescues listed children
RESCUE_BUNDLES = {
    # VIIRS: signal -> quality
    'ntl_mean': ['ntl_stale_days', 'ntl_trust_frac', 'ntl_peak'],

    # Food Security: price -> recency & shocks
    'price_maize': ['price_maize_recency_days', 'price_maize_shock'],
    'price_rice': ['price_rice_recency_days', 'price_rice_shock'],
    'price_oil': ['price_oil_recency_days', 'price_oil_shock'],
    'price_sorghum': ['price_sorghum_recency_days', 'price_sorghum_shock'],
    'price_cassava': ['price_cassava_recency_days', 'price_cassava_shock'],
    'price_groundnuts': ['price_groundnuts_recency_days', 'price_groundnuts_shock'],

    # Derived: shock -> level
    'price_maize_shock': ['price_maize'],
    'price_rice_shock': ['price_rice'],
    'price_oil_shock': ['price_oil'],
    'price_sorghum_shock': ['price_sorghum'],
    'price_cassava_shock': ['price_cassava'],
    'price_groundnuts_shock': ['price_groundnuts'],

    # Spatial lag -> local
    'cw_score_local_spatial_lag': ['cw_score_local'],
    'gdelt_event_count_spatial_lag': ['gdelt_event_count'],
    'acled_fatalities_spatial_lag': ['fatalities_14d_sum'],
}

# CLASS D: Bidirectional bundles — keep all if any survives
BIDIRECTIONAL_BUNDLES = {
    'mech_gold_pivot': ['mech_gold_pivot_uncertainty'],
    'mech_predatory_tax': ['mech_predatory_tax_uncertainty'],
    'mech_factional_infighting': ['mech_factional_infighting_uncertainty'],
    'mech_collective_punishment': ['mech_collective_punishment_uncertainty'],
}
