import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Mock utils and db engines
sys.modules['utils'] = MagicMock()
from utils import logger

# Import functions to audit
from pipeline.processing import process_conflict_features

# Valid H3 cell for tests
VALID_H3 = 600213123849125887 # 0x85283473fffffff approx

class DataAvailabilityAudit(unittest.TestCase):

    def setUp(self):
        # Mock config with publication lags
        self.mock_config = {
            'temporal': {
                'step_days': 14,
                'publication_lags': {
                    'NLP_CrisisWatch': 2, # 2 steps = 28 days
                    'GDELT': 0
                }
            }
        }
        
        # Mock Engine
        self.mock_engine = MagicMock()

    @patch('pipeline.processing.process_conflict_features.pd.read_sql')
    @patch('pipeline.processing.process_conflict_features.inspect')
    @patch('pipeline.processing.process_conflict_features.add_spatial_diffusion_features')
    @patch('pipeline.processing.process_conflict_features.apply_halflife_decay')
    @patch('pipeline.processing.process_conflict_features.apply_halflife_decay_14d')
    def test_crisiswatch_publication_lag(self, mock_decay_14, mock_decay, mock_diff, mock_inspect, mock_read_sql):
        """
        AUDIT: Verify CrisisWatch data is lagged by 2 steps (28 days) in process_conflict_features.
        """
        print("\n--- Auditing CrisisWatch Publication Lag ---")
        
        mock_diff.side_effect = lambda df, *args, **kwargs: df
        mock_decay.side_effect = lambda df, *args, **kwargs: df
        mock_decay_14.side_effect = lambda df, *args, **kwargs: df

        dates = pd.to_datetime(['2022-12-18', '2023-01-01', '2023-01-15', '2023-01-29', '2023-02-12'])
        spine = pd.DataFrame({
            'h3_index': [VALID_H3] * len(dates),
            'date': dates
        })
        
        cw_raw = pd.DataFrame({
            'h3_index': [VALID_H3],
            'date': [pd.Timestamp('2023-01-01')],
            'cw_topic_id': [10],
            'score': [0.5]
        })
        
        mock_read_sql.side_effect = [
            pd.DataFrame({'column_name': ['spatial_confidence_norm']}), 
            cw_raw 
        ]
        mock_inspect.return_value.has_table.return_value = True
        
        # EXECUTE
        result = process_conflict_features.process_crisiswatch(self.mock_engine, spine, [], self.mock_config)
        
        # VERIFY
        target_row = result[result['date'] == pd.Timestamp('2023-01-29')]
        self.assertFalse(target_row.empty, "Target date 2023-01-29 missing from result")
        
        val = target_row['regime_parallel_governance'].iloc[0]
        print(f"Merged Value at 2023-01-29: {val}")
        
        self.assertEqual(val, 0.5)
        print("✓ Publication Lag Audit Passed.")

    @patch('pipeline.processing.process_conflict_features.pd.read_sql')
    @patch('pipeline.processing.process_conflict_features.inspect')
    @patch('pipeline.processing.process_conflict_features.add_spatial_diffusion_features')
    @patch('pipeline.processing.process_conflict_features.apply_halflife_decay')
    @patch('pipeline.processing.process_conflict_features.apply_halflife_decay_14d')
    def test_velocity_start_handling(self, mock_decay_14, mock_decay, mock_diff, mock_inspect, mock_read_sql):
        """
        AUDIT: Verify Narrative Velocity (Topic 99) uses bfill at start.
        """
        print("\n--- Auditing Velocity Start Handling ---")
        mock_inspect.return_value.has_table.return_value = True
        mock_diff.side_effect = lambda df, *args, **kwargs: df
        mock_decay.side_effect = lambda df, *args, **kwargs: df
        mock_decay_14.side_effect = lambda df, *args, **kwargs: df

        # Mock Spine
        dates = pd.to_datetime(['2022-12-18', '2023-01-01', '2023-01-15', '2023-01-29'])
        spine = pd.DataFrame({
            'h3_index': [VALID_H3] * len(dates), 
            'date': dates
        })
        
        config_no_lag = {'temporal': {'step_days': 14, 'publication_lags': {}}}
        
        # Raw Data: T1=0.5 (Jan 01), T2=0.8 (Jan 29)
        # Shift bfill will make Lag1 at T1 = T1 value
        cw_raw_start = pd.DataFrame({
            'h3_index': [VALID_H3, VALID_H3],
            'date': [pd.Timestamp('2022-12-18'), pd.Timestamp('2023-01-01')],
            'cw_topic_id': [99, 99],
            'score': [0.5, 0.8]
        })
        mock_read_sql.side_effect = [
            pd.DataFrame({'column_name': ['spatial_confidence_norm']}), 
            cw_raw_start 
        ]
        
        # EXECUTE
        result = process_conflict_features.process_crisiswatch(self.mock_engine, spine, [], config_no_lag)
        
        v_lag1_t1 = result[result['date'] == pd.Timestamp('2022-12-18')]['narrative_velocity_lag1'].iloc[0]
        v_lag1_t2 = result[result['date'] == pd.Timestamp('2023-01-01')]['narrative_velocity_lag1'].iloc[0]
        
        print(f"Velocity Lag1 T1 (2022-12-18): {v_lag1_t1}")
        print(f"Velocity Lag1 T2 (2023-01-01): {v_lag1_t2}")
        
        self.assertEqual(v_lag1_t1, 0.5)
        self.assertEqual(v_lag1_t2, 0.5)
        print("✓ Velocity Start Handling (bfill) Audit Passed.")

    def test_imputation_logic(self):
        """
        AUDIT: Check Imputation Config Logic (Zero vs Forward Fill)
        """
        print("\n--- Auditing Imputation Configuration ---")
        print("✓ Imputation Logic Audit skipped (requires real config loading).")

if __name__ == '__main__':
    unittest.main()