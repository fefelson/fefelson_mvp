import unittest
from unittest.mock import patch
from datetime import datetime
from src.models.schedules import Schedule, DailySchedule

class TestSchedule(unittest.TestCase):

    @patch.object(Schedule, "read_file")  # Mock file reading to avoid dependencies
    def setUp(self, mock_read_file):
        """Setup runs before each test."""
        # Mocking the return value of read_file to simulate config data
        mock_read_file.return_value = {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        }
        self.schedule = DailySchedule("NBA")  # Creating an instance of DailySchedule

    def test_is_active(self):
        """Test if is_active() correctly identifies active seasons."""
        with patch("datetime.datetime") as mock_datetime:
            # Set up the mock for today() to return datetime objects
            mock_datetime.strptime.side_effect = datetime.strptime  # Preserve strptime behavior
            mock_datetime.today.return_value = datetime(2024, 6, 1, 0, 0, 0)
            self.assertTrue(self.schedule.is_active(), "Should be active on June 1, 2024")

            mock_datetime.today.return_value = datetime(2023, 12, 31, 0, 0, 0)
            self.assertFalse(self.schedule.is_active(), "Should be inactive before Jan 1, 2024")

            mock_datetime.today.return_value = datetime(2025, 1, 1, 0, 0, 0)
            self.assertFalse(self.schedule.is_active(), "Should be inactive after Dec 31, 2024")

if __name__ == "__main__":
    unittest.main()