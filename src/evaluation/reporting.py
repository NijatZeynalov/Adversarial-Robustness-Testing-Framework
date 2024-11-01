from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Report:
    """Container for evaluation results."""

    model_name: str
    attack_name: str
    metrics: Dict[str, float]
    parameters: Dict[str, any]
    timestamp: str


class Reporter:
    """Generates and manages evaluation reports."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
            self,
            report: Report,
            save: bool = True
    ) -> Dict:
        """Generate evaluation report."""
        try:
            report_dict = {
                'model_info': {
                    'name': report.model_name,
                    'attack': report.attack_name
                },
                'metrics': report.metrics,
                'parameters': report.parameters,
                'timestamp': report.timestamp
            }

            if save:
                self.save_report(report_dict)

            return report_dict

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {}

    def save_report(self, report_dict: Dict) -> None:
        """Save report to file."""
        try:
            filename = (
                f"{report_dict['model_info']['name']}_"
                f"{report_dict['model_info']['attack']}_"
                f"{report_dict['timestamp']}.json"
            )

            with open(self.output_dir / filename, 'w') as f:
                json.dump(report_dict, f, indent=4)

        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")

    def load_report(self, filename: str) -> Optional[Dict]:
        """Load report from file."""
        try:
            with open(self.output_dir / filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading report: {str(e)}")
            return None