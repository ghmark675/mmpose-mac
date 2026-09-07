import json
import tempfile
import unittest
from pathlib import Path

from keypoint_config import filter_predictions, load_keypoint_config


class KeypointConfigTest(unittest.TestCase):
    def test_json_export_is_compacted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"visible_joints": ["nose", "right_ankle"]}))
            config = load_keypoint_config(path)
            predictions = [
                {
                    "frame_id": 3,
                    "instances": [
                        {
                            "keypoints": [[i, i] for i in range(17)],
                            "keypoint_scores": list(range(17)),
                        }
                    ],
                }
            ]
            filter_predictions(predictions, config)
            instance = predictions[0]["instances"][0]
            self.assertEqual(instance["keypoint_indices"], [0, 16])
            self.assertEqual(instance["keypoints"], [[0, 0], [16, 16]])
            self.assertEqual(instance["keypoint_scores"], [0, 16])


if __name__ == "__main__":
    unittest.main()
