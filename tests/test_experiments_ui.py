import unittest

from app.app_pages.experiment_ui import execute_predictions
from streamlit.testing.v1 import AppTest


class ExperimentsUiTests(unittest.TestCase):
    def test_execute_predictions_runs_only_selected_models(self):
        calls = []

        def predictor(model_name, image):
            calls.append(model_name)
            return {
                "label": "Stop",
                "confidence_label": "Confidence",
                "confidence_display": "99.0%",
            }

        results = execute_predictions(["svm"], object(), predictor)

        self.assertEqual(calls, ["svm"])
        self.assertEqual(results["svm"]["status"], "success")

    def test_execute_predictions_keeps_one_model_failure_isolated(self):
        def predictor(model_name, image):
            if model_name == "svm":
                raise RuntimeError("test failure")
            return {
                "label": "Stop",
                "confidence_label": "Confidence",
                "confidence_display": "99.0%",
            }

        results = execute_predictions(["svm", "random_forest"], object(), predictor)

        self.assertEqual(results["svm"]["status"], "error")
        self.assertEqual(results["random_forest"]["status"], "success")

    def test_summary_card_renders_with_apptest(self):
        def script():
            from app.app_pages.experiment_ui import render_model_summary

            render_model_summary(
                "svm",
                {
                    "status": "success",
                    "result": {
                        "label": "Stop",
                        "confidence_label": "Decision margin",
                        "confidence_display": "1.250",
                    },
                },
            )

        app = AppTest.from_function(script).run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 2)
        self.assertTrue(app.success)


if __name__ == "__main__":
    unittest.main()
