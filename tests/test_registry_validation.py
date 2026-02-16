
import pytest
from src.models.registry import ModelRegistryWorkflow
from unittest.mock import MagicMock, patch

class TestRegistryValidation:
    @patch('src.models.registry.ModelEvaluator')
    def test_validate_model_pass(self, mock_evaluator_class):
        # Setup mock
        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator
        mock_evaluator.calculate_metrics.return_value = {'accuracy': 0.85}

        workflow = ModelRegistryWorkflow()

        # Test validation pass
        # We use a mock so we don't need real MLflow or data
        with patch.object(mock_evaluator, 'load_model'), \
             patch.object(mock_evaluator, 'load_test_data'), \
             patch.object(mock_evaluator, 'make_predictions'):

            result = workflow.validate_model(
                model_name="test_model",
                version=1,
                metric="test_accuracy",
                threshold=0.8
            )

            assert result is True
            mock_evaluator.load_model.assert_called_with("models:/test_model/1")

    @patch('src.models.registry.ModelEvaluator')
    def test_validate_model_fail(self, mock_evaluator_class):
        # Setup mock
        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator
        mock_evaluator.calculate_metrics.return_value = {'accuracy': 0.75}

        workflow = ModelRegistryWorkflow()

        # Test validation fail
        with patch.object(mock_evaluator, 'load_model'), \
             patch.object(mock_evaluator, 'load_test_data'), \
             patch.object(mock_evaluator, 'make_predictions'):

            result = workflow.validate_model(
                model_name="test_model",
                version=1,
                metric="test_accuracy",
                threshold=0.8
            )

            assert result is False

    @patch('src.models.registry.ModelRegistryWorkflow.validate_model')
    @patch('src.models.registry.ModelRegistryWorkflow.promote_to_staging')
    @patch('src.models.registry.ModelRegistryWorkflow.register_best_model')
    def test_full_deployment_workflow_validation_call(self, mock_register, mock_promote, mock_validate):
        # Setup mocks
        mock_version = MagicMock()
        mock_version.name = "test_model"
        mock_version.version = 1
        mock_register.return_value = mock_version
        mock_validate.return_value = True

        workflow = ModelRegistryWorkflow()

        workflow.full_deployment_workflow(
            model_name="test_model",
            metric="test_accuracy",
            min_threshold=0.8,
            skip_staging=False
        )

        # Verify promote was called
        mock_promote.assert_called_once()
        # Verify validate was called
        mock_validate.assert_called_once_with(
            model_name="test_model",
            version=1,
            metric="test_accuracy",
            threshold=0.8
        )
