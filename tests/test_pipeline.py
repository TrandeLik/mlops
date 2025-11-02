from src.trainer import Trainer
from unittest.mock import MagicMock


def test_training_pipeline_run_orchestration(mocker, mock_config, mock_image_dataset):

    mocker.patch("src.data_processing.load_and_validate_dataset", return_value={
        "train": mock_image_dataset, "val": mock_image_dataset
    })
    mocker.patch("src.model_utils.get_image_processor", return_value=MagicMock())
    mocker.patch("src.data_processing.create_transform_fn", return_value=lambda x: x)
    mocker.patch("src.data_processing.apply_transform", return_value={
        "train": mock_image_dataset, "val": mock_image_dataset
    })
    mock_create_dataloaders = mocker.patch("src.data_processing.create_dataloaders", return_value={
        "train": "dummy_train_loader", "val": "dummy_val_loader"
    })

    mock_get_model = mocker.patch("src.model_utils.get_model", return_value=MagicMock())
    mock_get_optimizer = mocker.patch("src.training_utils.get_optimizer", return_value=MagicMock())
    mock_run_training = mocker.patch("src.training_utils.run_training_loop", return_value=MagicMock(name="trained_model"))
    mock_save_model = mocker.patch("src.trainer.Trainer._save_model")


    pipeline = Trainer(mock_config)
    pipeline.train()

    mock_create_dataloaders.assert_called_once()

    mock_get_model.assert_called_once()
    mock_get_model.call_args.kwargs['id2label'] == {i: f'class_{i}' for i in range(10)}

    mock_get_optimizer.assert_called_once()
    
    mock_run_training.assert_called_once()
    assert mock_run_training.call_args.args[1] == "dummy_train_loader"
    assert mock_run_training.call_args.args[2] == "dummy_val_loader"
    
    mock_save_model.assert_called_once()

